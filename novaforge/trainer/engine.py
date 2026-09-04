from __future__ import annotations
import contextlib,random,time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from .adapters import ModelAdapter,SupervisedAdapter
from .config import TrainingConfig
from .governor import NovaGovernor
from .checkpoint import save_checkpoint_atomic,load_checkpoint
from .telemetry import TelemetryMeter
from .distributed import initialize_distributed,cleanup_distributed,wrap_model,validate_distributed_loader
from ..safety.watchdog import TrainingWatchdog
from ..hardware.profile import detect_hardware
from ..hardware.parallelism import plan_parallelism
class SafeOOMError(RuntimeError):pass
class TinyModel(nn.Module):
    def __init__(self,d=16):super().__init__();self.net=nn.Sequential(nn.Linear(d,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,x):return self.net(x)
def _device(local_rank=0):
    p=detect_hardware();return torch.device(f'cuda:{local_rank}' if p.cuda else 'mps' if p.mps else 'cpu')
def _rng_state():
    d={'python':random.getstate(),'torch':torch.get_rng_state()}
    if torch.cuda.is_available():d['cuda']=torch.cuda.get_rng_state_all()
    return d
def _restore_rng(d):
    if not d:return
    random.setstate(d['python']);torch.set_rng_state(d['torch'])
    if torch.cuda.is_available() and 'cuda' in d:torch.cuda.set_rng_state_all(d['cuda'])
def _set_determinism(enabled):
    if not enabled:return
    torch.use_deterministic_algorithms(True,warn_only=True)
    if torch.cuda.is_available():torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
def _parameter_count(model):return sum(p.numel() for p in model.parameters())
def _maybe_compile(model,config):
    result={'requested':bool(config.compile_model),'enabled':False,'reason':'not_requested'}
    if not config.compile_model:return model,result
    if not hasattr(torch,'compile'):result['reason']='torch_compile_unavailable';return model,result
    try:
        kwargs={}
        if config.compile_mode:kwargs['mode']=config.compile_mode
        model=torch.compile(model,**kwargs);result.update(enabled=True,reason='compiled');return model,result
    except Exception as e:result['reason']=f'compile_fallback:{type(e).__name__}:{e}';return model,result
def _autocast_context(device,config,profile):
    enabled=bool(config.amp and device.type=='cuda');dtype=torch.bfloat16 if enabled and profile.bf16 else torch.float16;return torch.autocast(device_type=device.type,dtype=dtype,enabled=enabled)
def _make_profiler(config):
    if config.profile_steps<=0:return None
    Path(config.profile_dir).mkdir(parents=True,exist_ok=True);activities=[torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():activities.append(torch.profiler.ProfilerActivity.CUDA)
    return torch.profiler.profile(activities=activities,schedule=torch.profiler.schedule(wait=0,warmup=0,active=config.profile_steps,repeat=1),on_trace_ready=torch.profiler.tensorboard_trace_handler(config.profile_dir),record_shapes=True,profile_memory=True,with_stack=False)
def validate_model(model,loader,adapter,device):
    if loader is None:return {'loss':None,'samples':0}
    was=model.training;model.eval();total=0.0;n=0
    try:
        for batch in loader:
            spec=adapter.unpack_batch(batch,device);value,count=adapter.validation_metric(model,spec);total+=value*count;n+=count
    finally:model.train(was)
    return {'loss':total/max(1,n) if n else None,'samples':n}
def train_model(model,train_loader,adapter,*,val_loader=None,config=None,checkpoint_path='artifacts/last.pt',resume_from=None,seed=7,strategy='auto'):
    config=config or TrainingConfig();config.validate();_set_determinism(config.deterministic);random.seed(seed);torch.manual_seed(seed);profile=detect_hardware();dist_ctx=initialize_distributed();device=_device(dist_ctx.local_rank);validate_distributed_loader(train_loader,dist_ctx);plan=plan_parallelism(profile,_parameter_count(model),batch_size=max(1,config.grad_accum_steps));selected=plan.strategy if strategy=='auto' else strategy
    if not dist_ctx.enabled and selected in {'ddp','fsdp','fsdp2'}:selected='single_cuda' if device.type=='cuda' else 'single_mps' if device.type=='mps' else 'single_cpu'
    model=model.to(device);model,compile_info=_maybe_compile(model,config);model=wrap_model(model,selected,dist_ctx,device) if dist_ctx.enabled else model;opt=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay);scheduler=torch.optim.lr_scheduler.ExponentialLR(opt,gamma=0.98);scaler=torch.amp.GradScaler('cuda',enabled=(config.amp and device.type=='cuda'));gov=NovaGovernor(seed);gov.state.grad_accum=config.grad_accum_steps;wd=TrainingWatchdog();meter=TelemetryMeter();update_step=0;micro_step=0;window_micro=0;window_samples=0;window_tokens=0;window_loss_sum=0.0;window_data_latency=0.0;window_started=None;start_epoch=0;best_val=float('inf');bad_epochs=0
    if resume_from:
        if dist_ctx.enabled:raise RuntimeError('Atomic single-file resume is disabled for distributed jobs; use distributed checkpointing for correctness.')
        ck=load_checkpoint(resume_from);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);gov.load_state_dict(ck['governor']);update_step=int(ck.get('update_step',ck.get('global_step',0)));micro_step=int(ck.get('micro_step',0));start_epoch=int(ck.get('epoch',0));best_val=float(ck.get('best_val',best_val));_restore_rng(ck.get('rng'))
        if ck.get('scaler'):scaler.load_state_dict(ck['scaler'])
    losses=[];validation_history=[];events_log=[];samples_seen=0;tokens_seen=0;wall_start=time.perf_counter();profiler=_make_profiler(config);prof_ctx=profiler if profiler is not None else contextlib.nullcontext();opt.zero_grad(set_to_none=True)
    try:
        with prof_ctx:
            for epoch in range(start_epoch,start_epoch+config.epochs):
                if hasattr(getattr(train_loader,'sampler',None),'set_epoch'):train_loader.sampler.set_epoch(epoch)
                iterator=iter(train_loader);data_ready=time.perf_counter();batch_in_epoch=0
                try:loader_len=len(train_loader)
                except Exception:loader_len=None
                while True:
                    try:batch=next(iterator)
                    except StopIteration:break
                    batch_in_epoch+=1;data_latency=time.perf_counter()-data_ready;step_started=time.perf_counter();micro_step+=1;window_micro+=1
                    if window_started is None:window_started=step_started
                    current_accum=max(1,int(gov.state.grad_accum))
                    try:
                        spec=adapter.unpack_batch(batch,device)
                        with _autocast_context(device,config,profile):raw_loss=adapter.forward_loss(model,spec);loss=raw_loss/current_accum
                        if not torch.isfinite(raw_loss):raise RuntimeError('Fail-safe: non-finite loss')
                        if scaler.is_enabled():scaler.scale(loss).backward()
                        else:loss.backward()
                        window_samples+=spec.sample_count;window_tokens+=spec.token_count or 0;window_loss_sum+=float(raw_loss.detach().cpu());window_data_latency+=data_latency
                    except RuntimeError as e:
                        if 'out of memory' in str(e).lower():opt.zero_grad(set_to_none=True);torch.cuda.empty_cache() if torch.cuda.is_available() else None;raise SafeOOMError('OOM detected; partial gradients discarded. Reduce microbatch, enable activation checkpointing, or use FSDP2/sharding.') from e
                        raise
                    is_boundary=window_micro>=current_accum;is_last=loader_len is not None and batch_in_epoch==loader_len
                    if is_boundary or is_last:
                        if scaler.is_enabled():scaler.unscale_(opt)
                        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),config.max_grad_norm);grad_v=float(grad.detach().cpu());loss_v=window_loss_sum/max(1,window_micro)
                        if scaler.is_enabled():scaler.step(opt);scaler.update()
                        else:opt.step()
                        opt.zero_grad(set_to_none=True);update_step+=1;samples_seen+=window_samples;tokens_seen+=window_tokens;tel=meter.record(update_step,micro_step,loss_v,grad_v,window_started or step_started,window_data_latency,window_samples,window_tokens or None,opt.param_groups[0]['lr'],device);ev=wd.inspect(loss_v,grad_v,tel.samples_per_sec,tel.data_latency_s,max(tel.memory_pressure,tel.gpu_memory_pressure or 0.0));events_log.extend([e.__dict__ for e in ev])
                        if any(e.severity=='CRITICAL' for e in ev):raise RuntimeError('Fail-safe watchdog: '+'; '.join(e.code for e in ev if e.severity=='CRITICAL'))
                        if config.enable_governor:
                            action=gov.choose(tel.to_dict(),True);gov.apply(action,True);reward=(-max(0.0,tel.loss_slope))*0.7+min(1.0,tel.samples_per_sec/10000.0)*0.3;gov.update_reward(action,reward)
                            for pg in opt.param_groups:pg['lr']=config.learning_rate*gov.state.lr_multiplier
                        losses.append(loss_v);window_micro=0;window_samples=0;window_tokens=0;window_loss_sum=0.0;window_data_latency=0.0;window_started=None
                        if config.checkpoint_every_updates and update_step%config.checkpoint_every_updates==0 and not dist_ctx.enabled:_save_training_checkpoint(checkpoint_path,model,opt,scheduler,scaler,gov,seed,epoch,update_step,micro_step,best_val,config,device,profile,compile_info)
                    if profiler is not None:profiler.step()
                    data_ready=time.perf_counter()
                scheduler.step()
                if val_loader is not None and ((epoch-start_epoch+1)%config.validation_every_epochs==0):
                    val=validate_model(model,val_loader,adapter,device);val['epoch']=epoch+1;validation_history.append(val)
                    if val['loss'] is not None:
                        if val['loss']<best_val-config.min_delta:best_val=val['loss'];bad_epochs=0
                        else:bad_epochs+=1
                        if config.early_stopping_patience and bad_epochs>=config.early_stopping_patience:break
    finally:cleanup_distributed(dist_ctx)
    elapsed=max(1e-9,time.perf_counter()-wall_start);digest=None
    if not dist_ctx.enabled:digest=_save_training_checkpoint(checkpoint_path,model,opt,scheduler,scaler,gov,seed,start_epoch+config.epochs,update_step,micro_step,best_val,config,device,profile,compile_info)
    return {'device':str(device),'strategy':selected,'distributed':dist_ctx.to_dict(),'epochs_requested':config.epochs,'update_steps':update_step,'micro_steps':micro_step,'final_loss':losses[-1] if losses else None,'samples_seen':samples_seen,'tokens_seen':tokens_seen or None,'samples_per_sec':round(samples_seen/elapsed,2),'validation':validation_history,'best_validation_loss':None if best_val==float('inf') else best_val,'checkpoint':checkpoint_path if digest else None,'checkpoint_sha256':digest,'compile':compile_info,'parallelism_plan':plan.to_dict(),'governor':gov.state_dict(),'watchdog_events':events_log}
def _save_training_checkpoint(path,model,opt,scheduler,scaler,gov,seed,epoch,update_step,micro_step,best_val,config,device,profile,compile_info):
    return save_checkpoint_atomic(path,{'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'governor':gov.state_dict(),'epoch':epoch,'update_step':update_step,'micro_step':micro_step,'global_step':update_step,'best_val':best_val,'seed':seed,'rng':_rng_state(),'device':str(device),'torch_version':torch.__version__,'config':config.to_dict(),'environment':profile.to_dict(),'compile':compile_info})
def train_demo(epochs=2,seed=7,checkpoint_path='artifacts/last.pt',resume_from=None):
    random.seed(seed);torch.manual_seed(seed);x=torch.randn(1024,16);y=x.sum(1,keepdim=True)+0.1*torch.randn(1024,1);vx=torch.randn(256,16);vy=vx.sum(1,keepdim=True)+0.1*torch.randn(256,1);train_loader=DataLoader(TensorDataset(x,y),batch_size=32,shuffle=True,pin_memory=torch.cuda.is_available());val_loader=DataLoader(TensorDataset(vx,vy),batch_size=64,shuffle=False,pin_memory=torch.cuda.is_available());out=train_model(TinyModel(),train_loader,SupervisedAdapter(nn.functional.mse_loss),val_loader=val_loader,config=TrainingConfig(epochs=epochs),checkpoint_path=checkpoint_path,resume_from=resume_from,seed=seed);out['epochs']=epochs;out['steps']=out['update_steps'];return out
