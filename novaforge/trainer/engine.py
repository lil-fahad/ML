from __future__ import annotations
import random, time
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .governor import NovaGovernor
from .checkpoint import save_checkpoint_atomic, load_checkpoint
from .telemetry import TelemetryMeter
from ..safety.watchdog import TrainingWatchdog
from ..hardware.profile import detect_hardware

class TinyModel(nn.Module):
    def __init__(self,d=16):
        super().__init__(); self.net=nn.Sequential(nn.Linear(d,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,x): return self.net(x)

def _device():
    p=detect_hardware(); return torch.device('cuda' if p.cuda else 'mps' if p.mps else 'cpu')

def _rng_state():
    d={'python':random.getstate(),'torch':torch.get_rng_state()}
    if torch.cuda.is_available(): d['cuda']=torch.cuda.get_rng_state_all()
    return d

def _restore_rng(d):
    if not d:return
    random.setstate(d['python']); torch.set_rng_state(d['torch'])
    if torch.cuda.is_available() and 'cuda' in d: torch.cuda.set_rng_state_all(d['cuda'])

def train_demo(epochs=2, seed=7, checkpoint_path='artifacts/last.pt', resume_from=None):
    random.seed(seed); torch.manual_seed(seed); dev=_device()
    x=torch.randn(1024,16); y=x.sum(dim=1,keepdim=True)+0.1*torch.randn(1024,1)
    loader=DataLoader(TensorDataset(x,y),batch_size=32,shuffle=True,pin_memory=(dev.type=='cuda'))
    model=TinyModel().to(dev); opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    scheduler=torch.optim.lr_scheduler.ExponentialLR(opt,gamma=0.98)
    scaler=torch.amp.GradScaler('cuda',enabled=(dev.type=='cuda'))
    gov=NovaGovernor(seed); wd=TrainingWatchdog(); meter=TelemetryMeter(); losses=[]; step=0; start=time.perf_counter()
    if resume_from:
        ck=load_checkpoint(resume_from)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler'])
        gov.load_state_dict(ck['governor']); step=int(ck['global_step']); _restore_rng(ck.get('rng'))
        if ck.get('scaler'): scaler.load_state_dict(ck['scaler'])
    for _ in range(epochs):
        for xb,yb in loader:
            step_start=time.perf_counter(); xb=xb.to(dev,non_blocking=(dev.type=='cuda')); yb=yb.to(dev,non_blocking=(dev.type=='cuda'))
            opt.zero_grad(set_to_none=True)
            amp_dtype=torch.bfloat16 if dev.type=='cuda' and torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(device_type=dev.type,dtype=amp_dtype,enabled=(dev.type=='cuda')):
                pred=model(xb); loss=nn.functional.mse_loss(pred,yb)
            if not torch.isfinite(loss): raise RuntimeError('Fail-safe: non-finite loss')
            if scaler.is_enabled(): scaler.scale(loss).backward(); scaler.unscale_(opt)
            else: loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            loss_v=float(loss.detach().cpu()); grad_v=float(grad.detach().cpu())
            events=wd.inspect(loss_v,grad_v)
            if any(e.severity=='CRITICAL' for e in events): raise RuntimeError('Fail-safe watchdog: '+'; '.join(e.code for e in events))
            if scaler.is_enabled(): scaler.step(opt); scaler.update()
            else: opt.step()
            losses.append(loss_v); step+=1
            tel=meter.record(step,loss_v,grad_v,step_start,len(xb),opt.param_groups[0]['lr'])
            action=gov.choose(tel.to_dict(),True)
            gov.apply(action,True)
            reward=(-max(0.0,tel.loss_slope))*0.7 + min(1.0,tel.samples_per_sec/10000.0)*0.3
            gov.update_reward(action,reward)
            for pg in opt.param_groups: pg['lr']=1e-3*gov.state.lr_multiplier
        scheduler.step()
    elapsed=time.perf_counter()-start
    payload={'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
             'governor':gov.state_dict(),'global_step':step,'seed':seed,'rng':_rng_state(),'device':str(dev),'torch_version':torch.__version__}
    digest=save_checkpoint_atomic(checkpoint_path,payload)
    return {'device':str(dev),'epochs':epochs,'steps':step,'final_loss':losses[-1],
            'samples_per_sec':round(1024*epochs/elapsed,2),'checkpoint':checkpoint_path,'checkpoint_sha256':digest,
            'governor':gov.state_dict()}
