from __future__ import annotations
import argparse,asyncio,json
from novaforge.hardware.profile import detect_hardware
from novaforge.hardware.os_optimizer import choose_os
from novaforge.hardware.parallelism import plan_parallelism
from novaforge.doctor import run_doctor
from novaforge.trainer.engine import train_demo,train_model
from novaforge.trainer.config import TrainingConfig
from novaforge.trainer.plugins import load_factory
from novaforge.benchmark import run_benchmark
def _training_config(a):return TrainingConfig(epochs=a.epochs,learning_rate=a.lr,grad_accum_steps=a.grad_accum,deterministic=a.deterministic,compile_model=a.compile,early_stopping_patience=a.early_stopping,checkpoint_every_updates=a.checkpoint_every,profile_steps=a.profile_steps,enable_governor=not a.no_governor)
def _add_train_args(p):
    p.add_argument('--epochs',type=int,default=2);p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--grad-accum',type=int,default=1);p.add_argument('--resume-from');p.add_argument('--checkpoint',default='artifacts/last.pt');p.add_argument('--deterministic',action='store_true');p.add_argument('--compile',action='store_true');p.add_argument('--early-stopping',type=int);p.add_argument('--checkpoint-every',type=int,default=0);p.add_argument('--profile-steps',type=int,default=0);p.add_argument('--no-governor',action='store_true')
def main():
    ap=argparse.ArgumentParser(prog='novaforge_ai.py');sp=ap.add_subparsers(dest='cmd',required=True);sp.add_parser('capabilities');sp.add_parser('os-choice');sp.add_parser('doctor');b=sp.add_parser('benchmark');b.add_argument('--repeats',type=int,default=3);pp=sp.add_parser('plan');pp.add_argument('--params',type=int,required=True);pp.add_argument('--seq-len',type=int,default=512);pp.add_argument('--batch-size',type=int,default=1);t=sp.add_parser('train');_add_train_args(t);tp=sp.add_parser('train-plugin');_add_train_args(tp);tp.add_argument('--factory',required=True);c=sp.add_parser('collect');c.add_argument('--topic',required=True);c.add_argument('--max-pages',type=int,default=10);a=ap.parse_args()
    if a.cmd=='capabilities':out=detect_hardware().to_dict()
    elif a.cmd=='os-choice':out=choose_os(detect_hardware())
    elif a.cmd=='doctor':out=run_doctor('.')
    elif a.cmd=='plan':out=plan_parallelism(detect_hardware(),a.params,a.seq_len,a.batch_size).to_dict()
    elif a.cmd=='train':out=train_demo(a.epochs,checkpoint_path=a.checkpoint,resume_from=a.resume_from)
    elif a.cmd=='train-plugin':
        bundle=load_factory(a.factory);out=train_model(bundle.model,bundle.train_loader,bundle.adapter,val_loader=bundle.val_loader,config=_training_config(a),checkpoint_path=a.checkpoint,resume_from=a.resume_from)
    elif a.cmd=='benchmark':out=run_benchmark(a.repeats)
    else:
        from novaforge.collector.browser import collect_topic
        out=asyncio.run(collect_topic(a.topic,a.max_pages))
    print(json.dumps(out,indent=2,default=str,ensure_ascii=False))
if __name__=='__main__':main()
