from __future__ import annotations
import argparse, asyncio, json
from novaforge.hardware.profile import detect_hardware
from novaforge.hardware.os_optimizer import choose_os
from novaforge.hardware.parallelism import plan_parallelism
from novaforge.doctor import run_doctor
from novaforge.trainer.engine import train_demo
from novaforge.benchmark import run_benchmark

def main():
    ap=argparse.ArgumentParser(prog='novaforge_ai.py'); sp=ap.add_subparsers(dest='cmd',required=True)
    sp.add_parser('capabilities'); sp.add_parser('os-choice'); sp.add_parser('doctor'); sp.add_parser('benchmark')
    pp=sp.add_parser('plan'); pp.add_argument('--params',type=int,required=True); pp.add_argument('--seq-len',type=int,default=512); pp.add_argument('--batch-size',type=int,default=1)
    t=sp.add_parser('train'); t.add_argument('--epochs',type=int,default=2); t.add_argument('--resume-from'); t.add_argument('--checkpoint',default='artifacts/last.pt')
    c=sp.add_parser('collect'); c.add_argument('--topic',required=True); c.add_argument('--max-pages',type=int,default=10)
    a=ap.parse_args()
    if a.cmd=='capabilities': out=detect_hardware().to_dict()
    elif a.cmd=='os-choice': out=choose_os(detect_hardware())
    elif a.cmd=='doctor': out=run_doctor('.')
    elif a.cmd=='plan': out=plan_parallelism(detect_hardware(),a.params,a.seq_len,a.batch_size).to_dict()
    elif a.cmd=='train': out=train_demo(a.epochs,checkpoint_path=a.checkpoint,resume_from=a.resume_from)
    elif a.cmd=='benchmark': out=run_benchmark(3)
    else:
        from novaforge.collector.browser import collect_topic
        out=asyncio.run(collect_topic(a.topic,a.max_pages))
    print(json.dumps(out,indent=2,default=str,ensure_ascii=False))
if __name__=='__main__': main()
