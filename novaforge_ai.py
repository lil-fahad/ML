from __future__ import annotations
import argparse, asyncio, json
from novaforge.hardware.profile import detect_hardware
from novaforge.hardware.os_optimizer import choose_os
from novaforge.doctor import run_doctor
from novaforge.trainer.engine import train_demo

def main():
    ap=argparse.ArgumentParser(prog="novaforge_ai.py")
    sp=ap.add_subparsers(dest="cmd",required=True)
    sp.add_parser("capabilities"); sp.add_parser("os-choice"); sp.add_parser("doctor")
    t=sp.add_parser("train"); t.add_argument("--epochs",type=int,default=2)
    c=sp.add_parser("collect"); c.add_argument("--topic",required=True); c.add_argument("--max-pages",type=int,default=10)
    sp.add_parser("benchmark")
    args=ap.parse_args()
    if args.cmd=="capabilities": print(json.dumps(detect_hardware().to_dict(),indent=2,default=str))
    elif args.cmd=="os-choice": print(json.dumps(choose_os(detect_hardware()),indent=2))
    elif args.cmd=="doctor": print(json.dumps(run_doctor("."),indent=2,default=str))
    elif args.cmd=="train": print(json.dumps(train_demo(args.epochs),indent=2))
    elif args.cmd=="benchmark": print(json.dumps(train_demo(1,checkpoint_path="artifacts/benchmark.pt"),indent=2))
    elif args.cmd=="collect":
        from novaforge.collector.browser import collect_topic
        print(json.dumps(asyncio.run(collect_topic(args.topic,args.max_pages)),indent=2,ensure_ascii=False))

if __name__=="__main__": main()
