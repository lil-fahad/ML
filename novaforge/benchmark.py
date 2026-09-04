from __future__ import annotations
from .trainer.engine import train_demo

def run_benchmark(repeats=3):
    runs=[]
    for i in range(repeats):
        runs.append(train_demo(1,seed=100+i,checkpoint_path=f'artifacts/bench_{i}.pt'))
    s=[r['samples_per_sec'] for r in runs]
    return {'repeats':repeats,'mean_samples_per_sec':round(sum(s)/len(s),2),'min_samples_per_sec':min(s),'max_samples_per_sec':max(s),'runs':runs}
