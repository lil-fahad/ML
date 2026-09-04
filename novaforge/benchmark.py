from __future__ import annotations
import statistics
from .trainer.engine import train_model
from .trainer.config import TrainingConfig
from .examples.regression import build

def _run_once(seed,governor,checkpoint):
    b=build(seed)
    return train_model(b.model,b.train_loader,b.adapter,val_loader=b.val_loader,config=TrainingConfig(epochs=1,enable_governor=governor),checkpoint_path=checkpoint,seed=seed)
def evaluate_candidate(baseline,candidate,quality_tolerance=0.02,min_speedup=1.0):
    b_sps=float(baseline['mean_samples_per_sec']); c_sps=float(candidate['mean_samples_per_sec']); b_loss=float(baseline['mean_final_loss']); c_loss=float(candidate['mean_final_loss']); speed=c_sps/b_sps if b_sps else 0.0; quality=c_loss/b_loss if b_loss else float('inf'); quality_ok=quality<=1.0+quality_tolerance; speed_ok=speed>=min_speedup; accepted=bool(quality_ok and speed_ok); reasons=[]
    if not quality_ok:reasons.append(f'quality regression: loss ratio {quality:.4f} exceeds {1+quality_tolerance:.4f}')
    if not speed_ok:reasons.append(f'throughput ratio {speed:.4f} below required {min_speedup:.4f}')
    if accepted:reasons.append('candidate satisfies both quality and throughput gates')
    return {'accepted':accepted,'speed_ratio':round(speed,4),'quality_loss_ratio':round(quality,4),'quality_tolerance':quality_tolerance,'min_speedup':min_speedup,'reasons':reasons}
def run_benchmark(repeats=3,quality_tolerance=0.02):
    nova=[];baseline=[]
    for i in range(repeats):baseline.append(_run_once(100+i,False,f'artifacts/baseline_{i}.pt'));nova.append(_run_once(100+i,True,f'artifacts/nova_{i}.pt'))
    def stats(runs):
        s=[r['samples_per_sec'] for r in runs];return {'mean_samples_per_sec':round(statistics.mean(s),2),'min_samples_per_sec':min(s),'max_samples_per_sec':max(s),'mean_final_loss':round(statistics.mean(r['final_loss'] for r in runs),6),'runs':runs}
    b,n=stats(baseline),stats(nova);verdict=evaluate_candidate(b,n,quality_tolerance);return {'repeats':repeats,'baseline':b,'novaforge':n,'verdict':verdict,'scientific_claim':'ACCEPTED improvement' if verdict['accepted'] else 'REJECTED candidate; do not claim improvement.'}
