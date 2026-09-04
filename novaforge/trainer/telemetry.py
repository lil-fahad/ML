from __future__ import annotations
from dataclasses import dataclass,asdict
import time,psutil,torch
@dataclass
class Telemetry:
    update_step:int; micro_step:int; loss:float; loss_slope:float; grad_norm:float; step_latency_s:float; data_latency_s:float; samples_per_sec:float; tokens_per_sec:float|None; memory_pressure:float; gpu_memory_pressure:float|None; lr:float
    def to_dict(self): return asdict(self)
class TelemetryMeter:
    def __init__(self): self.last_loss=None
    def gpu_pressure(self,device):
        if device.type!='cuda':return None
        try:return float(torch.cuda.memory_reserved(device)/max(1,torch.cuda.get_device_properties(device).total_memory))
        except Exception:return None
    def record(self,update_step,micro_step,loss,grad_norm,started,data_latency_s,samples,tokens,lr,device):
        dt=max(1e-9,time.perf_counter()-started); slope=0.0 if self.last_loss is None else loss-self.last_loss; self.last_loss=loss
        return Telemetry(update_step,micro_step,loss,slope,grad_norm,dt,float(data_latency_s),samples/dt,(tokens/dt if tokens is not None else None),psutil.virtual_memory().percent/100.0,self.gpu_pressure(device),lr)
