from __future__ import annotations
from dataclasses import dataclass, asdict
import time, psutil

@dataclass
class Telemetry:
    step:int
    loss:float
    loss_slope:float
    grad_norm:float
    step_latency_s:float
    samples_per_sec:float
    memory_pressure:float
    lr:float
    def to_dict(self): return asdict(self)

class TelemetryMeter:
    def __init__(self): self.last_loss=None
    def record(self, step, loss, grad_norm, started, batch_size, lr):
        dt=max(1e-9,time.perf_counter()-started)
        slope=0.0 if self.last_loss is None else loss-self.last_loss
        self.last_loss=loss
        return Telemetry(step,loss,slope,grad_norm,dt,batch_size/dt,psutil.virtual_memory().percent/100.0,lr)
