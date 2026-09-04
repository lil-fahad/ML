from __future__ import annotations
from dataclasses import dataclass
import math
@dataclass
class WatchdogEvent: severity:str; code:str; message:str
class TrainingWatchdog:
    def __init__(self,explode_factor=8.0,throughput_collapse_factor=0.2,stall_seconds=60.0): self.loss_history=[]; self.throughput_history=[]; self.explode_factor=explode_factor; self.throughput_collapse_factor=throughput_collapse_factor; self.stall_seconds=stall_seconds
    def inspect(self,loss,grad_norm=None,throughput=None,data_latency_s=None,memory_pressure=None):
        events=[]
        if not math.isfinite(loss): return [WatchdogEvent('CRITICAL','NONFINITE_LOSS',f'loss={loss}')]
        if grad_norm is not None and not math.isfinite(grad_norm): events.append(WatchdogEvent('CRITICAL','NONFINITE_GRAD',f'grad_norm={grad_norm}'))
        if self.loss_history:
            baseline=max(1e-12,sum(self.loss_history[-10:])/min(10,len(self.loss_history)))
            if loss>baseline*self.explode_factor: events.append(WatchdogEvent('CRITICAL','EXPLODING_LOSS',f'loss {loss:.4g} vs baseline {baseline:.4g}'))
        if throughput is not None and throughput>0:
            if len(self.throughput_history)>=5:
                base=sum(self.throughput_history[-10:])/min(10,len(self.throughput_history))
                if base>0 and throughput<base*self.throughput_collapse_factor: events.append(WatchdogEvent('WARNING','THROUGHPUT_COLLAPSE',f'{throughput:.2f} vs baseline {base:.2f}'))
            self.throughput_history.append(float(throughput))
        if data_latency_s is not None and data_latency_s>self.stall_seconds: events.append(WatchdogEvent('CRITICAL','DATALOADER_STALL',f'data latency {data_latency_s:.2f}s'))
        if memory_pressure is not None and memory_pressure>0.97: events.append(WatchdogEvent('WARNING','MEMORY_PRESSURE',f'memory pressure={memory_pressure:.3f}'))
        self.loss_history.append(float(loss)); return events
