from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass
class WatchdogEvent:
    severity: str
    code: str
    message: str

class TrainingWatchdog:
    def __init__(self, explode_factor: float = 8.0):
        self.loss_history: list[float] = []
        self.explode_factor = explode_factor
    def inspect(self, loss: float, grad_norm: float | None = None) -> list[WatchdogEvent]:
        events=[]
        if not math.isfinite(loss):
            return [WatchdogEvent("CRITICAL","NONFINITE_LOSS",f"loss={loss}")]
        if grad_norm is not None and not math.isfinite(grad_norm):
            events.append(WatchdogEvent("CRITICAL","NONFINITE_GRAD",f"grad_norm={grad_norm}"))
        if self.loss_history:
            baseline=max(1e-12,sum(self.loss_history[-10:])/min(10,len(self.loss_history)))
            if loss>baseline*self.explode_factor:
                events.append(WatchdogEvent("CRITICAL","EXPLODING_LOSS",f"loss {loss:.4g} vs baseline {baseline:.4g}"))
        self.loss_history.append(float(loss))
        return events
