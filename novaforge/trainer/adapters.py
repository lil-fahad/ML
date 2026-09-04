from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
import torch
from torch import nn
@dataclass
class BatchSpec:
    inputs: Any
    targets: Any
    sample_count: int
    token_count: int | None = None
class ModelAdapter:
    def unpack_batch(self,batch,device): raise NotImplementedError
    def forward_loss(self,model,spec): raise NotImplementedError
    def validation_metric(self,model,spec):
        with torch.no_grad(): loss=self.forward_loss(model,spec)
        return float(loss.detach().cpu()),spec.sample_count
class SupervisedAdapter(ModelAdapter):
    def __init__(self,loss_fn:Callable): self.loss_fn=loss_fn
    def unpack_batch(self,batch,device):
        if not isinstance(batch,(tuple,list)) or len(batch)<2: raise ValueError("SupervisedAdapter expects batch=(inputs, targets, ...)")
        x,y=batch[0],batch[1]
        x=x.to(device,non_blocking=device.type=="cuda") if hasattr(x,"to") else x
        y=y.to(device,non_blocking=device.type=="cuda") if hasattr(y,"to") else y
        n=int(x.shape[0]) if hasattr(x,"shape") and len(x.shape) else 1
        return BatchSpec(x,y,n)
    def forward_loss(self,model,spec): return self.loss_fn(model(spec.inputs),spec.targets)
