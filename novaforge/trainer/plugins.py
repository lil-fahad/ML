from __future__ import annotations
import importlib
from dataclasses import dataclass
from torch import nn
from .adapters import ModelAdapter
@dataclass
class TrainingBundle:
    model:nn.Module; train_loader:object; adapter:ModelAdapter; val_loader:object|None=None
def load_factory(spec,**kwargs):
    if ':' not in spec: raise ValueError("factory must be module:function")
    module_name,func_name=spec.split(':',1); fn=getattr(importlib.import_module(module_name),func_name); bundle=fn(**kwargs)
    if isinstance(bundle,dict): bundle=TrainingBundle(**bundle)
    if not isinstance(bundle,TrainingBundle): raise TypeError("factory must return TrainingBundle or compatible dict")
    return bundle
