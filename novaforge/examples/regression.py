from __future__ import annotations
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from novaforge.trainer.adapters import SupervisedAdapter
from novaforge.trainer.engine import TinyModel
from novaforge.trainer.plugins import TrainingBundle
def build(seed:int=7):
    g=torch.Generator().manual_seed(seed); x=torch.randn(1024,16,generator=g); y=x.sum(1,keepdim=True)+0.1*torch.randn(1024,1,generator=g); vx=torch.randn(256,16,generator=g); vy=vx.sum(1,keepdim=True)+0.1*torch.randn(256,1,generator=g)
    return TrainingBundle(TinyModel(),DataLoader(TensorDataset(x,y),batch_size=32,shuffle=True),SupervisedAdapter(nn.functional.mse_loss),DataLoader(TensorDataset(vx,vy),batch_size=64))
