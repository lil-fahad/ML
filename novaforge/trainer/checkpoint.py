from __future__ import annotations
import os, tempfile
from pathlib import Path
import torch

def save_checkpoint_atomic(path, payload: dict):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name,suffix=".tmp",dir=path.parent); os.close(fd)
    try:
        torch.save(payload,tmp)
        _=torch.load(tmp,map_location="cpu",weights_only=False)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def load_checkpoint(path):
    return torch.load(path,map_location="cpu",weights_only=False)
