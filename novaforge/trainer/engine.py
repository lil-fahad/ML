from __future__ import annotations
import random, time
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .governor import NovaGovernor
from .checkpoint import save_checkpoint_atomic
from ..safety.watchdog import TrainingWatchdog
from ..hardware.profile import detect_hardware

class TinyModel(nn.Module):
    def __init__(self,d=16):
        super().__init__(); self.net=nn.Sequential(nn.Linear(d,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,x): return self.net(x)

def _device():
    p=detect_hardware(); return torch.device("cuda" if p.cuda else "mps" if p.mps else "cpu")

def train_demo(epochs=2,seed=7,checkpoint_path="artifacts/last.pt"):
    random.seed(seed); torch.manual_seed(seed); dev=_device()
    x=torch.randn(1024,16); y=x.sum(dim=1,keepdim=True)+0.1*torch.randn(1024,1)
    loader=DataLoader(TensorDataset(x,y),batch_size=32,shuffle=True,pin_memory=(dev.type=="cuda"))
    model=TinyModel().to(dev); opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    gov=NovaGovernor(seed); wd=TrainingWatchdog(); losses=[]; step=0; start=time.perf_counter()
    for _ in range(epochs):
        for xb,yb in loader:
            xb=xb.to(dev); yb=yb.to(dev); opt.zero_grad(set_to_none=True)
            pred=model(xb); loss=nn.functional.mse_loss(pred,yb)
            if not torch.isfinite(loss): raise RuntimeError("Fail-safe: non-finite loss")
            loss.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            events=wd.inspect(float(loss.detach().cpu()),float(grad.detach().cpu()))
            if any(e.severity=="CRITICAL" for e in events): raise RuntimeError("Fail-safe watchdog: "+"; ".join(e.code for e in events))
            opt.step(); losses.append(float(loss.detach().cpu())); step+=1
            slope=(losses[-1]-losses[-2]) if len(losses)>1 else 0.0
            action=gov.choose({"loss_slope":slope,"memory_pressure":0.0},True); gov.apply(action,True)
            for pg in opt.param_groups: pg["lr"]=1e-3*gov.state.lr_multiplier
    elapsed=time.perf_counter()-start
    save_checkpoint_atomic(checkpoint_path,{"model":model.state_dict(),"optimizer":opt.state_dict(),"governor":gov.state_dict(),"global_step":step,"seed":seed})
    return {"device":str(dev),"epochs":epochs,"steps":step,"final_loss":losses[-1],"samples_per_sec":round(1024*epochs/elapsed,2),"checkpoint":checkpoint_path}
