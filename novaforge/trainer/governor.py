from __future__ import annotations
from dataclasses import dataclass, asdict
import math, random

@dataclass
class GovernorState:
    grad_accum: int = 1
    lr_multiplier: float = 1.0
    checkpoint_interval: int = 100
    steps_seen: int = 0

class NovaGovernor:
    def __init__(self, seed: int = 7):
        self.state=GovernorState(); self.rng=random.Random(seed)
        self.values={"noop":0.0,"lower_lr":0.0,"raise_accum":0.0}; self.counts={k:0 for k in self.values}
    def choose(self, telemetry: dict, accumulation_boundary: bool = True) -> str:
        self.state.steps_seen+=1
        if telemetry.get("memory_pressure",0)>0.92 and accumulation_boundary: return "raise_accum"
        if telemetry.get("loss_slope",0)>0.2: return "lower_lr"
        total=max(1,sum(self.counts.values())); scored={}
        for a,v in self.values.items():
            n=self.counts[a]; bonus=math.sqrt(2*math.log(total+1)/(n+1)); scored[a]=v+0.1*bonus
        return max(scored,key=scored.get)
    def apply(self, action: str, accumulation_boundary: bool = True):
        if action=="raise_accum" and accumulation_boundary: self.state.grad_accum=min(32,self.state.grad_accum*2)
        elif action=="lower_lr": self.state.lr_multiplier=max(0.25,self.state.lr_multiplier*0.8)
        return asdict(self.state)
    def update_reward(self, action: str, reward: float):
        self.counts[action]+=1; n=self.counts[action]; self.values[action]+=(reward-self.values[action])/n
    def state_dict(self): return {"state":asdict(self.state),"values":self.values,"counts":self.counts}
    def load_state_dict(self,d):
        self.state=GovernorState(**d["state"]); self.values=dict(d["values"]); self.counts=dict(d["counts"])
