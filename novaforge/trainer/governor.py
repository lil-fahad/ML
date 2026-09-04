from __future__ import annotations
from dataclasses import dataclass,asdict
import math,random
@dataclass
class GovernorState: grad_accum:int=1; lr_multiplier:float=1.0; checkpoint_interval:int=100; steps_seen:int=0
class NovaGovernor:
    ACTIONS=('noop','lower_lr','raise_lr','raise_accum','lower_accum')
    def __init__(self,seed=7): self.state=GovernorState(); self.rng=random.Random(seed); self.values={k:0.0 for k in self.ACTIONS}; self.counts={k:0 for k in self.ACTIONS}
    def allowed_actions(self,telemetry,accumulation_boundary=True):
        allowed=['noop','lower_lr','raise_lr']
        if accumulation_boundary:
            if self.state.grad_accum<32:allowed.append('raise_accum')
            if self.state.grad_accum>1:allowed.append('lower_accum')
        if telemetry.get('memory_pressure',0)>0.92 or (telemetry.get('gpu_memory_pressure') or 0)>0.92: allowed=[a for a in allowed if a not in {'raise_lr','lower_accum'}]
        if telemetry.get('loss_slope',0)>0.2: allowed=[a for a in allowed if a!='raise_lr']
        return allowed
    def choose(self,telemetry,accumulation_boundary=True):
        self.state.steps_seen+=1
        if (telemetry.get('memory_pressure',0)>0.96 or (telemetry.get('gpu_memory_pressure') or 0)>0.96) and accumulation_boundary:return 'raise_accum' if self.state.grad_accum<32 else 'lower_lr'
        if telemetry.get('loss_slope',0)>0.2:return 'lower_lr'
        allowed=self.allowed_actions(telemetry,accumulation_boundary); total=max(1,sum(self.counts.values())); scored={a:self.values[a]+0.1*math.sqrt(2*math.log(total+1)/(self.counts[a]+1)) for a in allowed}; return max(scored,key=scored.get)
    def apply(self,action,accumulation_boundary=True):
        if action not in self.ACTIONS:raise ValueError(f'Unknown governor action: {action}')
        if action=='raise_accum' and accumulation_boundary:self.state.grad_accum=min(32,self.state.grad_accum*2)
        elif action=='lower_accum' and accumulation_boundary:self.state.grad_accum=max(1,self.state.grad_accum//2)
        elif action=='lower_lr':self.state.lr_multiplier=max(0.25,self.state.lr_multiplier*0.8)
        elif action=='raise_lr':self.state.lr_multiplier=min(1.25,self.state.lr_multiplier*1.05)
        return asdict(self.state)
    def update_reward(self,action,reward):
        reward=max(-1.0,min(1.0,float(reward))); self.counts[action]+=1; n=self.counts[action]; self.values[action]+=(reward-self.values[action])/n
    def state_dict(self):return {'state':asdict(self.state),'values':self.values,'counts':self.counts,'rng_state':self.rng.getstate()}
    def load_state_dict(self,d):
        self.state=GovernorState(**d['state']); self.values={k:float(d.get('values',{}).get(k,0.0)) for k in self.ACTIONS}; self.counts={k:int(d.get('counts',{}).get(k,0)) for k in self.ACTIONS}
        if d.get('rng_state') is not None:self.rng.setstate(d['rng_state'])
