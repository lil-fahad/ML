from __future__ import annotations
from dataclasses import dataclass, asdict
from .profile import HardwareProfile

@dataclass
class ParallelismPlan:
    strategy: str
    world_size: int
    mixed_precision: str
    activation_checkpointing: bool
    torch_compile: bool
    reasons: list[str]
    estimated_parameter_gb: float

    def to_dict(self): return asdict(self)

def plan_parallelism(profile: HardwareProfile, parameter_count: int, seq_len: int = 512, batch_size: int = 1) -> ParallelismPlan:
    param_gb = parameter_count * 4 / (1024**3)
    reasons=[]
    mp = 'bf16' if profile.bf16 else 'fp16' if profile.fp16 else 'fp32'
    compile_ok = bool(profile.compile_available and profile.backend in {'cuda','cpu'})
    act_ckpt = False
    if profile.gpu_count <= 0:
        strategy='single_cpu' if profile.backend=='cpu' else 'single_mps'
        reasons.append('No CUDA multi-GPU fabric detected; use a single safe device path.')
    elif profile.gpu_count == 1:
        strategy='single_cuda'
        reasons.append('One CUDA device detected; distributed communication would add overhead.')
        if param_gb > 0.55 * max(profile.vram_gb or [0]): act_ckpt=True
    else:
        max_vram=max(profile.vram_gb or [0])
        if param_gb < max_vram * 0.45:
            strategy='ddp'
            reasons.append('Model fits comfortably per GPU; DDP is the most general multi-GPU strategy.')
        else:
            strategy='fsdp'
            act_ckpt=True
            reasons.append('Parameter footprint is large relative to per-GPU VRAM; shard model state with FSDP.')
    if seq_len >= 4096 or batch_size >= 16:
        act_ckpt = act_ckpt or profile.backend=='cuda'
        reasons.append('Long sequence or large batch raises activation pressure.')
    return ParallelismPlan(strategy, max(1,profile.gpu_count), mp, act_ckpt, compile_ok, reasons, round(param_gb,4))
