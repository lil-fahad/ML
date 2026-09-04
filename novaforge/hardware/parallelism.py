from __future__ import annotations
from dataclasses import dataclass,asdict
from .profile import HardwareProfile
@dataclass
class ParallelismPlan:
    strategy:str; world_size:int; mixed_precision:str; activation_checkpointing:bool; torch_compile:bool; reasons:list[str]; estimated_parameter_gb:float; estimated_training_state_gb:float; tp_selected:bool=False; pp_selected:bool=False
    def to_dict(self):return asdict(self)
def plan_parallelism(profile,parameter_count,seq_len=512,batch_size=1,architecture_known=False,tensor_parallel_compatible=False,pipeline_parallel_compatible=False):
    if parameter_count<0:raise ValueError('parameter_count must be >= 0')
    param_gb=parameter_count*4/(1024**3); state_gb=parameter_count*16/(1024**3); reasons=[]; mp='bf16' if profile.bf16 else 'fp16' if profile.fp16 else 'fp32'; compile_ok=bool(profile.compile_available and profile.backend in {'cuda','cpu'}); act=False
    if profile.gpu_count<=0:strategy='single_cpu' if profile.backend=='cpu' else 'single_mps'; reasons.append('No CUDA multi-GPU fabric detected; use a single safe device path.')
    elif profile.gpu_count==1:
        strategy='single_cuda'; max_vram=max(profile.vram_gb or [0]); reasons.append('One CUDA device detected; distributed communication would add overhead.'); act=state_gb>0.8*max_vram or param_gb>0.55*max_vram
    else:
        max_vram=max(profile.vram_gb or [0])
        if state_gb<max_vram*0.65:strategy='ddp'; reasons.append('Estimated training state fits per GPU; DDP is the most general multi-GPU strategy.')
        else:strategy='fsdp2'; act=True; reasons.append('Estimated training state exceeds comfortable per-GPU memory; use FSDP2 per-parameter sharding.')
    if seq_len>=4096 or batch_size>=16:act=act or profile.backend=='cuda'; reasons.append('Long sequence or large batch increases activation pressure.')
    if not architecture_known:reasons.append('TP/PP not selected because architecture-specific partition safety is unknown.')
    elif tensor_parallel_compatible and profile.gpu_count>=4:reasons.append('Tensor parallelism is compatible but not auto-selected without an architecture-specific mesh plan.')
    if architecture_known and pipeline_parallel_compatible:reasons.append('Pipeline parallelism requires an explicit stage balance plan; generic auto-partitioning is intentionally disabled.')
    return ParallelismPlan(strategy,max(1,profile.gpu_count),mp,act,compile_ok,reasons,round(param_gb,4),round(state_gb,4))
