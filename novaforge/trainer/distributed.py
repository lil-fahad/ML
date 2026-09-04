from __future__ import annotations
from dataclasses import dataclass,asdict
import os,torch
@dataclass
class DistributedContext:
    enabled:bool; rank:int=0; local_rank:int=0; world_size:int=1; backend:str|None=None; initialized_here:bool=False
    def to_dict(self): return asdict(self)
def detect_distributed_env():
    world=int(os.getenv("WORLD_SIZE","1")); return DistributedContext(world>1,int(os.getenv("RANK","0")),int(os.getenv("LOCAL_RANK","0")),world)
def initialize_distributed():
    ctx=detect_distributed_env()
    if not ctx.enabled:return ctx
    if not torch.distributed.is_available(): raise RuntimeError("WORLD_SIZE>1 but torch.distributed is unavailable")
    backend="nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available(): torch.cuda.set_device(ctx.local_rank)
    if not torch.distributed.is_initialized(): torch.distributed.init_process_group(backend=backend,init_method="env://"); ctx.initialized_here=True
    ctx.backend=backend; return ctx
def cleanup_distributed(ctx):
    if ctx.initialized_here and torch.distributed.is_initialized(): torch.distributed.destroy_process_group()
def wrap_model(model,strategy,ctx,device):
    if not ctx.enabled:return model
    if strategy=="ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP
        return DDP(model,device_ids=[ctx.local_rank] if device.type=="cuda" else None)
    if strategy in {"fsdp","fsdp2"}:
        try:
            from torch.distributed.fsdp import fully_shard
            fully_shard(model); return model
        except (ImportError,AttributeError):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            return FSDP(model,device_id=device if device.type=="cuda" else None)
    raise ValueError(f"Unsupported distributed strategy: {strategy}")
def validate_distributed_loader(loader,ctx):
    if not ctx.enabled:return
    from torch.utils.data.distributed import DistributedSampler
    if not isinstance(getattr(loader,"sampler",None),DistributedSampler): raise RuntimeError("Distributed training requires DistributedSampler (or an explicitly sharded iterable) to prevent silent duplicate-data training across ranks.")
