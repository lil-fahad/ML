from __future__ import annotations
from dataclasses import dataclass, asdict
import os, platform, shutil
import psutil
import torch

@dataclass
class HardwareProfile:
    os: str; os_version: str; arch: str; cpu_cores: int; ram_gb: float; storage_free_gb: float
    torch_version: str; cuda: bool; cuda_version: str | None; gpu_count: int; gpu_names: list[str]
    mps: bool; bf16: bool; fp16: bool; tf32: bool; compile_available: bool; distributed_available: bool; backend: str
    def to_dict(self): return asdict(self)

def detect_hardware() -> HardwareProfile:
    usage=shutil.disk_usage(os.getcwd()); cuda=bool(torch.cuda.is_available()); gpu_count=torch.cuda.device_count() if cuda else 0
    names=[torch.cuda.get_device_name(i) for i in range(gpu_count)] if cuda else []
    mps=bool(getattr(torch.backends,"mps",None) and torch.backends.mps.is_available())
    bf16=False; tf32=False
    if cuda:
        try: bf16=bool(torch.cuda.is_bf16_supported())
        except Exception: bf16=False
        tf32=bool(getattr(torch.backends.cuda.matmul,"allow_tf32",False))
    backend="cuda" if cuda else "mps" if mps else "cpu"
    return HardwareProfile(platform.system(),platform.version(),platform.machine(),os.cpu_count() or 1,
        round(psutil.virtual_memory().total/2**30,2),round(usage.free/2**30,2),torch.__version__,cuda,torch.version.cuda,
        gpu_count,names,mps,bf16,(cuda or mps),tf32,hasattr(torch,"compile"),torch.distributed.is_available(),backend)
