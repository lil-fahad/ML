from __future__ import annotations
from dataclasses import dataclass, asdict

@dataclass
class TrainingConfig:
    epochs: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    deterministic: bool = False
    compile_model: bool = False
    compile_mode: str | None = None
    amp: bool = True
    validation_every_epochs: int = 1
    early_stopping_patience: int | None = None
    min_delta: float = 0.0
    checkpoint_every_updates: int = 0
    profile_steps: int = 0
    enable_governor: bool = True
    profile_dir: str = "artifacts/profiler"
    def validate(self):
        if self.epochs < 1: raise ValueError("epochs must be >= 1")
        if self.learning_rate <= 0: raise ValueError("learning_rate must be > 0")
        if self.grad_accum_steps < 1: raise ValueError("grad_accum_steps must be >= 1")
        if self.max_grad_norm <= 0: raise ValueError("max_grad_norm must be > 0")
        if self.validation_every_epochs < 1: raise ValueError("validation_every_epochs must be >= 1")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1: raise ValueError("early_stopping_patience must be >= 1 or None")
        if self.checkpoint_every_updates < 0: raise ValueError("checkpoint_every_updates must be >= 0")
        if self.profile_steps < 0: raise ValueError("profile_steps must be >= 0")
    def to_dict(self): return asdict(self)
