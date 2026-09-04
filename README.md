# NovaForge AI 0.3.0

NovaForge AI fuses a browser-first evidence collector with a safe adaptive PyTorch training engine.

`Browser Research -> Evidence/Provenance -> Dataset Gate -> Model Plugin -> NovaTrain Engine -> Watchdog/Governor -> Checkpoints/Doctor/Benchmark`

## Current capabilities

- Cross-platform CPU/CUDA/MPS capability detection.
- Parallelism planner for single-device, DDP, and FSDP2. TP/PP are not auto-selected without an architecture-specific partition plan.
- Generic `TrainingBundle` / `ModelAdapter` plugin API; core training is no longer tied to TinyModel.
- Correct gradient accumulation including partial final windows; Governor changes accumulation only at optimizer-update boundaries.
- CUDA AMP with BF16 preference when supported, optional `torch.compile` with fallback, validation, early stopping, profiler hooks and telemetry.
- Fail-safe OOM behavior: partial gradients are discarded and training stops instead of retrying an unsafe backward pass.
- Atomic checkpoints with SHA-256 manifests, RNG/config/environment/Governor state, resume and corruption detection.
- Distributed environment initialization plus a correctness guard requiring `DistributedSampler` for multi-rank DataLoader training.
- PyTorch Distributed Checkpoint hooks for multi-rank state.
- Baseline-vs-Nova benchmark with a quality gate: speed alone is not accepted as an improvement.

## Install

```bash
pip install -e .
pip install -e '.[data]'
pip install -e '.[browser]'
python -m playwright install chromium
```

## CLI

```bash
python novaforge_ai.py capabilities
python novaforge_ai.py os-choice
python novaforge_ai.py doctor
python novaforge_ai.py plan --params 7000000000 --seq-len 4096
python novaforge_ai.py train --epochs 2
python novaforge_ai.py train-plugin --factory novaforge.examples.regression:build --epochs 2 --grad-accum 4
python novaforge_ai.py benchmark --repeats 3
python novaforge_ai.py collect --topic "digital human twin" --max-pages 20
```

For multi-GPU, launch the plugin command through `torchrun`; NovaForge detects `WORLD_SIZE/RANK/LOCAL_RANK`. The plugin must provide a correctly sharded loader/`DistributedSampler` so duplicate-data training cannot happen silently.

Correctness > safety > model quality > stability > performance > autonomy. NovaForge does not claim an optimization is better unless both quality and benchmark gates pass.
