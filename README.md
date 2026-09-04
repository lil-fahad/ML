# NovaForge AI

NovaForge AI combines a browser-first evidence collector with NovaTrain, an adaptive training engine.

Pipeline:

`Browser Research -> Evidence/Provenance -> Dataset Gate -> NovaTrain -> Checkpoints/Doctor/Benchmarks`

The project is local-first, cross-platform, and does not require search APIs. Browser collection is optional and uses Playwright when installed.

## CLI

```bash
python novaforge_ai.py capabilities
python novaforge_ai.py os-choice
python novaforge_ai.py doctor
python novaforge_ai.py collect --topic "digital human twin"
python novaforge_ai.py train --epochs 2
python novaforge_ai.py benchmark
```

## Safety

The collector does not bypass CAPTCHA, paywalls, login requirements, robots.txt, or private-network boundaries. The trainer prioritizes correctness and fail-safe behavior over speed.
