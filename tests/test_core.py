import torch
from novaforge.hardware.profile import detect_hardware
from novaforge.hardware.os_optimizer import choose_os
from novaforge.storage.dataset_gate import DatasetGate
from novaforge.safety.watchdog import TrainingWatchdog
from novaforge.trainer.governor import NovaGovernor
from novaforge.trainer.checkpoint import save_checkpoint_atomic, load_checkpoint
from novaforge.collector.browser import public_web_url

def test_hardware_profile():
    p=detect_hardware(); assert p.cpu_cores>=1; assert p.backend in {"cpu","cuda","mps"}

def test_os_optimizer_shape():
    d=choose_os(detect_hardware()); assert d["recommended_os"] in {"Linux","Windows","macOS"}

def test_dataset_gate_dedupe():
    g=DatasetGate(min_chars=3,min_quality=0,min_relevance=0); r={"text":"hello world","quality":1,"relevance":1}
    assert g.evaluate(r).accepted; assert not g.evaluate(r).accepted

def test_watchdog_nan():
    ev=TrainingWatchdog().inspect(float("nan")); assert ev and ev[0].severity=="CRITICAL"

def test_governor_bounds():
    g=NovaGovernor()
    for _ in range(20): g.apply("lower_lr")
    assert g.state.lr_multiplier>=0.25
    for _ in range(20): g.apply("raise_accum")
    assert g.state.grad_accum<=32

def test_checkpoint_roundtrip(tmp_path):
    p=tmp_path/"c.pt"; save_checkpoint_atomic(p,{"x":torch.tensor([1,2])}); d=load_checkpoint(p); assert d["x"].tolist()==[1,2]

def test_public_url_guard():
    assert public_web_url("https://example.com"); assert not public_web_url("http://127.0.0.1/x"); assert not public_web_url("http://192.168.1.1")
