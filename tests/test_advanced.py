import pytest
from novaforge.hardware.profile import detect_hardware
from novaforge.hardware.parallelism import plan_parallelism
from novaforge.trainer.engine import train_demo
from novaforge.trainer.checkpoint import load_checkpoint, CorruptCheckpointError, verify_checkpoint

def test_parallelism_plan():
    p=detect_hardware(); plan=plan_parallelism(p,10_000_000)
    assert plan.strategy in {'single_cpu','single_mps','single_cuda','ddp','fsdp'}
    assert plan.mixed_precision in {'fp32','fp16','bf16'}

def test_checkpoint_integrity_and_resume(tmp_path):
    p=tmp_path/'a.pt'
    first=train_demo(1,checkpoint_path=str(p)); assert verify_checkpoint(p)
    second=train_demo(1,checkpoint_path=str(tmp_path/'b.pt'),resume_from=str(p))
    assert second['steps']>first['steps']

def test_corrupt_checkpoint_detected(tmp_path):
    p=tmp_path/'a.pt'; train_demo(1,checkpoint_path=str(p))
    data=p.read_bytes(); p.write_bytes(data[:-32]+b'X'*32)
    with pytest.raises(CorruptCheckpointError): load_checkpoint(p)
