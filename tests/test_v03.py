import torch,pytest
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from novaforge.trainer.config import TrainingConfig
from novaforge.trainer.adapters import SupervisedAdapter
from novaforge.trainer.engine import TinyModel,train_model
from novaforge.trainer.distributed import detect_distributed_env
from novaforge.trainer.checkpoint import load_checkpoint,distributed_checkpoint_available
from novaforge.safety.watchdog import TrainingWatchdog
from novaforge.trainer.governor import NovaGovernor
def small_loaders(n=10,batch=2):
    x=torch.arange(n*16,dtype=torch.float32).reshape(n,16)/100; y=x.sum(1,keepdim=True); return DataLoader(TensorDataset(x,y),batch_size=batch,shuffle=False)
def test_training_config_rejects_invalid_accum():
    with pytest.raises(ValueError):TrainingConfig(grad_accum_steps=0).validate()
def test_generic_trainer_gradient_accum_no_batch_drop(tmp_path):
    out=train_model(TinyModel(),small_loaders(),SupervisedAdapter(nn.functional.mse_loss),config=TrainingConfig(epochs=1,grad_accum_steps=3),checkpoint_path=tmp_path/'x.pt'); assert out['micro_steps']==5 and out['update_steps']==2 and out['samples_seen']==10; ck=load_checkpoint(tmp_path/'x.pt'); assert ck['micro_step']==5 and ck['update_step']==2
def test_generic_validation_runs(tmp_path):
    out=train_model(TinyModel(),small_loaders(8,2),SupervisedAdapter(nn.functional.mse_loss),val_loader=small_loaders(4,2),config=TrainingConfig(epochs=1),checkpoint_path=tmp_path/'v.pt'); assert len(out['validation'])==1 and out['validation'][0]['samples']==4
def test_governor_reversible_and_safety_pruned():
    g=NovaGovernor();g.apply('raise_accum');assert g.state.grad_accum==2;g.apply('lower_accum');assert g.state.grad_accum==1;allowed=g.allowed_actions({'memory_pressure':0.99},True);assert 'raise_lr' not in allowed and 'lower_accum' not in allowed
def test_watchdog_detects_dataloader_stall():
    assert any(e.code=='DATALOADER_STALL' for e in TrainingWatchdog(stall_seconds=1).inspect(1.0,1.0,1.0,2.0))
def test_distributed_env_default(monkeypatch):
    for k in ('WORLD_SIZE','RANK','LOCAL_RANK'):monkeypatch.delenv(k,raising=False)
    assert detect_distributed_env().world_size==1
def test_dcp_feature_detection_is_boolean():assert isinstance(distributed_checkpoint_available(),bool)
def test_benchmark_quality_gate_rejects_fast_but_worse_candidate():
    from novaforge.benchmark import evaluate_candidate
    assert not evaluate_candidate({'mean_samples_per_sec':100,'mean_final_loss':1.0},{'mean_samples_per_sec':140,'mean_final_loss':1.1})['accepted']
