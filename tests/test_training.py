from novaforge.trainer.engine import train_demo

def test_cpu_training_smoke(tmp_path):
    out=train_demo(1,checkpoint_path=str(tmp_path/"m.pt"))
    assert out["steps"]>0
    assert out["final_loss"]>=0
