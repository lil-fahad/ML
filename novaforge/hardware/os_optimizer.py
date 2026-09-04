from __future__ import annotations
from .profile import HardwareProfile

def choose_os(p: HardwareProfile) -> dict:
    scores={"Linux":50.0,"Windows":50.0,"macOS":50.0}; reasons=[]
    if p.gpu_count>1 and p.cuda:
        scores["Linux"]+=35; scores["Windows"]+=10; scores["macOS"]-=30; reasons.append("NVIDIA multi-GPU favors Linux for the broadest distributed stack.")
    elif p.cuda:
        scores["Linux"]+=20; scores["Windows"]+=15; scores["macOS"]-=25; reasons.append("CUDA is best supported on Linux/Windows.")
    elif p.mps:
        scores["macOS"]+=45; scores["Linux"]-=25; scores["Windows"]-=25; reasons.append("Apple Silicon acceleration requires macOS/MPS.")
    else:
        reasons.append("CPU-only workloads rarely justify OS migration for performance alone.")
    recommended=max(scores,key=scores.get); current=p.os; delta=scores.get(recommended,0)-scores.get(current,0)
    should=delta>=20 and p.backend!="cpu"; confidence=round(min(0.98,0.5+max(0,delta)/100),2)
    return {"recommended_os":recommended,"current_os":current,"should_migrate":bool(should),"confidence":confidence,
            "recommended_backend":p.backend,"linux_score":scores["Linux"],"windows_score":scores["Windows"],"macos_score":scores["macOS"],"reasons":reasons}
