from __future__ import annotations
from .storage.dataset_gate import DatasetGate

def build_training_dataset(records: list[dict]) -> dict:
    gate=DatasetGate(); accepted=[]; rejected=[]
    for r in records:
        d=gate.evaluate(r)
        (accepted if d.accepted else rejected).append({"record":r,"decision":d.__dict__})
    return {"accepted":accepted,"rejected":rejected,"accepted_count":len(accepted),"rejected_count":len(rejected)}
