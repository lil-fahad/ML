from __future__ import annotations
import hashlib
from dataclasses import dataclass

@dataclass
class GateDecision:
    accepted: bool
    reason: str
    fingerprint: str

def fingerprint(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()

class DatasetGate:
    def __init__(self,min_chars=50,min_quality=0.45,min_relevance=0.05):
        self.min_chars=min_chars; self.min_quality=min_quality; self.min_relevance=min_relevance; self.seen=set()
    def evaluate(self,record: dict) -> GateDecision:
        text=str(record.get("text","")); fp=fingerprint(text)
        if len(text)<self.min_chars: return GateDecision(False,"too_short",fp)
        if float(record.get("quality",0))<self.min_quality: return GateDecision(False,"low_quality",fp)
        if float(record.get("relevance",0))<self.min_relevance: return GateDecision(False,"low_relevance",fp)
        if fp in self.seen: return GateDecision(False,"duplicate",fp)
        self.seen.add(fp); return GateDecision(True,"accepted",fp)
