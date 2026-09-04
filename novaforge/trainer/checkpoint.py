from __future__ import annotations
import hashlib, json, os, tempfile
from pathlib import Path
import torch

class CorruptCheckpointError(RuntimeError):
    pass

def _sha256(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def save_checkpoint_atomic(path, payload: dict):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name,suffix='.tmp',dir=path.parent); os.close(fd); tmp=Path(tmp)
    manifest=path.with_suffix(path.suffix+'.sha256')
    try:
        torch.save(payload,tmp)
        _=torch.load(tmp,map_location='cpu',weights_only=False)
        digest=_sha256(tmp)
        os.replace(tmp,path)
        mtmp=manifest.with_suffix(manifest.suffix+'.tmp')
        mtmp.write_text(json.dumps({'sha256':digest,'bytes':path.stat().st_size}),encoding='utf-8')
        os.replace(mtmp,manifest)
        return digest
    finally:
        if tmp.exists(): tmp.unlink()

def verify_checkpoint(path):
    path=Path(path); manifest=path.with_suffix(path.suffix+'.sha256')
    if not path.exists() or not manifest.exists(): return False
    try:
        expected=json.loads(manifest.read_text(encoding='utf-8'))['sha256']
    except Exception:
        return False
    return _sha256(path)==expected

def load_checkpoint(path, verify=True):
    path=Path(path)
    if verify and not verify_checkpoint(path):
        raise CorruptCheckpointError(f'Checkpoint integrity verification failed: {path}')
    return torch.load(path,map_location='cpu',weights_only=False)
