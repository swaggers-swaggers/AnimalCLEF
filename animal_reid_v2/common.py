"""Shared state, deterministic runs, and atomic resumable artifacts."""
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'animal-clef-2026'
RUN = ROOT / 'runs' / 'rebuild'
RUN.mkdir(parents=True, exist_ok=True)
STOPPED = False

def _stop(*_):
    global STOPPED
    STOPPED = True

signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)

def check_stop():
    if STOPPED or (RUN / 'STOP').exists():
        raise SystemExit('Stopped safely at an artifact boundary; run again to resume.')

def seed(value=20261104):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.set_num_threads(8)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

def atomic_torch(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)

def json_write(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(tmp, path)

def atomic_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **values)
    os.replace(tmp, path)

def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def log(message):
    from datetime import datetime
    print(f'[{datetime.now().isoformat(timespec="seconds")}] {message}', flush=True)
