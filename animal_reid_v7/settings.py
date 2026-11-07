import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
CV = ROOT.parent
DATA = CV / 'animal-clef-2026'
V3 = CV / 'animal_reid_v3/runs/strong'
V4 = CV / 'animal_reid_v4/runs/crossview'
V6 = CV / 'animal_reid_v6/runs/metric_graph'
RUN = ROOT / 'runs/advanced'
RUN.mkdir(parents=True, exist_ok=True)
SPECIES = ['LynxID2025', 'SalamanderID2025', 'SeaTurtleID2022']


def write(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)); tmp.replace(path)


def save(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); torch.save(obj, tmp); tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(x)
    return h.hexdigest()


def log(s): print(time.strftime('[%H:%M:%S] ') + s, flush=True)


def stop():
    if (RUN / 'STOP').exists(): raise SystemExit('Stopped at safe boundary')


def metadata():
    return pd.read_csv(DATA / 'metadata.csv'), pd.read_csv(V3 / 'splits.csv')


def roles():
    m, s = metadata(); r = []
    for name in SPECIES:
        frame = s[(s.dataset == name) & (s.partition == 'calibration')]
        ids = np.sort(frame.identity.unique()); np.random.default_rng(70106).shuffle(ids)
        fit = set(ids[:int(.8 * len(ids))])
        for row in frame.itertuples():
            r.append({'image_id': int(row.image_id), 'identity': row.identity, 'dataset': name,
                      'role': 'fit' if row.identity in fit else 'calibrate'})
    r = pd.DataFrame(r); path = RUN / 'roles.csv'
    if path.exists(): pd.testing.assert_frame_equal(pd.read_csv(path), r)
    else: r.to_csv(path, index=False)
    assert r.groupby('identity').role.nunique().max() == 1
    assert not set(r.identity) & set(s.loc[s.partition != 'calibration', 'identity'])
    return r
