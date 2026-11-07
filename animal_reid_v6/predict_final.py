"""Replay the actually selected V6 model on CPU without retraining adapters.

Requires the V3 cache and the V6 calibrated/frozen configuration. GPU experiments
are not needed by the selected control-ensemble branch.
"""
import json
import pickle
import numpy as np
import pandas as pd
from experiment import V3, RUN, SPECIES, metadata, load_banks, predict, digest
from model import normalized_mean


def main():
    frozen = json.loads((RUN / 'frozen.json').read_text())
    cfgs = frozen['selected']
    with (RUN / 'calibrators.pkl').open('rb') as f:
        calibrators = pickle.load(f)
    base = json.loads((V3 / 'selected_config.json').read_text())
    out = pd.read_csv(V3 / 'submission.csv')
    m, _ = metadata()
    ids, x = load_banks()
    lookup = {int(k): i for i, k in enumerate(ids)}
    for name in SPECIES:
        cfg = cfgs[name]
        if cfg == {'kind': 'baseline'}:
            continue
        assert cfg.get('branch') == 'control', 'Use full experiment for a learned branch'
        o = np.load(V3 / 'scores' / f'production_{name}_{base[name]["family"]}.npz')
        qi, ri = o['qi'], o['ri']
        a = normalized_mean(x[[lookup[int(k)] for k in qi]])
        b = normalized_mean(x[[lookup[int(k)] for k in ri]])
        iso = calibrators[name]['control']
        nq = iso.predict((a @ a.T).ravel()).reshape(len(a), len(a)).astype(np.float32)
        nr = iso.predict((a @ b.T).ravel()).reshape(len(a), len(b)).astype(np.float32)
        np.fill_diagonal(nq, 1.)
        assert base[name]['rerank'] == 0.
        case = {'sq': o['sq'], 'sr': o['sr'], 'refs': m.loc[ri, 'identity'].to_numpy(),
                'extra': {'control': (nq, nr)}}
        labels = predict(case, cfg, base[name])
        values = {int(i): f'cluster_{name}_{label}' for i, label in zip(qi, labels)}
        mask = out.image_id.isin(qi)
        out.loc[mask, 'cluster'] = out.loc[mask, 'image_id'].map(values)
    path = RUN / 'submission_reproduced.csv'
    out.to_csv(path, index=False)
    pd.testing.assert_frame_equal(out, pd.read_csv(RUN / 'submission.csv'))
    assert digest(path) == digest(RUN / 'submission.csv')
    print(f'CPU inference exactly reproduced {len(out)} rows, SHA-256 {digest(path)}', flush=True)


if __name__ == '__main__':
    main()
