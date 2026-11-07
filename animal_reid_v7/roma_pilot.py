"""Assess dense correspondence as identity evidence on calibration pairs only."""
import json
import time
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import to_tensor
from sklearn.metrics import average_precision_score, roc_auc_score
from settings import *
from images import body

COLUMNS = ['overlap_mean', 'overlap_q90', 'fraction_02', 'fraction_05', 'fraction_08', 'fraction_095',
           'cycle_002', 'cycle_005', 'ncc_mean', 'ncc_good', 'joint_05', 'joint_08']


def select_pairs():
    original = np.load(V3 / 'training_v3.npz'); m, s = metadata(); indexed = m.set_index('image_id')
    allowed = set(s.loc[s.partition == 'calibration', 'image_id']); rows = []
    rng = np.random.default_rng(71006)
    for si, name in enumerate(SPECIES):
        p = original['pairs'][original['dataset'] == si]; y = original['y'][original['dataset'] == si]
        g = original['g'][original['dataset'] == si]
        positive = np.flatnonzero(y == 1); negative = np.flatnonzero(y == 0)
        pos = rng.choice(positive, min(24, len(positive)), replace=False)
        # Include difficult global look-alikes, not just trivial random negatives.
        hard = negative[np.argsort(-g[negative, 1])[:max(48, len(negative)//5)]]
        neg = rng.choice(hard, 24, replace=False)
        for i in np.r_[pos, neg]:
            a, b = map(int, p[i]); assert a in allowed and b in allowed
            rows.append({'a': a, 'b': b, 'same': int(y[i]), 'dataset': name,
                         'orientation_a': str(indexed.loc[a, 'orientation']),
                         'orientation_b': str(indexed.loc[b, 'orientation'])})
    return rows


def warp_image(x, warp):
    return F.grid_sample(x, warp, mode='bilinear', padding_mode='zeros', align_corners=False)


def features(pred, im0, im1, mask0, mask1, backwards=False):
    direct, reverse = ('BA', 'AB') if backwards else ('AB', 'BA')
    warp = pred['warp_' + direct].float()
    rev = pred['warp_' + reverse].float()
    overlap = pred['overlap_' + direct][..., 0].float()
    h, w = overlap.shape[-2:]
    a = F.interpolate(im0, (h, w), mode='bilinear', align_corners=False)
    b = warp_image(F.interpolate(im1, (h, w), mode='bilinear', align_corners=False), warp)
    ma = F.interpolate(mask0, (h, w), mode='nearest')[:, 0] > .5
    mb = warp_image(F.interpolate(mask1, (h, w), mode='nearest'), warp)[:, 0] > .5
    valid = ma & mb & (warp.abs().amax(-1) < .995)
    yy, xx = torch.meshgrid((torch.arange(h, device=warp.device)+.5)*2/h-1,
                           (torch.arange(w, device=warp.device)+.5)*2/w-1, indexing='ij')
    grid = torch.stack([xx, yy], -1)[None]
    returned = warp_image(rev.permute(0, 3, 1, 2), warp).permute(0, 2, 3, 1)
    error = (returned-grid).square().sum(-1).sqrt()
    gray0 = a.mean(1, keepdim=True); gray1 = b.mean(1, keepdim=True)
    avg = lambda x: F.avg_pool2d(x, 7, stride=1, padding=3)
    mu0, mu1 = avg(gray0), avg(gray1)
    var0 = (avg(gray0.square())-mu0.square()).clamp_min(1e-6)
    var1 = (avg(gray1.square())-mu1.square()).clamp_min(1e-6)
    ncc = ((avg(gray0*gray1)-mu0*mu1)/(var0*var1).sqrt()).clamp(-1, 1)[:, 0]
    textured = valid & (var0[:, 0] > .0004) & (var1[:, 0] > .0004)
    if valid.sum() < 10: return np.zeros(len(COLUMNS), np.float32)
    ov = overlap[valid]; cy = error[valid]
    corr = ncc[textured]
    joint = overlap * ncc.clamp_min(0) * torch.exp(-error/.03)
    return np.array([float(ov.mean()), float(ov.quantile(.9)), *[float((ov>t).float().mean()) for t in [.2,.5,.8,.95]],
                     float((cy<.02).float().mean()), float((cy<.05).float().mean()),
                     float(corr.mean()) if len(corr) else 0., float((corr>.5).float().mean()) if len(corr) else 0.,
                     float((joint[textured]>.5).float().mean()) if len(corr) else 0.,
                     float((joint[textured]>.8).float().mean()) if len(corr) else 0.], np.float32)


def main():
    from romav2 import RoMaV2
    torch.set_num_threads(8); torch.set_float32_matmul_precision('highest'); torch.manual_seed(71006)
    model = RoMaV2(RoMaV2.Cfg(setting='fast')); model.bidirectional = True
    plan = {'model': 'RoMa v2.0.1', 'commit': '95c9968145c8906b7b59383258e9f73b02853d89',
            'setting': 'fast', 'bidirectional': True, 'columns': COLUMNS,
            'pairs': select_pairs(), 'purpose': 'calibration-only feasibility screening; no audit or test labels'}
    path = RUN / 'roma_pilot_plan.json'
    if path.exists(): assert json.loads(path.read_text()) == plan
    else: write(plan, path)
    m, _ = metadata(); indexed = {int(row.image_id): row for row in m.itertuples()}; cache = {}
    resultfile = RUN / 'roma_pilot_rows.json'; rows = json.loads(resultfile.read_text()) if resultfile.exists() else []
    done = {(r['a'], r['b']) for r in rows}
    def load(key):
        if key not in cache:
            im, mask = body(indexed[key]); im = im.resize((512,512)); mask = cv2.resize(mask.astype(np.uint8), (512,512), interpolation=cv2.INTER_NEAREST)
            cache[key] = (im, to_tensor(im).cuda()[None], torch.tensor(mask.astype(np.float32),device='cuda')[None,None])
        return cache[key]
    for i, pair in enumerate(plan['pairs']):
        if (pair['a'], pair['b']) in done: continue
        stop(); im0, x0, m0 = load(pair['a']); im1, x1, m1 = load(pair['b']); started = time.monotonic()
        with torch.inference_mode():
            pred = model.match(im0, im1)
            values = (features(pred,x0,x1,m0,m1)+features(pred,x1,x0,m1,m0,True))/2
        assert np.isfinite(values).all()
        rows.append({**pair, 'values': values.tolist(), 'seconds': time.monotonic()-started})
        write(rows, resultfile)
        if i % 12 == 0: log(f'RoMa pilot {i+1}/{len(plan["pairs"])}')
    report = {}
    for name in SPECIES:
        group = [r for r in rows if r['dataset'] == name]; y = np.array([r['same'] for r in group]); x = np.array([r['values'] for r in group])
        report[name] = {key: {'auc': float(roc_auc_score(y, x[:,i])), 'ap': float(average_precision_score(y,x[:,i]))} for i,key in enumerate(COLUMNS)}
    report['mean_seconds_per_pair'] = float(np.mean([r['seconds'] for r in rows]))
    report['model_sha256'] = sha(ROOT / 'models/torch_home/hub/checkpoints/romav2.0.1.pt')
    write(report, RUN / 'roma_pilot_report.json'); log(json.dumps(report))


if __name__ == '__main__': main()
