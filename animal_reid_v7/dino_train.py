import argparse
import copy
import json
import random
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
from settings import *
from images import body, square, AUGMENT, EvaluationImages
from dino_model import SpatialDINO, queue_contrastive

PLAN = {'epochs': 24, 'steps': 80, 'batch': 16, 'identity_images': 4, 'train_size': 384,
        'rank': 16, 'last_blocks': 8, 'optimizer': 'AdamW', 'backbone_lr': .00015, 'head_lr': .001,
        'weight_decay': .01, 'queue_per_species': 256, 'cosface_margin': .15, 'cosface_scale': 24,
        'selection': 'per-species tune retrieval mAP every 2 epochs, including epoch 0',
        'fit': '80% of calibration identities; remaining 20% reserved for score fitting',
        'input': 'native-aspect body crop; no color enhancement; augment at inference resolution',
        'base_weights': 'local official DINOv3-L timm checkpoint', 'test_training': False}


@torch.inference_mode()
def extract(model, frame, flip=False):
    model.eval(); values = []; ids = []
    for keys, x in DataLoader(EvaluationImages(frame), batch_size=12, num_workers=4, pin_memory=True):
        stop(); x = x.cuda(non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z = model(x)
            if flip: z = F.normalize(z + model(x.flip(-1)), dim=1)
        values.append(z.cpu().numpy()); ids.extend(keys.tolist())
    return np.asarray(ids), np.concatenate(values)


def map_score(z, y):
    scores = z @ z.T; aps = []
    for i in range(len(y)):
        valid = np.arange(len(y)) != i; pos = y[valid] == y[i]
        if pos.any(): aps.append(average_precision_score(pos, scores[i, valid]))
    return float(np.mean(aps))


def train(smoke=False):
    torch.manual_seed(70106); np.random.seed(70106); random.seed(70106); torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    assert torch.cuda.is_available()
    m, split = metadata(); r = roles()
    planpath = RUN / 'dino_plan.json'
    if not smoke:
        if planpath.exists(): assert json.loads(planpath.read_text()) == PLAN
        else: write(PLAN, planpath)
    training = m[m.image_id.isin(r.loc[r.role == 'fit', 'image_id'])]
    rows = list(training.itertuples()); keys = sorted(training.identity.unique()); label = {k: i for i, k in enumerate(keys)}
    groups = {k: np.array([i for i, row in enumerate(rows) if row.identity == k]) for k in keys}
    species_keys = {s: sorted(training.loc[training.dataset == s, 'identity'].unique()) for s in SPECIES}
    valid = {}
    for name in SPECIES:
        frame = m[m.image_id.isin(split.loc[(split.dataset == name) & (split.partition == 'tune'), 'image_id'])]
        valid[name] = frame.groupby(['identity', 'orientation'], dropna=False, sort=True).head(3).groupby('identity', sort=True).head(6)
    model = SpatialDINO().cuda(); head = nn.Linear(2048, len(keys), bias=False).cuda()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{'params': params, 'lr': .00015}, {'params': head.parameters(), 'lr': .001}], weight_decay=.01)
    history = []; best = {s: -1. for s in SPECIES}; start = 0
    memory = {s: (torch.empty(0, 2048, device='cuda'), torch.empty(0, dtype=torch.long, device='cuda')) for s in SPECIES}
    last = RUN / 'dino_last.pt'
    if last.exists() and not smoke:
        ck = torch.load(last, map_location='cuda', weights_only=False)
        model.load_adapter(ck['adapter']); head.load_state_dict(ck['head']); opt.load_state_dict(ck['optimizer'])
        best = ck['best']; history = ck['history']; start = ck['epoch']; memory = ck['memory']
        torch.set_rng_state(ck['rng_cpu'].cpu()); torch.cuda.set_rng_state(ck['rng_cuda'].cpu())
        np.random.set_state(ck['numpy_rng']); random.setstate(ck['python_rng'])
    elif not smoke:
        detail = {}
        for name in SPECIES:
            _, z = extract(model, valid[name]); detail[name] = map_score(z, valid[name].identity.to_numpy()); best[name] = detail[name]
            save({'adapter': model.adapter_state(), 'epoch': 0, 'map': detail[name]}, RUN / f'dino_best_{name}.pt')
        save({'adapter': model.adapter_state()}, RUN / 'dino_initial.pt')
        history.append({'epoch': 0, 'retrieval': detail}); log(f'DINO initialization retrieval {detail}')
    log(f'DINOv3-L image training: {len(training)} images, {len(keys)} identities, {sum(p.numel() for p in params):,} trainable parameters')
    # Cache only 384px PIL crops on CPU; all learning still consumes augmented images.
    images = [square(body(row)[0]) for row in rows]
    started = time.monotonic()
    for epoch in range(start, 1 if smoke else 24):
        rng = np.random.default_rng(170106 + epoch); model.train(); losses = []
        for step in range(3 if smoke else 80):
            stop(); name = SPECIES[(epoch * 80 + step) % 3]
            chosen = rng.choice(species_keys[name], 4, replace=False)
            ix = np.concatenate([rng.choice(groups[k], 4, replace=len(groups[k]) < 4) for k in chosen])
            x = torch.stack([AUGMENT(images[i]) for i in ix]).cuda(); y = torch.tensor([label[rows[i].identity] for i in ix], device='cuda')
            ratio = (epoch * 80 + step) / (24 * 80); warm = min(1., (epoch * 80 + step + 1) / 80)
            decay = warm * (.1 + .9 * (1 + np.cos(np.pi * ratio)) / 2)
            opt.param_groups[0]['lr'] = .00015 * decay; opt.param_groups[1]['lr'] = .001 * decay
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z = model(x); cosine = F.linear(z, F.normalize(head.weight.float(), dim=1))
                logits = 24 * (cosine - .15 * F.one_hot(y, len(keys)))
                loss = F.cross_entropy(logits, y) + .3 * queue_contrastive(z, y, *memory[name])
            if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
            loss.backward(); nn.utils.clip_grad_norm_(params + list(head.parameters()), 5.); opt.step()
            memory[name] = (torch.cat([memory[name][0], z.detach()])[-256:], torch.cat([memory[name][1], y])[-256:])
            losses.append(float(loss.detach()))
            if step % 20 == 0: log(f'DINO epoch {epoch+1}, step {step+1}, loss {losses[-1]:.4f}')
        if smoke:
            updates = [float(p.detach().abs().max()) for n, p in model.named_parameters() if n.endswith('.b')]
            assert updates and min(updates) > 0, 'A low-rank branch failed to receive gradients'
            result = {'passed': True, 'steps': 3, 'losses': losses, 'seconds': time.monotonic()-started,
                      'peak_gpu_gb': torch.cuda.max_memory_allocated()/1024**3,
                      'trainable_parameters': sum(p.numel() for p in params),
                      'all_32_low_rank_branches_updated': len(updates) == 32,
                      'smallest_nonzero_update': min(updates)}
            write(result, RUN / 'dino_smoke.json'); log(str(result)); return
        row = {'epoch': epoch+1, 'loss': float(np.mean(losses))}
        if (epoch+1) % 2 == 0:
            detail = {}
            for name in SPECIES:
                _, z = extract(model, valid[name]); detail[name] = map_score(z, valid[name].identity.to_numpy())
                if detail[name] > best[name]:
                    best[name] = detail[name]; save({'adapter': model.adapter_state(), 'epoch': epoch+1, 'map': best[name]}, RUN / f'dino_best_{name}.pt')
            row['retrieval'] = detail
        history.append(row)
        save({'adapter': model.adapter_state(), 'head': head.state_dict(), 'optimizer': opt.state_dict(),
              'best': best, 'history': history, 'epoch': epoch+1, 'memory': memory,
              'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state(),
              'numpy_rng': np.random.get_state(), 'python_rng': random.getstate()}, last)
        write({'plan': PLAN, 'history': history, 'best': best, 'gpu': torch.cuda.get_device_name(0),
               'peak_gpu_gb': torch.cuda.max_memory_allocated()/1024**3}, RUN / 'dino_history.json')
        log(f'DINO epoch {epoch+1}/24 completed; best mAP {best}')
    log('DINO image-level training completed')


def infer():
    m, _ = metadata(); model = SpatialDINO().cuda()
    initial = torch.load(RUN / 'dino_initial.pt', map_location='cpu', weights_only=True)
    for name in SPECIES:
        frame = m[m.dataset == name]
        for branch in ['initial', 'trained']:
            path = RUN / f'dino_{branch}_{name}.npz'
            if path.exists(): continue
            ck = initial if branch == 'initial' else torch.load(RUN / f'dino_best_{name}.pt', map_location='cpu', weights_only=True)
            model.load_adapter(ck['adapter']); ids, z = extract(model, frame, flip=True)
            np.savez_compressed(path, ids=ids, features=z.astype(np.float32), epoch=np.array(ck.get('epoch', 0)))
            log(f'DINO {name} {branch}: {len(ids)} images extracted')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=['smoke', 'train', 'infer'], default='train'); a = p.parse_args()
    if a.stage == 'infer': infer()
    else: train(a.stage == 'smoke')
