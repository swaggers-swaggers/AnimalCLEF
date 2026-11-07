"""V6: calibration-only GPU metric adapters and identity-resampled graph selection.

All old caches are read-only inputs. Audit labels are used only after freeze.
The historical audit is a development holdout, not a fresh independent test.
"""
import argparse
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent
CV = ROOT.parent
V3 = CV / 'animal_reid_v3/runs/strong'
DATA = CV / 'animal-clef-2026'
RUN = ROOT / 'runs/metric_graph'
RUN.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(CV / 'animal_reid_v3'))
from graph import cluster as old_cluster, attach, rerank
from pipeline import metrics, validation_context
from model import MetricAdapter, supervised_contrastive, normalized_mean, identity_retrieval

SPECIES = ['LynxID2025', 'SalamanderID2025', 'SeaTurtleID2022']
BANKS = ['miewid', 'mega', 'miew_arc', 'dinov3']
PLAN = {
    'version': 1, 'seed': 20261106, 'banks': BANKS,
    'adapter': {'epochs': 16, 'steps_per_epoch': 50, 'identities_per_batch': 16,
                'images_per_identity': 4, 'width': 512, 'hidden': 256,
                'learning_rate': .0005, 'weight_decay': .01,
                'loss': 'supervised contrastive + 0.5 CosFace + 0.5 pair-similarity distillation',
                'checkpoint_epochs': [0, 4, 8, 12, 16],
                'checkpoint_selection': 'tune identity retrieval mAP; includes initialization',
                'fit_identities': '75% of calibration identities, seeded permutation',
                'score_calibration': 'remaining 25% calibration identities; no adapter training',
                'inherited_backbone': 'V3 miew_arc previously trained on all calibration identities'},
    'neural_weights': [0., .1, .25, .4],
    'threshold_offsets': [-.1, 0., .1],
    'topk': [10, 20, 40], 'resolution': [.02, .08, .2],
    'neighborhood': ['mutual', 'union'],
    'selection': 'best full tune candidates (top 8) rerun on 4 identity subsets; '
                 'maximize equal-weight full/subset mean ARI; require full gain >= .002, '
                 'neither full mode loses more than .01, and >=3/4 subset wins',
    'subset_seed': 61006, 'subset_identity_fraction': .75,
    'reference_attachment': 'keep V3 per-species attachment settings fixed',
    'lizard': 'preserve V3 exactly: no local labels',
    'audit': 'historically seen development holdout; freeze before loading audit labels/scores',
    'official_score': None, 'target_cluster_count': None,
}


def write(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stop():
    if (RUN / 'STOP').exists():
        raise SystemExit('STOP requested; complete artifacts retained.')


def log(message):
    print(time.strftime('[%H:%M:%S] ') + message, flush=True)


def metadata():
    return pd.read_csv(DATA / 'metadata.csv').set_index('image_id', drop=False), pd.read_csv(V3 / 'splits.csv')


def initialize():
    path = RUN / 'experiment_plan.json'
    if path.exists():
        assert json.loads(path.read_text()) == PLAN, 'Use a new run for a changed plan'
    else:
        write(PLAN, path)
    m, splits = metadata()
    assert m.index.is_unique and m.loc[m.split == 'test', 'identity'].isna().all()
    roles = []
    for name in SPECIES:
        cal = splits[(splits.dataset == name) & (splits.partition == 'calibration')]
        keys = np.sort(cal.identity.unique())
        np.random.default_rng(20261106).shuffle(keys)
        fit = set(keys[:int(len(keys) * .75)])
        for row in cal.itertuples():
            roles.append({'image_id': int(row.image_id), 'identity': row.identity,
                          'dataset': name, 'role': 'adapter' if row.identity in fit else 'calibrator'})
    roles = pd.DataFrame(roles)
    assert roles.groupby('identity').role.nunique().max() == 1
    for name in SPECIES:
        sets = [set(splits.loc[(splits.dataset == name) & (splits.partition == p), 'identity'])
                for p in ['calibration', 'tune', 'audit']]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    path = RUN / 'calibration_roles.csv'
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(path), roles)
    else:
        roles.to_csv(path, index=False)
    write({'identity_disjoint': True, 'adapter_calibrator_disjoint': True,
           'counts': roles.groupby(['dataset', 'role']).agg(images=('image_id', 'size'),
                     identities=('identity', 'nunique')).reset_index().to_dict('records')}, RUN / 'split_checks.json')


def source_manifest():
    files = [DATA / 'metadata.csv', DATA / 'sample_submission.csv', V3 / 'splits.csv',
             V3 / 'selected_config.json', V3 / 'submission.csv', V3 / 'fusion_v3.pkl']
    files += sorted((V3 / 'scores').glob('*.npz'))
    files += [CV / 'animal_reid_v3' / n for n in ['graph.py', 'pipeline.py']]
    for bank in BANKS:
        files += sorted((V3 / 'features' / bank).glob('part_*.pt'))
    return {str(p.relative_to(CV)): digest(p) for p in files}


def load_banks():
    arrays = []; ids = None
    for bank in BANKS:
        parts = [torch.load(p, map_location='cpu', weights_only=True)
                 for p in sorted((V3 / 'features' / bank).glob('part_*.pt'))]
        keys = torch.cat([p['ids'] for p in parts]).numpy()
        order = np.argsort(keys)
        f = torch.cat([p['features'] for p in parts]).float().numpy()[order]
        if ids is None:
            ids = keys[order]
        else:
            assert np.array_equal(ids, keys[order]), bank
        assert len(np.unique(keys)) == 15483 and np.isfinite(f).all()
        f /= np.maximum(np.linalg.norm(f, axis=-1, keepdims=True), 1e-8)
        arrays.append(f / np.sqrt(len(BANKS)))
        log(f'Loaded aligned bank: {bank}')
    return ids, np.concatenate(arrays, axis=-1).astype(np.float32)


@torch.inference_mode()
def embed(model, x):
    model.eval()
    values = []
    for part in x.split(256):
        shape = part.shape
        values.append(F.normalize(model(part.flatten(0, 1)).reshape(shape[0], shape[1], -1).mean(1), dim=1).cpu().numpy())
    return np.concatenate(values)


def train():
    initialize()
    assert torch.cuda.is_available(), 'GPU required: run outside the restricted environment'
    random.seed(20261106); np.random.seed(20261106); torch.manual_seed(20261106)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    write({'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
           'cuda': torch.version.cuda, 'adapter_training_device': 'cuda'}, RUN / 'environment.json')
    manifest = source_manifest()
    if (RUN / 'source_manifest.json').exists():
        assert manifest == json.loads((RUN / 'source_manifest.json').read_text())
    else:
        write(manifest, RUN / 'source_manifest.json')
    ids, all_x = load_banks()
    lookup = {int(key): i for i, key in enumerate(ids)}
    m, splits = metadata(); roles = pd.read_csv(RUN / 'calibration_roles.csv')
    for name in SPECIES:
        stop()
        feature_path = RUN / f'embeddings_{name}.npz'
        if feature_path.exists():
            log(f'Already complete: {name}'); continue
        torch.manual_seed(20261106)
        frame = m[m.dataset == name]
        positions = np.array([lookup[int(key)] for key in frame.image_id])
        x = torch.tensor(all_x[positions], device='cuda')
        local_lookup = {int(key): i for i, key in enumerate(frame.image_id)}
        fitting = roles[(roles.dataset == name) & (roles.role == 'adapter')]
        keys = sorted(fitting.identity.unique())
        groups = [np.array([local_lookup[int(k)] for k in fitting.loc[fitting.identity == key, 'image_id']]) for key in keys]
        val_ids = splits.loc[(splits.dataset == name) & (splits.partition == 'tune'), 'image_id']
        val = m.loc[val_ids].groupby('identity', sort=True).head(8)
        vi = np.array([local_lookup[int(k)] for k in val.image_id])
        model = MetricAdapter(x.shape[-1]).cuda()
        head = nn.Linear(512, len(keys), bias=False).cuda()
        opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=.0005, weight_decay=.01)
        history = []; best = -1.; start = 0
        bestfile = RUN / f'adapter_{name}.pt'; lastfile = RUN / f'last_{name}.pt'
        if lastfile.exists():
            ck = torch.load(lastfile, map_location='cuda', weights_only=False)
            model.load_state_dict(ck['model']); head.load_state_dict(ck['head']); opt.load_state_dict(ck['optimizer'])
            history = ck['history']; best = ck['best']; start = ck['epoch']
            torch.set_rng_state(ck['rng_cpu'].cpu()); torch.cuda.set_rng_state(ck['rng_cuda'].cpu())
        else:
            value = identity_retrieval(embed(model, x[vi]), val.identity.to_numpy())
            history.append({'epoch': 0, 'tune_retrieval_map': value}); best = value
            torch.save({'state_dict': model.cpu().state_dict(), 'dimensions': x.shape[-1], 'epoch': 0}, bestfile)
            model.cuda()
        log(f'{name}: {len(keys)} adapter identities, {len(fitting)} images; GPU training started')
        for epoch in range(start, 16):
            rng = np.random.default_rng(61106 + epoch)
            model.train(); losses = []
            for step in range(50):
                stop()
                chosen = rng.choice(len(keys), min(16, len(keys)), replace=False)
                ix = np.concatenate([rng.choice(groups[k], 4, replace=len(groups[k]) < 4) for k in chosen])
                views = rng.integers(0, 2, len(ix))
                batch = x[ix, views]; y = torch.tensor(np.repeat(chosen, 4), device='cuda')
                warm = min(1., (epoch * 50 + step + 1) / 50)
                lr = .0005 * warm * (.1 + .9 * (1 + np.cos(np.pi * (epoch * 50 + step) / 800)) / 2)
                for group in opt.param_groups: group['lr'] = lr
                opt.zero_grad(set_to_none=True)
                z = model(batch)
                cosine = F.linear(z, F.normalize(head.weight, dim=1))
                logits = 20 * (cosine - .15 * F.one_hot(y, len(keys)))
                teacher = batch @ batch.T
                loss = supervised_contrastive(z, y) + .5 * F.cross_entropy(logits, y) + .5 * F.mse_loss(z @ z.T, teacher)
                if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
                loss.backward(); nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 5.)
                opt.step(); losses.append(float(loss.detach()))
            entry = {'epoch': epoch + 1, 'loss': float(np.mean(losses))}
            if (epoch + 1) % 4 == 0:
                value = identity_retrieval(embed(model, x[vi]), val.identity.to_numpy())
                entry['tune_retrieval_map'] = value
                if value > best:
                    best = value
                    state = {k: v.cpu() for k, v in model.state_dict().items()}
                    torch.save({'state_dict': state, 'dimensions': x.shape[-1], 'epoch': epoch + 1}, bestfile)
            history.append(entry)
            ck = {'model': model.state_dict(), 'head': head.state_dict(), 'optimizer': opt.state_dict(),
                  'epoch': epoch + 1, 'best': best, 'history': history,
                  'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state()}
            tmp = lastfile.with_suffix('.tmp'); torch.save(ck, tmp); tmp.replace(lastfile)
            write({'history': history, 'best_map': best, 'fit_identity_count': len(keys)}, RUN / f'training_{name}.json')
            log(f'{name}: epoch {epoch + 1}/16 complete, loss {entry["loss"]:.4f}')
        checkpoint = torch.load(bestfile, map_location='cuda', weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        learned = embed(model, x)
        # Raw four-bank ensemble is a no-training control.
        control = normalized_mean(all_x[positions])
        np.savez_compressed(feature_path, ids=frame.image_id.to_numpy(), learned=learned, control=control,
                            epoch=np.array(checkpoint['epoch']))
        del x, model, head, opt
        torch.cuda.empty_cache()
    log('All GPU adapters and inference complete')


def calibrate():
    m, _ = metadata(); roles = pd.read_csv(RUN / 'calibration_roles.csv'); result = {}
    for name in SPECIES:
        o = np.load(RUN / f'embeddings_{name}.npz'); lookup = {int(k): i for i, k in enumerate(o['ids'])}
        frame = m.loc[roles.loc[(roles.dataset == name) & (roles.role == 'calibrator'), 'image_id']]
        # Deterministic cap controls dominance of heavily photographed identities.
        frame = frame.groupby('identity', sort=True).head(12)
        ix = np.array([lookup[int(k)] for k in frame.image_id]); a, b = np.triu_indices(len(ix), 1)
        truth = frame.identity.to_numpy()[a] == frame.identity.to_numpy()[b]
        assert truth.any() and (~truth).any()
        result[name] = {}
        for branch in ['learned', 'control']:
            z = o[branch][ix]; s = (z @ z.T)[a, b]
            result[name][branch] = IsotonicRegression(out_of_bounds='clip').fit(s, truth.astype(float))
    with (RUN / 'calibrators.pkl').open('wb') as f: pickle.dump(result, f)
    return result


def load_cases(partition, name, calibrators):
    m, splits = metadata(); cfg = json.loads((V3 / 'selected_config.json').read_text())[name]
    embeddings = np.load(RUN / f'embeddings_{name}.npz')
    lookup = {int(k): i for i, k in enumerate(embeddings['ids'])}; result = []
    for mode in (['test'] if partition == 'test' else ['discovery', 'mixed']):
        path = V3 / 'scores' / (f'production_{name}_{cfg["family"]}.npz' if partition == 'test' else f'{partition}_{mode}_{name}.npz')
        o = np.load(path); qi, ri = o['qi'], o['ri']
        q = m.loc[qi]; r = m.loc[ri]
        if partition != 'test':
            allowed = set(splits.loc[(splits.dataset == name) & (splits.partition == partition), 'image_id'])
            assert set(qi) | set(ri) <= allowed and not set(qi) & set(ri)
            expected_q, expected_r = validation_context(m.loc[sorted(allowed)].reset_index(drop=True), np.random.default_rng(8391), mode)
            assert expected_q.image_id.tolist() == list(qi) and expected_r.image_id.tolist() == list(ri)
        else:
            assert (q.split == 'test').all() and (r.split == 'train').all()
        base = o['sq'] if partition == 'test' else o['qq_' + cfg['family']]
        sr = o['sr'] if partition == 'test' else o['qr_' + cfg['family']]
        score = rerank(base, 12, cfg['rerank']) if cfg['rerank'] else base
        extra = {}
        for branch in ['learned', 'control']:
            zq = embeddings[branch][[lookup[int(k)] for k in qi]]
            zr = embeddings[branch][[lookup[int(k)] for k in ri]]
            iso = calibrators[name][branch]
            a = iso.predict((zq @ zq.T).ravel()).reshape(len(q), len(q)).astype(np.float32)
            b = iso.predict((zq @ zr.T).ravel()).reshape(len(q), len(r)).astype(np.float32) if len(r) else np.empty((len(q), 0), np.float32)
            np.fill_diagonal(a, 1.); extra[branch] = (a, b)
        result.append({'mode': mode, 'qi': qi, 'ri': ri,
                       'truth': q.identity.to_numpy() if partition != 'test' else None,
                       'refs': r.identity.to_numpy(), 'sq': score, 'sr': sr, 'extra': extra})
    return result


def graph_cluster(score, cfg):
    import igraph as ig
    n = len(score)
    if n < 2: return np.arange(n)
    values = score.copy(); np.fill_diagonal(values, -1.)
    k = min(cfg['topk'], n - 1)
    neighbors = np.argsort(-values, axis=1, kind='stable')[:, :k]
    keep = np.zeros((n, n), bool); keep[np.arange(n)[:, None], neighbors] = True
    keep = (keep & keep.T) if cfg['neighborhood'] == 'mutual' else (keep | keep.T)
    a, b = np.where(np.triu(keep & (values >= cfg['threshold']), 1))
    if not len(a): return np.arange(n)
    weights = (values[a, b] - cfg['threshold'] + .02) / (1 - cfg['threshold'] + .02)
    ig.set_random_number_generator(random.Random(20261104))
    graph = ig.Graph(n=n, edges=list(zip(a.tolist(), b.tolist())), directed=False)
    return np.asarray(graph.community_leiden(objective_function='CPM', weights=weights.tolist(),
                      resolution=cfg['resolution'], n_iterations=-1).membership)


def predict(case, candidate, base):
    weight = candidate.get('weight', 0.)
    sq, sr = case['sq'], case['sr']
    if weight:
        nq, nr = case['extra'][candidate['branch']]
        sq = (1 - weight) * sq + weight * nq
        sr = (1 - weight) * sr + weight * nr
    if candidate['kind'] == 'baseline':
        labels = old_cluster(sq, base); threshold = base['threshold']
    else:
        labels = graph_cluster(sq, candidate); threshold = candidate['threshold']
    return attach(labels, sr, case['refs'], base['attach_threshold'], base['attach_margin'],
                  sq, threshold, base.get('anchor_mode', 'safe'))


def evaluate(cases, candidate, base):
    values = [float(adjusted_rand_score(c['truth'], predict(c, candidate, base))) for c in cases]
    return {'mean': float(np.mean(values)), 'modes': values}


def subset_cases(cases, seed):
    identities = np.unique(cases[0]['truth']); rng = np.random.default_rng(seed)
    keep = rng.choice(identities, max(2, int(.75 * len(identities))), replace=False)
    result = []
    for c in cases:
        a = np.flatnonzero(np.isin(c['truth'], keep)); b = np.flatnonzero(np.isin(c['refs'], keep))
        result.append({'mode': c['mode'], 'qi': c['qi'][a], 'ri': c['ri'][b], 'truth': c['truth'][a],
                       'refs': c['refs'][b], 'sq': c['sq'][np.ix_(a, a)], 'sr': c['sr'][np.ix_(a, b)],
                       'extra': {k: (v[0][np.ix_(a, a)], v[1][np.ix_(a, b)]) for k, v in c['extra'].items()}})
    return result


def candidates(base):
    yield {'kind': 'baseline'}
    # A metric-only branch isolates neural benefits from graph changes.
    for branch, weight in itertools.product(['learned', 'control'], [.1, .25, .4]):
        yield {'kind': 'baseline', 'branch': branch, 'weight': weight}
    for delta, k, resolution, neighborhood in itertools.product(PLAN['threshold_offsets'], PLAN['topk'], PLAN['resolution'], PLAN['neighborhood']):
        graph = {'kind': 'adaptive_leiden', 'threshold': float(np.clip(base['threshold'] + delta, .05, .95)),
                 'topk': k, 'resolution': resolution, 'neighborhood': neighborhood}
        yield graph
        # Same search budgets for trained metric and frozen ensemble control.
        for branch, weight in itertools.product(['learned', 'control'], [.1, .25, .4]):
            yield {**graph, 'branch': branch, 'weight': weight}


def tune():
    assert not (RUN / 'frozen.json').exists(), 'Selection is frozen; use verify/deliver, or a new experiment'
    calibrators = calibrate(); configs = json.loads((V3 / 'selected_config.json').read_text()); selected = {}; reports = {}
    for name in SPECIES:
        cases = load_cases('tune', name, calibrators); base = configs[name]
        baseline = evaluate(cases, {'kind': 'baseline'}, base)
        prior = json.loads((V3 / 'tune_report.json').read_text())[name]['metrics']
        assert np.allclose(baseline['modes'], [prior[k]['ari'] for k in ['discovery', 'mixed']], atol=1e-12)
        trials = []
        for i, candidate in enumerate(candidates(base)):
            stop(); value = evaluate(cases, candidate, base)
            trials.append({'config': candidate, **value})
            if i % 100 == 0: log(f'{name}: tune candidate {i}')
        trials.sort(key=lambda row: -row['mean'])
        subsets = [subset_cases(cases, 61006 + i) for i in range(4)]
        baseline_sub = [evaluate(c, {'kind': 'baseline'}, base)['mean'] for c in subsets]
        best = {'config': {'kind': 'baseline'}, **baseline, 'subsets': baseline_sub,
                'robust_objective': .5 * baseline['mean'] + .5 * float(np.mean(baseline_sub))}
        shortlist = trials[:8]
        # Include best graph-only, metric-only, and frozen-control candidates as ablations.
        for predicate in [lambda c: not c.get('weight'), lambda c: c.get('branch') == 'learned',
                          lambda c: c.get('branch') == 'control', lambda c: c['kind'] == 'baseline']:
            candidate = next(row for row in trials if predicate(row['config']))
            if candidate not in shortlist: shortlist.append(candidate)
        for row in shortlist:
            row['subsets'] = [evaluate(c, row['config'], base)['mean'] for c in subsets]
            row['robust_objective'] = .5 * row['mean'] + .5 * float(np.mean(row['subsets']))
            row['eligible'] = bool(row['mean'] >= baseline['mean'] + .002 and
                                  np.min(np.array(row['modes']) - baseline['modes']) >= -.01 and
                                  np.sum(np.array(row['subsets']) > np.array(baseline_sub) + 1e-8) >= 3)
            if row['eligible'] and row['robust_objective'] > best['robust_objective']:
                best = copy.deepcopy(row)
        selected[name] = best['config']
        reports[name] = {'baseline': baseline, 'baseline_subsets': baseline_sub, 'selected': best,
                         'shortlist': shortlist, 'trials': trials}
        write(reports, RUN / 'tune_search.json'); write(selected, RUN / 'selected_config.json')
        log(f'{name}: selected on tune only; baseline {baseline["mean"]:.4f}, selected {best["mean"]:.4f}')
    files = [ROOT / 'experiment.py', ROOT / 'model.py', RUN / 'calibrators.pkl', RUN / 'selected_config.json',
             RUN / 'experiment_plan.json', RUN / 'calibration_roles.csv', RUN / 'source_manifest.json']
    files += list(RUN.glob('embeddings_*.npz')) + list(RUN.glob('adapter_*.pt'))
    write({'files': {str(p.relative_to(CV)): digest(p) for p in files}, 'selected': selected,
           'selection_used_audit': False}, RUN / 'frozen.json')


def verify_frozen():
    frozen = json.loads((RUN / 'frozen.json').read_text())
    for key, sha in frozen['files'].items():
        assert digest(CV / key) == sha, f'Frozen input changed: {key}'
    assert source_manifest() == json.loads((RUN / 'source_manifest.json').read_text()), 'V3 input changed'
    return frozen


def deliver():
    frozen = verify_frozen()
    with (RUN / 'calibrators.pkl').open('rb') as f: calibrators = pickle.load(f)
    configs = json.loads((V3 / 'selected_config.json').read_text()); report = {}; predictions = []
    baseline_values = []; selected_values = []
    for name in SPECIES:
        base = configs[name]; selected = frozen['selected'][name]; report[name] = {}
        cases = load_cases('audit', name, calibrators)
        for case in cases:
            a = predict(case, {'kind': 'baseline'}, base); b = predict(case, selected, base)
            ma, mb = metrics(case['truth'], a), metrics(case['truth'], b)
            old = json.loads((V3 / 'audit_report.json').read_text())[name]['metrics'][case['mode']]['ari']
            assert abs(ma['ari'] - old) < 1e-12
            baseline_values.append(ma['ari']); selected_values.append(mb['ari'])
            report[name][case['mode']] = {'baseline': ma, 'selected': mb}
            for version, labels in [('baseline', a), ('selected', b)]:
                predictions.extend({'dataset': name, 'mode': case['mode'], 'version': version,
                                    'image_id': int(i), 'identity': y, 'cluster': str(p)}
                                   for i, y, p in zip(case['qi'], case['truth'], labels))
        log(f'{name}: frozen development evaluation complete')
    report['baseline_macro_ari'] = float(np.mean(baseline_values)); report['selected_macro_ari'] = float(np.mean(selected_values))
    report['historically_seen_development_holdout'] = True; report['official_ari'] = None
    write(report, RUN / 'audit_report.json'); pd.DataFrame(predictions).to_csv(RUN / 'audit_predictions.csv', index=False)
    out = pd.read_csv(V3 / 'submission.csv'); details = {}
    for name in SPECIES:
        case = load_cases('test', name, calibrators)[0]; cfg = frozen['selected'][name]
        # Reproduce original partition membership independently of arbitrary cluster IDs.
        a = predict(case, {'kind': 'baseline'}, configs[name])
        old = out.set_index('image_id').loc[case['qi'], 'cluster'].to_numpy()
        assert adjusted_rand_score(old, a) == 1.
        pred = predict(case, cfg, configs[name]); values = {int(i): f'cluster_{name}_{p}' for i, p in zip(case['qi'], pred)}
        mask = out.image_id.isin(case['qi']); out.loc[mask, 'cluster'] = out.loc[mask, 'image_id'].map(values)
        _, counts = np.unique(pred, return_counts=True)
        details[name] = {'images': len(pred), 'clusters': len(counts), 'singletons': int((counts == 1).sum()), 'largest_cluster': int(counts.max())}
    template = pd.read_csv(DATA / 'sample_submission.csv'); m, _ = metadata()
    assert out.image_id.tolist() == template.image_id.tolist() and len(out) == 2409 and out.image_id.is_unique and out.cluster.notna().all()
    joined = out.merge(m.reset_index(drop=True)[['image_id', 'dataset']], on='image_id', validate='one_to_one')
    assert joined.groupby('cluster').dataset.nunique().max() == 1
    assert all(c.startswith('cluster_' + d + '_') for c, d in zip(joined.cluster, joined.dataset))
    lizard = set(m.loc[m.dataset == 'TexasHornedLizards', 'image_id'])
    old = pd.read_csv(V3 / 'submission.csv'); assert out[out.image_id.isin(lizard)].equals(old[old.image_id.isin(lizard)])
    out.to_csv(RUN / 'submission.csv', index=False)
    verify_frozen()
    write({'passed': True, 'rows': len(out), 'baseline_tune_and_audit_replayed': True,
           'baseline_test_partitions_replayed': True, 'frozen_inputs_unchanged': True,
           'lizard_unchanged': True, 'submission_sha256': digest(RUN / 'submission.csv'),
           'species': details, 'official_ari': None}, RUN / 'verification.json')
    log(f'Done: development macro ARI {report["baseline_macro_ari"]:.6f} -> {report["selected_macro_ari"]:.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--stage', choices=['train', 'tune', 'deliver', 'all'], default='all')
    args = parser.parse_args()
    import fcntl
    lock = (RUN / 'experiment.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for stage in (['train', 'tune', 'deliver'] if args.stage == 'all' else [args.stage]):
        write({'stage': stage, 'status': 'running'}, RUN / 'progress.json')
        globals()[stage]()
    write({'stage': args.stage, 'status': 'complete'}, RUN / 'progress.json')
