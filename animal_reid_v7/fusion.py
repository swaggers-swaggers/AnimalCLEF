"""Select image-level and correspondence evidence on tune, then freeze once."""
import argparse
import copy
import itertools
import json
import pickle
import sqlite3
import sys
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import adjusted_rand_score
from settings import *
sys.path.insert(0,str(CV/'animal_reid_v6'))
import experiment as v6
from graph import cluster, attach
from pipeline import metrics


@lru_cache(maxsize=12)
def embeddings(name,branch):
    o=np.load(RUN/f'dino_{branch}_{name}.npz')
    return {int(k):i for i,k in enumerate(o['ids'])},o['features']


def spot_read(pairs,strict=True):
    db=sqlite3.connect(f'file:{RUN / "spots.sqlite"}?mode=ro',uri=True); values=[]; available=[]
    for pair in pairs:
        a,b=sorted(map(int,pair)); row=db.execute('SELECT value FROM matches WHERE a=? AND b=?',(a,b)).fetchone()
        if row is None and strict: raise KeyError(f'Missing real spot matching evidence: {a},{b}')
        values.append(np.frombuffer(row[0],np.float32) if row else np.zeros(10,np.float32)); available.append(row is not None)
    db.close(); return np.asarray(values).reshape(-1,10),np.asarray(available)


def dense_read(pairs):
    db=sqlite3.connect(f'file:{RUN / "dense.sqlite"}?mode=ro',uri=True); values=[]; available=[]
    for pair in pairs:
        a,b=sorted(map(int,pair)); row=db.execute('SELECT value FROM matches WHERE a=? AND b=?',(a,b)).fetchone()
        values.append(np.frombuffer(row[0],np.float32) if row else np.zeros(12,np.float32)); available.append(row is not None)
    db.close(); return np.asarray(values).reshape(-1,12),np.asarray(available)


def fit():
    m,_=metadata(); r=roles(); result={}; summary={}
    for name in SPECIES:
        frame=m[m.image_id.isin(r.loc[(r.dataset==name)&(r.role=='calibrate'),'image_id'])].groupby('identity',sort=True).head(8)
        ids=frame.image_id.to_numpy(); y=frame.identity.to_numpy(); a,b=np.triu_indices(len(ids),1)
        same=(y[a]==y[b]).astype(float); result[name]={}
        for branch in ['initial','trained']:
            lookup,z=embeddings(name,branch); f=z[[lookup[int(k)] for k in ids]]
            result[name][branch]=IsotonicRegression(out_of_bounds='clip').fit((f@f.T)[a,b],same)
        if name=='SalamanderID2025':
            local,_=spot_read(np.c_[ids[a],ids[b]])
            for c in [1,2,3,5,7]: result[name]['spot_'+str(c)]=IsotonicRegression(out_of_bounds='clip').fit(local[:,c],same)
            if (RUN/'roma_calibration_complete.json').exists():
                from lightgbm import LGBMClassifier
                pairs=np.load(RUN/'roma_pairs_calibration.npz')['pairs']; dense,available=dense_read(pairs)
                assert available.all()
                truth=dict(zip(m.image_id,m.identity)); dense_y=np.array([truth[int(i)]==truth[int(j)] for i,j in pairs])
                result[name]['roma']=LGBMClassifier(n_estimators=100,num_leaves=5,max_depth=3,min_child_samples=20,
                    learning_rate=.035,reg_lambda=8,n_jobs=8,verbosity=-1,random_state=770106).fit(dense,dense_y)
                result[name]['roma_ncc']=IsotonicRegression(out_of_bounds='clip').fit(dense[:,8],dense_y)
        summary[name]={'images':len(ids),'identities':len(np.unique(y)),'pairs':len(a),'positives':int(same.sum())}
    with (RUN/'score_calibrators.pkl').open('wb') as f: pickle.dump(result,f)
    write(summary,RUN/'score_calibration.json')
    return result


def cases(stage,name,calibrators,with_spots=True,with_dense=True):
    with (V6/'calibrators.pkl').open('rb') as f: oldcal=pickle.load(f)
    basecfg=json.loads((V3/'selected_config.json').read_text())[name]
    v6cfg=json.loads((V6/'selected_config.json').read_text())[name]
    old=v6.load_cases(stage,name,oldcal); result=[]
    for c in old:
        q,r=c['qi'],c['ri']; sq,sr=c['sq'],c['sr']
        if v6cfg.get('weight',0):
            nq,nr=c['extra'][v6cfg['branch']]; w=v6cfg['weight']; sq=(1-w)*sq+w*nq; sr=(1-w)*sr+w*nr
        scores={'base':(sq,sr)}
        for branch in ['initial','trained']:
            lookup,z=embeddings(name,branch); zq=z[[lookup[int(k)] for k in q]]; zr=z[[lookup[int(k)] for k in r]]
            s=zq@zq.T; t=zq@zr.T
            iso=calibrators[name][branch]
            scores[branch]=(iso.predict(s.ravel()).reshape(s.shape).astype(np.float32),
                            iso.predict(t.ravel()).reshape(t.shape).astype(np.float32) if len(r) else t)
            scores[branch+'_raw']=(np.clip(s,0,1),np.clip(t,0,1))
        if name=='SalamanderID2025' and with_spots:
            a,b=np.triu_indices(len(q),1); features,_=spot_read(np.c_[q[a],q[b]])
            qrfeatures=np.empty((0,10)); rows=cols=np.empty(0,int)
            if len(r):
                k=min(32,len(r)); neighbors=np.argpartition(-c['sr'],k-1,axis=1)[:,:k]
                rows=np.repeat(np.arange(len(q)),k); cols=neighbors.ravel()
                qrfeatures,_=spot_read(np.c_[q[rows],r[cols]])
            for col in [1,2,3,5,7]:
                iso=calibrators[name]['spot_'+str(col)]; ss=np.eye(len(q),dtype=np.float32)
                ss[a,b]=iso.predict(features[:,col]); ss[b,a]=ss[a,b]
                rr=sr.copy()
                if len(r): rr[rows,cols]=iso.predict(qrfeatures[:,col])
                scores['spot_'+str(col)]=(ss,rr)
            # Confidence count transformed monotonically; threshold chosen only on tune.
            ss=np.eye(len(q),dtype=np.float32); ss[a,b]=1-np.exp(-np.expm1(features[:,2])/8); ss[b,a]=ss[a,b]
            rr=sr.copy()
            if len(r): rr[rows,cols]=1-np.exp(-np.expm1(qrfeatures[:,2])/8)
            scores['spot_raw']=(ss,rr)
        if with_dense and name=='SalamanderID2025' and 'roma' in calibrators[name]:
            a,b=np.triu_indices(len(q),1); dense,valid=dense_read(np.c_[q[a],q[b]])
            rows,cols=np.indices((len(q),len(r))); rows=rows.ravel(); cols=cols.ravel()
            rdense,rvalid=dense_read(np.c_[q[rows],r[cols]]) if len(r) else (np.empty((0,12)),np.zeros(0,bool))
            for source in ['roma','roma_ncc']:
                clf=calibrators[name][source]
                estimate=lambda x: clf.predict_proba(x)[:,1] if source=='roma' else clf.predict(x[:,8])
                ss=sq.copy(); rr=sr.copy()
                if valid.any(): ss[a[valid],b[valid]]=estimate(dense[valid]); ss[b[valid],a[valid]]=ss[a[valid],b[valid]]
                if rvalid.any(): rr[rows[rvalid],cols[rvalid]]=estimate(rdense[rvalid])
                scores[source]=(ss,rr)
        for ss,rr in scores.values():
            np.fill_diagonal(ss,1); assert np.isfinite(ss).all() and np.isfinite(rr).all()
        result.append({'mode':c['mode'],'qi':q,'ri':r,'truth':c['truth'],'refs':c['refs'],
                       'baseline':v6.predict(c,v6cfg,basecfg),'scores':scores,'basecfg':basecfg,'v6cfg':v6cfg})
    return result


def source_config(cases):
    names=list(cases[0]['scores'])
    yield {'source':'base','weight':0.}
    for name in names:
        if name=='base': continue
        for weight in [1.,.5,.25]: yield {'source':name,'weight':weight}
    if 'spot_2' in names:
        for w in [.25,.5,.75]: yield {'source':'spot_dino','weight':1.,'dino_weight':w}
    if 'spot_2' in names and 'roma' in names:
        for w in [.25,.5,.75]: yield {'source':'spot_roma','weight':1.,'dense_weight':w}


def score(case,cfg):
    if cfg['source']=='base': return case['scores']['base']
    if cfg['source']=='spot_dino':
        w=cfg['dino_weight']; a,b=case['scores']['spot_2']; x,y=case['scores']['trained']
        return (1-w)*a+w*x,(1-w)*b+w*y
    if cfg['source']=='spot_roma':
        w=cfg['dense_weight']; a,b=case['scores']['spot_2']; x,y=case['scores']['roma']
        return (1-w)*a+w*x,(1-w)*b+w*y
    a,b=case['scores'][cfg['source']]; x,y=case['scores']['base']; w=cfg['weight']
    return (1-w)*x+w*a,(1-w)*y+w*b


def predict(case,cfg,tree=None):
    if cfg.get('kind')=='baseline':
        sq,sr=case['scores']['base']; base=case['basecfg']; old=case['v6cfg']
        labels=cluster(sq,base) if old['kind']=='baseline' else v6.graph_cluster(sq,old)
        threshold=base['threshold'] if old['kind']=='baseline' else old['threshold']
        return attach(labels,sr,case['refs'],base['attach_threshold'],base['attach_margin'],sq,threshold,base.get('anchor_mode','safe'))
    sq,sr=score(case,cfg)
    if cfg['algorithm'] in ['average','complete']:
        if tree is None:
            distance=np.clip(1-(sq+sq.T)/2,0,1); np.fill_diagonal(distance,0)
            tree=linkage(squareform(distance,checks=False),method=cfg['algorithm'])
        labels=fcluster(tree,t=max(1-cfg['threshold'],1e-8),criterion='distance')-1
    else: labels=cluster(sq,cfg)
    base=case['basecfg']
    return attach(labels,sr,case['refs'],cfg.get('attach_threshold',base['attach_threshold']),
                  base['attach_margin'],sq,cfg['threshold'],base.get('anchor_mode','safe'))


def measure(cases,cfg,trees=None):
    values=[float(adjusted_rand_score(c['truth'],predict(c,cfg,trees[i] if trees else None))) for i,c in enumerate(cases)]
    return {'mean':float(np.mean(values)),'modes':values}


def subsets(cases,seed):
    keys=np.unique(cases[0]['truth']); chosen=np.random.default_rng(seed).choice(keys,max(2,int(.75*len(keys))),replace=False)
    result=[]
    for c in cases:
        a=np.flatnonzero(np.isin(c['truth'],chosen)); b=np.flatnonzero(np.isin(c['refs'],chosen))
        # Recompute baseline graph using the original V6 config on the subset.
        baseline=c['baseline'][a]
        result.append({'mode':c['mode'],'qi':c['qi'][a],'ri':c['ri'][b],'truth':c['truth'][a],
                       'refs':c['refs'][b],'baseline':baseline,'basecfg':c['basecfg'],'v6cfg':c['v6cfg'],
                       'scores':{k:(v[0][np.ix_(a,a)],v[1][np.ix_(a,b)]) for k,v in c['scores'].items()}})
    return result


def tune(neural_only=False,prepare=False):
    assert not (RUN/'frozen.json').exists(),'Frozen selection cannot be changed'
    if not neural_only and not prepare:
        assert (RUN/'roma_tune_complete.json').exists(), 'Finish planned dense tune inference first'
        assert json.loads((RUN/'native_checks.json').read_text())['passed']
    calibrators=fit() if not neural_only else fit_neural()
    selected={}; reports={}; thresholds=np.unique(np.r_[.02,.05,np.arange(.1,1.,.05),.975,.99])
    plan={'sources':'new image DINO initial/trained, native spots, sparse RoMa, spot/DINO and spot/RoMa mixtures; baseline V6',
          'algorithms':['average','complete','leiden'],'thresholds':thresholds.tolist(),
          'weights':[1.,.5,.25],'attachment':'baseline threshold or disabled; original margin/rule',
          'shortlist':'top 12 distinct two-mode partitions, with duplicate thresholds collapsed',
          'gate':'full mean gain >=.005; neither full mode loses >.02; wins >=4/6 identity subsets',
          'subset_comparison':'both pipelines reclustered on identical identity subsets',
          'selection':'equal-weight full and subset mean; all tune only','neural_only':neural_only}
    write(plan,RUN/('neural_tune_plan.json' if neural_only else 'tune_plan.json'))
    for name in SPECIES:
        if prepare and name=='SalamanderID2025': continue
        cs=cases('tune',name,calibrators,not neural_only); base=measure(cs,{'kind':'baseline'}); trials=[]
        # Independent species can finish while dense salamander matching runs.
        # Reuse only an exact match of scores, identities, plan, and selector code.
        fingerprint=hashlib.sha256((sha(Path(__file__))+json.dumps(plan,sort_keys=True)).encode())
        for c in cs:
            fingerprint.update(json.dumps({k:c[k] for k in ['basecfg','v6cfg']},sort_keys=True).encode())
            for key in ['qi','ri','truth','refs']: fingerprint.update(str(c[key].tolist()).encode())
            for key,(a,b) in sorted(c['scores'].items()):
                fingerprint.update(key.encode()); fingerprint.update(a.tobytes()); fingerprint.update(b.tobytes())
        key=fingerprint.hexdigest(); partial=RUN/f'tune_partial_{name}.json'
        if not neural_only and partial.exists():
            saved=json.loads(partial.read_text())
            if saved['fingerprint']==key:
                reports[name]=saved['report']; selected[name]=reports[name]['selected']['config']
                log(f'{name}: reused identical completed selection')
                continue
        for spec in source_config(cs):
            for algorithm in ['average','complete','leiden']:
                trees=[]
                for c in cs:
                    sq,_=score(c,spec); d=np.clip(1-(sq+sq.T)/2,0,1); np.fill_diagonal(d,0)
                    trees.append(linkage(squareform(d,checks=False),method=algorithm) if algorithm in ['average','complete'] else None)
                for threshold in thresholds:
                    stop(); cfg={**spec,'algorithm':algorithm,'threshold':float(threshold)}
                    for at in [cs[0]['basecfg']['attach_threshold'],1.1]:
                        cfg['attach_threshold']=at; trials.append({'config':cfg.copy(),**measure(cs,cfg,trees)})
            log(f'{name} tuning {spec}')
        trials.sort(key=lambda v:-v['mean']); chunks=[subsets(cs,73106+i) for i in range(6)]
        base_sub=[measure(c,{'kind':'baseline'})['mean'] for c in chunks]
        best={'config':{'kind':'baseline'},**base,'subsets':base_sub,'objective':.5*base['mean']+.5*float(np.mean(base_sub))}
        shortlist=[]; seen=set()
        for row in trials:
            signature=[]
            for c in cs:
                labels=predict(c,row['config']); mapping={}; canonical=[]
                for label in labels:
                    if label not in mapping: mapping[label]=len(mapping)
                    canonical.append(mapping[label])
                signature.append(tuple(canonical))
            key_partition=tuple(signature)
            if key_partition in seen: continue
            seen.add(key_partition); shortlist.append(row)
            if len(shortlist)==12: break
        for row in shortlist:
            row['subsets']=[measure(c,row['config'])['mean'] for c in chunks]
            row['eligible']=bool(row['mean']>=base['mean']+.005 and min(np.array(row['modes'])-base['modes'])>=-.02 and
                                 sum(np.array(row['subsets'])>np.array(base_sub)+1e-8)>=4)
            row['objective']=.5*row['mean']+.5*float(np.mean(row['subsets']))
            if row['eligible'] and row['objective']>best['objective']: best=copy.deepcopy(row)
        reports[name]={'baseline':base,'selected':best,'baseline_subsets':base_sub,'shortlist':shortlist,'trials':trials}
        if not neural_only: write({'fingerprint':key,'report':reports[name]},partial)
        selected[name]=best['config']; log(f'{name}: V6 {base["mean"]:.4f}, selected {best["mean"]:.4f}')
        write(reports,RUN/('neural_tune_search.json' if neural_only else 'tune_search.json'))
        write(selected,RUN/('neural_selected_config.json' if neural_only else 'selected_config.json'))
    if not neural_only:
        write(reports,RUN/'tune_search.json'); write(selected,RUN/'selected_config.json')
    if not neural_only and not prepare:
        # Freeze the exact calibration/tune correspondence evidence. The live
        # databases may later grow with label-free audit/test inference.
        from patterns import pair_plan
        plans=pair_plan()
        for part in ['calibration','tune']:
            pairs=plans[part]; values,valid=spot_read(pairs); assert valid.all()
            np.savez_compressed(RUN/f'frozen_spots_{part}.npz',pairs=pairs,values=values)
            pairs=np.load(RUN/f'roma_pairs_{part}.npz')['pairs']; values,valid=dense_read(pairs); assert valid.all()
            np.savez_compressed(RUN/f'frozen_dense_{part}.npz',pairs=pairs,values=values)
        files=[ROOT/f for f in ['fusion.py','dino_model.py','dino_train.py','images.py','patterns.py','settings.py','roma_stage.py','roma_pilot.py']]
        files += [RUN/f for f in ['roles.csv','selected_config.json','score_calibrators.pkl','tune_plan.json','tune_search.json','spot_features_manifest.json']]
        files += list(RUN.glob('dino_*.npz')) + list(RUN.glob('dino_best_*.pt'))
        files += list(RUN.glob('frozen_*.npz'))
        files += [RUN/f for f in ['roma_calibration_complete.json','roma_tune_complete.json','native_checks.json']]
        files += [V6/'submission.csv',V6/'selected_config.json',V6/'calibrators.pkl',V3/'selected_config.json',V3/'splits.csv']
        files += list(V6.glob('embeddings_*.npz')) + list((V3/'scores').glob('*.npz'))
        write({'inputs':{str(p.relative_to(CV)):sha(p) for p in files},'selected':selected,'audit_used_for_selection':False},RUN/'frozen.json')


def fit_neural():
    m,_=metadata(); r=roles(); result={}
    for name in SPECIES:
        frame=m[m.image_id.isin(r.loc[(r.dataset==name)&(r.role=='calibrate'),'image_id'])].groupby('identity',sort=True).head(8)
        ids=frame.image_id.to_numpy(); y=frame.identity.to_numpy(); a,b=np.triu_indices(len(ids),1); result[name]={}
        for branch in ['initial','trained']:
            lookup,z=embeddings(name,branch); f=z[[lookup[int(k)] for k in ids]]
            result[name][branch]=IsotonicRegression(out_of_bounds='clip').fit((f@f.T)[a,b],y[a]==y[b])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['neural_tune','prepare','tune'],default='neural_tune'); a=p.parse_args()
    tune(a.stage=='neural_tune',a.stage=='prepare')
