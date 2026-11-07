"""Training-side selection of global/local experts and species graph operating points."""
import argparse
import itertools
import json
import pickle
import gc
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, precision_recall_curve
from common import ROOT,RUN,BASE_RUN,DATA,seed,log,atomic_npz,json_write,check_stop,fingerprint
from extract import load_features
from matching import LocalMatcher,LOCAL_COLUMNS
from prepare import prepare
from pipeline import SPECIES,capped,validation_context,metrics,diverse_references
from graph import cluster,tree,rerank,attach

BANKS=['miewid','mega','miew_pp','miew_arc','dinov3','dinov3_pool']
BRANCHES=['aliked','disk','rootsift','kaze','dino_patch','superpoint_pattern','superpoint_regions']
FAMILIES=['learned','no_dino','wildfusion','pattern','global','dino_only','legacy','blend']
PATTERN=set(SPECIES[1::2])
GLOBAL_DIM=23
NO_DINO_GLOBAL=np.r_[np.arange(12),np.arange(18,GLOBAL_DIM)]

class Bank:
    def __init__(self):
        self.features={};self.lookup=None
        for name in BANKS:
            f=load_features('dinov3' if name=='dinov3_pool' else name)
            ids=f['ids'].numpy();assert len(ids)==15483 and len(np.unique(ids))==15483,name
            if self.lookup is None:self.lookup={int(x):i for i,x in enumerate(ids)};self.ids=ids
            else:assert np.array_equal(ids,self.ids),name
            self.features[name]=f['pooled' if name=='dinov3_pool' else 'features'].float().numpy()
        metadata=pd.read_csv(DATA/'metadata.csv').set_index('image_id').loc[self.ids]
        orientation=metadata.orientation.fillna('').astype(str).str.lower()
        self.orientation=pd.Categorical(orientation).codes.astype(np.int32)+1
        self.orientation[orientation.isin(['','nan','unknown','unspecified']).to_numpy()]=0
        dates=pd.to_datetime(metadata.date,errors='coerce',format='mixed').to_numpy(dtype='datetime64[D]')
        self.date_known=~np.isnat(dates);self.days=np.where(self.date_known,dates.astype(np.int64),0).astype(np.float64)
        log('Six aligned global feature banks loaded')
    def meta(self,a,b):
        oa=self.orientation[a];ob=self.orientation[b];known=(oa>0)&(ob>0);dated=self.date_known[a]&self.date_known[b]
        delta=np.log1p(np.abs(self.days[a]-self.days[b]))/8.;delta=np.where(dated,delta,0)
        return np.stack([known&(oa==ob),known&(oa!=ob),~known,delta,dated],-1).astype(np.float32)
    def pair(self,pairs):
        ix=np.array([[self.lookup[int(a)],self.lookup[int(b)]] for a,b in pairs]);out=[]
        for f in self.features.values():
            dot=np.einsum('bvd,bwd->bvw',f[ix[:,0]],f[ix[:,1]])
            out.extend([dot[:,0,0],dot.max((1,2)),dot.mean((1,2))])
        return np.c_[np.stack(out,1),self.meta(ix[:,0],ix[:,1])].astype(np.float32)
    def matrices(self,ids0,ids1=None):
        if ids1 is None:ids1=ids0
        a=np.array([self.lookup[int(x)] for x in ids0]);b=np.array([self.lookup[int(x)] for x in ids1]);out=[]
        for f in self.features.values():
            values=[f[a,i]@f[b,j].T for i,j in itertools.product(range(2),repeat=2)]
            out.extend([values[0],np.maximum.reduce(values),np.mean(values,axis=0)])
        return np.concatenate([np.stack(out,-1),self.meta(a[:,None],b[None,:])],-1).astype(np.float32)

def local_features(pairs,name,matcher,needed=None):
    values=[]
    for branch in BRANCHES:
        if (needed is not None and branch not in needed) or (name not in PATTERN and branch in BRANCHES[3:]):values.append(np.zeros((len(pairs),9),np.float32))
        else:values.append(matcher.compute(pairs,branch))
    return np.concatenate(values,1)

def fit(m,split,bank,matcher):
    path=RUN/'training_v3.npz'
    if path.exists():o=np.load(path);pairs=o['pairs'];y=o['y'];dataset=o['dataset'];g=o['g'];local=o['local']
    else:
        old=np.load(BASE_RUN/'training_plan.npz');pairs0=old['pairs'];allpairs=[];datasets=[];truth=dict(zip(m.image_id,m.identity));rng=np.random.default_rng(4398)
        for si,name in enumerate(SPECIES[:3]):
            frame=capped(m[m.image_id.isin(split.loc[(split.dataset==name)&(split.partition=='calibration'),'image_id'])],10,rng)
            ids=frame.image_id.to_numpy();g=bank.matrices(ids);keep=np.zeros(g.shape[:2],bool)
            for channel in (10,13,16):
                v=g[:,:,channel].copy();np.fill_diagonal(v,-2);k=min(8,len(ids)-1);ix=np.argpartition(-v,k-1,axis=1)[:,:k];keep[np.arange(len(ids))[:,None],ix]=True
            a,b=np.where(np.triu(keep|keep.T,1));hard=np.c_[ids[a],ids[b]];hard=hard[np.array([truth[x]!=truth[z] for x,z in hard])]
            if len(hard)>1400:hard=hard[rng.choice(len(hard),1400,replace=False)]
            original=pairs0[old['dataset']==si];p=np.unique(np.sort(np.r_[original,hard],axis=1),axis=0)
            allpairs.append(p);datasets.extend([si]*len(p))
        pairs=np.concatenate(allpairs);dataset=np.array(datasets);y=np.array([truth[a]==truth[b] for a,b in pairs],np.int32)
        assert set(pairs.flatten()).issubset(set(split.loc[split.partition=='calibration','image_id']))
        g=bank.pair(pairs);local=np.zeros((len(pairs),len(BRANCHES)*9),np.float32)
        for si,name in enumerate(SPECIES[:3]):local[dataset==si]=local_features(pairs[dataset==si],name,matcher)
        atomic_npz(path,pairs=pairs,y=y,dataset=dataset,g=g,local=local)
    models={};nondino=np.r_[np.arange(36),np.arange(45,len(BRANCHES)*9)]
    for si,name in enumerate(SPECIES):
        mask=dataset==si if si<3 else dataset==1
        xg=g[mask];xl=local[mask];target=y[mask]
        params=dict(n_estimators=280,num_leaves=9,max_depth=4,learning_rate=.025,min_child_samples=35,reg_lambda=12.,colsample_bytree=.9,n_jobs=8,verbosity=-1,random_state=20261104)
        clf=lambda x:LGBMClassifier(**params).fit(x,target)
        isos=[]
        for channel in range(1,18,3):isos.append(IsotonicRegression(out_of_bounds='clip').fit(xg[:,channel],target))
        for j in range(len(BRANCHES)):isos.append(IsotonicRegression(out_of_bounds='clip').fit(xl[:,j*9],target))
        models[name]={'global':clf(xg),'no_dino_global':clf(xg[:,NO_DINO_GLOBAL]),'learned':clf(np.c_[xg,xl]),'no_dino':clf(np.c_[xg[:,NO_DINO_GLOBAL],xl[:,nondino]]),'pattern':clf(xl[:,27:]),'dino_only':clf(xg[:,12:]),'isos':isos}
    with (RUN/'fusion_v3.pkl').open('wb') as f:pickle.dump(models,f)
    json_write({'pairs':len(y),'positive_pairs':int(y.sum()),'calibration_identities_only':True,'global_banks':BANKS,'local_branches':BRANCHES,'lizard_fit_source':'calibration salamander pattern pairs'},RUN/'training_summary.json')
    log('Supervised expert fitting completed');return models

def candidates(g,ids0,ids1,name,matcher,symmetric):
    n0,n1=g.shape[:2]
    if symmetric and name in PATTERN:
        a,b=np.triu_indices(n0,1);return a,b,np.c_[np.asarray(ids0)[a],np.asarray(ids1)[b]]
    keep=np.zeros((n0,n1),bool)
    for channel in range(1,18,3):
        v=g[:,:,channel].copy()
        if symmetric:np.fill_diagonal(v,-2)
        k=min(48 if symmetric else 24,n1-int(symmetric))
        if k<=0:continue
        ix=np.argpartition(-v,k-1,axis=1)[:,:k];keep[np.arange(n0)[:,None],ix]=True
    if not symmetric and name=='SalamanderID2025':
        # Texture retrieval can recover references missed by every global descriptor.
        a,b=np.indices((n0,n1));pairs=np.c_[np.asarray(ids0)[a.ravel()],np.asarray(ids1)[b.ravel()]]
        for branch in ['rootsift','kaze']:
            local=matcher.compute(pairs,branch);texture=(local[:,0]+.3*local[:,3]).reshape(n0,n1);k=min(32,n1)
            ix=np.argpartition(-texture,k-1,axis=1)[:,:k];keep[np.arange(n0)[:,None],ix]=True
    if symmetric:keep|=keep.T;np.fill_diagonal(keep,False);a,b=np.where(np.triu(keep,1))
    else:a,b=np.where(keep)
    return a,b,np.c_[np.asarray(ids0)[a],np.asarray(ids1)[b]]

def predict_scores(g,ids0,ids1,name,model,matcher,symmetric):
    flat=g.reshape(-1,GLOBAL_DIM);shape=g.shape[:2]
    scores={'global':model['global'].predict_proba(flat)[:,1].reshape(shape).astype(np.float32),'dino_only':model['dino_only'].predict_proba(flat[:,12:])[:,1].reshape(shape).astype(np.float32)}
    for family in ['learned','no_dino','pattern','wildfusion']:scores[family]=scores['global'].copy()
    scores['no_dino']=model['no_dino_global'].predict_proba(flat[:,NO_DINO_GLOBAL])[:,1].reshape(shape).astype(np.float32)
    a,b,pairs=candidates(g,ids0,ids1,name,matcher,symmetric)
    local=local_features(pairs,name,matcher);selected=g[a,b];nondino=np.r_[np.arange(36),np.arange(45,len(BRANCHES)*9)]
    for family,x in [('learned',np.c_[selected,local]),('no_dino',np.c_[selected[:,NO_DINO_GLOBAL],local[:,nondino]]),('pattern',local[:,27:])]:
        values=model[family].predict_proba(x)[:,1];scores[family][a,b]=values
        if symmetric:scores[family][b,a]=values
    global_iso=np.stack([model['isos'][j].predict(selected[:,1+3*j]) for j in range(6)],1)
    local_iso=np.stack([model['isos'][j+6].predict(local[:,j*9]) for j in range(len(BRANCHES))],1)
    if name in PATTERN:
        values=.15*global_iso.mean(1)+.85*np.average(local_iso,axis=1,weights=[.5,.5,1,1,1,2,2])
    else:values=.5*global_iso.mean(1)+.5*local_iso[:,:3].mean(1)
    scores['wildfusion'][a,b]=values
    if symmetric:scores['wildfusion'][b,a]=values
    for s in scores.values():
        if symmetric:np.fill_diagonal(s,1.)
        assert np.isfinite(s).all()
    return scores

def context_scores(q,r,name,tag,bank,model,matcher):
    folder=RUN/'scores';folder.mkdir(exist_ok=True);file=folder/f'{tag}_{name}.npz';qi=q.image_id.to_numpy();ri=r.image_id.to_numpy()
    model_sha=fingerprint(RUN/'fusion_v3.pkl')
    if file.exists():
        o=np.load(file);assert np.array_equal(o['qi'],qi) and np.array_equal(o['ri'],ri)
        assert o['model_sha256'].item()==model_sha,'Score cache belongs to a different fusion model'
        return {f:o['qq_'+f] for f in FAMILIES},{f:o['qr_'+f] for f in FAMILIES}
    qq=predict_scores(bank.matrices(qi),qi,qi,name,model,matcher,True)
    qr=predict_scores(bank.matrices(qi,ri),qi,ri,name,model,matcher,False) if len(ri) else {f:np.empty((len(qi),0),np.float32) for f in FAMILIES}
    oldcfg=json.loads((BASE_RUN/'selected_config.json').read_text())[name]
    olddir=BASE_RUN/('scores' if oldcfg['candidate_policy']=='sparse' else 'scores_qq128')
    oldname=f'test_{name}_{oldcfg["family"]}' if tag=='test' else f'{tag.split("_",1)[0]}_{name}_{tag.split("_",1)[1]}_{oldcfg["family"]}'
    old=np.load(olddir/(oldname+'.npz'));assert np.array_equal(old['qi'],qi) and np.array_equal(old['ri'],ri)
    qq['legacy']=old['sq'];qr['legacy']=old['sr']
    qq['blend']=.65*qq['learned']+.35*qq['legacy'];qr['blend']=.65*qr['learned']+.35*qr['legacy']
    atomic_npz(file,qi=qi,ri=ri,model_sha256=np.array(model_sha),**{'qq_'+f:s for f,s in qq.items()},**{'qr_'+f:s for f,s in qr.items()})
    return qq,qr

def build_contexts(m,split,bank,models,matcher,partition):
    contexts={}
    for name in SPECIES[:3]:
        frame=m[m.image_id.isin(split.loc[(split.dataset==name)&(split.partition==partition),'image_id'])];cases=[]
        for mode in ['discovery','mixed']:
            q,r=validation_context(frame,np.random.default_rng(8391),mode)
            qq,qr=context_scores(q,r,name,partition+'_'+mode,bank,models[name],matcher)
            cases.append((mode,q,r,qq,qr))
        contexts[name]=cases;log(f'{partition} {name}: matching complete; aggregate statistics deferred')
    return contexts

def transformed(score,amount):return rerank(score,12,amount) if amount else score

def choose(contexts):
    configs={};search={};ablations={}
    for name,cases in contexts.items():
        trials=[]
        for family,amount,algorithm in itertools.product(FAMILIES,[0.,.25],['average','complete','cw','leiden']):
            prepared=[]
            for mode,q,r,qq,qr in cases:
                score=transformed(qq[family],amount);z=tree(score,algorithm) if algorithm in ['average','complete'] else None
                prepared.append((q,r,score,qr[family],z))
            for threshold in np.r_[.05,.1,np.linspace(.15,.95,25),.975,.99,.995]:
                check_stop();cfg={'family':family,'rerank':amount,'algorithm':algorithm,'threshold':float(threshold),'attach_threshold':1.1,'attach_margin':.1}
                values=[]
                for q,r,score,_,z in prepared:
                    labels=cluster(score,cfg,z);values.append(adjusted_rand_score(q.identity,labels)+.05*adjusted_mutual_info_score(q.identity,labels))
                trials.append({'config':cfg.copy(),'objective':float(np.mean(values)-.025*np.std(values))})
        trials.sort(key=lambda x:-x['objective']);cfg=trials[0]['config'].copy()
        ablations[name]={family:next(x for x in trials if x['config']['family']==family) for family in FAMILIES}
        _,q,r,qq,qr=cases[1];score=transformed(qq[cfg['family']],cfg['rerank']);labels=cluster(score,cfg);best=-1e9
        for threshold,margin,anchor_mode in itertools.product([1.1,.97,.9,.8,.65,.5],[.25,.15,.05],['safe','original']):
            pred=attach(labels,qr[cfg['family']],r.identity.to_numpy(),threshold,margin,score,cfg['threshold'],anchor_mode)
            objective=adjusted_rand_score(q.identity,pred)+.05*adjusted_mutual_info_score(q.identity,pred)
            if objective>best:best=objective;cfg['attach_threshold']=threshold;cfg['attach_margin']=margin;cfg['anchor_mode']=anchor_mode
        configs[name]=cfg;search[name]=trials[:15];log(f'{name}: graph selection saved (no intermediate scores reported)')
    # Zero-shot transfer uses precision-constrained validation, never published test population counts.
    mode,q,r,qq,qr=contexts['SalamanderID2025'][0];a,b=np.triu_indices(len(q),1);same=q.identity.to_numpy()[a]==q.identity.to_numpy()[b]
    precision,recall,threshold=precision_recall_curve(same,qq['pattern'][a,b]);f=(1+.25)*precision[:-1]*recall[:-1]/np.maximum(.25*precision[:-1]+recall[:-1],1e-8)
    good=precision[:-1]>=.99
    if good.any():t=float(threshold[np.argmax(np.where(good,f,-1))])
    else:t=float(min(.995,max(.8,np.quantile(qq['pattern'][a,b][~same],.9995))))
    configs[SPECIES[3]]={'family':'pattern','rerank':0.,'algorithm':'complete','threshold':t,'attach_threshold':1.1,'attach_margin':.25,'source':'99% precision constraint on tuning salamander pattern pairs; F0.5 tie-break'}
    json_write(configs,RUN/'selected_config.json');json_write(search,RUN/'search_summary.json');json_write(ablations,RUN/'ablation_tuning.json');return configs

def measure(contexts,configs,path):
    report={'previous_audit_used_for_architecture_diagnosis':True,'official_ari':None};values=[]
    for name,cases in contexts.items():
        cfg=configs[name];report[name]={'config':cfg,'metrics':{}}
        for mode,q,r,qq,qr in cases:
            score=transformed(qq[cfg['family']],cfg['rerank']);labels=cluster(score,cfg);labels=attach(labels,qr[cfg['family']],r.identity.to_numpy(),cfg['attach_threshold'],cfg['attach_margin'],score,cfg['threshold'],cfg.get('anchor_mode','safe'))
            metric=metrics(q.identity.to_numpy(),labels);report[name]['metrics'][mode]=metric;values.append(metric['ari'])
    report['macro_ari']=float(np.mean(values));json_write(report,path);return report

def freeze():
    files=['vision.py','finetune.py','strong.py','graph.py','matching.py','common.py','pipeline.py','models.py','extract.py','prepare.py']
    return {'code':{f:fingerprint(ROOT/f) for f in files},'models':fingerprint(RUN/'fusion_v3.pkl'),'config':fingerprint(RUN/'selected_config.json'),'split':fingerprint(RUN/'splits.csv'),'arcface':fingerprint(RUN/'miew_arc_best.pt'),'dino':json.loads((ROOT/'models/dinov3/manifest.json').read_text())}

def production_scores(q,r,name,bank,model,matcher,cfg):
    family=cfg['family'];file=RUN/'scores'/f'production_{name}_{family}.npz';qi=q.image_id.to_numpy();ri=r.image_id.to_numpy()
    use_reference=cfg['attach_threshold']<=1 and len(r)>0
    if file.exists():
        o=np.load(file);assert np.array_equal(o['qi'],qi) and np.array_equal(o['ri'],ri)
        assert bool(o['reference_used'].item())==bool(use_reference)
        assert o['model_sha256'].item()==fingerprint(RUN/'fusion_v3.pkl')
        return o['sq'],o['sr']
    oldcfg=json.loads((BASE_RUN/'selected_config.json').read_text())[name]
    olddir=BASE_RUN/('scores' if oldcfg['candidate_policy']=='sparse' else 'scores_qq128')
    old=np.load(olddir/f'test_{name}_{oldcfg["family"]}.npz');assert np.array_equal(old['qi'],qi) and np.array_equal(old['ri'],ri)
    def calculate(ids1,symmetric):
        if family=='legacy':return old['sq' if symmetric else 'sr']
        g=bank.matrices(qi,ids1);flat=g.reshape(-1,GLOBAL_DIM)
        if family in ['global','dino_only']:
            x=flat if family=='global' else flat[:,12:]
            score=model[family].predict_proba(x)[:,1].reshape(g.shape[:2]).astype(np.float32)
        else:
            baseline='no_dino_global' if family=='no_dino' else 'global'
            x=flat[:,NO_DINO_GLOBAL] if family=='no_dino' else flat
            score=model[baseline].predict_proba(x)[:,1].reshape(g.shape[:2]).astype(np.float32)
            a,b,pairs=candidates(g,qi,ids1,name,matcher,symmetric)
            needed=BRANCHES[3:] if family=='pattern' else [x for x in BRANCHES if x!='dino_patch'] if family=='no_dino' else BRANCHES
            local=local_features(pairs,name,matcher,needed);selected=g[a,b]
            if family=='pattern':values=model['pattern'].predict_proba(local[:,27:])[:,1]
            elif family=='no_dino':
                nondino=np.r_[np.arange(36),np.arange(45,len(BRANCHES)*9)];values=model['no_dino'].predict_proba(np.c_[selected[:,NO_DINO_GLOBAL],local[:,nondino]])[:,1]
            elif family in ['learned','blend']:values=model['learned'].predict_proba(np.c_[selected,local])[:,1]
            else:
                gi=np.stack([model['isos'][j].predict(selected[:,1+3*j]) for j in range(6)],1)
                li=np.stack([model['isos'][j+6].predict(local[:,j*9]) for j in range(len(BRANCHES))],1)
                values=.15*gi.mean(1)+.85*np.average(li,axis=1,weights=[.5,.5,1,1,1,2,2]) if name in PATTERN else .5*gi.mean(1)+.5*li[:,:3].mean(1)
            score[a,b]=values
            if symmetric:score[b,a]=values
            if family=='blend':score=.65*score+.35*old['sq' if symmetric else 'sr']
        if symmetric:np.fill_diagonal(score,1.)
        return score
    sq=calculate(qi,True);sr=calculate(ri,False) if use_reference else np.empty((len(q),0),np.float32)
    assert np.isfinite(sq).all() and np.isfinite(sr).all()
    atomic_npz(file,sq=sq,sr=sr,qi=qi,ri=ri,reference_used=np.array(use_reference),model_sha256=np.array(fingerprint(RUN/'fusion_v3.pkl')))
    return sq,sr

def submit(m,bank,models,matcher,configs):
    results=[];detail={}
    for name in SPECIES:
        q=m[(m.dataset==name)&(m.split=='test')];r=m[(m.dataset==name)&(m.split=='train')]
        if len(r):r=diverse_references(r,bank)
        cfg=configs[name];sq,sr=production_scores(q,r,name,bank,models[name],matcher,cfg);score=transformed(sq,cfg['rerank']);labels=cluster(score,cfg)
        labels=attach(labels,sr,r.identity.to_numpy(),cfg['attach_threshold'],cfg['attach_margin'],score,cfg['threshold'],cfg.get('anchor_mode','safe'))
        labels=np.array([f'cluster_{name}_{x}' for x in labels]);results.extend(zip(q.image_id,labels));_,count=np.unique(labels,return_counts=True)
        detail[name]={'images':len(q),'clusters':len(count),'singletons':int((count==1).sum()),'largest_cluster':int(count.max())}
        log(f'Test {name}: matching and prediction completed; aggregate statistics deferred')
    template=pd.read_csv(DATA/'sample_submission.csv');out=template[['image_id']].merge(pd.DataFrame(results,columns=['image_id','cluster']),on='image_id',how='left',validate='one_to_one')
    assert len(out)==2409 and out.cluster.notna().all() and out.image_id.tolist()==template.image_id.tolist()
    joined=m[m.split=='test'].merge(out,on='image_id',validate='one_to_one');assert joined.groupby('cluster').dataset.nunique().max()==1
    assert all(c.startswith('cluster_'+d+'_') for d,c in zip(joined.dataset,joined.cluster))
    out.to_csv(RUN/'submission.csv',index=False);json_write({'complete':True,'rows':len(out),'species':detail,'submission_sha256':fingerprint(RUN/'submission.csv'),'official_ari':None},RUN/'completion_check.json')

def main(stage):
    seed();torch.hub.set_dir(str(ROOT/'models/torch_hub'));m,split=prepare();bank=Bank();matcher=LocalMatcher()
    if stage=='fit':fit(m,split,bank,matcher)
    else:
        with (RUN/'fusion_v3.pkl').open('rb') as f:models=pickle.load(f)
        if stage=='tune':
            contexts=build_contexts(m,split,bank,models,matcher,'tune');cfg=choose(contexts);measure(contexts,cfg,RUN/'tune_report.json');json_write(freeze(),RUN/'audit_plan.json')
        elif stage=='audit':
            assert freeze()==json.loads((RUN/'audit_plan.json').read_text())
            contexts=build_contexts(m,split,bank,models,matcher,'audit');measure(contexts,json.loads((RUN/'selected_config.json').read_text()),RUN/'audit_report.json')
        else:
            assert freeze()==json.loads((RUN/'audit_plan.json').read_text());submit(m,bank,models,matcher,json.loads((RUN/'selected_config.json').read_text()))
    matcher.close();log(f'V3 {stage} completed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['fit','tune','audit','submit'],required=True);a=p.parse_args();main(a.stage)
