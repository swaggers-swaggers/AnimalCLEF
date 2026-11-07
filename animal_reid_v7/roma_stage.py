"""Sparse, candidate-based dense verification. Missing evidence stays unknown."""
import argparse
from collections import OrderedDict
import sqlite3
import time
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
from settings import *
from images import body
from roma_pilot import features, COLUMNS

NAME='SalamanderID2025'


def nominate(s,ids0,ids1,k,symmetric):
    values=s.copy()
    if symmetric: np.fill_diagonal(values,-np.inf)
    k=min(k,len(ids1)-int(symmetric))
    if k<=0: return np.empty((0,2),np.int64)
    order=np.argpartition(-values,k-1,axis=1)[:,:k]
    pairs=np.c_[np.repeat(ids0,k),ids1[order.ravel()]]
    return np.unique(np.sort(pairs,axis=1),axis=0)


def plan(stage):
    from fusion import cases, fit, spot_read, embeddings
    if stage=='calibration' and (RUN/'roma_pairs_calibration.npz').exists():
        # Calibration pair screening was fixed before precise spot rematching;
        # retain the same identities/pairs, not a new adaptive sample.
        return np.load(RUN/'roma_pairs_calibration.npz')['pairs']
    if stage=='calibration':
        m,_=metadata(); r=roles()
        c=m[m.image_id.isin(r.loc[(r.dataset==NAME)&(r.role=='calibrate'),'image_id'])].groupby('identity',sort=True).head(8)
        ids=c.image_id.to_numpy(); truth=c.identity.to_numpy(); a,b=np.triu_indices(len(ids),1)
        pairs=np.c_[ids[a],ids[b]]; positive=truth[a]==truth[b]
        sp,_=spot_read(pairs); lookup,z=embeddings(NAME,'trained'); f=z[[lookup[int(k)] for k in ids]]
        sim=(f@f.T)[a,b]; negatives=np.flatnonzero(~positive); rng=np.random.default_rng(770106)
        hard1=negatives[np.argsort(-sp[negatives,2])[:400]]
        hard2=negatives[np.argsort(-sim[negatives])[:400]]
        random=rng.choice(negatives,min(800,len(negatives)),replace=False)
        selected=np.unique(np.r_[np.flatnonzero(positive),hard1,hard2,random]); pairs=pairs[selected]
    else:
        import pickle
        if (RUN/'frozen.json').exists():
            with (RUN/'score_calibrators.pkl').open('rb') as f: calibrators=pickle.load(f)
        else: calibrators=fit()
        contexts=cases(stage,NAME,calibrators,True,False); candidates=[]
        for c in contexts:
            for source in ['base','spot_2','trained_raw']:
                sq,sr=c['scores'][source]
                candidates.append(nominate(sq,c['qi'],c['qi'],4,True))
                if len(c['ri']): candidates.append(nominate(sr,c['qi'],c['ri'],2,False))
        pairs=np.unique(np.sort(np.concatenate(candidates),axis=1),axis=0)
    path=RUN/f'roma_pairs_{stage}.npz'
    if path.exists():
        np.testing.assert_array_equal(np.load(path)['pairs'],pairs)
    else: np.savez_compressed(path,pairs=pairs)
    return pairs


class DenseMatcher:
    def __init__(self):
        from romav2 import RoMaV2
        torch.set_num_threads(8); torch.set_float32_matmul_precision('highest'); torch.manual_seed(770106)
        self.model=RoMaV2(RoMaV2.Cfg(setting='fast')); self.model.bidirectional=True
        m,_=metadata(); self.rows={int(r.image_id):r for r in m.itertuples()}; self.cache=OrderedDict()
        self.db=sqlite3.connect(RUN/'dense.sqlite')
        self.db.execute('CREATE TABLE IF NOT EXISTS matches(a INTEGER,b INTEGER,value BLOB,PRIMARY KEY(a,b))'); self.db.commit()

    def image(self,key):
        import cv2
        if key in self.cache:
            self.cache.move_to_end(key); return self.cache[key]
        im,mask=body(self.rows[key]); im=im.resize((512,512)); mask=cv2.resize(mask.astype(np.uint8),(512,512),interpolation=cv2.INTER_NEAREST)
        result=(to_tensor(im).cuda(),torch.tensor(mask.astype(np.float32),device='cuda')[None])
        self.cache[key]=result
        if len(self.cache)>128: self.cache.popitem(last=False)
        return result

    @torch.inference_mode()
    def compute(self,pairs,batch_size=4):
        missing=[]
        for pair in pairs:
            a,b=sorted(map(int,pair))
            if self.db.execute('SELECT 1 FROM matches WHERE a=? AND b=?',(a,b)).fetchone() is None: missing.append((a,b))
        start=time.monotonic()
        for offset in range(0,len(missing),batch_size):
            stop(); batch=missing[offset:offset+batch_size]
            p=[self.image(a) for a,b in batch]; q=[self.image(b) for a,b in batch]
            x0=torch.stack([x[0] for x in p]); m0=torch.stack([x[1] for x in p]); x1=torch.stack([x[0] for x in q]); m1=torch.stack([x[1] for x in q])
            pred=self.model.match(x0,x1)
            for i,pair in enumerate(batch):
                pr={k:v[i:i+1] if v is not None else None for k,v in pred.items()}
                value=(features(pr,x0[i:i+1],x1[i:i+1],m0[i:i+1],m1[i:i+1])+features(pr,x1[i:i+1],x0[i:i+1],m1[i:i+1],m0[i:i+1],True))/2
                assert np.isfinite(value).all()
                self.db.execute('INSERT INTO matches VALUES (?,?,?)',(*pair,value.astype(np.float32).tobytes()))
            self.db.commit()
            if offset%200==0: log(f'Dense matching {offset+len(batch)}/{len(missing)}, {(time.monotonic()-start)/max(1,offset+len(batch)):.3f} sec/pair')


def main(stage):
    pairs=plan(stage); log(f'Dense {stage}: {len(pairs)} canonical candidate pairs')
    model=DenseMatcher(); model.compute(pairs)
    write({'complete':True,'pairs':len(pairs),'columns':COLUMNS,'selector':'top4 QQ/top2 QR in baseline, spots and trained image DINO; calibration includes positives/hard/random negatives',
           'model_sha256':sha(ROOT/'models/torch_home/hub/checkpoints/romav2.0.1.pt')},RUN/f'roma_{stage}_complete.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['calibration','tune','audit','test'],required=True); a=p.parse_args(); main(a.stage)
