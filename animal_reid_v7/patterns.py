"""High-resolution spot matching with native LightGlue, without padded tokens.

Truncate at most 31 weakest real keypoints for exact-length batching. Two views
handle head/tail ambiguity. Every saved feature is a real point on the spot ROI.
"""
import argparse
from collections import defaultdict
import itertools
import sqlite3
import os
import gc
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import to_tensor
from settings import *
from images import native

NAME = 'SalamanderID2025'
COLUMNS = ['count_01', 'count_05', 'count_065', 'count_08', 'count_09', 'count_095',
           'confidence_sum', 'affine_inliers', 'coverage', 'real_points_min']
FEATURES = RUN / 'spot_features'


def pattern(row):
    im, mask = native(row); raw = np.asarray(im)
    # Work in native image coordinates before rotation/resizing.
    if max(raw.shape[:2]) > 1600:
        ratio = 1600 / max(raw.shape[:2]); size = (round(raw.shape[1]*ratio), round(raw.shape[0]*ratio))
        raw = cv2.resize(raw,size); mask = cv2.resize(mask.astype(np.uint8),size,interpolation=cv2.INTER_NEAREST)>0
    lab = cv2.cvtColor(raw, cv2.COLOR_RGB2LAB)
    if mask.sum() > 100:
        threshold, _ = cv2.threshold(lab[:,:,2][mask],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        spots = ((lab[:,:,2] > max(132, threshold)) & mask).astype(np.uint8)
        spots = cv2.morphologyEx(spots,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        radius = max(3, round(max(raw.shape[:2])*.008)) | 1
        spots = cv2.dilate(spots,np.ones((radius,radius),np.uint8)) & mask
        if spots.sum() < 100: spots = mask.astype(np.uint8)
    else: spots = mask.astype(np.uint8)
    yy,xx=np.where(mask)
    angle=0.
    if len(xx)>100:
        _,v=np.linalg.eigh(np.cov(np.c_[xx,yy].T)); d=v[:,-1]; angle=np.degrees(np.arctan2(d[1],d[0]))
    h,w=raw.shape[:2]; side=int(np.ceil(np.hypot(h,w)))
    transform=cv2.getRotationMatrix2D((w/2,h/2),angle,1.)
    transform[:,2] += np.array([(side-w)/2,(side-h)/2])
    rotated=cv2.warpAffine(raw,transform,(side,side)); fg=cv2.warpAffine(spots,transform,(side,side),flags=cv2.INTER_NEAREST)>0
    yy,xx=np.where(fg)
    if len(xx):
        x0,x1=max(0,xx.min()-8),min(side,xx.max()+9); y0,y1=max(0,yy.min()-8),min(side,yy.max()+9)
        rotated=rotated[y0:y1,x0:x1]; fg=fg[y0:y1,x0:x1]
    rotated[~fg]=0
    ratio=1024/max(rotated.shape[:2]); w=max(1,round(rotated.shape[1]*ratio)); h=max(1,round(rotated.shape[0]*ratio))
    canvas=np.zeros((1024,1024,3),np.uint8); outmask=np.zeros((1024,1024),bool)
    x=(1024-w)//2; y=(1024-h)//2
    canvas[y:y+h,x:x+w]=cv2.resize(rotated,(w,h)); outmask[y:y+h,x:x+w]=cv2.resize(fg.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0
    return canvas,outmask


def weights():
    folder=ROOT/'models/torch_home/hub/checkpoints'; folder.mkdir(parents=True,exist_ok=True)
    for name in ['superpoint_v1.pth','superpoint_lightglue_v0-1_arxiv.pth']:
        target=folder/name
        if not target.exists(): target.symlink_to(CV/'animal_reid_v3/models/torch_hub/checkpoints'/name)


@torch.inference_mode()
def extract():
    from lightglue import SuperPoint
    weights(); torch.set_num_threads(8); cv2.setNumThreads(1)
    model=SuperPoint(max_num_keypoints=1024,detection_threshold=.0005).eval().cuda()
    m,_=metadata(); frame=m[m.dataset==NAME]; FEATURES.mkdir(exist_ok=True)
    code_sha=sha(Path(__file__)); tiles=[]
    for i,row in enumerate(frame.itertuples()):
        stop(); path=FEATURES/f'{row.image_id}.pt'
        if path.exists(): continue
        im,mask=pattern(row); result=[]
        for rotate in [False,True]:
            a=np.rot90(im,2).copy() if rotate else im; fg=np.rot90(mask,2).copy() if rotate else mask
            f=model({'image':to_tensor(a).cuda()[None]})
            kp=f['keypoints'][0]; ix=kp.round().long().clamp(0,1023); valid=torch.tensor(fg,device='cuda')[ix[:,1],ix[:,0]]
            order=torch.argsort(f['keypoint_scores'][0][valid],descending=True)[:512]
            n=len(order); keep=n//32*32 if n>=32 else n
            result.append({'keypoints':kp[valid][order[:keep]].cpu().half(),
                           'descriptors':f['descriptors'][0][valid][order[:keep]].cpu().half()})
        save({'id':int(row.image_id),'views':result,'size':1024,'code_sha':code_sha},path)
        if i%100==0: log(f'Native spot extraction {i+1}/{len(frame)}')
    write({'complete':True,'images':len(frame),'resolution':1024,'max_real_points':512,'native_lightglue':True,
           'feature_sha':{p.name:sha(p) for p in FEATURES.glob('*.pt')}},RUN/'spot_features_manifest.json')


def pair_plan():
    """All query pairs; fixed reference shortlist avoids a million irrelevant pairs."""
    m,s=metadata(); indexed=m.set_index('image_id',drop=False); result={}
    r=roles(); cal=indexed.loc[r.loc[(r.dataset==NAME)&(r.role=='calibrate'),'image_id']]
    ids=cal.groupby('identity',sort=True).head(8).image_id.to_numpy()
    a,b=np.triu_indices(len(ids),1); result['calibration']=np.c_[ids[a],ids[b]]
    for stage in ['tune','audit']:
        o=np.load(V3/'scores'/f'{stage}_discovery_{NAME}.npz'); ids=o['qi']
        a,b=np.triu_indices(len(ids),1); result[stage]=np.c_[ids[a],ids[b]]
    o=np.load(V3/'scores'/f'production_{NAME}_blend.npz'); qi,ri=o['qi'],o['ri']
    a,b=np.triu_indices(len(qi),1); result['test_qq']=np.c_[qi[a],qi[b]]
    sr=o['sr']; k=min(32,len(ri)); near=np.argpartition(-sr,k-1,axis=1)[:,:k]
    result['test_qr']=np.c_[np.repeat(qi,k),ri[near.ravel()]]
    for k,v in result.items(): result[k]=np.unique(np.sort(v,axis=1),axis=0)
    return result


class SpotMatcher:
    def __init__(self):
        from lightglue import LightGlue
        weights(); torch.set_num_threads(8); cv2.setNumThreads(1)
        torch.set_float32_matmul_precision('highest')
        self.model=LightGlue(features='superpoint',depth_confidence=-1,width_confidence=-1,flash=False,mp=False).eval().cuda()
        self.bank={}; self.db=sqlite3.connect(RUN/'spots.sqlite')
        self.db.execute('CREATE TABLE IF NOT EXISTS matches(a INTEGER,b INTEGER,value BLOB,PRIMARY KEY(a,b))'); self.db.commit()

    def load(self,key,view):
        if key not in self.bank:
            o=torch.load(FEATURES/f'{key}.pt',weights_only=True,map_location='cpu')
            self.bank[key]=[{k:v.cuda() for k,v in x.items()} for x in o['views']]
        return self.bank[key][view]

    @torch.inference_mode()
    def compute(self,pairs,batch_size=64):
        pairs=np.unique(np.sort(np.asarray(pairs,dtype=np.int64).reshape(-1,2),axis=1),axis=0)
        missing=[tuple(map(int,p)) for p in pairs if self.db.execute('SELECT 1 FROM matches WHERE a=? AND b=?',tuple(map(int,p))).fetchone() is None]
        if not missing: return
        values=np.zeros((len(missing),2,len(COLUMNS)),np.float32)
        for view in [0,1]:
            groups=defaultdict(list)
            for i,(a,b) in enumerate(missing):
                x=self.load(a,0); y=self.load(b,view); groups[(len(x['keypoints']),len(y['keypoints']))].append(i)
            done=0
            for (n0,n1),indices in sorted(groups.items()):
                if not min(n0,n1): continue
                for start in range(0,len(indices),batch_size):
                    stop(); batch=indices[start:start+batch_size]
                    d0={k:torch.stack([self.load(missing[i][0],0)[k] for i in batch]) for k in ['keypoints','descriptors']}
                    d1={k:torch.stack([self.load(missing[i][1],view)[k] for i in batch]) for k in ['keypoints','descriptors']}
                    for d in [d0,d1]:
                        d['keypoints']=d['keypoints'].float(); d['descriptors']=d['descriptors'].float()
                        d['image_size']=torch.tensor([[1024,1024]],device='cuda').expand(len(batch),-1)
                    out=self.model({'image0':d0,'image1':d1})
                    matches=out['matches0'].cpu().numpy(); confidence=out['matching_scores0'].float().cpu().numpy()
                    kp0=d0['keypoints'].cpu().numpy(); kp1=d1['keypoints'].cpu().numpy()
                    for j,i in enumerate(batch):
                        good=matches[j]>=0; p=confidence[j][good]; a=kp0[j][good]; b=kp1[j][matches[j][good]]
                        strong=p>=.65; a=a[strong]; b=b[strong]; nin=0; coverage=0.
                        if len(a)>=4:
                            cv2.setRNGSeed(71006)
                            _,mask=cv2.estimateAffine2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=10,maxIters=500,confidence=.99)
                            if mask is not None: nin=int(mask.sum())
                            coverage=min(cv2.contourArea(cv2.convexHull(a)),cv2.contourArea(cv2.convexHull(b)))/1024**2
                        values[i,view]=[*[np.log1p((p>=t).sum()) for t in [.1,.5,.65,.8,.9,.95]],np.log1p(p.sum()),np.log1p(nin),coverage,min(n0,n1)/512]
                    done+=len(batch)
                    if done%5000<len(batch): log(f'Spot matching view {view}, {done}/{len(missing)} pairs')
        # Keep evidence from the stronger orientation, instead of mixing columns from different matches.
        choice=values[:,:,2].argmax(1); result=values[np.arange(len(values)),choice]
        for pair,value in zip(missing,result): self.db.execute('INSERT INTO matches VALUES (?,?,?)',(*pair,value.tobytes()))
        self.db.commit(); log(f'Committed {len(missing)} native spot pairs')

    def read(self,pairs):
        out=[]
        for p in pairs:
            a,b=sorted(map(int,p)); row=self.db.execute('SELECT value FROM matches WHERE a=? AND b=?',(a,b)).fetchone()
            if row is None: raise KeyError((a,b))
            out.append(np.frombuffer(row[0],np.float32))
        return np.asarray(out)


def match(stage):
    matcher=SpotMatcher(); plan=pair_plan(); write({k:len(v) for k,v in plan.items()},RUN/'spot_pair_plan.json')
    selected={'tune':['calibration','tune'],'audit':['audit'],
              'test':['test_qq','test_qr'],'complete':['audit','test_qq','test_qr']}[stage]
    for key in selected:
        # Commit chunks so interrupted long matching can resume.
        for offset in range(0,len(plan[key]),4096): matcher.compute(plan[key][offset:offset+4096])
    matcher.db.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['extract','tune','audit','test','complete'],required=True); a=p.parse_args()
    extract() if a.stage=='extract' else match(a.stage)
