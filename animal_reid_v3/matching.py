"""Batched LightGlue with transactional, symmetric pair caching."""
import gc
import sqlite3
import sys
import time
import cv2
import numpy as np
import torch
from common import ROOT, RUN, log, check_stop
from extract import load_features
sys.path.insert(0, str(ROOT.parent / 'references' / 'lightglue'))

LOCAL_COLUMNS = ['log_foreground_matches','log_strong_foreground','confidence', 'log_inliers', 'inlier_ratio', 'coverage', 'log_center','log_raw_matches','foreground_ratio']

class LocalMatcher:
    def __init__(self):
        self.db = sqlite3.connect(RUN / 'matches.sqlite')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS matches (branch TEXT, a INTEGER, b INTEGER, value BLOB, PRIMARY KEY(branch,a,b))')
        self.branch = None
        self.model = None
        self.features = None

    def load(self, name):
        from lightglue import LightGlue
        if name == self.branch:return
        self.model = None; self.features = None
        gc.collect();torch.cuda.empty_cache()
        torch.hub.set_dir(str(ROOT / 'models' / 'torch_hub'))
        f = load_features(name)
        if len(f['ids'])==0 or len(set(f['ids'].tolist()))!=len(f['ids']):
            raise RuntimeError(f'{name}: extraction incomplete; complete extraction before pair matching')
        self.lookup = {int(k):i for i,k in enumerate(f['ids'])}
        self.lengths=f['lengths'].numpy()
        self.features = {k: v.cuda() for k,v in f.items() if k in ('keypoints','descriptors')}
        self.foreground=np.ones(f['keypoints'].shape[:2],bool)
        for i,image_id in enumerate(f['ids'].tolist()):
            path=RUN/'masks'/f'{image_id}.png'
            if path.exists():
                mask=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
                points=f['keypoints'][i].float().numpy().round().astype(int).clip(0,511)
                self.foreground[i]=mask[points[:,1],points[:,0]]>0
        feature_name='superpoint' if name.startswith('superpoint_') else name
        self.model = LightGlue(features=feature_name, depth_confidence=-1, width_confidence=-1, flash=True, mp=True).eval().cuda() if name not in ('rootsift','kaze','dino_patch') else None
        self.branch = name

    @torch.inference_mode()
    def compute(self, pairs, name, batch_size=128):
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1,2)
        pairs = np.sort(pairs, axis=1)
        output = np.zeros((len(pairs), len(LOCAL_COLUMNS)), np.float32)
        missing = {}
        cursor = self.db.cursor()
        for i, (a,b) in enumerate(pairs):
            row = cursor.execute('SELECT value FROM matches WHERE branch=? AND a=? AND b=?',('fg_v1_'+name,int(a),int(b))).fetchone()
            if row:output[i] = np.frombuffer(row[0],dtype=np.float32)
            else:missing.setdefault((int(a),int(b)),[]).append(i)
        if not missing:return output
        self.load(name)
        groups={}
        for pair in missing:
            lengths=(self.features['keypoints'].shape[1],)*2 if name in ('rootsift','kaze','dino_patch') else tuple(int(self.lengths[self.lookup[x]]) for x in pair)
            if name.startswith('superpoint_'):lengths=tuple(((x+63)//64)*64 for x in lengths)
            groups.setdefault(lengths,[]).append(pair)
        todo=[]
        for lengths,group in groups.items():
            todo.extend((lengths,group[o:o+batch_size]) for o in range(0,len(group),batch_size))
        started = time.monotonic()
        log(f'{name}: matching {len(missing)} uncached pairs (batch={batch_size}, length groups={len(groups)})')
        for (length0,length1),batch in todo:
            check_stop()
            if min(length0,length1)==0:
                value=np.zeros(len(LOCAL_COLUMNS),np.float32)
                for pair in batch:
                    cursor.execute('INSERT OR REPLACE INTO matches VALUES (?,?,?,?)',('fg_v1_'+name,*pair,value.tobytes()))
                    output[missing[pair]]=value
                self.db.commit();continue
            i0 = torch.tensor([self.lookup[a] for a,b in batch], device='cuda')
            i1 = torch.tensor([self.lookup[b] for a,b in batch], device='cuda')
            d0 = {k: v[i0,:length0] for k,v in self.features.items()}
            d1 = {k: v[i1,:length1] for k,v in self.features.items()}
            d0['keypoints'] = d0['keypoints'].float()
            d1['keypoints'] = d1['keypoints'].float()
            d0['image_size'] = d1['image_size'] = torch.tensor([[512,512]],device='cuda').expand(len(batch),-1)
            with torch.autocast('cuda',dtype=torch.float16):
                if name in ('rootsift','kaze','dino_patch'):
                    # RootSIFT descriptors have unit L2 norm. Mutual Lowe ratio matching.
                    similarity=torch.bmm(d0['descriptors'],d1['descriptors'].transpose(1,2)).float()
                    distance=(2-2*similarity).clamp_min(0)
                    if min(length0,length1)<2:
                        result={'matches0':torch.full((len(batch),length0),-1,device='cuda',dtype=torch.long),'matching_scores0':torch.zeros((len(batch),length0),device='cuda')}
                    else:
                        best,index=distance.topk(2,dim=2,largest=False)
                        backward=distance.argmin(1)
                        match=index[:,:,0]
                        mutual=backward.gather(1,match)==torch.arange(length0,device='cuda')[None]
                        ratio=best[:,:,0]/best[:,:,1].clamp_min(1e-8)
                        valid=mutual&(ratio<.75**2)&(best[:,:,1]>1e-8)
                        if name=='dino_patch':
                            valid=mutual&(best[:,:,0]<.7)&((best[:,:,1]-best[:,:,0])>.025)
                        valid &= torch.arange(length0,device='cuda')[None] < torch.as_tensor(self.lengths[[self.lookup[a] for a,b in batch]],device='cuda')[:,None]
                        valid &= match < torch.as_tensor(self.lengths[[self.lookup[b] for a,b in batch]],device='cuda')[:,None]
                        result={'matches0':torch.where(valid,match,-1),'matching_scores0':(1-ratio).clamp(0,1)}
                elif name.startswith('superpoint_'):
                    result = self.masked_lightglue(d0,d1,torch.as_tensor(self.lengths[[self.lookup[a] for a,b in batch]],device='cuda'),torch.as_tensor(self.lengths[[self.lookup[b] for a,b in batch]],device='cuda'))
                else:result = self.model({'image0':d0, 'image1':d1})
            matches = result['matches0'].cpu().numpy()
            confidence = result['matching_scores0'].float().cpu().numpy()
            kp0 = d0['keypoints'].cpu().numpy();kp1 = d1['keypoints'].cpu().numpy()
            for k,pair in enumerate(batch):
                valid = matches[k]>=0
                raw_n=int(valid.sum())
                valid &= self.foreground[self.lookup[pair[0]],:length0] & self.foreground[self.lookup[pair[1]]][matches[k].clip(0)]
                conf = confidence[k][valid]
                p0 = kp0[k][valid];p1 = kp1[k][matches[k][valid]]
                n = len(conf);nin = 0;coverage = 0.;center = 0
                if n>=4:
                    # Similarity geometry is evidence, never a mandatory hard gate for deformable animals.
                    cv2.setRNGSeed(20261104)
                    if name in ('kaze','dino_patch') or name.startswith('superpoint_'):
                        _,mask = cv2.estimateAffine2D(p0,p1,method=cv2.RANSAC,ransacReprojThreshold=5.,maxIters=500,confidence=.99)
                    else:
                        _,mask = cv2.estimateAffinePartial2D(p0,p1,method=cv2.RANSAC,ransacReprojThreshold=8.,maxIters=300,confidence=.99)
                    if mask is not None:nin = int(mask.sum())
                    coverage = min(float(cv2.contourArea(cv2.convexHull(p0))),float(cv2.contourArea(cv2.convexHull(p1)))) / 512**2
                    center = int((((p0[:,0]>80)&(p0[:,0]<432)&(p0[:,1]>40)&(p0[:,1]<472)) & ((p1[:,0]>80)&(p1[:,0]<432)&(p1[:,1]>40)&(p1[:,1]<472))).sum())
                value = np.array([np.log1p(n),np.log1p((conf>.9).sum()),float(conf.mean()) if n else 0.,np.log1p(nin),nin/max(n,1),coverage,np.log1p(center),np.log1p(raw_n),n/max(raw_n,1)],dtype=np.float32)
                if not np.isfinite(value).all():raise ValueError(f'Nonfinite matching evidence for {pair}')
                cursor.execute('INSERT OR REPLACE INTO matches VALUES (?,?,?,?)',('fg_v1_'+name,*pair,value.tobytes()))
                output[missing[pair]] = value
            self.db.commit()
        log(f'{name}: matched in {time.monotonic()-started:.1f}s')
        return output

    def masked_lightglue(self,d0,d1,lengths0,lengths1):
        """Exact variable-length attention and assignment in padded GPU batches."""
        from lightglue.lightglue import normalize_keypoints
        import torch.nn.functional as F
        model=self.model;x0=model.input_proj(d0['descriptors']);x1=model.input_proj(d1['descriptors'])
        m0=torch.arange(x0.shape[1],device='cuda')[None]<lengths0[:,None]
        m1=torch.arange(x1.shape[1],device='cuda')[None]<lengths1[:,None]
        e0=model.posenc(normalize_keypoints(d0['keypoints'],d0['image_size']).clone());e1=model.posenc(normalize_keypoints(d1['keypoints'],d1['image_size']).clone())
        for layer in model.transformers:x0,x1=layer(x0,x1,e0,e1,mask0=m0[:,None,:,None],mask1=m1[:,None,:,None])
        head=model.log_assignment[-1];z0=head.matchability(x0);z1=head.matchability(x1)
        a=head.final_proj(x0);b=head.final_proj(x1);sim=torch.bmm(a,b.transpose(1,2))/(a.shape[-1]**.5)
        valid=m0[:,:,None]&m1[:,None,:];sim=sim.masked_fill(~valid,-1e4)
        scores=F.log_softmax(sim,2)+F.log_softmax(sim,1)+F.logsigmoid(z0)+F.logsigmoid(z1).transpose(1,2)
        scores=scores.masked_fill(~valid,-1e4)
        best,match=scores.max(2);back=scores.argmax(1)
        mutual=back.gather(1,match)==torch.arange(x0.shape[1],device='cuda')[None]
        confidence=best.exp();good=mutual&m0&(confidence>model.conf.filter_threshold)
        return {'matches0':torch.where(good,match,-1),'matching_scores0':torch.where(good,confidence,0.)}

    def close(self):
        self.db.close(); self.model=None;self.features=None
        gc.collect();torch.cuda.empty_cache()
