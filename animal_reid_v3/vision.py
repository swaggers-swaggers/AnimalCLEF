"""Species preprocessing, foreground DINO tokens, and signed KAZE descriptors."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import timm
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from safetensors.torch import load_file
from common import ROOT, RUN, DATA, log, seed, atomic_torch, check_stop
from extract import read_image, load_features
from models import load_global

PATTERN_SPECIES=['SalamanderID2025','TexasHornedLizards']

def specimen(row, enhance=True):
    im=np.asarray(read_image(row).resize((512,512),Image.Resampling.BICUBIC)).copy()
    path=RUN/'masks'/f'{row.image_id}.png'
    mask=cv2.imread(str(path),0)>0 if path.exists() else np.ones((512,512),bool)
    if row.dataset=='TexasHornedLizards':
        # A conservative central belly ROI removes fingers and spiny flank texture.
        yy,xx=np.where(mask)
        if len(xx):
            cx=(xx.min()+xx.max())/2;cy=(yy.min()+yy.max())/2
            rx=max((xx.max()-xx.min())*.32,30);ry=max((yy.max()-yy.min())*.36,40)
            y,x=np.ogrid[:512,:512];mask &= ((x-cx)/rx)**2+((y-cy)/ry)**2<1
    yy,xx=np.where(mask)
    if row.dataset in PATTERN_SPECIES and len(xx)>500:
        box=(max(0,int(xx.min())-12),max(0,int(yy.min())-12),min(512,int(xx.max())+13),min(512,int(yy.max())+13))
    else:box=(0,0,512,512)
    if enhance:
        if row.dataset=='LynxID2025':
            lab=cv2.cvtColor(im,cv2.COLOR_RGB2LAB);lab[:,:,0]=cv2.createCLAHE(2.,(4,4)).apply(lab[:,:,0]);im=cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)
            im=cv2.addWeighted(im,1.3,cv2.GaussianBlur(im,(0,0),1.),-.3,0)
        elif row.dataset=='SeaTurtleID2022':
            x=im.astype(np.float32)/255.;x[:,:,0]*=1.15;im=(np.clip(x,0,1)**1.15*255).astype(np.uint8)
        elif row.dataset=='SalamanderID2025':
            lab=cv2.cvtColor(im,cv2.COLOR_RGB2LAB);lab[:,:,0]=cv2.createCLAHE(1.5,(4,4)).apply(lab[:,:,0]);im=cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)
    a,b,c,d=box
    return Image.fromarray(im[b:d,a:c]),mask[b:d,a:c],box

class Specimens(Dataset):
    def __init__(self,frame,size):self.rows=list(frame.itertuples());self.size=size
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i];im,mask,box=specimen(row)
        x=TF.normalize(TF.to_tensor(im.resize((self.size,self.size),Image.Resampling.BICUBIC)),[.485,.456,.406],[.229,.224,.225])
        mask=torch.from_numpy(cv2.resize(mask.astype(np.uint8),(self.size//16,self.size//16),interpolation=cv2.INTER_NEAREST).astype(bool))
        return int(row.image_id),x,mask,torch.tensor(box),row.dataset in PATTERN_SPECIES

def dino_model():
    model=timm.create_model('vit_large_patch16_dinov3.lvd1689m',pretrained=False,img_size=384,num_classes=0)
    model.load_state_dict(load_file(str(ROOT/'models/dinov3/model.safetensors')),strict=True)
    return model.eval().cuda()

@torch.inference_mode()
def extract_global(name, limit=None):
    m=pd.read_csv(DATA/'metadata.csv');folder=RUN/'features'/name;folder.mkdir(exist_ok=True)
    done=set()
    for p in folder.glob('part_*.pt'):done.update(torch.load(p,weights_only=True)['ids'].tolist())
    todo=m[~m.image_id.isin(done)]
    if limit:todo=todo.groupby('dataset',sort=False).head(limit)
    if todo.empty:return
    if name=='dinov3':model=dino_model();size=384;batch=8
    else:
        model,size=load_global('miewid');batch=16
        if name=='miew_arc':model.load_state_dict(torch.load(RUN/'miew_arc_best.pt',weights_only=True)['state_dict'])
    loader=DataLoader(Specimens(todo,size),batch_size=batch,num_workers=6,pin_memory=True)
    patchfolder=RUN/'features/dino_patch';patchfolder.mkdir(exist_ok=True)
    log(f'{name}: extracting {len(todo)} images with species preprocessing')
    for ids,x,mask,boxes,pattern in loader:
        check_stop();x=x.cuda(non_blocking=True);emb=[];pooled=[];original_tokens=None
        for flip in (False,True):
            with torch.autocast('cuda',dtype=torch.float16):
                if name=='dinov3':
                    tokens=model.forward_features(x.flip(-1) if flip else x)
                    p=tokens[:,model.num_prefix_tokens:].float()
                    fg=mask.flip(-1) if flip else mask
                    fg=fg.flatten(1).cuda().float();fg=torch.where(fg.sum(1,keepdim=True)>0,fg,torch.ones_like(fg))
                    emb.append(F.normalize(tokens[:,0].float(),dim=1))
                    pooled.append(F.normalize((p*fg[:,:,None]).sum(1)/fg.sum(1,keepdim=True).clamp_min(1),dim=1))
                    if not flip:original_tokens=F.normalize(p,dim=-1).cpu()
                else:emb.append(F.normalize(model(x.flip(-1) if flip else x).float(),dim=1))
        result={'ids':ids,'features':torch.stack(emb,1).cpu().half()}
        if pooled:result['pooled']=torch.stack(pooled,1).cpu().half()
        if not torch.isfinite(result['features']).all():raise RuntimeError('Nonfinite global features')
        if name=='dinov3' and pattern.any():
            ix=torch.nonzero(pattern,as_tuple=True)[0];allkp=[];alldesc=[];lengths=[]
            for k in ix.tolist():
                # A spatial grid prevents a single textured patch dominating descriptors.
                candidates=np.arange(24*24);inside=mask[k].numpy().flatten()
                grid=candidates.reshape(24,24)[::2,::2].flatten()
                grid=grid[inside[grid]]
                rest=np.setdiff1d(candidates[inside],grid)
                chosen=np.r_[grid,rest]
                if len(chosen)>128:chosen=chosen[np.linspace(0,len(chosen)-1,128).astype(int)]
                n=len(chosen);lengths.append(n)
                desc=original_tokens[k,chosen];a,b,c,d=boxes[k].tolist()
                kp=np.c_[((chosen%24+.5)/24)*(c-a)+a,((chosen//24+.5)/24)*(d-b)+b]
                allkp.append(F.pad(torch.tensor(kp,dtype=torch.float16),(0,0,0,128-n)))
                alldesc.append(F.pad(desc.half(),(0,0,0,128-n)))
            atomic_torch({'ids':ids[ix],'keypoints':torch.stack(allkp),'descriptors':torch.stack(alldesc),'lengths':torch.tensor(lengths)},patchfolder/f'part_{int(ids[0]):06d}.pt')
        atomic_torch(result,folder/f'part_{int(ids[0]):06d}.pt')
    del model;gc.collect();torch.cuda.empty_cache();log(f'{name}: extraction completed')

def kaze_one(row):
    im=np.asarray(read_image(row).resize((512,512),Image.Resampling.BICUBIC));gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
    gray=cv2.createCLAHE(2.,(6,6)).apply(gray)
    path=RUN/'masks'/f'{row.image_id}.png';mask=cv2.imread(str(path),0) if path.exists() else np.full((512,512),255,np.uint8)
    if row.dataset=='TexasHornedLizards':
        _,roi,box=specimen(row,False);mask[:]=0;a,b,c,d=box;mask[b:d,a:c]=roi.astype(np.uint8)*255
    detector=cv2.KAZE_create(extended=False,upright=False,threshold=.0003,nOctaves=3)
    kp,desc=detector.detectAndCompute(gray,mask)
    order=np.argsort([-p.response for p in kp])[:512];n=len(order)
    points=np.zeros((512,2),np.float16);features=np.zeros((512,64),np.float16)
    if n:
        points[:n]=np.array([kp[i].pt for i in order]);v=desc[order];v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-8);features[:n]=v
    return int(row.image_id),points,features,n

def extract_kaze():
    cv2.setNumThreads(1);m=pd.read_csv(DATA/'metadata.csv');m=m[m.dataset.isin(PATTERN_SPECIES)]
    folder=RUN/'features/kaze';folder.mkdir(exist_ok=True);done=set()
    for p in folder.glob('part_*.pt'):done.update(torch.load(p,weights_only=True)['ids'].tolist())
    rows=list(m[~m.image_id.isin(done)].itertuples());log(f'KAZE: extracting {len(rows)} images')
    with ThreadPoolExecutor(max_workers=6) as pool:
        for start in range(0,len(rows),64):
            check_stop();result=list(pool.map(kaze_one,rows[start:start+64]));ids,kp,desc,lengths=zip(*result)
            atomic_torch({'ids':torch.tensor(ids),'keypoints':torch.tensor(np.stack(kp)),'descriptors':torch.tensor(np.stack(desc)),'lengths':torch.tensor(lengths)},folder/f'part_{ids[0]:06d}.pt')
    log('KAZE: extraction completed')

def pattern_image(row):
    native=read_image(row)
    _,body,box=specimen(row,False);a,b,c,d=box
    full=np.zeros((512,512),np.uint8);full[b:d,a:c]=body.astype(np.uint8)*255
    # Preserve the specimen's physical pattern shape before any local extraction.
    ratio=512/max(native.size);width=max(1,round(native.width*ratio));height=max(1,round(native.height*ratio))
    left=(512-width)//2;top=(512-height)//2
    raw=np.zeros((512,512,3),np.uint8);raw[top:top+height,left:left+width]=np.asarray(native.resize((width,height),Image.Resampling.BICUBIC))
    letter=np.array([[width/512,0,left],[0,height/512,top],[0,0,1.]])
    oldmask=full;full=np.zeros((512,512),np.uint8);full[top:top+height,left:left+width]=cv2.resize(oldmask,(width,height),interpolation=cv2.INTER_NEAREST)
    angle=0.
    if row.dataset=='SalamanderID2025':
        yy,xx=np.where(full>0)
        if len(xx)>100:
            _,vectors=np.linalg.eigh(np.cov(np.c_[xx,yy].T));direction=vectors[:,-1];angle=np.degrees(np.arctan2(direction[1],direction[0]))
        hsv=cv2.cvtColor(raw,cv2.COLOR_RGB2HSV)
        yellow=((hsv[:,:,0]>=12)&(hsv[:,:,0]<=50)&(hsv[:,:,1]>65)&(hsv[:,:,2]>60)&(full>0)).astype(np.uint8)
        if yellow.sum()>100:
            yellow=cv2.dilate(yellow,np.ones((5,5),np.uint8),iterations=1);full=full*yellow
    rotation=cv2.getRotationMatrix2D((256,256),angle,1.);rotation[:,2]+=128
    im=cv2.warpAffine(raw,rotation,(768,768));mask=cv2.warpAffine(full,rotation,(768,768),flags=cv2.INTER_NEAREST)
    yy,xx=np.where(mask>0)
    if len(xx)>100:a,b,c,d=max(0,xx.min()-8),max(0,yy.min()-8),min(768,xx.max()+9),min(768,yy.max()+9)
    else:a,b,c,d=128,128,640,640;mask[:]=255
    im[mask==0]=0
    factor=512/max(c-a,d-b);tw=max(1,round((c-a)*factor));th=max(1,round((d-b)*factor));px=(512-tw)//2;py=(512-th)//2
    image_canvas=np.zeros((512,512,3),np.uint8);mask_canvas=np.zeros((512,512),np.uint8)
    image_canvas[py:py+th,px:px+tw]=cv2.resize(im[b:d,a:c],(tw,th))
    mask_canvas[py:py+th,px:px+tw]=cv2.resize(mask[b:d,a:c],(tw,th),interpolation=cv2.INTER_NEAREST)
    im,mask=image_canvas,mask_canvas
    affine=np.vstack([rotation,[0,0,1.]])
    scale=np.array([[tw/(c-a),0,px-a*tw/(c-a)],[0,th/(d-b),py-b*th/(d-b)],[0,0,1.]])
    return im,mask,np.linalg.inv(scale@affine@letter).astype(np.float32)

@torch.inference_mode()
def extract_pattern(branch='superpoint_pattern'):
    import sys
    sys.path.insert(0,str(ROOT.parent/'references/lightglue'))
    from lightglue import SuperPoint
    torch.hub.set_dir(str(ROOT/'models/torch_hub'))
    model=SuperPoint(max_num_keypoints=512,detection_threshold=.001).eval().cuda()
    m=pd.read_csv(DATA/'metadata.csv');m=m[m.dataset.isin(PATTERN_SPECIES)]
    folder=RUN/'features'/branch;folder.mkdir(exist_ok=True);done=set()
    for p in folder.glob('part_*.pt'):done.update(torch.load(p,weights_only=True)['ids'].tolist())
    log(f'SuperPoint patterns: extracting {len(m)-len(done)} images')
    batch=[]
    for row in m[~m.image_id.isin(done)].itertuples():
        check_stop();im,mask,inverse=pattern_image(row);x=TF.to_tensor(im).cuda()[None];points=[];descs=[]
        for flip in (False,True):
            offset=np.array([0.,0.]);scale=np.array([1.,1.]);view=x
            if branch=='superpoint_regions':
                if row.dataset=='SalamanderID2025':
                    left=332 if flip else 0;right=512 if flip else 180;top=0;bottom=512
                else:
                    left,top,right,bottom=128,96,384,416
                yy,xx=np.where(mask[top:bottom,left:right]>0)
                if len(xx)>20:
                    ox,oy=left,top;left=max(0,ox+int(xx.min())-3);right=min(512,ox+int(xx.max())+4);top=max(0,oy+int(yy.min())-3);bottom=min(512,oy+int(yy.max())+4)
                factor=512/max(right-left,bottom-top);vw=max(1,round((right-left)*factor));vh=max(1,round((bottom-top)*factor));px=(512-vw)//2;py=(512-vh)//2
                view=F.pad(F.interpolate(x[:,:,top:bottom,left:right],(vh,vw),mode='bilinear',align_corners=False),(px,512-vw-px,py,512-vh-py))
                scale=np.array([(right-left)/vw,(bottom-top)/vh]);offset=np.array([left-px*scale[0],top-py*scale[1]])
            f=model.extract(view.flip((-2,-1)) if flip else view,resize=None)
            kp=f['keypoints'][0].float().cpu().numpy();desc=f['descriptors'][0].cpu()
            if flip:kp=511-kp
            kp=kp*scale+offset
            xy=kp.round().astype(int).clip(0,511);valid=mask[xy[:,1],xy[:,0]]>0
            kp=kp[valid][:256];desc=desc[torch.from_numpy(valid)][:256]
            kp=(np.c_[kp,np.ones(len(kp))]@inverse.T)[:,:2]
            points.append(torch.tensor(kp).half());descs.append(desc.half())
        kp=torch.cat(points);desc=torch.cat(descs);n=len(kp)
        batch.append((int(row.image_id),F.pad(kp,(0,0,0,512-n)),F.pad(desc,(0,0,0,512-n)),n))
        if len(batch)==32:
            ids,kp,desc,n=zip(*batch);atomic_torch({'ids':torch.tensor(ids),'keypoints':torch.stack(kp),'descriptors':torch.stack(desc),'lengths':torch.tensor(n)},folder/f'part_{ids[0]:06d}.pt');batch=[]
    if batch:
        ids,kp,desc,n=zip(*batch);atomic_torch({'ids':torch.tensor(ids),'keypoints':torch.stack(kp),'descriptors':torch.stack(desc),'lengths':torch.tensor(n)},folder/f'part_{ids[0]:06d}.pt')
    log('SuperPoint pattern extraction completed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--branch',choices=['dinov3','miew_pp','miew_arc','kaze','superpoint_pattern','superpoint_regions'],required=True);p.add_argument('--limit',type=int);a=p.parse_args();seed()
    if a.branch=='kaze':extract_kaze()
    elif a.branch.startswith('superpoint_'):extract_pattern(a.branch)
    else:extract_global(a.branch,a.limit)
