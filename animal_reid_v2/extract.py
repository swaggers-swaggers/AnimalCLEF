"""Resumable multi-view wildlife embeddings and learned local keypoints."""
import argparse
import gc
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from common import ROOT, RUN, DATA, atomic_torch, check_stop, seed, log, json_write, fingerprint

sys.path.insert(0, str(ROOT.parent / 'references' / 'lightglue'))

def read_image(row):
    with Image.open(DATA / row.path) as f:
        image = ImageOps.exif_transpose(f).convert('RGB')
    # Remove large black padding in camera-trap crops, preserving the full animal.
    if row.dataset == 'LynxID2025':
        a = np.asarray(image)
        mask = a.max(2) > 12
        yy, xx = np.where(mask)
        if len(xx):
            box = (max(0, xx.min()-4), max(0, yy.min()-4), min(image.width, xx.max()+5), min(image.height, yy.max()+5))
            if (box[2]-box[0])*(box[3]-box[1]) > image.width*image.height*.25:
                image = image.crop(box)
    return image

class Images(Dataset):
    def __init__(self, frame, size, normalize=True):
        self.rows = list(frame.itertuples())
        self.size = size
        self.normalize = normalize

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        im = read_image(row)
        x = TF.to_tensor(im.resize((self.size, self.size), Image.Resampling.BICUBIC))
        if self.normalize:
            x = TF.normalize(x, [.485,.456,.406], [.229,.224,.225])
        return int(row.image_id), x

@torch.inference_mode()
def globals_extract(metadata, name, batch_size=16):
    from models import load_global
    folder = RUN / 'features' / name
    folder.mkdir(parents=True, exist_ok=True)
    completed = set()
    for path in folder.glob('part_*.pt'):
        completed.update(torch.load(path, weights_only=True)['ids'].tolist())
    todo = metadata[~metadata.image_id.isin(completed)]
    if todo.empty:
        log(f'{name}: all embeddings cached'); return
    model, size = load_global(name)
    loader = DataLoader(Images(todo, size), batch_size=batch_size, num_workers=6, pin_memory=True)
    log(f'{name}: extracting {len(todo)} images, two views, size={size}')
    for ids, x in loader:
        check_stop()
        x = x.cuda(non_blocking=True)
        with torch.autocast('cuda', dtype=torch.float16):
            a = F.normalize(model(x).float(), dim=1)
            b = F.normalize(model(x.flip(-1)).float(), dim=1)
        # Retain original and flip separately: max-of-view similarities preserve sided patterns.
        features = torch.stack([a, b], 1).cpu().half()
        assert torch.isfinite(features).all()
        atomic_torch({'ids':ids, 'features':features}, folder / f'part_{int(ids[0]):06d}.pt')
    del model
    gc.collect(); torch.cuda.empty_cache()
    log(f'{name}: finished')

@torch.inference_mode()
def locals_extract(metadata, name='aliked', keypoints=512):
    from lightglue import ALIKED, DISK
    folder = RUN / 'features' / name
    folder.mkdir(parents=True, exist_ok=True)
    completed = set()
    for path in folder.glob('part_*.pt'):
        completed.update(torch.load(path, weights_only=True)['ids'].tolist())
    todo = metadata[~metadata.image_id.isin(completed)]
    if todo.empty:
        log(f'{name}: all keypoints cached');return
    if name == 'aliked':
        model = ALIKED(max_num_keypoints=keypoints, detection_threshold=0.0).eval().cuda()
    else:
        model = DISK(max_num_keypoints=keypoints).eval().cuda()
    loader = DataLoader(Images(todo, 512, normalize=False), batch_size=8, num_workers=6, pin_memory=True)
    log(f'{name}: extracting {len(todo)} images with {keypoints} keypoints')
    for ids, x in loader:
        check_stop()
        if name=='disk':
            found=model.model(x.cuda(non_blocking=True),n=keypoints,window_size=5,score_threshold=0.0,pad_if_not_divisible=True)
            lengths=torch.tensor([len(f.keypoints) for f in found])
            result={
                'keypoints':torch.stack([F.pad(f.keypoints,(0,0,0,keypoints-len(f.keypoints))) for f in found]),
                'descriptors':torch.stack([F.pad(f.descriptors,(0,0,0,keypoints-len(f.keypoints))) for f in found]),
                'keypoint_scores':torch.stack([F.pad(f.detection_scores,(0,keypoints-len(f.keypoints))) for f in found]),
            }
        else:
            result = model({'image': x.cuda(non_blocking=True)})
            lengths=torch.full((len(ids),),keypoints)
        values = {k: v.cpu().half() for k,v in result.items() if k in ('keypoints','descriptors','keypoint_scores')}
        assert values['keypoints'].shape[1] == keypoints
        assert all(torch.isfinite(v).all() for v in values.values())
        atomic_torch({'ids': ids, 'lengths':lengths, **values}, folder / f'part_{int(ids[0]):06d}.pt')
    del model
    gc.collect();torch.cuda.empty_cache()
    log(f'{name}: finished')

def load_features(name):
    items = [torch.load(p, weights_only=True) for p in sorted((RUN / 'features' / name).glob('part_*.pt'))]
    if not items: raise FileNotFoundError(f'No features for {name}')
    for item in items:
        if 'keypoints' in item and 'lengths' not in item:
            item['lengths']=torch.full((len(item['ids']),),item['keypoints'].shape[1])
    result = {k: torch.cat([x[k] for x in items]) for k in items[0]}
    order = result['ids'].argsort()
    return {k:v[order] for k,v in result.items()}

def rootsift_extract(metadata,keypoints=512):
    import cv2
    from concurrent.futures import ThreadPoolExecutor
    cv2.setNumThreads(1)
    folder=RUN/'features'/'rootsift';folder.mkdir(parents=True,exist_ok=True)
    done=set()
    for path in folder.glob('part_*.pt'):done.update(torch.load(path,weights_only=True)['ids'].tolist())
    rows=list(metadata[~metadata.image_id.isin(done)].itertuples())
    log(f'RootSIFT: extracting {len(rows)} images on CPU')
    def one(row):
        image=np.asarray(read_image(row).resize((512,512),Image.Resampling.BICUBIC))
        kpts,desc=cv2.SIFT_create(nfeatures=keypoints,contrastThreshold=.02).detectAndCompute(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),None)
        order=np.argsort([-p.response for p in kpts])[:keypoints]
        xy=np.zeros((keypoints,2),np.float32);des=np.zeros((keypoints,128),np.float32);scores=np.zeros(keypoints,np.float32)
        if len(order):
            d=desc[order];d=d/np.maximum(d.sum(1,keepdims=True),1e-12);d=np.sqrt(d)
            xy[:len(order)]=np.array([kpts[i].pt for i in order]);des[:len(order)]=d;scores[:len(order)]=[kpts[i].response for i in order]
        return int(row.image_id),len(order),xy,des,scores
    with ThreadPoolExecutor(max_workers=6) as pool:
        for offset in range(0,len(rows),64):
            check_stop();values=list(pool.map(one,rows[offset:offset+64]))
            ids,lengths,kp,desc,scores=zip(*values)
            atomic_torch({'ids':torch.tensor(ids),'lengths':torch.tensor(lengths),'keypoints':torch.tensor(np.stack(kp)).half(),'descriptors':torch.tensor(np.stack(desc)).half(),'keypoint_scores':torch.tensor(np.stack(scores)).half()},folder/f'part_{ids[0]:06d}.pt')
    log('RootSIFT: finished')

def main():
    p = argparse.ArgumentParser();p.add_argument('--branch', choices=['miewid','mega','aliked','disk','rootsift'], required=True)
    p.add_argument('--limit',type=int);p.add_argument('--batch-size',type=int,default=16);args=p.parse_args()
    seed();torch.hub.set_dir(str(ROOT / 'models' / 'torch_hub'))
    if args.branch!='rootsift' and not torch.cuda.is_available():raise RuntimeError('CUDA required. Run outside the restricted environment.')
    metadata = pd.read_csv(DATA / 'metadata.csv')
    if args.limit: metadata=metadata.head(args.limit)
    if args.branch=='rootsift':rootsift_extract(metadata)
    elif args.branch in ('miewid','mega'):globals_extract(metadata,args.branch,args.batch_size)
    else:locals_extract(metadata,args.branch)

if __name__ == '__main__':main()
