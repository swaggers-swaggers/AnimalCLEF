"""SAM2 foreground prompts use image content only, never identity annotations."""
import argparse
import sys
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from common import ROOT, RUN, DATA, seed, log, check_stop, json_write
from extract import read_image
sys.path.insert(0,str(ROOT.parent/'references'/'sam2'))

def prompts(image, species):
    h,w=image.shape[:2]
    if species=='SalamanderID2025':
        hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        yellow=((hsv[:,:,0]>=15)&(hsv[:,:,0]<=42)&(hsv[:,:,1]>95)&(hsv[:,:,2]>75)).astype(np.uint8)
        yellow[:int(.04*h)]=0;yellow[int(.96*h):]=0;yellow[:,:int(.04*w)]=0;yellow[:,int(.96*w):]=0
        # Find a yellow point near the specimen center; SAM infers the full body.
        yy,xx=np.where(yellow)
        if len(xx)>8:
            dist=((xx/w-.5)**2+(yy/h-.5)**2)
            best=dist.argmin();point=np.array([[xx[best],yy[best]]],dtype=np.float32)
        else:point=np.array([[w*.5,h*.5]],dtype=np.float32)
        return point,np.array([1]),None
    # Lizard belly is centrally framed. Box excludes much of the photographer's hands.
    return np.array([[w*.5,h*.5]],np.float32),np.array([1]),np.array([w*.18,h*.08,w*.82,h*.92],np.float32)

@torch.inference_mode()
def run(limit=None):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model=build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml',str(ROOT/'models/sam2.1_hiera_small.pt'),device='cuda',apply_postprocessing=False)
    predictor=SAM2ImagePredictor(model)
    m=pd.read_csv(DATA/'metadata.csv');m=m[m.dataset.isin(['SalamanderID2025','TexasHornedLizards'])]
    if limit:m=m.groupby('dataset').head(limit)
    folder=RUN/'masks';folder.mkdir(exist_ok=True);report=[]
    log(f'SAM2: foreground masks for {len(m)} images')
    for row in m.itertuples():
        check_stop();path=folder/f'{row.image_id}.png'
        if path.exists():continue
        # Matching uses exactly the same square-resized coordinate frame.
        im=np.array(read_image(row).resize((512,512),Image.Resampling.BICUBIC))
        point,labels,box=prompts(im,row.dataset)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            predictor.set_image(im)
            masks,scores,_=predictor.predict(point_coords=point,point_labels=labels,box=box,multimask_output=True)
        fractions=masks.reshape(len(masks),-1).mean(1)
        valid=(fractions>.03)&(fractions<.9)
        choice=int(np.argmax(np.where(valid,scores,-1)))
        mask=masks[choice]
        fallback=not bool(valid.any())
        if fallback:mask=np.ones((512,512),bool)
        temp=path.with_suffix('.tmp.png');Image.fromarray(mask.astype(np.uint8)*255).save(temp);temp.replace(path)
        report.append({'image_id':int(row.image_id),'score':float(scores[choice]),'foreground_fraction':float(mask.mean()),'fallback':fallback})
    existing=folder/'quality.json'
    old=__import__('json').loads(existing.read_text()) if existing.exists() else []
    json_write(old+report,existing);log('SAM2: foreground masks completed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);args=p.parse_args();seed();run(args.limit)
