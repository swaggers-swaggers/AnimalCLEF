"""Verify padded LightGlue against unpadded inference on actual image features."""
import sys
import numpy as np
import torch
from common import ROOT,RUN,json_write,seed
from extract import load_features
from matching import LocalMatcher
sys.path.insert(0,str(ROOT.parent/'references/lightglue'))
from lightglue import LightGlue

@torch.inference_mode()
def main():
    seed();torch.hub.set_dir(str(ROOT/'models/torch_hub'))
    f=load_features('superpoint_pattern');available=torch.nonzero(f['lengths']>=40,as_tuple=True)[0][:2]
    assert len(available)==2
    torch.backends.cuda.matmul.allow_tf32=False
    results={}
    for mixed in [False,True]:
        model=LightGlue(features='superpoint',depth_confidence=-1,width_confidence=-1,flash=mixed,mp=mixed).cuda().eval()
        matcher=object.__new__(LocalMatcher);matcher.model=model
        l0=torch.tensor([37,61],device='cuda');l1=torch.tensor([49,73],device='cuda')
        ix=int(available[0]);capacity=min(96,int(f['lengths'][ix]));l0=l0.clamp_max(capacity);l1=l1.clamp_max(capacity)
        data=[]
        for lengths in [l0,l1]:
            kp=f['keypoints'][ix,:capacity].float().cuda()[None].repeat(2,1,1);desc=f['descriptors'][ix,:capacity].float().cuda()[None].repeat(2,1,1)
            if mixed:desc=desc.half()
            for i,n in enumerate(lengths.tolist()):kp[i,n:]=0;desc[i,n:]=0
            data.append({'keypoints':kp,'descriptors':desc,'image_size':torch.tensor([[512,512]],device='cuda').repeat(2,1)})
        with torch.autocast('cuda',dtype=torch.float16,enabled=mixed):batched=matcher.masked_lightglue(*data,l0,l1)
        differences=[];total=0
        for i,(a,b) in enumerate(zip(l0.tolist(),l1.tolist())):
            d0={k:v[i:i+1,:a] if k!='image_size' else v[i:i+1] for k,v in data[0].items()};d1={k:v[i:i+1,:b] if k!='image_size' else v[i:i+1] for k,v in data[1].items()}
            reference=model({'image0':d0,'image1':d1})
            assert torch.equal(reference['matches0'][0],batched['matches0'][i,:a])
            good=reference['matches0'][0]>=0;total+=int(good.sum())
            if good.any():differences.append(float((reference['matching_scores0'][0,good]-batched['matching_scores0'][i,:a][good]).abs().max()))
            assert (batched['matches0'][i,a:]==-1).all()
        difference=max(differences,default=0)
        print({'mixed_precision':mixed,'positive_matches':total,'max_confidence_difference':difference},flush=True)
        assert total>10 and difference<(.01 if mixed else 1e-4)
        results['fp16' if mixed else 'fp32']={'matches_exact':True,'positive_matches':total,'max_confidence_difference':difference}
        del model,matcher
    json_write({'passed':True,'padding_never_matches':True,'comparisons':results},RUN/'gpu_checks.json')
    print('Variable-length LightGlue checked against unpadded inference')

if __name__=='__main__':main()
