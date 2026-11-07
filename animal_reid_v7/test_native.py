"""Real-feature native batch vs one-pair equivalence, with no dummy points."""
import numpy as np
import torch
from patterns import SpotMatcher,FEATURES
from settings import RUN,write


def main():
    torch.set_num_threads(8); model=SpotMatcher(); ids=[]
    for path in sorted(FEATURES.glob('*.pt')):
        key=int(path.stem); f=model.load(key,0)
        if len(f['keypoints'])==512: ids.append(key)
        if len(ids)==4: break
    assert len(ids)==4
    def data(keys):
        d={k:torch.stack([model.load(key,0)[k] for key in keys]) for k in ['keypoints','descriptors']}
        d['keypoints']=d['keypoints'].float();d['descriptors']=d['descriptors'].float()
        d['image_size']=torch.tensor([[1024,1024]],device='cuda').expand(len(keys),-1)
        return d
    with torch.inference_mode():
        batched=model.model({'image0':data(ids[:2]),'image1':data(ids[2:])})
        errors=[]; raw_errors=[]; low_confidence_switches=[]
        for i in range(2):
            single=model.model({'image0':data([ids[i]]),'image1':data([ids[i+2]])})
            scores=single['matching_scores0'][0]; other=batched['matching_scores0'][i]
            accepted=(single['matches0'][0]>=0)|(batched['matches0'][i]>=0)
            # Mutual-best selection can switch an unused below-.1 candidate to zero.
            # Production uses only accepted correspondences, so test those contracts.
            assert torch.equal(single['matches0'][0],batched['matches0'][i])
            error=float((scores[accepted]-other[accepted]).abs().max()) if accepted.any() else 0.
            errors.append(error); raw_errors.append(float((scores-other).abs().max()))
            low_confidence_switches.append(int(((scores-other).abs()>.005).sum()))
            assert error<.005,error
            for threshold in [.1,.5,.65,.8,.9,.95]:
                assert int(((scores>=threshold)&accepted).sum())==int(((other>=threshold)&accepted).sum()),threshold
            assert not (((scores-other).abs()>.005)&accepted).any()
    write({'passed':True,'real_points_per_image':512,'padded_points':0,'accepted_confidence_max_error':errors,
           'raw_including_rejected_max_error':raw_errors,'rejected_candidate_switches':low_confidence_switches,
           'all_accepted_match_ids_equal':True,'all_six_production_confidence_counts_equal':True,'image_ids':ids},RUN/'native_checks.json')
    print(errors)


if __name__=='__main__':main()
