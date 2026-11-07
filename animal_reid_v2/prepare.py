"""Data checks and sealed identity-disjoint calibration/tuning/audit splits."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from common import ROOT, DATA, RUN, seed, json_write, fingerprint, log

def prepare():
    m = pd.read_csv(DATA/'metadata.csv')
    assert m.image_id.is_unique
    assert (m[m.split=='train'].identity.notna()).all()
    assert (m[m.split=='test'].identity.isna()).all()
    sample = pd.read_csv(DATA/'sample_submission.csv')
    assert set(sample.image_id) == set(m.loc[m.split=='test','image_id'])
    missing = [p for p in m.path if not (DATA/p).is_file()]
    if missing:raise FileNotFoundError(missing[:10])
    records=[]
    rng=np.random.default_rng(20261104)
    for species, frame in m[m.split=='train'].groupby('dataset'):
        identities=frame.identity.drop_duplicates().to_numpy(copy=True);rng.shuffle(identities)
        n=len(identities)
        assignment = {identity: ('calibration' if i<int(n*.5) else 'tune' if i<int(n*.75) else 'audit') for i,identity in enumerate(identities)}
        for row in frame.itertuples():
            records.append({'image_id':int(row.image_id),'identity':row.identity,'dataset':species,'partition':assignment[row.identity]})
    split = pd.DataFrame(records)
    for species, group in split.groupby('dataset'):
        sets=[set(group.loc[group.partition==p,'identity']) for p in ('calibration','tune','audit')]
        assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    path=RUN/'splits.csv'
    if path.exists():pd.testing.assert_frame_equal(pd.read_csv(path),split)
    else:split.to_csv(path,index=False)
    report={'metadata_sha256':fingerprint(DATA/'metadata.csv'),'sample_sha256':fingerprint(DATA/'sample_submission.csv'), 'images':len(m),'test_images':len(sample),'missing_images':len(missing),'identity_disjoint':True,'split_counts':split.groupby(['dataset','partition']).agg(images=('image_id','size'),identities=('identity','nunique')).reset_index().to_dict('records')}
    json_write(report,RUN/'data_audit.json');log(f'Data verified: {len(m)} images, sealed splits saved')
    return m,split

if __name__=='__main__':seed();prepare()
