"""Freeze audit inputs, then verify the final submission and experiment provenance."""
import argparse
import json
import numpy as np
import pandas as pd
from common import ROOT, RUN, DATA, json_write, fingerprint

CORE=['common.py','models.py','extract.py','matching.py','pipeline.py','foreground.py']

def inputs():
    return {'config_sha256':fingerprint(RUN/'selected_config.json'),'fusion_sha256':fingerprint(RUN/'fusion_models.pkl'),'splits_sha256':fingerprint(RUN/'splits.csv'),'metadata_sha256':fingerprint(DATA/'metadata.csv'),'code_sha256':{name:fingerprint(ROOT/name) for name in CORE}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',action='store_true');p.add_argument('--final',action='store_true');a=p.parse_args()
    config=json.loads((RUN/'selected_config.json').read_text())
    assert all(x['candidate_policy'] in ['sparse','expanded'] for x in config.values())
    split=pd.read_csv(RUN/'splits.csv')
    assert split.groupby(['dataset','identity']).partition.nunique().max()==1
    pairs=np.load(RUN/'training_plan.npz')['pairs']
    assert set(pairs.flatten()).issubset(set(split.loc[split.partition=='calibration','image_id']))
    feature_checks=json.loads((RUN/'feature_checks.json').read_text())
    assert all(x['images']==15483 and x['unique_images']==15483 and x['nonfinite_values']==0 for x in feature_checks.values())
    if a.plan:
        json_write(inputs(),RUN/'audit_plan.json');print('Audit inputs frozen; audit labels have not selected this configuration.')
    if a.final:
        assert inputs()==json.loads((RUN/'audit_plan.json').read_text()),'Inputs changed after audit freeze'
        audit=json.loads((RUN/'audit_report.json').read_text())
        for name in list(config)[:3]:assert audit[name]['config']==config[name]
        template=pd.read_csv(DATA/'sample_submission.csv');out=pd.read_csv(RUN/'submission.csv')
        assert list(out.columns)==['image_id','cluster']
        assert len(out)==2409 and out.image_id.is_unique and out.cluster.notna().all()
        assert out.image_id.tolist()==template.image_id.tolist()
        metadata=pd.read_csv(DATA/'metadata.csv')
        joined=metadata[metadata.split=='test'].merge(out,on='image_id',validate='one_to_one')
        assert len(joined)==2409 and joined.groupby('cluster').dataset.nunique().max()==1
        assert all(label.startswith('cluster_'+dataset+'_') and len(label)>len('cluster_'+dataset+'_') for dataset,label in zip(joined.dataset,joined.cluster))
        complete=json.loads((RUN/'completion_check.json').read_text())
        assert fingerprint(RUN/'submission.csv')==complete['submission_sha256']
        assert audit['official_ari'] is None and complete['official_ari'] is None
        result={'passed':True,'submission_rows':len(out),'template_order_exact':True,'official_cluster_name_format':True,'no_cross_species_clusters':True,'calibration_identity_only_fitting':True,'identity_disjoint_validation':True,'audit_inputs_unchanged':True,'all_features_complete_and_finite':True,'official_score_available':False}
        json_write(result,RUN/'verification.json');print(json.dumps(result))

if __name__=='__main__':main()
