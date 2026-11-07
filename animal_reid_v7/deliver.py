"""Evaluate the frozen pipeline once; produce and validate the submission."""
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from settings import *
from fusion import cases, predict, measure, metrics, spot_read, dense_read


def verify_frozen():
    frozen=json.loads((RUN/'frozen.json').read_text())
    for path,expected in frozen['inputs'].items():
        assert sha(CV/path)==expected, f'Frozen input changed: {path}'
    for kind,reader in [('spots',spot_read),('dense',dense_read)]:
        for stage in ['calibration','tune']:
            data=np.load(RUN/f'frozen_{kind}_{stage}.npz')
            values,valid=reader(data['pairs']); assert valid.all()
            np.testing.assert_array_equal(values,data['values'])
    return frozen


def selected_cases(stage,name,cal,cfg):
    source=cfg.get('source','base')
    return cases(stage,name,cal,source.startswith('spot'),source.startswith('roma') or source=='spot_roma')


def extended_metrics(truth,prediction,orientation):
    result=metrics(truth,prediction)
    a,b=np.triu_indices(len(truth),1); positive=truth[a]==truth[b]; matched=prediction[a]==prediction[b]
    tp=int((positive&matched).sum()); fp=int((~positive&matched).sum()); fn=int((positive&~matched).sum())
    result.update(true_positive_pairs=tp,false_merge_pairs=fp,missed_positive_pairs=fn,
                  pair_precision=tp/max(1,tp+fp),pair_recall=tp/max(1,tp+fn))
    known=pd.notna(orientation)&(orientation!='unknown')
    cross=known[a]&known[b]&(orientation[a]!=orientation[b])
    total=int((positive&cross).sum()); recovered=int((positive&cross&matched).sum())
    result.update(cross_orientation_positive_pairs=total,cross_orientation_recovered_pairs=recovered,
                  cross_orientation_pair_recall=recovered/total if total else None)
    return result


def main():
    frozen=verify_frozen(); config=frozen['selected']
    with (RUN/'score_calibrators.pkl').open('rb') as f: calibrators=pickle.load(f)
    previous=json.loads((V6/'audit_report.json').read_text())
    tuning=json.loads((RUN/'tune_search.json').read_text())
    report={}; rows=[]; base_values=[]; new_values=[]
    meta,_=metadata(); orientations=meta.set_index('image_id').orientation
    # Replaying tune detects stale caches or configuration changes before audit.
    for name in SPECIES:
        cs=selected_cases('tune',name,calibrators,config[name])
        got=measure(cs,config[name]); expected=tuning[name]['selected']
        np.testing.assert_allclose(got['modes'],expected['modes'],atol=1e-12,rtol=0)
        report[name]={}
        for c in selected_cases('audit',name,calibrators,config[name]):
            base=predict(c,{'kind':'baseline'}); pred=predict(c,config[name])
            view=orientations.loc[c['qi']].to_numpy()
            old=extended_metrics(c['truth'],base,view); new=extended_metrics(c['truth'],pred,view)
            assert abs(old['ari']-previous[name][c['mode']]['selected']['ari'])<1e-12
            base_values.append(old['ari']); new_values.append(new['ari'])
            report[name][c['mode']]={'baseline':old,'selected':new}
            for version,labels in [('V6',base),('V7',pred)]:
                rows.extend({'dataset':name,'mode':c['mode'],'version':version,'image_id':int(i),
                             'identity':y,'cluster':str(p)} for i,y,p in zip(c['qi'],c['truth'],labels))
        log(f'{name}: frozen development evaluation complete')
    report.update(baseline_macro_ari=float(np.mean(base_values)),selected_macro_ari=float(np.mean(new_values)),
                  delta=float(np.mean(new_values)-np.mean(base_values)),historically_seen_development_holdout=True,
                  audit_used_for_v7_selection=False,official_ari=None)
    write(report,RUN/'audit_report.json'); pd.DataFrame(rows).to_csv(RUN/'audit_predictions.csv',index=False)
    old=pd.read_csv(V6/'submission.csv'); out=old.copy(); details={}
    for name in SPECIES:
        cfg=config[name]; c=selected_cases('test',name,calibrators,cfg)[0]
        base=predict(c,{'kind':'baseline'}); labels=predict(c,cfg)
        reference=old.set_index('image_id').loc[c['qi'],'cluster'].to_numpy()
        assert adjusted_rand_score(reference,base)==1.
        if cfg.get('kind')!='baseline':
            mapping={int(i):f'cluster_{name}_{label}' for i,label in zip(c['qi'],labels)}
            mask=out.image_id.isin(c['qi']); out.loc[mask,'cluster']=out.loc[mask,'image_id'].map(mapping)
        _,counts=np.unique(labels,return_counts=True)
        details[name]={'images':len(labels),'clusters':len(counts),'singletons':int((counts==1).sum()),
                       'largest_cluster':int(counts.max()),'partition_ari_against_v6':float(adjusted_rand_score(reference,labels))}
    template=pd.read_csv(DATA/'sample_submission.csv'); m,_=metadata()
    assert out.columns.tolist()==template.columns.tolist()
    assert len(out)==2409 and out.image_id.tolist()==template.image_id.tolist()
    assert out.image_id.is_unique and out.cluster.notna().all()
    joined=out.merge(m[['image_id','dataset']],on='image_id',validate='one_to_one')
    assert joined.groupby('cluster').dataset.nunique().max()==1
    assert all(c.startswith('cluster_'+d+'_') for c,d in zip(joined.cluster,joined.dataset))
    keep=m.loc[(m.dataset=='TexasHornedLizards')|m.dataset.isin([n for n,c in config.items() if c.get('kind')=='baseline']),'image_id']
    pd.testing.assert_frame_equal(out[out.image_id.isin(keep)],old[old.image_id.isin(keep)])
    out.to_csv(RUN/'submission.csv',index=False)
    verify_frozen()
    write({'passed':True,'rows':len(out),'frozen_inputs_unchanged':True,'calibration_and_tune_evidence_unchanged':True,
           'selected_tune_replayed':True,'v6_audit_and_test_replayed':True,'unchanged_species_rows_preserved':True,
           'species':details,'submission_sha256':sha(RUN/'submission.csv'),'official_ari':None},RUN/'verification.json')
    log(f'Development macro ARI: {report["baseline_macro_ari"]:.6f} -> {report["selected_macro_ari"]:.6f}')


if __name__=='__main__': main()
