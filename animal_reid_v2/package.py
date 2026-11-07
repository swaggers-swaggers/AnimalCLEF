"""Small reproducibility bundle; excludes original images and backbone weights."""
import hashlib
import json
from pathlib import Path
import zipfile
from common import ROOT, RUN, json_write, fingerprint

def main():
    check=json.loads((RUN/'completion_check.json').read_text())
    assert check['complete'] and check['rows']==2409
    assert fingerprint(RUN/'submission.csv')==check['submission_sha256']
    project=ROOT.parent
    files=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.py','.md','.sh','.lock')]
    names=['submission.csv','实验结果.md','audit_comparison.png','audit_report.json','tune_report.json','selected_config.json','training_summary.json','data_audit.json','feature_checks.json','environment.json','splits.csv','completion_check.json','fusion_models.pkl','audit_plan.json','verification.json','batch_benchmark.json']
    files.extend(RUN/name for name in names if (RUN/name).exists())
    if (RUN/'submission_format_check.json').exists():files.append(RUN/'submission_format_check.json')
    files.extend(p for p in [ROOT/'models/manifest.json',ROOT/'models/mega_config.json'] if p.exists())
    for name in ['lightglue','sam2']:
        folder=project/'references'/name
        files.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and (p.suffix in ('.py','.yaml','.yml','.toml') or p.name.startswith('LICENSE')))
    files.extend(p for p in [project/'references/winner_readme.md',project/'references/winner/LICENSE',project/'references/miewid_files.json',project/'references/mega_files.json'] if p.exists())
    manifest={str(p.relative_to(project)):fingerprint(p) for p in files}
    json_write(manifest,RUN/'bundle_manifest.json');files.append(RUN/'bundle_manifest.json')
    out=project/'AnimalCLEF2026_重构交付包.zip'
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(set(files)):z.write(p,str(p.relative_to(project)))
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
        assert 'animal_reid_v2/runs/rebuild/submission.csv' in z.namelist()
    print(out,out.stat().st_size)

if __name__=='__main__':main()
