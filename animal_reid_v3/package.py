"""Repackage verified results without repeating model evaluation."""
import hashlib
import json
import zipfile
from common import ROOT,RUN,BASE_RUN,json_write,fingerprint

def package():
    check=json.loads((RUN/'verification.json').read_text());complete=json.loads((RUN/'completion_check.json').read_text())
    assert check['passed'] and complete['complete']
    assert fingerprint(RUN/'submission.csv')==complete['submission_sha256']
    files=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.py','.md','.sh','.lock']]
    files += [p for p in RUN.iterdir() if p.is_file() and p.name!='bundle_manifest.json' and p.suffix in ['.json','.csv','.pkl','.png','.md']]
    files += [RUN/'miew_arc_best.pt']
    files += [p for p in (ROOT/'models/dinov3').iterdir() if p.suffix in ['.md','.json']]
    oldroot=ROOT.parent/'animal_reid_v2';files += [p for p in oldroot.iterdir() if p.is_file() and p.suffix in ['.py','.md','.sh','.lock']]
    files += [BASE_RUN/n for n in ['selected_config.json','audit_report.json','tune_report.json','splits.csv','training_plan.npz','fusion_models.pkl','completion_check.json']]
    for name in ['lightglue','sam2']:
        files += [p for p in (ROOT.parent/'references'/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts and (p.suffix in ['.py','.yaml','.yml','.toml'] or p.name.startswith('LICENSE'))]
    files=sorted(set(files));manifest={str(p.relative_to(ROOT.parent)):fingerprint(p) for p in files};json_write(manifest,RUN/'bundle_manifest.json')
    archive=ROOT.parent/'AnimalCLEF2026_强化架构交付包.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p in files+[RUN/'bundle_manifest.json']:z.write(p,str(p.relative_to(ROOT.parent)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,digest in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
    return archive

if __name__=='__main__':print(package())
