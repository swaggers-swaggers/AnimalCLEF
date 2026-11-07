"""Wait for every selected-branch pair before evaluation and final packaging."""
import sqlite3
import subprocess
import sys
import time
import numpy as np
from settings import *
from patterns import pair_plan
from fusion import spot_read,dense_read


def main():
    frozen=json.loads((RUN/'frozen.json').read_text())
    assert frozen['selected']['SalamanderID2025']['source']=='spot_1'
    plans=pair_plan(); required={tuple(map(int,p)) for pairs in plans.values() for p in pairs}
    previous=-1; last_change=time.monotonic()
    while True:
        stop()
        with sqlite3.connect(f'file:{RUN/"spots.sqlite"}?mode=ro',uri=True) as db:
            present=set(db.execute('SELECT a,b FROM matches'))
        complete=len(required&present)
        if complete!=previous:
            log(f'Waiting for complete matching: {complete}/{len(required)} saved')
            previous=complete; last_change=time.monotonic()
        if complete==len(required): break
        if time.monotonic()-last_change>1200:
            raise RuntimeError('No matching progress for 20 minutes; evaluation has not started')
        time.sleep(30)
    checks={}
    for stage,pairs in plans.items():
        values,valid=spot_read(pairs)
        assert valid.all() and np.isfinite(values).all()
        checks['spots_'+stage]=len(pairs)
    for stage in ['calibration','tune']:
        pairs=np.load(RUN/f'roma_pairs_{stage}.npz')['pairs']; values,valid=dense_read(pairs)
        assert valid.all() and np.isfinite(values).all()
        checks['dense_'+stage]=len(pairs)
    write({'passed':True,'complete_finite_pairs':checks,'all_selected_branch_matching_complete_before_evaluation':True},RUN/'matching_checks.json')
    log('All required matching complete; starting frozen evaluation')
    # V3 imports prepend their directory to sys.path. Run explicit V7 paths
    # to prevent selecting the older, identically named deliver module.
    subprocess.run([sys.executable,str(ROOT/'deliver.py')],check=True)
    subprocess.run([sys.executable,str(ROOT/'make_report.py')],check=True)
    write({'status':'complete','submission':str(RUN/'submission.csv'),'sha256':sha(RUN/'submission.csv')},RUN/'completion.json')


if __name__=='__main__': main()
