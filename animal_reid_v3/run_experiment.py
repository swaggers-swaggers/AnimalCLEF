"""Exclusive, resumable GPU experiment with one log per stage."""
import argparse
import fcntl
import subprocess
import sys
from common import ROOT,RUN,log,check_stop,json_write,fingerprint

STAGES=[
 ('dino',['vision.py','--branch','dinov3']),
 ('kaze',['vision.py','--branch','kaze']),
 ('patterns',['vision.py','--branch','superpoint_pattern']),
 ('regions',['vision.py','--branch','superpoint_regions']),
 ('gpu_checks',['test_gpu.py']),
 ('preprocessed',['vision.py','--branch','miew_pp']),
 ('finetune',['finetune.py']),
 ('adapted',['vision.py','--branch','miew_arc']),
 ('fit',['strong.py','--stage','fit']),
 ('tune',['strong.py','--stage','tune']),
 ('audit',['strong.py','--stage','audit']),
 ('submit',['strong.py','--stage','submit']),
 ('deliver',['deliver.py'])]

def main():
    p=argparse.ArgumentParser();p.add_argument('--from-stage',choices=[n for n,_ in STAGES],default='dino');a=p.parse_args()
    lock=(RUN/'experiment.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    core=['vision.py','finetune.py','strong.py','graph.py','matching.py','common.py','pipeline.py','models.py','extract.py','prepare.py','test_gpu.py']
    hashes={f:fingerprint(ROOT/f) for f in core};json_write(hashes,RUN/'execution_code.json')
    start=[n for n,_ in STAGES].index(a.from_stage)
    for name,args in STAGES[start:]:
        assert hashes=={f:fingerprint(ROOT/f) for f in core},'Source changed during execution; resume explicitly after reviewing cache validity'
        check_stop();json_write({'stage':name,'status':'running'},RUN/'progress.json');log(f'V3 stage {name} started')
        with (RUN/f'{name}.log').open('a') as out:result=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
        if result.returncode:
            json_write({'stage':name,'status':'stopped','exit_code':result.returncode},RUN/'progress.json');raise SystemExit(f'{name} failed; inspect its log and resume from this stage')
        log(f'V3 stage {name} completed')
    json_write({'stage':'complete','status':'complete'},RUN/'progress.json')

if __name__=='__main__':main()
