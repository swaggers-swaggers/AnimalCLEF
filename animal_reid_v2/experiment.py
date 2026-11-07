"""One-process experiment launcher with exclusive lock and resumable stages."""
import argparse
import fcntl
import os
import subprocess
import sys
import time
from common import ROOT, RUN, log, json_write

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--from-stage',choices=['extract','fit','tune','audit','submit'],default='extract');args=parser.parse_args()
    lock=(RUN/'experiment.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Another experiment already holds the lock.')
    lock.write(str(os.getpid()));lock.flush()
    if (RUN/'STOP').exists():raise SystemExit('Remove runs/rebuild/STOP to resume.')
    stages=[]
    if args.from_stage=='extract':
        stages.extend((f'extract_{branch}',['extract.py','--branch',branch]) for branch in ('miewid','mega','aliked','disk','rootsift'))
    names=['fit','tune','audit','submit'];start=names.index(args.from_stage) if args.from_stage in names else 0
    stages.extend((stage,['pipeline.py','--stage',stage]) for stage in names[start:])
    for stage,command in stages:
        if stage=='audit':subprocess.run([sys.executable,'verify.py','--plan'],cwd=ROOT,check=True)
        log(f'Stage {stage} started')
        with (RUN/f'{stage}.log').open('a') as out:
            result=subprocess.run([sys.executable,*command],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
        if result.returncode:raise SystemExit(f'Stage {stage} stopped, exit code {result.returncode}. See its log and resume from this stage.')
        log(f'Stage {stage} finished')
    if args.from_stage in ('extract','fit','tune','audit','submit'):
        subprocess.run([sys.executable,'finalize_submission.py'],cwd=ROOT,check=True)
        subprocess.run([sys.executable,'plot_results.py'],cwd=ROOT,check=True)
        subprocess.run([sys.executable,'report.py'],cwd=ROOT,check=True)
        subprocess.run([sys.executable,'verify.py','--final'],cwd=ROOT,check=True)
        subprocess.run([sys.executable,'package.py'],cwd=ROOT,check=True)
    log('Experiment completed')

if __name__=='__main__':main()
