"""Detached queue, replay/calibration gates, and durable per-run inventories."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from baseline.data_protocol import sha256
from baseline.training import write_json
from tools.artifact_inventory import create

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,required=True);p.add_argument('--deadline',type=float,required=True)
    args=p.parse_args()
    lock=ROOT/f'worker-control/worker-{args.worker}.lock'
    lock.parent.mkdir(exist_ok=True)
    with lock.open('x') as f:f.write(str(os.getpid()))
    status=lock.with_suffix('.json')
    state=dict(status='qualifying',completed=[],worker=args.worker,started_at=time.time())
    def save():write_json(status,state)
    save()
    try:
        inventory=json.loads((HERE/'source_inventory.json').read_text())
        for item in inventory['files']:
            assert sha256(ROOT/item['path'])==item['sha256'],item['path']
        cohort=json.loads((HERE/'cohort.json').read_text())
        queue=[r for r in cohort if r['worker']==args.worker]
        def execute(row,output,mechanics=False):
            state.update(output=str(output));save()
            command=[sys.executable,'-m','baseline.lr_decisions','--config',str(ROOT/'runs'/row['run']/'config.json'),'--cache',str(ROOT/'inputs/cache'),'--output',str(output),'--deadline',str(args.deadline)]
            if mechanics:command+=['--mechanics','3','--replay']
            with (lock.parent/f'worker-{args.worker}.training.log').open('a') as log:
                subprocess.run(command,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'},stdout=log,stderr=subprocess.STDOUT,check=True)
        qualifications={}
        for size in ['14M','70M']:
            calibration=next(r for r in cohort if r['size']==size and r['multiplier']==1)
            suffix='-14M' if size=='14M' else ''
            output=ROOT/f'worker-control/calibration-{args.worker}{suffix}'
            execute(calibration,output,True)
            summary=json.loads((output/'summary.json').read_text())
            expected=json.loads((HERE/'references'/f"{calibration['block']}.json").read_text())['first_updates']
            actual=[json.loads(l) for l in (output/'events.jsonl').read_text().splitlines() if json.loads(l)['event']=='update']
            differences=[]
            for actual_row,expected_row in zip(actual,expected):
                diff={k:abs(actual_row[k]-expected_row[k]) for k in ['loss','gradient_norm','adjacent_cosine']}
                assert diff['loss']<=1e-4+1e-5*abs(expected_row['loss']),diff
                assert diff['adjacent_cosine']<=1e-3,diff
                assert diff['gradient_norm']<=1e-6+1e-3*abs(expected_row['gradient_norm']),diff
                differences.append(diff)
            qualifications[size]=dict(seconds=summary['update_seconds'],historical_replay_differences=differences,maximum_cuda_allocated_bytes=max(r['peak_cuda_allocated_bytes'] for r in actual))
            assert qualifications[size]['maximum_cuda_allocated_bytes']<43*1024**3
            create(output,output/'inventory.json')
        estimated=sum(128*max(qualifications[r['size']]['seconds'])*1.2+90 for r in queue)
        assert estimated<args.deadline-time.time(),(estimated,args.deadline-time.time())
        qualification=dict(verified=True,by_size=qualifications,qualified_remaining_seconds=estimated,remaining_deadline_seconds=args.deadline-time.time())
        write_json(HERE/f'calibration-worker{args.worker}.json',qualification)
        write_json(ROOT/f'worker-control/calibration-{args.worker}.json',qualification)
        spec=importlib.util.spec_from_file_location('verify_h024',HERE/'03_verify.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        state.update(status='running');save()
        for row in queue:
            output=ROOT/'runs'/row['run']/'artifacts/attempt-01/full-pass'
            execute(row,output)
            write_json(output/'verification.json',module.verify(output))
            create(output,output/'inventory.json')
            state['completed'].append(row['run']);save()
        state.update(status='completed',finished_at=time.time());save()
    except Exception as exc:
        state.update(status='failed',error=repr(exc),finished_at=time.time());save();raise

if __name__=='__main__':main()
