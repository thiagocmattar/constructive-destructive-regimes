"""Detached single-task/seed queue; resume unchanged arms, respect one deadline."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from src.cifar_eos.core import file_hash,stamp,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=['resnet','transformer'],required=True)
    p.add_argument('--seed',type=int,choices=[0,1,2],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline-epoch',type=float,required=True)
    p.add_argument('--smoke',action='store_true',help='Three-update reference, full control and smallest sampled fork at state zero; no paper evidence.')
    args=p.parse_args()
    cfg=json.loads((HERE/'config.json').read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    jobs=[dict(name='reference',mode='reference')]
    for state in cfg['states']:
        # Full batch first provides a deterministic control before stochastic forks.
        for batch in reversed(cfg['batches'][args.task]):
            repeats=1 if batch==cfg['batches'][args.task][-1] else cfg['stochastic_repetitions']
            for rep in range(repeats):
                jobs.append(dict(name=f'state-{state}-b{batch}-rep{rep:02d}',mode='fork',state=state,batch=batch,rep=rep))
    if args.smoke:
        jobs=[jobs[0]]+[j for j in jobs[1:] if j['state']==0 and j['rep']==0 and j['batch'] in (cfg['batches'][args.task][0],cfg['batches'][args.task][-1])]
    identity=dict(task=args.task,seed=args.seed,config_sha256=file_hash(HERE/'config.json'),
                  input_manifest_sha256=file_hash(HERE/'input_manifest.json'),mechanics_only=args.smoke,jobs=jobs)
    packet=args.output/'worker_manifest.json'
    if packet.exists():
        previous=json.loads(packet.read_text()); assert previous['identity']==identity
    write_json(packet,dict(identity=identity,status='running',started_utc=stamp(),deadline_epoch=args.deadline_epoch))
    status='smoke_complete' if args.smoke else 'complete'
    for i,job in enumerate(jobs):
        output=args.output/job['name']; summary=output/'summary.json'
        if summary.exists():
            assert json.loads(summary.read_text())['status']==('smoke_complete' if args.smoke else 'complete')
            continue
        remaining=args.deadline_epoch-time.time()
        if remaining<120 or (HERE/'STOP').exists():status='bounded_stop';break
        command=[sys.executable,'-u',str(HERE/'01_measure.py'),'--task',args.task,'--seed',str(args.seed),
            '--mode',job['mode'],'--output',str(output),'--max-seconds',str(max(1,remaining-60))]
        if args.smoke:command+=['--smoke-updates','3']
        if job['mode']=='fork':
            command+=['--state',str(job['state']),'--batch',str(job['batch']),'--rep',str(job['rep']),
                '--reference',str(args.output/'reference'/f'state-{job["state"]:04d}.pt')]
        write_json(args.output/'worker_progress.json',dict(utc=stamp(),job_index=i,job=job,job_count=len(jobs)))
        with (args.output/(job['name']+'.log')).open('a',encoding='utf-8') as log:
            process=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if process.returncode:
            status='failed';break
        if not summary.exists():status='bounded_stop';break
    write_json(packet,dict(identity=identity,status=status,finished_utc=stamp(),deadline_epoch=args.deadline_epoch,
        complete_jobs=sum((args.output/j['name']/'summary.json').exists() for j in jobs),job_count=len(jobs)))
    inventory={p.relative_to(args.output).as_posix():dict(bytes=p.stat().st_size,sha256=file_hash(p))
               for p in args.output.rglob('*') if p.is_file() and p.name!='worker_inventory.json'}
    write_json(args.output/'worker_inventory.json',inventory)
    print(json.dumps(dict(status=status,files=len(inventory))),flush=True)
    if status=='failed':sys.exit(1)


if __name__=='__main__':main()
