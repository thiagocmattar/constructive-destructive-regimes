"""Execute two already-declared Run214 forks on an extra qualified worker."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline-epoch',type=float,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.root))
    from src.cifar_eos.core import file_hash,stamp,write_json
    run=a.root/'runs/214-2026-11-02-sgd-batch-extension'
    jobs=[dict(name=f'state-2000-b20-rep{rep:02d}',mode='fork',state=2000,batch=20,rep=rep) for rep in [6,7]]
    identity=dict(task='resnet',seed=2,mechanics_only=False,role='late-continuation-helper',jobs=jobs,
        config_sha256=file_hash(run/'config.json'),input_manifest_sha256=file_hash(run/'input_manifest.json'))
    a.output.mkdir(parents=True,exist_ok=True)
    write_json(a.output/'worker_manifest.json',dict(identity=identity,status='running',started_utc=stamp()))
    status='complete'
    for index,job in enumerate(jobs):
        remaining=a.deadline_epoch-time.time()
        if remaining<120:status='bounded_stop';break
        write_json(a.output/'worker_progress.json',dict(utc=stamp(),job_index=index,job=job,job_count=2))
        command=[sys.executable,'-u',str(run/'01_measure.py'),'--task','resnet','--seed','2',
            '--mode','fork','--state','2000','--batch','20','--rep',str(job['rep']),
            '--reference',str(a.output/'reference/state-2000.pt'),'--output',str(a.output/job['name']),
            '--max-seconds',str(remaining-60)]
        with (a.output/(job['name']+'.log')).open('ab') as stream:
            result=subprocess.run(command,cwd=a.root,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:status='failed';break
        if not (a.output/job['name']/'summary.json').exists():status='bounded_stop';break
    write_json(a.output/'worker_manifest.json',dict(identity=identity,status=status,finished_utc=stamp(),
        complete_jobs=sum((a.output/j['name']/'summary.json').exists() for j in jobs),job_count=2))
    inventory={p.relative_to(a.output).as_posix():dict(bytes=p.stat().st_size,sha256=file_hash(p))
        for p in a.output.rglob('*') if p.is_file() and p.name!='worker_inventory.json'}
    write_json(a.output/'worker_inventory.json',inventory)
    print(json.dumps(dict(status=status,jobs=2)),flush=True)


if __name__=='__main__':main()
