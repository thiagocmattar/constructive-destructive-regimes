"""Run unchanged small-batch forks on Spark, with an additional full-GD runtime bridge."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline-epoch',type=float,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.root))
    from src.cifar_eos.core import file_hash,stamp,write_json
    run=a.root/'runs/214-2026-11-02-sgd-batch-extension'
    cfg=json.loads((run/'config.json').read_text())
    assert cfg['learning_rates']['resnet']==.1 and cfg['physical']['resnet']==1000
    jobs=[dict(name=f'state-{state}-b{batch}-rep{rep:02d}',state=state,batch=batch,rep=rep)
        for state in [0,2000] for batch in [5,1] for rep in range(8)]
    jobs.insert(16,dict(name='initial-full-runtime-bridge',state=0,batch=50,rep=0,bridge_only=True))
    a.output.mkdir(parents=True,exist_ok=True)
    identity=dict(task='resnet',seed=2,jobs=jobs,config_sha256=file_hash(run/'config.json'),
        input_manifest_sha256=file_hash(run/'input_manifest.json'),
        role='Additional hardware parallelism; unchanged scientific recipe, explicit runtime variation.')
    write_json(a.output/'helper_manifest.json',dict(identity=identity,status='running',started_utc=stamp()))
    status='complete'
    for index,job in enumerate(jobs):
        target=a.output/job['name']
        if (target/'summary.json').exists():continue
        reference=a.reference/f'state-{job["state"]:04d}.pt'
        while not reference.exists() and time.time()<a.deadline_epoch-120:time.sleep(5)
        remaining=a.deadline_epoch-time.time()
        if remaining<120 or (run/'STOP').exists():status='bounded_stop';break
        write_json(a.output/'helper_progress.json',dict(utc=stamp(),index=index,job=job,jobs=len(jobs)))
        command=[sys.executable,'-u',str(run/'01_measure.py'),'--task','resnet','--seed','2',
            '--mode','fork','--state',str(job['state']),'--batch',str(job['batch']),
            '--rep',str(job['rep']),'--reference',str(reference),'--output',str(target),
            '--max-seconds',str(remaining-60)]
        with (a.output/(job['name']+'.log')).open('ab') as stream:
            result=subprocess.run(command,cwd=a.root,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:status='failed';break
        if not (target/'summary.json').exists():status='bounded_stop';break
    write_json(a.output/'helper_manifest.json',dict(identity=identity,status=status,finished_utc=stamp(),
        complete_jobs=sum((a.output/j['name']/'summary.json').exists() for j in jobs)))
    inventory={p.relative_to(a.output).as_posix():dict(bytes=p.stat().st_size,sha256=file_hash(p))
        for p in a.output.rglob('*') if p.is_file() and p.name!='helper_inventory.json'}
    write_json(a.output/'helper_inventory.json',inventory)
    print(json.dumps(dict(status=status,jobs=len(jobs))),flush=True)


if __name__=='__main__':main()
