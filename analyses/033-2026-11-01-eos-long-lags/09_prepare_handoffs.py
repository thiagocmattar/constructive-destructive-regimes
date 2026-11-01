"""Freeze new recovery inputs and packets without changing executing runs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import torch

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent
RUN=ROOT/'runs/208-2026-11-01-eos-checkpoint-handoffs';sys.path.insert(0,str(ROOT))
from src.cifar_eos.core import tensor_hash
from src.eos_tasks.task import model_for

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8',newline='\n')

def main():
    assert not (RUN/'input_manifest.json').exists(),'Executed inputs remain immutable'
    old=ROOT/'runs/206-2026-11-01-eos-long-lags';cfg=json.loads((old/'config.json').read_text())
    captures=json.loads((HERE/'handoff_captures.json').read_text())['captures'];jobs=[];files={}
    def add(path,workers):files.setdefault(path.relative_to(ROOT).as_posix(),set()).update(workers)
    for record in captures:
        w=record['worker'];origin=ROOT/record['snapshot'];prefix=json.loads((origin/'result.json').read_text())
        job=dict(next(j for j in cfg['jobs'] if j['parent_run_id']==record['parent_run_id'] and j['anchor']==record['anchor']))
        capture=json.loads((origin/'capture.json').read_text())
        for item in capture['files']:
            p=origin/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];add(p,[w])
        state=torch.load(origin/'end.pt',map_location='cpu',weights_only=False)
        shared=json.loads((ROOT/cfg['original_inputs']).read_text())
        model=model_for(job['task'],ROOT/cfg['task_cache'],shared['wikitext']['vocabulary_size']);model.load_state_dict(state['model'])
        assert state['step']==job['anchor']+prefix['last_lag']
        job.update(worker=w,resume_lag=prefix['last_lag'],physical=512 if job['task']=='transformer' else 1000,
            checkpoint=(origin/'end.pt').relative_to(ROOT).as_posix(),checkpoint_sha256=sha(origin/'end.pt'),
            resume_tensor_sha256=tensor_hash(model.parameters()),boundary_gradient=(origin/'audit-current.npy').relative_to(ROOT).as_posix(),
            prefix_rows=prefix['rows'],prefix_result=(origin/'result.json').relative_to(ROOT).as_posix())
        jobs.append(job);add(ROOT/job['anchor_gradient'],[w]);add(ROOT/job['data'],[w]);add(origin/'capture.json',[w])
    workers=[j['worker'] for j in jobs]
    for item in json.loads((old/'input_manifest.json').read_text())['files']:
        if item['path'].startswith(('src/','cache/eos-multitask/reference/')) or item['path']==cfg['original_inputs']:add(ROOT/item['path'],workers)
    for p in [RUN/'01_worker.py',ROOT/'src/eos_tasks/lag_profile.py']:add(p,workers)
    cfg.update(run_id='208',jobs=jobs,purpose='Qualified checkpoint handoffs to shorten parallel tail; unchanged objective, stages, cosine and stopping rule',
        inputs_manifest=RUN.relative_to(ROOT).as_posix()+'/input_manifest.json',stop_request=RUN.relative_to(ROOT).as_posix()+'/stop-request',
        source_prefix='Verified Run206/207 measured prefix and unapplied checkpoint gradient; recompute within original1e-5 tolerance before stepping; hardware changes can perturb subsequent FP32 trajectories')
    save(RUN/'config.json',cfg);add(RUN/'config.json',workers)
    save(RUN/'input_manifest.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        files=[dict(path=n,bytes=(ROOT/n).stat().st_size,sha256=sha(ROOT/n),workers=sorted(ws)) for n,ws in sorted(files.items())]))
    packets=[]
    for w in workers:
        path=ROOT/f'cache/analysis033-handoff-worker{w}.tar.gz';chosen=[ROOT/n for n,ws in files.items() if w in ws]+[RUN/'input_manifest.json']
        with tarfile.open(path,'w:gz',compresslevel=1) as tar:
            for p in chosen:tar.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
        packets.append(dict(worker=w,path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=sha(path),files=len(chosen)))
    save(HERE/'handoff_prelaunch_review.json',dict(status='passed',config_sha256=sha(RUN/'config.json'),
        shared_runner_sha256=sha(ROOT/'src/eos_tasks/lag_profile.py'),captures=len(jobs),checkpoints_and_prefixes_verified=True,
        qualification='Each new worker must match its captured full gradient at1e-5 before any update; old sources stop only after new progress exceeds its resume lag'))
    save(HERE/'handoff_packets.json',dict(packets=packets,config_sha256=sha(RUN/'config.json')))
    print(json.dumps(dict(prepared=len(jobs),resume_lags={j['worker']:j['resume_lag'] for j in jobs})))

if __name__=='__main__':main()
