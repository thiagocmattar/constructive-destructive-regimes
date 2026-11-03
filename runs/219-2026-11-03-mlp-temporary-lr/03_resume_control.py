"""Resume the unchanged control after a recorded Windows status-file lock."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('run219_resume_source',HERE/'01_temporary_lr.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
ORIGINAL_WRITE=r.b.write_json


def retry_write(path,value,writer=None):
    writer=writer or ORIGINAL_WRITE
    for retry in range(10):
        try:return writer(path,value)
        except PermissionError:
            if retry==9:raise
            time.sleep(min(.025*(retry+1),.2))


def main():
    # Focused I/O behavior check only, without any model or policy change.
    calls=[]
    def temporary_lock(path,value):
        calls.append(value)
        if len(calls)<3:raise PermissionError('synthetic transient lock')
        return value
    assert retry_write(None,{'preserved':True},temporary_lock)=={'preserved':True} and len(calls)==3
    r.b.write_json=retry_write
    original=HERE/'artifacts/attempts/001-local'
    prior=json.loads((original/'manifest.json').read_text())
    started=time.perf_counter();remaining=600-prior['worker_elapsed_seconds']
    attempt=HERE/'artifacts/attempts/002-control-recovery';attempt.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((HERE/'input_manifest.json').read_text());r.b.verify_inputs(r.ROOT,manifest)
    source=original/'conditions/unchanged/recovery.pt';source_sha=r.b.file_hash(source)
    assert r.b.file_hash(r.PARENT)==r.PARENT_SHA
    _,task,opt,obs,parent_identity,data=r.b.prepare(r.ROOT,r.BASE_HERE,torch.device('cuda:0'))
    identity=dict(run_id='219',condition='unchanged',parent_sha256=r.PARENT_SHA,parent_identity=parent_identity,
        config_sha256=r.b.file_hash(HERE/'config.json'),source_manifest_sha256=r.b.file_hash(HERE/'input_manifest.json'),rule='unchanged')
    policy=r.load_branch(source,task.model,opt,obs,identity,data,'unchanged')
    start_state=obs.completed_updates
    receipt=dict(status='running',reason='Resume unchanged scientific inputs after WinError5 status replacement failure',
        source_attempt='001-local',source_checkpoint=str(source.relative_to(r.ROOT)),source_sha256=source_sha,
        resume_state=start_state,original_failed_updates=prior['results']['unchanged']['updates'],
        infrastructure_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scientific_input_manifest_sha256=r.b.file_hash(HERE/'input_manifest.json'),
        focused_check='two transient write failures then success, value unchanged',started_utc=r.b.stamp())
    retry_write(attempt/'authorization_and_recovery.json',receipt)
    import shutil
    shutil.copyfile(Path(__file__),attempt/'03_resume_control.py')
    torch.cuda.reset_peak_memory_stats()
    result=r.run(task,opt,obs,identity,data,'unchanged',attempt/'unchanged',1024,policy,
        lambda:time.perf_counter()-started>=remaining or (HERE/'STOP').exists())
    evaluation_start=time.perf_counter();evaluation=task.evaluate('test');torch.cuda.synchronize()
    previous=r.jsonl(original/'conditions/unchanged/boundaries.jsonl');new=r.jsonl(attempt/'unchanged/boundaries.jsonl')
    old_by_state={row['state']:row for row in previous}
    overlap=[row for row in new if row['state'] in old_by_state]
    exact=all((row['loss'],row['gradient_norm'],row['eta_kappa_hat'])==(old_by_state[row['state']]['loss'],old_by_state[row['state']]['gradient_norm'],old_by_state[row['state']]['eta_kappa_hat']) for row in overlap)
    assert exact
    canonical=[row for row in previous if row['state']<start_state]+new
    assert [row['state'] for row in canonical]==list(range(657,1025))
    historical=r.jsonl(r.ROOT/'runs/218-2026-11-03-mlp-adam-step-actions/artifacts/attempts/001-20261103T172300Z-local-pilot/branches/A0/boundaries.jsonl')
    overlap218=all((canonical[i]['loss'],canonical[i]['gradient_norm'])==(row['loss'],row['gradient_norm']) for i,row in enumerate(historical))
    receipt.update(status=result['status'],resumed_updates=1024-start_state,policy_state=policy,
        worker_elapsed_seconds=time.perf_counter()-started,runner_seconds=result['elapsed_seconds'],finished_utc=r.b.stamp(),
        repeated_overlap_states=len(overlap),repeated_overlap_exact=exact,overlap_Run218_exact=overlap218,
        replayed_updates=prior['results']['unchanged']['updates']-(start_state-657),endpoint_test=evaluation,
        endpoint_test_seconds=time.perf_counter()-evaluation_start,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    retry_write(attempt/'manifest.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
