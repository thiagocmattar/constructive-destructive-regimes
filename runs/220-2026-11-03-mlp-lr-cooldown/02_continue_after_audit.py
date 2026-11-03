"""Infrastructure-only continuation: exclude observer UTC metadata from replay equality."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import time
import traceback

import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('run220_frozen',HERE/'01_cooldown.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
helper_spec=importlib.util.spec_from_file_location('run220_audit_equality',r.ROOT/'runs/218-2026-11-03-mlp-adam-step-actions/03_local_pilot.py')
helper=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)


def numeric_observer(observer):
    result=copy.deepcopy(observer)
    for row in result['history']:row['pre'].pop('utc',None)
    return result


def audit_pair(direct,resumed):
    for key in ['model','optimizer','rng','policy_state']:helper.equal(direct[key],resumed[key])
    helper.equal(numeric_observer(direct['observer']),numeric_observer(resumed['observer']))


def main():
    began=time.perf_counter();first=HERE/'artifacts/attempts/001-local'
    prior=json.loads((first/'manifest.json').read_text(encoding='utf-8'))
    cfg=json.loads((HERE/'config.json').read_text(encoding='utf-8'))
    r.b.verify_inputs(r.ROOT,json.loads((HERE/'input_manifest.json').read_text(encoding='utf-8')))
    assert r.b.file_hash(r.r.PARENT)==r.r.PARENT_SHA
    attempt=HERE/'artifacts/attempts/002-audit-continuation';attempt.mkdir(parents=True,exist_ok=False)
    receipt=dict(cfg,status='running',started_utc=r.b.stamp(),reason='Qualification compared fresh UTC timestamps. Scientific inputs unchanged.',
        infrastructure_source_sha256=r.b.file_hash(__file__),scientific_manifest_sha256=r.b.file_hash(HERE/'input_manifest.json'),
        source_attempt='001-local',prior_worker_seconds=prior['worker_elapsed_seconds'])
    r.write_json(attempt/'authorization_and_recovery.json',receipt);shutil.copyfile(__file__,attempt/Path(__file__).name)
    def stop():return time.perf_counter()-began+prior['worker_elapsed_seconds']>=600 or (HERE/'STOP').exists()
    def prepare(condition):
        if stop():raise TimeoutError('Cumulative worker600-second bound reached.')
        torch.cuda.reset_peak_memory_stats()
        _,task,opt,obs,parent_identity,data=r.b.prepare(r.ROOT,r.r.BASE_HERE,torch.device('cuda:0'))
        r.b.load_checkpoint(r.r.PARENT,task.model,opt,obs,parent_identity,data)
        identity=dict(run_id='220',condition=condition,parent_sha256=r.r.PARENT_SHA,parent_identity=parent_identity,
            config_sha256=r.b.file_hash(HERE/'config.json'),source_manifest_sha256=r.b.file_hash(HERE/'input_manifest.json'))
        return task,opt,obs,identity,data
    outcomes={}
    try:
        # Audit correction excludes only UTC fields, never gradients, scalar values or state.
        sample=dict(history=[dict(pre=dict(utc='a',loss=1.))],previous_gradient=torch.ones(2))
        other=copy.deepcopy(sample);other['history'][0]['pre']['utc']='b'
        helper.equal(numeric_observer(sample),numeric_observer(other))
        other['history'][0]['pre']['loss']=2.
        try:helper.equal(numeric_observer(sample),numeric_observer(other))
        except AssertionError:pass
        else:raise AssertionError('Scientific scalar difference was incorrectly ignored.')
        for condition in ['unchanged','always_half','temporary']:
            old=r.r.jsonl(r.PRIOR/'artifacts/attempts/001-local/conditions'/condition/'boundaries.jsonl')[:5]
            new=r.r.jsonl(first/'qualification'/condition/'direct4/boundaries.jsonl')
            assert [(x['state'],x['loss'],x['gradient_norm']) for x in old]==[(x['state'],x['loss'],x['gradient_norm']) for x in new]
        direct=torch.load(first/'qualification/hold8/direct24/recovery.pt',map_location='cpu',weights_only=False)
        resumed=torch.load(first/'qualification/hold8/resume24/recovery.pt',map_location='cpu',weights_only=False)
        audit_pair(direct,resumed);del direct,resumed
        condition='hold8_cosine8'
        for name,until,resume in [('direct24',681,None),('prefix12',669,None),('resume24',681,'prefix12')]:
            task,opt,obs,identity,data=prepare(condition);policy=r.initial_policy()
            if resume:policy=r.load_branch(attempt/'qualification'/condition/resume/'recovery.pt',task,opt,obs,identity,data,condition)
            r.run(task,opt,obs,identity,data,condition,attempt/'qualification'/condition/name,until,policy,stop)
            del task,opt,obs;torch.cuda.empty_cache()
        direct=torch.load(attempt/'qualification'/condition/'direct24/recovery.pt',map_location='cpu',weights_only=False)
        resumed=torch.load(attempt/'qualification'/condition/'resume24/recovery.pt',map_location='cpu',weights_only=False)
        audit_pair(direct,resumed);del direct,resumed
        qualification=dict(status='passed',updates=108,gradient_calls=117,
            audit_correction='Exclude only observer.history[*].pre.utc; altered scientific scalar still fails synthetic check.',
            checks=['three original policies first five Run219 boundaries exact',
                    'both new policies24 direct versus12+12 resumed weights/moments/RNG/policy/full numeric observer exact'])
        r.write_json(attempt/'qualification.json',qualification)
        print(r.b.stamp(),'Qualification passed; five production conditions begin.',flush=True)
        for condition in r.CONDITIONS:
            task,opt,obs,identity,data=prepare(condition);policy=r.initial_policy();folder=attempt/'conditions'/condition
            status=r.run(task,opt,obs,identity,data,condition,folder,1024,policy,stop)
            if status['status']!='completed':raise RuntimeError('Condition did not complete.')
            torch.cuda.synchronize();evaluation_start=time.perf_counter();evaluation=task.evaluate('test');torch.cuda.synchronize()
            outcome=dict(status=status['status'],updates=policy['updates'],runner_seconds=status['elapsed_seconds'],
                endpoint_test=evaluation,endpoint_test_seconds=time.perf_counter()-evaluation_start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),policy_state=policy)
            outcomes[condition]=outcome;r.write_json(folder/'execution.json',outcome);r.write_json(attempt/'results.json',outcomes)
            print(r.b.stamp(),condition,json.dumps(outcome),flush=True)
            del task,opt,obs;torch.cuda.empty_cache()
        assert r.b.file_hash(r.r.PARENT)==r.r.PARENT_SHA
        receipt.update(status='completed',worker_elapsed_seconds=time.perf_counter()-began,finished_utc=r.b.stamp(),results=outcomes,
            total_active_worker_seconds=time.perf_counter()-began+prior['worker_elapsed_seconds'],qualification=qualification)
        r.write_json(attempt/'manifest.json',receipt);r.write_json(HERE/'completion.json',receipt)
        r.write_json(HERE/'status.json',dict(status='completed',attempt='002-audit-continuation',earlier_audit_failure_preserved=True,finished_utc=r.b.stamp()))
        print('Total active worker seconds:',receipt['total_active_worker_seconds'],flush=True)
    except BaseException as error:
        receipt.update(status='failed',worker_elapsed_seconds=time.perf_counter()-began,finished_utc=r.b.stamp(),
            exception=repr(error),traceback=traceback.format_exc(),results=outcomes)
        r.write_json(attempt/'manifest.json',receipt);r.write_json(HERE/'status.json',receipt);raise


if __name__=='__main__':main()
