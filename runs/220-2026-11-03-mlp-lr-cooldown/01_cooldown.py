r"""Run220: matched Adam continuations with eight-update LR holds and recovery."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
PRIOR=ROOT/'runs/219-2026-11-03-mlp-temporary-lr'
spec=importlib.util.spec_from_file_location('run220_prior',PRIOR/'01_temporary_lr.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
b=r.b
CONDITIONS=('unchanged','always_half','temporary','hold8','hold8_cosine8')
ORIGINAL_WRITE=b.write_json


def write_json(path,value):
    # Preserve the scientific state through the observed transient Windows lock.
    for retry in range(10):
        try:return ORIGINAL_WRITE(path,value)
        except PermissionError:
            if retry==9:raise
            time.sleep(min(.025*(retry+1),.2))


def initial_policy():
    return dict(updates=0,hold_remaining=0,recovery_position=0,last_lr=.001,
        half_updates=0,below_base_updates=0,switches_within=0,switches_with_entry=0,
        triggers=0,retriggers=0,undefined_updates=0)


def schedule(condition,value,policy):
    """Pure pre-update decision. A trigger includes this update in an eight-step hold."""
    if condition not in CONDITIONS:raise ValueError('Unknown condition.')
    if value is not None and not math.isfinite(value):raise ValueError('Nonfinite diagnostic.')
    p=copy.deepcopy(policy);trigger=value is not None and value>=1.8
    active=p['hold_remaining']>0 or (condition=='hold8_cosine8' and (p['recovery_position']==-1 or 0<p['recovery_position']<8))
    if condition=='unchanged':lr,phase=.001,'fixed_base'
    elif condition=='always_half':lr,phase=.0005,'fixed_half'
    elif condition=='temporary':lr,phase=(.0005,'trigger') if trigger else (.001,'base')
    else:
        if trigger:
            p['retriggers']+=int(active);p['hold_remaining']=8;p['recovery_position']=0
        if p['hold_remaining']>0:
            lr,phase=.0005,'hold';p['hold_remaining']-=1
            if p['hold_remaining']==0 and condition=='hold8_cosine8':p['recovery_position']=-1
        elif condition=='hold8_cosine8' and p['recovery_position']!=0 and p['recovery_position']<8:
            k=1 if p['recovery_position']==-1 else p['recovery_position']+1
            p['recovery_position']=k
            lr=.0005+.0005*(1-math.cos(math.pi*k/8))/2;phase='cosine_recovery'
        else:lr,phase=.001,'base'
    # A completed hold pending its first ramp update also counts as active.
    if condition in ('temporary','hold8','hold8_cosine8'):p['triggers']+=int(trigger)
    p['switches_with_entry']+=int(lr!=p['last_lr'])
    p['switches_within']+=int(lr!=p['last_lr'] and p['updates']>0)
    p['updates']+=1;p['half_updates']+=int(lr==.0005);p['below_base_updates']+=int(lr<.001)
    p['undefined_updates']+=int(value is None);p['last_lr']=lr
    return lr,phase,trigger,p


def step(opt,obs,gradient,condition,policy,pre):
    state=obs.completed_updates;b.check_optimizer(opt,state)
    if gradient.dtype!=torch.float32 or not torch.equal(gradient,b.flat([p.grad.detach() for p in opt.param_groups[0]['params']])):
        raise ValueError('Raw optimizer input changed.')
    value,reason=r.eta_kappa_hat(gradient,obs.previous_gradient)
    assert pre['state']==state and (value is None)==(pre['raw_projected_change'] is None)
    if value is not None:assert abs(value-pre['raw_projected_change'])<=2e-13*(1+abs(value))
    lr,phase,trigger,next_policy=schedule(condition,value,policy)
    decision=dict(state=state,manuscript_gradient_index=state+1,available_before_update=state+1,
        eta_kappa_hat=value,undefined_reason=reason,threshold=1.8,applied_lr=lr,lr_over_base=lr/.001,
        phase=phase,trigger=trigger,hold_remaining_after=next_policy['hold_remaining'],
        recovery_position_after=next_policy['recovery_position'])
    before=b.flat(opt.param_groups[0]['params']).detach().double();opt.param_groups[0]['lr']=lr
    try:
        opt.step();after=b.flat(opt.param_groups[0]['params']).detach().double()
        if not bool(torch.isfinite(after).all()):raise ValueError('Nonfinite parameters.')
        post=obs.accept(gradient,opt,before,after,pre)
    finally:opt.param_groups[0]['lr']=.001
    post.pop('cumulative_step_rounding_residual_norm',None)
    post['decision']=copy.deepcopy(decision);obs.history[-1]['post']=copy.deepcopy(post)
    policy.clear();policy.update(next_policy)
    return decision,post


def run(task,opt,obs,identity,data,condition,output,until,policy,stop):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();status=dict(status='running',run_id='220',condition=condition,
        start_state=obs.completed_updates,requested_end_state=until,started_utc=b.stamp())
    write_json(output/'status.json',status);write_json(output/'identity.json',identity)
    def save(label=None):
        payload=b.checkpoint(task.model,opt,obs,identity,data,label or 'rolling_recovery')
        payload.update(checkpoint_kind='run220_branch',condition=condition,policy_state=copy.deepcopy(policy))
        path=output/'recovery.pt' if label is None else output/'checkpoints'/f'{label}-state-{obs.completed_updates:06d}.pt'
        b.save_checkpoint(path,payload,immutable=label is not None)
    try:
        save('start');save()
        while True:
            state=obs.completed_updates
            if stop():save();status['status']='interrupted';break
            if state==until:save('end');save()
            loss,gradient=task.gradient()
            if not np.isfinite(loss):raise ValueError('Nonfinite objective.')
            pre=obs.observe(state,gradient);value,reason=r.eta_kappa_hat(gradient,obs.previous_gradient)
            pre.update(loss=loss,eta_kappa_hat=value,eta_kappa_hat_undefined_reason=reason,utc=b.stamp())
            b.append_json(output/'boundaries.jsonl',pre)
            if state==until:status['status']='completed';break
            decision,post=step(opt,obs,gradient,condition,policy,pre)
            b.append_json(output/'decisions.jsonl',decision);b.append_json(output/'metrics.jsonl',post)
            if policy['updates']%32==0:save()
            b.append_json(output/'events.jsonl',dict(state=obs.completed_updates,condition=condition,
                loss=loss,loss_state=state,elapsed_seconds=time.perf_counter()-started,policy=policy,utc=b.stamp()))
            status['completed_updates']=obs.completed_updates;write_json(output/'status.json',status)
    except BaseException as error:
        status.update(status='failed',exception=repr(error),traceback=traceback.format_exc());raise
    finally:
        status.update(completed_updates=obs.completed_updates,policy_state=policy,
            elapsed_seconds=time.perf_counter()-started,finished_utc=b.stamp())
        write_json(output/'status.json',status);write_json(output/'manifest.json',dict(status,identity=identity))
    return status


def load_branch(path,task,opt,obs,identity,data,condition):
    saved=b.load_checkpoint(path,task.model,opt,obs,identity,data)
    if saved.get('checkpoint_kind')!='run220_branch' or saved['condition']!=condition:raise ValueError('Branch identity differs.')
    if saved['policy_state']['updates']!=saved['completed_updates']-657:raise ValueError('Policy clock differs.')
    return copy.deepcopy(saved['policy_state'])


def main():
    started=time.perf_counter();deadline=started+600
    cfg=json.loads((HERE/'config.json').read_text(encoding='utf-8'))
    manifest=json.loads((HERE/'input_manifest.json').read_text(encoding='utf-8'));b.verify_inputs(ROOT,manifest)
    assert b.file_hash(r.PARENT)==r.PARENT_SHA
    attempt=HERE/'artifacts/attempts/001-local';attempt.mkdir(parents=True,exist_ok=False)
    plan=dict(cfg,status='running',started_utc=b.stamp(),source_commit='941f3cb3dc217151ecaf41669e30aeae87643b45',
        authorization='User: Go on, after the explicit eight-update hold and eight-update cosine-recovery proposal.',
        input_manifest_sha256=b.file_hash(HERE/'input_manifest.json'))
    write_json(attempt/'authorization_and_plan.json',plan)
    for item in manifest['files']:
        path=ROOT/item['path']
        if path.suffix in ('.py','.json'):
            target=attempt/'executed_inputs'/item['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    shutil.copyfile(HERE/'input_manifest.json',attempt/'executed_input_manifest.json')
    def stop():return time.perf_counter()>=deadline or (HERE/'STOP').exists()
    def prepare(condition):
        if stop():raise TimeoutError('600-second bound reached.')
        torch.cuda.reset_peak_memory_stats()
        _,task,opt,obs,parent_identity,data=b.prepare(ROOT,r.BASE_HERE,torch.device('cuda:0'))
        b.load_checkpoint(r.PARENT,task.model,opt,obs,parent_identity,data)
        identity=dict(run_id='220',condition=condition,parent_sha256=r.PARENT_SHA,parent_identity=parent_identity,
            config_sha256=b.file_hash(HERE/'config.json'),source_manifest_sha256=b.file_hash(HERE/'input_manifest.json'))
        return task,opt,obs,identity,data
    helper_spec=importlib.util.spec_from_file_location('run220_equality',ROOT/'runs/218-2026-11-03-mlp-adam-step-actions/03_local_pilot.py')
    helper=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)
    outcomes={}
    try:
        print(b.stamp(),'Qualification begins.',flush=True)
        for condition in CONDITIONS:
            cases=[('direct24',681,None),('prefix12',669,None),('resume24',681,'prefix12')] if condition.startswith('hold8') else [('direct4',661,None)]
            for name,until,resume in cases:
                task,opt,obs,identity,data=prepare(condition);policy=initial_policy()
                if resume:policy=load_branch(attempt/'qualification'/condition/resume/'recovery.pt',task,opt,obs,identity,data,condition)
                run(task,opt,obs,identity,data,condition,attempt/'qualification'/condition/name,until,policy,stop)
                del task,opt,obs;torch.cuda.empty_cache()
            if condition.startswith('hold8'):
                direct=torch.load(attempt/'qualification'/condition/'direct24/recovery.pt',map_location='cpu',weights_only=False)
                resumed=torch.load(attempt/'qualification'/condition/'resume24/recovery.pt',map_location='cpu',weights_only=False)
                for key in ['model','optimizer','rng','policy_state','observer']:helper.equal(direct[key],resumed[key])
                del direct,resumed
            else:
                previous=r.jsonl(PRIOR/'artifacts/attempts/001-local/conditions'/condition/'boundaries.jsonl')[:5]
                current=r.jsonl(attempt/'qualification'/condition/'direct4/boundaries.jsonl')
                assert [(x['state'],x['loss'],x['gradient_norm']) for x in current]==[(x['state'],x['loss'],x['gradient_norm']) for x in previous]
        write_json(attempt/'qualification.json',dict(status='passed',updates=108,gradient_calls=117,
            checks=['three original conditions replay first five Run219 observations exactly',
                    'both new conditions: 24 direct versus12+12 resumed weights/moments/RNG/policy/full observer exact']))
        print(b.stamp(),'Qualification passed; five production conditions begin.',flush=True)
        for condition in CONDITIONS:
            task,opt,obs,identity,data=prepare(condition);policy=initial_policy();folder=attempt/'conditions'/condition
            status=run(task,opt,obs,identity,data,condition,folder,1024,policy,stop)
            if status['status']!='completed':raise RuntimeError('Condition did not complete.')
            torch.cuda.synchronize();evaluation_start=time.perf_counter();evaluation=task.evaluate('test');torch.cuda.synchronize()
            outcome=dict(status=status['status'],updates=policy['updates'],runner_seconds=status['elapsed_seconds'],
                endpoint_test=evaluation,endpoint_test_seconds=time.perf_counter()-evaluation_start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),policy_state=policy)
            outcomes[condition]=outcome;write_json(folder/'execution.json',outcome);write_json(attempt/'results.json',outcomes)
            print(b.stamp(),condition,json.dumps(outcome),flush=True)
            del task,opt,obs;torch.cuda.empty_cache()
        assert b.file_hash(r.PARENT)==r.PARENT_SHA
        plan.update(status='completed',worker_elapsed_seconds=time.perf_counter()-started,finished_utc=b.stamp(),results=outcomes)
        write_json(attempt/'manifest.json',plan);write_json(HERE/'completion.json',plan)
        write_json(HERE/'status.json',dict(status='completed',attempt='001-local',finished_utc=b.stamp()))
        print('Worker seconds:',plan['worker_elapsed_seconds'],flush=True)
    except BaseException as error:
        plan.update(status='failed',worker_elapsed_seconds=time.perf_counter()-started,finished_utc=b.stamp(),
            exception=repr(error),traceback=traceback.format_exc(),results=outcomes)
        write_json(attempt/'manifest.json',plan);write_json(HERE/'status.json',plan);raise


if __name__=='__main__':main()
