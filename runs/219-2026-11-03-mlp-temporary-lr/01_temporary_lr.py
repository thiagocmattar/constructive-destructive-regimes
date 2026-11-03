r"""Run219: ordinary Adam with a fixed temporary rule using \eta\widehat\kappa_t."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE_HERE=ROOT/'runs/217-2026-11-03-mlp-adam-control-foundation'
assert hashlib.sha256((BASE_HERE/'01_control.py').read_bytes()).hexdigest()=='01a434fc1bb42090d1ccedcf7cd86b4c022b117e2d5dd78bf06e9bd71ba7bdd0'
spec=importlib.util.spec_from_file_location('run219_base',BASE_HERE/'01_control.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
CONDITIONS=('unchanged','always_half','temporary')
PARENT=ROOT/'runs/218-2026-11-03-mlp-adam-step-actions/artifacts/attempts/001-20261103T172300Z-local-pilot/prefix/checkpoints/main_fork-state-000657.pt'
PARENT_SHA='359097d026b45f8f0b8a33665aca4ad0382e52835745ceaeccf027ed29539ba4'


def eta_kappa_hat(current,previous):
    r"""Direct raw-pair \eta\widehat\kappa_t, available before outgoing motion."""
    if not bool(torch.isfinite(current).all()): raise ValueError('Nonfinite current raw gradient.')
    if previous is None:return None,'missing_previous_gradient'
    if not bool(torch.isfinite(previous).all()):raise ValueError('Nonfinite previous raw gradient.')
    previous=previous.detach().double();current=current.detach().double()
    denominator=float(previous.dot(previous))
    if denominator==0:return None,'zero_previous_gradient_norm'
    return float(previous.dot(previous-current))/denominator,None


def choose_lr(condition,value,undefined_reason=None):
    if condition not in CONDITIONS:raise ValueError('Unknown condition.')
    if condition=='unchanged':return .001,'fixed_base'
    if condition=='always_half':return .0005,'fixed_half'
    if value is None:return .001,undefined_reason or 'undefined_available_diagnostic'
    if not np.isfinite(value):raise ValueError('Nonfinite available diagnostic.')
    return (.0005,'threshold_met') if value>=1.8 else (.001,'below_threshold')


def initial_policy():
    return dict(updates=0,reduced_updates=0,switches_with_entry=0,switches_within=0,last_lr=.001,undefined_updates=0)


def step(optimizer,observer,gradient,condition,policy,pre):
    state=observer.completed_updates
    b.check_optimizer(optimizer,state)
    if gradient.dtype!=torch.float32 or not torch.equal(gradient,b.flat([p.grad.detach() for p in optimizer.param_groups[0]['params']])):
        raise ValueError('Raw capture/input precision or identity differs.')
    value,reason=eta_kappa_hat(gradient,observer.previous_gradient)
    assert pre['state']==state and (value is None)==(pre['raw_projected_change'] is None)
    if value is not None:assert abs(value-pre['raw_projected_change'])<=2e-13*(1+abs(value))
    lr,why=choose_lr(condition,value,reason)
    decision=dict(state=state,manuscript_gradient_index=state+1,available_before_update=state+1,
                  eta_kappa_hat=value,undefined_reason=reason,threshold=1.8,applied_lr=lr,lr_over_base=lr/.001,reason=why)
    before=b.flat(optimizer.param_groups[0]['params']).detach().double()
    optimizer.param_groups[0]['lr']=lr
    try:
        optimizer.step()
        after=b.flat(optimizer.param_groups[0]['params']).detach().double()
        if not bool(torch.isfinite(after).all()):raise ValueError('Nonfinite outgoing parameters.')
        post=observer.accept(gradient,optimizer,before,after,pre)
    finally:optimizer.param_groups[0]['lr']=.001
    # A constant-LR cumulative formula cannot represent a varying LR history.
    post.pop('cumulative_step_rounding_residual_norm',None)
    observer.history[-1]['post'].pop('cumulative_step_rounding_residual_norm',None)
    switched=lr!=policy['last_lr']
    policy['switches_with_entry']+=int(switched)
    policy['switches_within']+=int(switched and policy['updates']>0)
    policy['updates']+=1;policy['reduced_updates']+=int(lr==.0005)
    policy['undefined_updates']+=int(value is None);policy['last_lr']=lr
    post['decision']=copy.deepcopy(decision);observer.history[-1]['post']=copy.deepcopy(post)
    return decision,post


def run(task,optimizer,observer,identity,data,condition,output,until,policy,stop):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();status=dict(status='running',run_id='219',condition=condition,start_state=observer.completed_updates,
                requested_end_state=until,started_utc=b.stamp(),completed_updates=observer.completed_updates)
    b.write_json(output/'status.json',status);b.write_json(output/'identity.json',identity)
    def save(label=None):
        payload=b.checkpoint(task.model,optimizer,observer,identity,data,label or 'rolling_recovery')
        payload.update(checkpoint_kind='run219_branch',condition=condition,policy_state=copy.deepcopy(policy))
        path=output/'recovery.pt' if label is None else output/'checkpoints'/f'{label}-state-{observer.completed_updates:06d}.pt'
        b.save_checkpoint(path,payload,immutable=label is not None)
    try:
        save('start');save()
        while True:
            state=observer.completed_updates
            if stop():save();status['status']='interrupted';break
            if state==until:save('end');save()
            loss,gradient=task.gradient()
            if not np.isfinite(loss):raise ValueError('Nonfinite full-objective loss.')
            pre=observer.observe(state,gradient);value,reason=eta_kappa_hat(gradient,observer.previous_gradient)
            pre.update(loss=loss,eta_kappa_hat=value,eta_kappa_hat_undefined_reason=reason,utc=b.stamp())
            b.append_json(output/'boundaries.jsonl',pre)
            if state==until:status['status']='completed';break
            decision,post=step(optimizer,observer,gradient,condition,policy,pre)
            b.append_json(output/'decisions.jsonl',decision);b.append_json(output/'metrics.jsonl',post)
            if policy['updates']%32==0:save()
            event=dict(state=observer.completed_updates,condition=condition,loss=loss,loss_state=state,
                       elapsed_seconds=time.perf_counter()-started,policy=policy,utc=b.stamp())
            b.append_json(output/'events.jsonl',event)
            status['completed_updates']=observer.completed_updates;b.write_json(output/'status.json',status)
    except BaseException as error:
        status.update(status='failed',exception=repr(error),traceback=traceback.format_exc());raise
    finally:
        status.update(completed_updates=observer.completed_updates,policy_state=policy,elapsed_seconds=time.perf_counter()-started,finished_utc=b.stamp())
        b.write_json(output/'status.json',status);b.write_json(output/'manifest.json',dict(status,identity=identity))
    return status


def load_branch(path,model,optimizer,observer,identity,data,condition):
    saved=b.load_checkpoint(path,model,optimizer,observer,identity,data)
    if saved.get('checkpoint_kind')!='run219_branch' or saved['condition']!=condition:
        raise ValueError('Branch condition differs.')
    policy=saved['policy_state']
    if policy['updates']!=saved['completed_updates']-657:raise ValueError('Policy/optimizer clocks differ.')
    return copy.deepcopy(policy)


def jsonl(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main():
    started=time.perf_counter();deadline=started+600
    cfg=json.loads((HERE/'config.json').read_text());manifest=json.loads((HERE/'input_manifest.json').read_text())
    b.verify_inputs(ROOT,manifest);assert b.file_hash(PARENT)==PARENT_SHA
    attempt=HERE/'artifacts/attempts/001-local';attempt.mkdir(parents=True,exist_ok=False)
    plan=dict(cfg,status='running',started_utc=b.stamp(),parent_git_commit='5107aff',
        authorization='Author: Go on. you can chat with the subagent and launch it; explicit approval via coordinating chat',
        input_manifest_sha256=b.file_hash(HERE/'input_manifest.json'))
    b.write_json(attempt/'authorization_and_plan.json',plan)
    import shutil
    for item in manifest['files']:
        source=ROOT/item['path']
        if source.suffix in ('.py','.json','.tex','.md'):
            target=attempt/'executed_inputs'/item['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    def stop():return time.perf_counter()>=deadline or (HERE/'STOP').exists()
    def prepare(condition):
        if stop():raise TimeoutError('Approved600-second local bound reached.')
        torch.cuda.reset_peak_memory_stats()
        _,task,opt,obs,parent_identity,data=b.prepare(ROOT,BASE_HERE,torch.device('cuda:0'))
        b.load_checkpoint(PARENT,task.model,opt,obs,parent_identity,data)
        identity=dict(run_id='219',condition=condition,parent_sha256=PARENT_SHA,parent_identity=parent_identity,
            config_sha256=b.file_hash(HERE/'config.json'),source_manifest_sha256=b.file_hash(HERE/'input_manifest.json'),
            rule='half_if_available_ge_1.8_else_base' if condition=='temporary' else condition)
        return task,opt,obs,identity,data
    outcomes={}
    try:
        print(b.stamp(),'Local qualification begins.',flush=True)
        for condition,foldername,until,resume in [('unchanged','unchanged4',661,None),('always_half','half4',661,None),
            ('temporary','temporary4',661,None),('temporary','temporary2',659,None),('temporary','temporary_resume4',661,'temporary2')]:
            task,opt,obs,identity,data=prepare(condition);policy=initial_policy()
            if resume:policy=load_branch(attempt/'qualification'/resume/'recovery.pt',task.model,opt,obs,identity,data,condition)
            folder=attempt/'qualification'/foldername
            run(task,opt,obs,identity,data,condition,folder,until,policy,stop)
            del task,opt,obs;torch.cuda.empty_cache()
        old=ROOT/'runs/218-2026-11-03-mlp-adam-step-actions/artifacts/attempts/001-20261103T172300Z-local-pilot/branches'
        for new,arm in [('unchanged4','A0'),('half4','A1')]:
            assert [(r['state'],r['loss'],r['gradient_norm']) for r in jsonl(attempt/'qualification'/new/'boundaries.jsonl')]==[(r['state'],r['loss'],r['gradient_norm']) for r in jsonl(old/arm/'boundaries.jsonl')[:5]]
        from importlib.util import spec_from_file_location,module_from_spec
        helper_spec=spec_from_file_location('old_equality',ROOT/'runs/218-2026-11-03-mlp-adam-step-actions/03_local_pilot.py')
        helper=module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)
        direct=torch.load(attempt/'qualification/temporary4/recovery.pt',map_location='cpu',weights_only=False)
        resumed=torch.load(attempt/'qualification/temporary_resume4/recovery.pt',map_location='cpu',weights_only=False)
        for key in ['model','optimizer','rng','policy_state']:helper.equal(direct[key],resumed[key])
        for key in ['previous_gradient','previous_increment','post']:helper.equal(direct['observer'][key],resumed['observer'][key])
        native=torch.load(attempt/'qualification/unchanged4/recovery.pt',map_location='cpu',weights_only=False)
        historical=torch.load(old/'A0/checkpoints/replay_check-state-000661.pt',map_location='cpu',weights_only=False)
        for key in ['model','optimizer','rng']:helper.equal(native[key],historical[key])
        b.write_json(attempt/'qualification.json',dict(status='passed',updates=16,gradient_calls=21,
             checks=['unchanged and always-half first5 boundary loss/norm observations exact against Run218',
                     'unchanged state661 weights/moments/RNG exact against existing replay state',
                     'temporary4 versus2+2 resumed weights/moments/RNG/policy/raw and actual histories exact']))
        del direct,resumed,native,historical
        print(b.stamp(),'Qualification passed; production begins.',flush=True)
        for condition in CONDITIONS:
            begin=time.perf_counter();task,opt,obs,identity,data=prepare(condition)
            folder=attempt/'conditions'/condition;policy=initial_policy()
            try:
                result=run(task,opt,obs,identity,data,condition,folder,1024,policy,stop)
                elapsed=time.perf_counter()-begin
                evaluation_begin=time.perf_counter();evaluation=task.evaluate('test');torch.cuda.synchronize()
                outcome=dict(status=result['status'],updates=policy['updates'],elapsed_seconds=elapsed,runner_seconds=result['elapsed_seconds'],
                    endpoint_test=evaluation,endpoint_test_seconds=time.perf_counter()-evaluation_begin,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),policy_state=policy)
                observations=jsonl(folder/'boundaries.jsonl')
                if condition in ('unchanged','always_half'):
                    historical_rows=jsonl(old/('A0' if condition=='unchanged' else 'A1')/'boundaries.jsonl')
                    outcome['overlap_Run218_exact']= [(r['state'],r['loss'],r['gradient_norm']) for r in observations[:129]]==[(r['state'],r['loss'],r['gradient_norm']) for r in historical_rows]
                outcomes[condition]=outcome;b.write_json(folder/'execution.json',outcome)
                print(b.stamp(),condition,outcome,flush=True)
            except BaseException as error:
                outcomes[condition]=dict(status='failed',updates=policy['updates'],exception=repr(error),traceback=traceback.format_exc())
                b.write_json(folder/'execution.json',outcomes[condition])
            b.write_json(attempt/'results.json',outcomes)
            del task,opt,obs;torch.cuda.empty_cache()
        assert b.file_hash(PARENT)==PARENT_SHA
        plan.update(status='completed' if all(v['status']=='completed' for v in outcomes.values()) else 'partial',
            worker_elapsed_seconds=time.perf_counter()-started,finished_utc=b.stamp(),results=outcomes)
        b.write_json(attempt/'manifest.json',plan)
        print('Worker seconds:',plan['worker_elapsed_seconds'],flush=True)
    except BaseException as error:
        plan.update(status='failed',worker_elapsed_seconds=time.perf_counter()-started,exception=repr(error),traceback=traceback.format_exc())
        b.write_json(attempt/'manifest.json',plan);raise


if __name__=='__main__':main()
