"""Approved local seed0 pilot: existing drivers, minimal qualification, no cloud."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time
import traceback
from datetime import datetime,timezone

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('pilot_actions',HERE/'01_actions.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
spec=importlib.util.spec_from_file_location('loss_events',ROOT/'analyses/036-2026-11-02-controlled-adam-diagnostics/02_extract_loss_spikes.py')
events=importlib.util.module_from_spec(spec);spec.loader.exec_module(events)
b=a.base


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def equal(x,y):
    if isinstance(x,torch.Tensor): assert torch.equal(x,y)
    elif isinstance(x,np.ndarray): assert np.array_equal(x,y)
    elif isinstance(x,dict):
        assert x.keys()==y.keys()
        for key in x: equal(x[key],y[key])
    elif isinstance(x,(tuple,list)):
        assert len(x)==len(y)
        for xx,yy in zip(x,y): equal(xx,yy)
    else: assert x==y


def main():
    start=time.perf_counter();deadline=start+1800
    attempt=HERE/'artifacts/attempts'/('001-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-local-pilot')
    attempt.mkdir(parents=True,exist_ok=False)
    b.write_json(HERE/'local_pilot_attempt.json',dict(path=attempt.relative_to(ROOT).as_posix(),started_utc=b.stamp()))
    plan=dict(status='running',started_utc=b.stamp(),authorization='Author explicitly answered Approve local pilot in coordinating chat',
              source_thread='01a1a1cb-b2b4-74c0-afdb-549829d405e2',run_id='218',analysis_id='040',
              source_commit='92954bded59dbb23c8fce633fb0d5be26ac1a46b',device='cuda:0',local_only=True,cloud_spend_usd=0,
              hard_seconds=1800,expected_minutes=[5,15],seed=0,prefix_updates=657,branch_updates=128,
              arms=list(a.SUPPORTED),deferred=['A7'],nominal_updates=1681,nominal_training_images=8405000,
              qualification_updates_separate=True,training_boundary_and_heldout_evaluations_separate=True)
    b.write_json(attempt/'authorization_and_plan.json',plan)
    implementation=a.implementation_identity()
    b.write_json(attempt/'implementation_identity.json',implementation)
    for item in implementation['files']:
        source=ROOT/item['path']
        if source.suffix in ('.py','.json','.md'):
            dest=attempt/'executed_inputs'/item['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    shutil.copyfile(Path(__file__),attempt/'executed_inputs/03_local_pilot.py')
    b.write_json(attempt/'launcher_identity.json',dict(path=Path(__file__).relative_to(ROOT).as_posix(),sha256=b.file_hash(__file__)))
    phase='qualification'; completed={}
    def emit(message):
        print(b.stamp(),message,flush=True)
        b.append_json(attempt/'pilot_events.jsonl',dict(utc=b.stamp(),phase=phase,message=message,elapsed_seconds=time.perf_counter()-start))
    def stop(): return time.perf_counter()>=deadline or (HERE/'STOP').exists()
    def prepare():
        if stop(): raise TimeoutError('Approved local30-minute bound reached.')
        torch.cuda.reset_peak_memory_stats()
        return b.prepare(ROOT,a.BASE_HERE,torch.device('cuda:0'))
    def stats(folder,begin):
        return dict(elapsed_seconds=time.perf_counter()-begin,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),path=folder.relative_to(ROOT).as_posix())
    try:
        emit('Minimum GPU save/resume qualification starts:4 direct versus2+2 resumed updates.')
        qstart=time.perf_counter(); cfg,task,opt,obs,pid,data=prepare()
        direct=attempt/'qualification/direct4'; b.run_control(task,opt,obs,direct,pid,data,4,{},2,stop)
        del task,opt,obs;torch.cuda.empty_cache()
        _,task,opt,obs,pid2,data2=prepare(); assert pid2==pid
        split=attempt/'qualification/split2';b.run_control(task,opt,obs,split,pid2,data2,2,{},2,stop)
        saved=split/'recovery.pt';del task,opt,obs;torch.cuda.empty_cache()
        _,task,opt,obs,pid2,data2=prepare();b.load_checkpoint(saved,task.model,opt,obs,pid2,data2)
        resumed=attempt/'qualification/resumed4';b.run_control(task,opt,obs,resumed,pid2,data2,4,{},2,stop)
        x=torch.load(direct/'recovery.pt',map_location='cpu',weights_only=False)
        y=torch.load(resumed/'recovery.pt',map_location='cpu',weights_only=False)
        for key in ['model','optimizer','rng']: equal(x[key],y[key])
        for key in ['previous_gradient','previous_increment','post']:equal(x['observer'][key],y['observer'][key])
        dr=rows(direct/'boundaries.jsonl');rr=rows(resumed/'boundaries.jsonl')
        assert [(r['state'],r['loss'],r['gradient_norm']) for r in dr[2:]]==[(r['state'],r['loss'],r['gradient_norm']) for r in rr]
        qualification=stats(direct,qstart);qualification.update(status='passed',optimizer_updates=8,full_gradient_calls=11,
                   equality='bitwise weights/moments/RNG/raw-history/actual-history and loss',runtime_identity=pid['runtime'])
        b.write_json(attempt/'qualification.json',qualification)
        emit('GPU save/resume bitwise check passed; qualification seconds='+str(round(qualification['elapsed_seconds'],2)))
        del task,opt,obs,x,y;torch.cuda.empty_cache()
        phase='common_prefix';begin=time.perf_counter();_,task,opt,obs,pid,data=prepare()
        prefix=attempt/'prefix';b.run_control(task,opt,obs,prefix,pid,data,657,{657:'main_fork'},32,stop)
        assert obs.completed_updates==657 and json.loads((prefix/'status.json').read_text())['status']=='completed'
        parent=prefix/'checkpoints/main_fork-state-000657.pt';parent_hash=b.file_hash(parent)
        prefix_stat=stats(prefix,begin);b.write_json(attempt/'prefix_execution.json',prefix_stat)
        test_begin=time.perf_counter();fork_eval=task.evaluate('test');torch.cuda.synchronize()
        b.write_json(attempt/'fork_evaluation.json',dict(state=657,split='test',**fork_eval,seconds=time.perf_counter()-test_begin))
        emit('Common prefix complete:657 updates, '+str(round(prefix_stat['elapsed_seconds'],2))+' seconds; immutable fork '+parent_hash[:12])
        del task,opt,obs;torch.cuda.empty_cache()
        # Short same-fork replay is mechanics work, separate from the128-update A0.
        phase='same_fork_replay';begin=time.perf_counter();_,task,opt,obs,pid2,data2=prepare()
        identity,branch=a.branch_from_parent(parent,parent_hash,pid,task.model,opt,obs,data2,'A0',implementation,657)
        short=attempt/'qualification/fork_A0_4';a.run_branch(task,opt,obs,branch,identity,data2,short,4,checkpoint_every=4,stop_requested=stop)
        assert branch['updates']==4
        replay_stat=stats(short,begin);replay_stat.update(optimizer_updates=4,full_gradient_calls=5)
        b.write_json(attempt/'same_fork_replay.json',replay_stat)
        del task,opt,obs;torch.cuda.empty_cache()
        for arm in a.SUPPORTED:
            if arm!='A0' and not completed['A0']['historical_qualified']:
                completed[arm]=dict(status='not_executed_historical_gate');continue
            if stop():
                completed[arm]=dict(status='not_executed_local_bound');continue
            phase=arm;begin=time.perf_counter();_,task,opt,obs,pid2,data2=prepare()
            assert pid2==pid
            identity,branch=a.branch_from_parent(parent,parent_hash,pid,task.model,opt,obs,data2,arm,implementation,657)
            folder=attempt/'branches'/arm;saved_states=set()
            def boundary_stop():
                state=obs.completed_updates
                if arm=='A0' and state in (661,689,690) and state not in saved_states:
                    label={661:'replay_check',689:'pre_rise',690:'onset'}[state]
                    a.save_branch(folder/'checkpoints'/f'{label}-state-{state:06d}.pt',task.model,opt,obs,identity,data2,branch,label,True)
                    saved_states.add(state)
                return stop()
            emit('Starts from identical authenticated fork.')
            try:
                result=a.run_branch(task,opt,obs,branch,identity,data2,folder,128,32,boundary_stop)
                execution=stats(folder,begin);execution.update(status=result['status'],updates=branch['updates'],parent_sha256=parent_hash,
                   identity_sha256=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest())
                if result['status']=='completed':
                    evaluation_begin=time.perf_counter();evaluation=task.evaluate('test');torch.cuda.synchronize()
                    execution['heldout_evaluation']=dict(state=785,split='test',**evaluation,seconds=time.perf_counter()-evaluation_begin)
                if arm=='A0':
                    short_rows=rows(short/'boundaries.jsonl');full_rows=rows(folder/'boundaries.jsonl')
                    assert [(r['state'],r['loss'],r['gradient_norm']) for r in short_rows]==[(r['state'],r['loss'],r['gradient_norm']) for r in full_rows[:5]]
                    x=torch.load(short/'checkpoints/endpoint-state-000661.pt',map_location='cpu',weights_only=False)
                    y=torch.load(folder/'checkpoints/replay_check-state-000661.pt',map_location='cpu',weights_only=False)
                    for key in ['model','optimizer','rng']:equal(x[key],y[key])
                    equal(x['observer']['previous_gradient'],y['observer']['previous_gradient'])
                    equal(x['observer']['previous_increment'],y['observer']['previous_increment'])
                    execution['contemporary_replay']='bitwise parameters/moments/RNG/previous-gradient/increment; all5 boundary losses exact'
                    sequence=[r['loss'] for r in rows(prefix/'boundaries.jsonl')[:657]]+[r['loss'] for r in full_rows]
                    candidates=events.excursions(sequence)
                    closest=min(candidates,key=lambda e:abs(e['before_state']-689)+abs(e['peak_state']-703))
                    ratio=sequence[closest['peak_state']]/sequence[closest['before_state']]
                    execution['historical_event']=dict(closest,peak_over_prerise=ratio,
                        before_difference=closest['before_state']-689,peak_difference=closest['peak_state']-703,
                        recovery_difference=None if closest['recovery_state'] is None else closest['recovery_state']-729,
                        ratio_relative_difference=ratio/4.1762988431-1)
                    execution['historical_qualified']=(abs(closest['before_state']-689)<=2 and abs(closest['peak_state']-703)<=2
                        and closest['recovery_state'] is not None and abs(closest['recovery_state']-729)<=2
                        and abs(ratio/4.1762988431-1)<=.05)
                    del x,y
                    emit('Historical event qualification '+str(execution['historical_qualified'])+': '+json.dumps(execution['historical_event']))
                completed[arm]=execution;b.write_json(folder/'execution.json',execution)
                emit('Finished '+str(branch['updates'])+' updates in '+str(round(execution['elapsed_seconds'],2))+' seconds.')
            except BaseException as error:
                completed[arm]=dict(status='failed',exception=repr(error),traceback=traceback.format_exc(),updates=branch['updates'],elapsed_seconds=time.perf_counter()-begin)
                b.write_json(folder/'execution.json',completed[arm]);emit('Failed: '+repr(error))
                if arm=='A0': raise
            b.write_json(attempt/'results.json',completed)
            del task,opt,obs;torch.cuda.empty_cache()
        assert b.file_hash(parent)==parent_hash
        phase='complete';plan.update(status='completed' if all(v['status']=='completed' for v in completed.values()) else 'partial',
            finished_utc=b.stamp(),elapsed_seconds=time.perf_counter()-start,results=completed,parent_sha256=parent_hash,
            qualification_updates=12,actual_main_updates=657+sum(v.get('updates',0) for v in completed.values()))
        b.write_json(attempt/'manifest.json',plan);emit('Pilot ended: '+plan['status']+'; total seconds='+str(round(plan['elapsed_seconds'],2)))
    except BaseException as error:
        plan.update(status='failed',phase=phase,finished_utc=b.stamp(),elapsed_seconds=time.perf_counter()-start,
                    exception=repr(error),traceback=traceback.format_exc(),results=completed)
        b.write_json(attempt/'manifest.json',plan);raise


if __name__=='__main__':main()
