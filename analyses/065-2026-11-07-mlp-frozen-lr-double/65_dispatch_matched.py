"""Dispatch fixed whole branches after durable shared-prefix completion."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('fleet',HERE/'58_matched_fleet.py');f=importlib.util.module_from_spec(s);s.loader.exec_module(f)
ASSIGNMENTS={
 'resnet':{'resnet-1':['control','lr_0p75','lr_0p5','beta2_0p95','beta2_0p9','cap_q25'],
           'resnet-0':['lr_0p1','beta2_0p5','clip_0p5','clip_1p0','cap_q50','cap_q75']},
 'transformer':{'transformer-1':['control','cap_q25'],'transformer-2':['lr_0p75','cap_q50'],
                'transformer-3':['lr_0p5','cap_q75'],'transformer-4':['lr_0p1','beta2_0p95'],
                'transformer-5':['beta2_0p9','clip_0p5'],'transformer-6':['beta2_0p5','clip_1p0']}}


def dispatch(case):
    assignments=ASSIGNMENTS[case];leader=case+'-1'
    cfg=json.loads((f.ROOT/f.RUNS[case]/'config.json').read_text())
    assert sorted(a for arms in assignments.values() for a in arms)==sorted(a['id'] for a in cfg['arms'])
    f.dump(f.ROOT/f.RUNS[case]/'dispatch.json',dict(assignments=assignments,leader=leader,utc=f.x.stamp()))
    while True:
        p=f.get(leader);assert not p.get('deleted_epoch') and not p.get('budget_stopped')
        state=f.status(leader)
        if state.get('prefix_exit')=='0':break
        if state.get('prefix_exit') is not None:raise RuntimeError(state)
        if time.time()>p['deadline_epoch']-1800:raise RuntimeError('Insufficient remaining bounded lease')
        time.sleep(30)
    if f.get(leader)['phase']=='prefix':f.shared(leader)
    def start(name,arms):
        while True:
            try:p=f.get(name)
            except FileNotFoundError:time.sleep(15);continue
            if p['phase'] in ['ready','prefix_ready']:break
            if p['phase']=='arms':return
            if p.get('deleted_epoch') or p.get('budget_stopped'):raise RuntimeError(name+' unavailable')
            time.sleep(15)
        if p['phase']=='ready':f.distribute(name)
        f.launch(name,'arms',arms)
    with ThreadPoolExecutor(max_workers=len(assignments)) as pool:
        jobs=[pool.submit(start,name,arms) for name,arms in assignments.items()]
        for job in jobs:job.result()
    # Every process independently gates on exact predecessor/fork and native
    # replay before treatment. Check cross-device eight-step identities too.
    fingerprints={}
    while len(fingerprints)<len(assignments):
        for name in assignments:
            if name in fingerprints:continue
            state=f.status(name)
            q=state.get('qualification',{})
            if q.get('eight_step_fingerprint'):fingerprints[name]=q['eight_step_fingerprint']
            if state.get('arms_exit') not in [None,'0']:raise RuntimeError(state)
        if len(set(fingerprints.values()))>1:
            for name in assignments:
                p=f.get(name);f.x.ssh(p,'kill -TERM -- -'+str(p['pid'])+' 2>/dev/null || true\n')
            raise RuntimeError('Cross-worker fingerprint mismatch; branches stopped for audit')
        if len(fingerprints)<len(assignments):time.sleep(15)
    f.dump(f.ROOT/f.RUNS[case]/'cross_worker_qualification.json',dict(status='passed',eight_step_fingerprints=fingerprints,utc=f.x.stamp()))
    f.emit(event='all_branches_dispatched_and_qualified',case=case,workers=len(assignments),arms=12)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case',choices=list(ASSIGNMENTS));dispatch(p.parse_args().case)
