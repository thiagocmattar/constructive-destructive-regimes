"""Verify six warm LR branches and reduce complete fixed-window comparisons."""
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
NAMES=['preactivation1','hidden1','preactivation2','hidden2','output']
CACHE={}


def sha(p):
    p=p.resolve();s=p.stat();key=(str(p),s.st_size,s.st_mtime_ns)
    if key not in CACHE:
        with p.open('rb') as f:CACHE[key]=hashlib.file_digest(f,'sha256').hexdigest()
    return CACHE[key]


def dump(p,value):p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',newline='\n')
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s]


def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def dense(raw,boundaries,rep):
    checks={}
    for name in NAMES:
        a=np.load(raw/(name+'.npy'),mmap_mode='r');flat=a.reshape(len(a),-1);assert len(a)==55
        norm2=np.zeros(54);dots=np.zeros(53)
        for start in range(0,flat.shape[1],32768):
            delta=np.diff(np.asarray(flat[:,start:start+32768],dtype=np.float64),axis=0)
            norm2+=np.sum(delta*delta,axis=1);dots+=np.sum(delta[1:]*delta[:-1],axis=1)
        rho=dots/np.sqrt(norm2[1:]*norm2[:-1]);rms=np.sqrt(norm2[1:]/flat.shape[1])
        np.testing.assert_allclose(rho,[rep[t][name]['rho'] for t in range(448,501)],rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(rms,[rep[t][name]['change_rms'] for t in range(448,501)],rtol=1e-9,atol=1e-12)
        checks[name]=dict(states=[448,500],coordinates=flat.shape[1],cosine_and_rms=True)
    a=np.load(raw/'raw_gradient.npy',mmap_mode='r')
    for i in range(1,55):
        g=np.asarray(a[i],dtype=np.float64);p=np.asarray(a[i-1],dtype=np.float64);q=boundaries[447+i]
        for z,old,kn,kr in [(g,p,'raw_gradient_norm','rho'),(g[614600:],p[614600:],'active_gradient_norm','active_gradient_rho'),
                              (g[:614600],p[:614600],'first_gradient_norm','first_gradient_rho')]:
            np.testing.assert_allclose([np.linalg.norm(z),float(z@old)/(np.linalg.norm(z)*np.linalg.norm(old))],
                [q[kn],q[kr]],rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(1-float(g@p)/float(p@p),q['eta_kappa_hat'],rtol=1e-9,atol=1e-10)
    return dict(representations=checks,raw_first_later_gradient_norms_cosines_and_eta_states=[448,501])


def verify(run):
    cfg=json.loads((run/'config.json').read_text());raw=run/'artifacts/spark/payload/worker'
    inputs=json.loads((run/'input_manifest.json').read_text())['files']
    for item in inputs:
        p=ROOT/item.get('local_path',item['path'])
        assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],str(p)
    retrieval=json.loads((run/'retrieval.json').read_text());assert retrieval['verified'] and retrieval['status']=='completed'
    inventory=json.loads((raw.parent/'transfer_inventory.json').read_text())['files']
    for item in inventory:
        p=raw.parent/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],str(p)
    manifest=json.loads((raw/'manifest.json').read_text());assert manifest['status']=='completed' and manifest['completed_state']==801
    qualification=json.loads((raw/'qualification.json').read_text())
    assert qualification['status']=='passed' and qualification['exact_boundary_states']==list(range(448,513))
    assert qualification['full_state512_exact'] and qualification['first_step_all_moments_clocks_rng_exact']
    load=lambda name:torch.load(raw/name,map_location='cpu',weights_only=False)
    fork=load('fork448.pt');end=load('end.pt');qualified=load('qualification512.pt')
    native=load('native449.pt');changed=load('intervention449.pt')
    parent=torch.load(ROOT/'runs/244-2026-11-07-mlp-freeze-first-linear/artifacts/spark/payload/worker/fork.pt',map_location='cpu',weights_only=False)
    historical=torch.load(ROOT/'runs/223-2026-11-04-multispike-interventions/artifacts/spark/payload/worker/prefix/end.pt',map_location='cpu',weights_only=False)
    for key in ['model','optimizer','rng','previous_gradient','current_gradient']:
        equal(fork[key],parent[key]);equal(qualified[key],historical[key])
    for key in ['optimizer','rng','previous_gradient']:equal(changed[key],native[key])
    assert all(int(s['step'])==448 for s in fork['optimizer']['state'].values())
    assert all(int(s['step'])==801 for s in end['optimizer']['state'].values())
    equal(fork['optimizer']['param_groups'],end['optimizer']['param_groups'])
    for n,p in end['model'].items():assert torch.isfinite(p).all() and not torch.equal(p,fork['model'][n])
    for n,p in changed['model'].items():
        factor=(cfg['first_lr'] if n.startswith('1.') else cfg['later_lr'])/.001
        expected=fork['model'][n].double()+factor*(native['model'][n].double()-fork['model'][n].double())
        torch.testing.assert_close(p.double(),expected,rtol=0,atol=2e-7)
        if factor==1:equal(p,native['model'][n])
    boundary={q['state']:q for q in rows(raw/'boundaries.jsonl')}
    rep={q['state']:q['streams'] for q in rows(raw/'representations.jsonl')}
    updates=rows(raw/'updates.jsonl')
    assert list(boundary)==list(range(448,802)) and list(rep)==list(range(447,801))
    assert [q['state'] for q in updates]==list(range(448,801))
    for q in updates:
        for layer in ['1','3','5']:
            item=q['layers'][layer]
            assert item['lr']==(cfg['first_lr'] if layer=='1' else cfg['later_lr'])
            assert item['actual_increment_norm']>0 and item['denominator_rms']>0
        np.testing.assert_allclose(q['actual_increment_norm']**2,sum(v['actual_increment_norm']**2 for v in q['layers'].values()),rtol=1e-10)
    result=dict(status='passed',input_files_checked=len(inputs),retrieved_files_checked=len(inventory),
        full_fork_matches_parent=True,full_qualification512_matches=True,all_layers_updated=True,
        first_changed_step_native_moments_and_scaled_updates=True,dense_checks=dense(raw,boundary,rep),
        worker_manifest=manifest,qualification=qualification,generator_sha256=sha(Path(__file__)))
    dump(run/'verification.json',result)
    return boundary,rep,updates,result


def summarize(arm,curve,a,b):
    t=np.arange(a,b+1);ix=t-448;loss=np.array(curve['loss'])[ix];delta=np.diff(loss)
    rms=np.array(curve['layers']['output']['change_rms'])[ix];rho=np.array(curve['layers']['output']['rho'])[ix]
    return dict(arm=arm,window=f'{a}-{b}',loss_mean=float(loss.mean()),loss_max=float(loss.max()),
        peak_state=int(t[np.argmax(loss)]),loss_final=float(loss[-1]),positive_loss_variation=float(np.maximum(delta,0).sum()),
        largest_upward_change=float(max(0,delta.max())),upward_steps=int(np.sum(delta>0)),
        output_negative=int(np.sum(rho<0)),output_rms_max=float(rms.max()),output_rms_energy=float(np.sum(rms**2)))


def main():
    plan=json.loads((HERE/'layer_lr_plan.json').read_text())
    prior=json.loads((HERE/'comparison.json').read_text());audit=json.loads((HERE/'figure_verification.json').read_text())
    assert prior['status']=='verified' and sha(HERE/'comparison.json')==audit['source_sha256']
    curves={arm:copy.deepcopy(prior['curves'][arm]) for arm in ['control','freeze']}
    for c in curves.values():c.update(first_lr=[.001]*353,later_lr=[.001]*353)
    curves['freeze']['first_lr']=[0.]*353
    sources=[HERE/'layer_lr_plan.json',HERE/'comparison.json'];verifications={}
    for record in plan['arms']:
        arm=record['arm'];run=ROOT/record['path'];raw=run/'artifacts/spark/payload/worker'
        b,r,u,v=verify(run);verifications[arm]=v
        mapping={'loss':'loss','eta_kappa_hat':'eta_kappa_hat','gradient_rho':'rho','raw_gradient_norm':'raw_gradient_norm',
                 'active_gradient_rho':'active_gradient_rho','active_gradient_norm':'active_gradient_norm',
                 'first_gradient_rho':'first_gradient_rho','first_gradient_norm':'first_gradient_norm'}
        curve={k:[b[t][source] for t in range(448,801)] for k,source in mapping.items()}
        curve.update(first_lr=[record['first_lr']]*353,later_lr=[record['later_lr']]*353,
            layers={n:{k:[r[t][n][k] for t in range(448,801)] for k in ['rho','change_rms']} for n in NAMES},updates=u)
        for key in ['loss','eta_kappa_hat','gradient_rho','raw_gradient_norm','active_gradient_rho','active_gradient_norm']:
            np.testing.assert_allclose(curve[key][0],curves['control'][key][0],rtol=0,atol=1e-12)
        curves[arm]=curve
        sources += [run/name for name in ['config.json','input_manifest.json','retrieval.json','verification.json']]
        sources += [raw/name for name in ['inventory.json','boundaries.jsonl','representations.jsonl','updates.jsonl']]
        print(json.dumps(dict(verified=arm,seconds=v['worker_manifest']['elapsed_seconds'])),flush=True)
    metrics=[summarize(arm,c,a,b) for a,b in [(448,800),(450,500),(769,800)] for arm,c in curves.items()]
    targets={}
    for arm,c in curves.items():
        loss=np.array(c['loss']);hits=[448+i for i in range(len(loss)-9) if np.all(loss[i:i+10]<.02)]
        targets[arm]=dict(threshold=.02,consecutive_states=10,first_qualifying_start=hits[0] if hits else None,
                         first_confirmation_state=hits[0]+9 if hits else None,censored=not bool(hits))
    for m in metrics:
        base=next(x for x in metrics if x['window']==m['window'] and x['arm']=='control')
        m['mean_loss_change_percent_vs_baseline']=100*(m['loss_mean']/base['loss_mean']-1)
    result=dict(status='verified',states=list(range(448,801)),curves=curves,metrics=metrics,targets=targets,
        plan=plan,run_verifications=verifications,sources={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
        generator_sha256=sha(Path(__file__)),limits='One seed/fork; training only; unmatched total update norms across location controls')
    dump(HERE/'layer_lr_comparison.json',result)
    with (HERE/'layer_lr_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]),lineterminator='\n');writer.writeheader();writer.writerows(metrics)
    print(json.dumps(dict(metrics=metrics,targets=targets),indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
