"""Independent gate mathematics, paired norm checks and full-window reductions."""
import copy
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('prior_lr_reduce',HERE/'07_reduce_layer_lr.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
sha,dump,rows,equal=prior.sha,prior.dump,prior.rows,prior.equal
NAMES=prior.NAMES


def inventory(run):
    for item in json.loads((run/'input_manifest.json').read_text())['files']:
        p=ROOT/item.get('local_path',item['path'])
        assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],str(p)
    receipt=json.loads((run/'retrieval.json').read_text());assert receipt['verified'] and receipt['status']=='completed'
    raw=run/'artifacts/spark/payload/worker'
    records=json.loads((raw.parent/'transfer_inventory.json').read_text())['files']
    for item in records:
        p=raw.parent/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],str(p)
    return raw,len(records)


def calibrate(run,x):
    raw,count=inventory(run);t=json.loads((raw/'thresholds.json').read_text())
    a=np.load(raw/'absolute_updates.npy',mmap_mode='r');s=np.load(raw/'neuron_rms.npy')
    assert a.shape==(32,614600) and s.shape==(32,200) and t['window']==[416,447]
    assert float(np.quantile(a,.9,method='linear'))==t['coordinate_q90']
    for key,q in [('q90',.9),('q75',.75),('q50',.5)]:assert float(np.quantile(s,q,method='linear'))==t['neuron'][key]
    for state in [416,447]:
        audit=torch.load(raw/f'calibration_audit_{state}.pt',map_location='cpu',weights_only=False)
        dw,db=audit['weight'].numpy(),audit['bias'].numpy();z=x@dw.T+db
        np.testing.assert_allclose(np.sqrt(np.mean(z*z,axis=0)),s[state-416],rtol=1e-10,atol=1e-12)
        np.testing.assert_array_equal(np.abs(np.concatenate([dw.ravel(),db])),a[state-416])
    parent=torch.load(ROOT/'runs/244-2026-11-07-mlp-freeze-first-linear/artifacts/spark/payload/worker/fork.pt',map_location='cpu',weights_only=False)
    equal(parent,torch.load(raw/'qualified448.pt',map_location='cpu',weights_only=False))
    result=dict(status='passed',all_quantiles_reproduced=True,independent_neuron_rms_audits=[416,447],full_fork_exact=True,
        retrieved_files=count,thresholds=t,generator_sha256=sha(Path(__file__)))
    dump(run/'verification.json',result);return t


def gate_audit(raw,state,x,kind,threshold,target):
    p=torch.load(raw/f'gate_audit_{state}.pt',map_location='cpu',weights_only=False)
    ow,ob=p['before_weight'].numpy().astype(np.float64),p['before_bias'].numpy().astype(np.float64)
    dw,db=p['proposed_weight'].numpy(),p['proposed_bias'].numpy()
    aw,ab=p['actual_weight'].numpy(),p['actual_bias'].numpy()
    proposed=np.concatenate([dw.ravel(),db]);actual=np.concatenate([aw.ravel(),ab]);stats=p['stats']
    z=x@dw.T+db;scores=np.sqrt(np.mean(z*z,axis=0))
    np.testing.assert_allclose(scores,p['proposed_neuron_rms'].numpy(),rtol=1e-10,atol=1e-12)
    if kind=='discard':
        intended=np.where(np.abs(proposed)<=threshold,proposed,0.);affected=float(np.mean(np.abs(proposed)>threshold))
    elif kind=='cap':
        intended=np.clip(proposed,-threshold,threshold);affected=float(np.mean(np.abs(proposed)>threshold))
    elif kind=='neuron':
        factors=np.divide(threshold,scores,out=np.ones_like(scores),where=scores>0).clip(max=1)
        intended=np.concatenate([(dw*factors[:,None]).ravel(),db*factors]);affected=float(np.mean(scores>threshold))
        np.testing.assert_allclose(factors,p['factors'].numpy(),rtol=1e-10,atol=1e-12)
    else:
        factor=target/np.linalg.norm(proposed) if np.linalg.norm(proposed) else 0.
        intended=proposed*factor
        # Exact equality to one is discontinuous under CPU/GPU summation.
        # Reproduce the native binary flag from its recorded operands; audit
        # the continuous norm/factor independently with NumPy below.
        native_factor=target/stats['proposed_norm'] if stats['proposed_norm'] else 0.
        affected=float(native_factor!=1)
        assert np.all(p['factors'].numpy()==native_factor)
        np.testing.assert_allclose(p['factors'].numpy(),factor,rtol=1e-10,atol=1e-12)
    nw,nb=intended[:dw.size].reshape(dw.shape),intended[dw.size:]
    rounded=np.concatenate([((ow+nw).astype(np.float32).astype(np.float64)-ow).ravel(),
                            (ob+nb).astype(np.float32).astype(np.float64)-ob])
    np.testing.assert_allclose(actual,rounded,rtol=0,atol=3e-8)
    qerror=np.linalg.norm(actual-intended)
    np.testing.assert_allclose([stats['proposed_norm'],stats['actual_norm'],stats['intended_norm'],stats['removed_norm'],stats['rounding_error_norm'],stats['affected_fraction']],
        [np.linalg.norm(proposed),np.linalg.norm(actual),np.linalg.norm(intended),np.linalg.norm(proposed-intended),qerror,affected],rtol=1e-9,atol=1e-11)
    z_actual=x@aw.T+ab;actual_scores=np.sqrt(np.mean(z_actual*z_actual,axis=0))
    np.testing.assert_allclose(actual_scores,p['actual_neuron_rms'].numpy(),rtol=1e-10,atol=1e-12)
    error=x@(aw-nw).T+(ab-nb);error_scores=np.sqrt(np.mean(error*error,axis=0))
    if kind=='neuron':assert np.all(actual_scores<=threshold+error_scores+1e-11)
    if kind=='uniform':assert abs(np.linalg.norm(actual)-target)<=qerror+1e-11
    if kind=='discard':assert np.all(actual[np.abs(proposed)>threshold]==0)
    return dict(state=state,independent_numpy_policy=True,maximum_rounding_reproduction_error=float(np.max(np.abs(actual-rounded))),
                actual_norm=float(np.linalg.norm(actual)),actual_neuron_rms_max=float(actual_scores.max()))


def verify(run,x,thresholds):
    raw,count=inventory(run);cfg=json.loads((run/'config.json').read_text())
    resolved=json.loads((raw/'resolved_policy.json').read_text())
    threshold=thresholds['coordinate_q90'] if cfg['threshold_key']=='coordinate_q90' else thresholds['neuron'][cfg['threshold_key']]
    assert resolved['threshold']==threshold
    manifest=json.loads((raw/'manifest.json').read_text());assert manifest['status']=='completed' and manifest['completed_state']==801
    checks=json.loads((raw/'qualification.json').read_text())
    assert checks['exact_boundary_states']==list(range(448,513)) and checks['full_state512_exact'] and checks['first_step_all_moments_clocks_rng_exact']
    load=lambda name:torch.load(raw/name,map_location='cpu',weights_only=False)
    fork,end,qualified,native,changed=[load(n) for n in ['fork448.pt','end.pt','qualification512.pt','native449.pt','intervention449.pt']]
    parent=torch.load(ROOT/'runs/244-2026-11-07-mlp-freeze-first-linear/artifacts/spark/payload/worker/fork.pt',map_location='cpu',weights_only=False)
    historical=torch.load(ROOT/'runs/223-2026-11-04-multispike-interventions/artifacts/spark/payload/worker/prefix/end.pt',map_location='cpu',weights_only=False)
    for key in ['model','optimizer','rng','previous_gradient','current_gradient']:
        equal(fork[key],parent[key]);equal(qualified[key],historical[key])
    for key in ['optimizer','rng','previous_gradient']:equal(changed[key],native[key])
    for name,p in changed['model'].items():
        if not name.startswith('1.'):equal(p,native['model'][name])
    assert all(int(s['step'])==801 for s in end['optimizer']['state'].values())
    assert all(torch.isfinite(p).all() for p in end['model'].values())
    equal(fork['optimizer']['param_groups'],end['optimizer']['param_groups'])
    boundary={q['state']:q for q in rows(raw/'boundaries.jsonl')};rep={q['state']:q['streams'] for q in rows(raw/'representations.jsonl')}
    updates=rows(raw/'updates.jsonl');assert list(boundary)==list(range(448,802)) and list(rep)==list(range(447,801))
    assert [q['state'] for q in updates]==list(range(448,801))
    targets={}
    if cfg['kind']=='uniform':
        targets={q['state']:q['gate']['actual_norm'] for q in rows(ROOT/cfg['target_run']/'artifacts/spark/payload/worker/updates.jsonl')}
    for q in updates:
        g=q['gate'];assert g['kind']==cfg['kind'] and g['threshold']==threshold
        np.testing.assert_allclose(g['actual_norm'],q['layers']['1']['actual_increment_norm'],rtol=1e-12,atol=1e-12)
        assert all(l['lr']==.001 for l in q['layers'].values())
        np.testing.assert_allclose(q['actual_increment_norm']**2,sum(l['actual_increment_norm']**2 for l in q['layers'].values()),rtol=1e-10)
        if cfg['kind']=='neuron':assert g['actual_neuron_rms_max']<=threshold+g['rounding_neuron_rms_max']+1e-11
        if cfg['kind']=='uniform':
            assert g['target_norm']==targets[q['state']]
            assert abs(g['actual_norm']-g['target_norm'])<=g['rounding_error_norm']+1e-11
        assert 0<=g['affected_fraction']<=1
    audits=[gate_audit(raw,t,x,cfg['kind'],threshold,targets.get(t)) for t in [448,474,800]]
    first=load('gate_audit_448.pt')
    for name,key in [('1.weight','weight'),('1.bias','bias')]:
        equal(first['proposed_'+key],native['model'][name].double()-fork['model'][name].double())
        equal(first['actual_'+key],changed['model'][name].double()-fork['model'][name].double())
    record=dict(status='passed',source_files_checked=len(json.loads((run/'input_manifest.json').read_text())['files']),
        retrieved_files_checked=count,full_fork_and_qualification_exact=True,first_step_native_moments_later_parameters=True,
        gate_audits=audits,all_steps_gate_checks=True,dense_checks=prior.dense(raw,boundary,rep),
        final_changed_parameters={n:int((p!=fork['model'][n]).sum()) for n,p in end['model'].items()},
        manifest=manifest,resolved_policy=resolved,generator_sha256=sha(Path(__file__)))
    dump(run/'verification.json',record)
    return boundary,rep,updates,record


def main():
    plan=json.loads((HERE/'selective_plan.json').read_text())
    data=torch.load(ROOT/'cache/cifar10-eos/data.pt',map_location='cpu',weights_only=True)
    x=data['x_train'].numpy().reshape(5000,3072).astype(np.float64);del data
    thresholds=calibrate(ROOT/plan['calibration_run'],x)
    old=json.loads((HERE/'layer_lr_comparison.json').read_text());assert old['status']=='verified'
    audit=json.loads((HERE/'figure03_verification.json').read_text());assert audit['source_sha256']==sha(HERE/'layer_lr_comparison.json')
    curves={a:copy.deepcopy(old['curves'][a]) for a in ['control','first_quarter']};verifications={}
    sources=[HERE/'selective_plan.json',HERE/'layer_lr_comparison.json',ROOT/plan['calibration_run']/'verification.json']
    for item in plan['arms']:
        run=ROOT/item['path'];arm=item['arm'];b,r,u,v=verify(run,x,thresholds);verifications[arm]=v
        mapping={'loss':'loss','eta_kappa_hat':'eta_kappa_hat','gradient_rho':'rho','raw_gradient_norm':'raw_gradient_norm',
                 'active_gradient_rho':'active_gradient_rho','active_gradient_norm':'active_gradient_norm'}
        c={k:[b[t][s] for t in range(448,801)] for k,s in mapping.items()}
        c['layers']={n:{k:[r[t][n][k] for t in range(448,801)] for k in ['rho','change_rms']} for n in NAMES}
        gatekeys=['affected_fraction','proposed_norm','actual_norm','removed_norm','rounding_error_norm','retained_norm_fraction',
                  'proposed_neuron_rms_max','actual_neuron_rms_max','rounding_neuron_rms_max','actual_neuron_cap_excess','factor_min','factor_mean','factor_max','target_norm']
        c['gate']={k:[q['gate'][k] for q in u] for k in gatekeys}
        for key in mapping:np.testing.assert_allclose(c[key][0],curves['control'][key][0],rtol=0,atol=1e-12)
        curves[arm]=c
        sources += [run/n for n in ['config.json','input_manifest.json','retrieval.json','verification.json']]
        sources += [run/'artifacts/spark/payload/worker'/n for n in ['inventory.json','boundaries.jsonl','representations.jsonl','updates.jsonl','resolved_policy.json']]
        print(json.dumps(dict(verified=arm,seconds=v['manifest']['elapsed_seconds'])),flush=True)
    metrics=[prior.summarize(a,c,start,end) for start,end in [(448,800),(450,500),(769,800)] for a,c in curves.items()]
    for m in metrics:
        base=next(v for v in metrics if v['window']==m['window'] and v['arm']=='control')
        m['mean_loss_change_percent_vs_baseline']=100*(m['loss_mean']/base['loss_mean']-1)
    targets={}
    gate_summaries={}
    for arm,c in curves.items():
        loss=np.array(c['loss']);hits=[448+i for i in range(344) if np.all(loss[i:i+10]<.02)]
        targets[arm]=dict(first_start=hits[0] if hits else None,confirmation=hits[0]+9 if hits else None,censored=not bool(hits))
        if 'gate' in c:
            g=c['gate'];gate_summaries[arm]=dict(affected_fraction_mean=float(np.mean(g['affected_fraction'])),
                retained_norm_fraction_mean=float(np.mean(g['retained_norm_fraction'])),
                first_hidden_negative=int(np.sum(np.array(c['layers']['hidden1']['rho'])<0)),
                applied_first_path_length=float(np.sum(g['actual_norm'])),
                amplification_steps=sum(v is not None and v>1 for v in g['factor_mean']),
                max_rounding_norm_error=max(g['rounding_error_norm']),
                max_neuron_cap_excess=max((v for v in g['actual_neuron_cap_excess'] if v is not None),default=None))
    pairs=[]
    for item in plan['arms']:
        if item['kind']!='uniform':continue
        control=item['arm'];selective=control.removeprefix('uniform_')
        a=np.array(curves[selective]['gate']['actual_norm']);b=np.array(curves[control]['gate']['actual_norm'])
        pairs.append(dict(selective=selective,control=control,max_norm_abs_difference=float(np.max(np.abs(a-b))),
            max_norm_relative_difference=float(np.max(np.abs(a-b)/np.maximum(a,1e-300))),
            positive_fraction=int(np.sum(b>a)),scope='first-layer weights+biases only; later layers follow their own native trajectories'))
    result=dict(status='verified',states=list(range(448,801)),plan=plan,thresholds=thresholds,curves=curves,metrics=metrics,
        targets=targets,gate_summaries=gate_summaries,norm_pairs=pairs,run_verifications=verifications,
        sources={p.relative_to(ROOT).as_posix():sha(p) for p in sources},generator_sha256=sha(Path(__file__)))
    dump(HERE/'selective_comparison.json',result)
    with (HERE/'selective_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]),lineterminator='\n');writer.writeheader();writer.writerows(metrics)
    print(json.dumps(dict(metrics=metrics,gate_summaries=gate_summaries,norm_pairs=pairs),indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
