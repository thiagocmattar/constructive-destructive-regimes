"""Independent NumPy audits and matched-window reduction of H048 continuations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from functools import lru_cache
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
RUNS={c:ROOT/f'runs/{n}-2026-11-07-{c}-representation-interventions' for n,c in [(269,'mlp'),(270,'resnet'),(271,'transformer')]}
torch.set_num_threads(4)
threadpool_limits(4)


@lru_cache(maxsize=None)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(line) for line in p.read_text().splitlines()]
def array(x):return x.detach().cpu().numpy().astype(np.float64)
def norm(x):return float(np.sqrt(np.sum(x*x)))
def dot(a,b):return float(np.sum(a*b))
def cosine(a,b):return dot(a,b)/(norm(a)*norm(b)) if norm(a)*norm(b) else None


def same_state(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same_state(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same_state(x,y)
    else:assert a==b


@lru_cache(maxsize=1)
def mlp_inputs():return torch.load(ROOT/'cache/cifar10-eos/data.pt',map_location='cpu',weights_only=False)['x_train'].numpy().reshape(5000,-1).astype(np.float64)


def check(a,b,rtol=2e-9,atol=2e-11):
    if a is None or b is None:assert a is b
    else:np.testing.assert_allclose(a,b,rtol=rtol,atol=atol)


def factor(m,arm):
    kind=arm['kind']
    if kind=='fixed':return arm['factor']
    if kind=='shock':return m['previous_ema']/m['norm'] if m['rho'] is not None and m['rho']<0 and m['norm']>m['previous_ema'] else 1.
    if kind=='secant':return min(1.,arm.get('target',2.)/m['eta_kappa_hat']) if m['eta_kappa_hat'] is not None and m['eta_kappa_hat']>arm.get('target',2.) else 1.
    raise ValueError(kind)


def audit(folder,arm,layout,boundaries,updates):
    count=0
    for p in folder.glob('audit_*.pt'):
        a=torch.load(p,map_location='cpu',weights_only=False);t=a['state'];b=boundaries[t];u=updates[t]
        g=array(a['raw_gradient']);old=array(a['previous_gradient']);before=array(a['before_parameters'])
        proposed=array(a['proposed_parameters'])-before;actual=array(a['actual_parameters'])-before
        previous_actual=before-array(a['previous_parameters'][-1])
        check(cosine(actual,previous_actual) if t!=min(updates) else None,u['actual_update_rho'])
        check(norm(g),b['raw_gradient_norm']);check(cosine(g,old),b['rho']);check(1-dot(g,old)/dot(old,old),b['eta_kappa_hat'])
        for n,span in dict(layout['groups'],**{'__global__':dict(start=0,end=len(g))}).items():
            sl=slice(span['start'],span['end']);v=g[sl];previous=old[sl];m=b['blocks'][n]
            check(norm(v),m['norm']);check(norm(previous),m['previous_norm']);check(dot(v,previous),m['dot_previous'])
            check(cosine(v,previous),m['rho']);check(1-dot(v,previous)/dot(previous,previous) if norm(previous) else None,m['eta_kappa_hat'])
        check(norm(proposed),u['proposed_norm']);check(norm(actual),u['actual_norm']);check(dot(g,actual),u['raw_gradient_dot_actual']);check(cosine(g,actual),u['raw_gradient_actual_cosine'])
        expected=before+proposed;offsets={};offset=0
        for name,shape in a['parameter_names']:
            size=int(np.prod(shape));offsets[name]=(slice(offset,offset+size),shape);offset+=size
        for n,gate in u['gates'].items():
            if 'projection_kind' in gate:
                prior=before-array(a['previous_parameters'][-1]);sl=slice(layout['groups'][n]['start'],layout['groups'][n]['end'])
                ws,shape=offsets[n+'.weight'];bs,_=offsets[n+'.bias'];x=mlp_inputs()
                z=x@proposed[ws].reshape(shape).T+proposed[bs];v=x@prior[ws].reshape(shape).T+prior[bs]
                check(dot(z.reshape(-1),v.reshape(-1))/dot(v.reshape(-1),v.reshape(-1)) if norm(v) else 0.,gate['coefficient'])
                check(float(np.sqrt(np.mean(v*v))),gate['previous_response_rms']);check(float(np.sqrt(np.mean(z*z))),gate['proposal_response_rms'])
                check(cosine(z.reshape(-1),v.reshape(-1)),gate['response_cosine'])
                candidate=proposed[sl]-gate['removed_coefficient']*prior[sl]
                if 'normalization_factor' in gate:
                    check(norm(candidate),gate['candidate_parameter_norm']);check(norm(proposed[sl]),gate['native_parameter_norm'])
                    check(gate['normalization_factor'],min(1.,norm(proposed[sl])/norm(candidate)) if norm(candidate) else 1.)
                expected[sl]=before[sl]+candidate*gate.get('normalization_factor',1.)
            elif 'factor' in gate:
                sl=slice(layout['groups'][n]['start'],layout['groups'][n]['end']);expected[sl]=before[sl]+gate['factor']*proposed[sl]
            else:
                span=layout['groups'][n];sl=slice(span['start'],span['end']);tau=gate['threshold']
                if not tau:continue
                if arm['kind']=='discard':expected[sl]=before[sl]+np.where(np.abs(proposed[sl])<=tau,proposed[sl],0.)
                elif arm['kind']=='cap':expected[sl]=before[sl]+np.clip(proposed[sl],-tau,tau)
                else:
                    scales=np.asarray(gate['factors'])
                    for name in span['names']:
                        if name.endswith('out_proj.weight') or name.endswith('out_proj.bias'):continue
                        ps,shape=offsets[name];mult=scales.reshape([-1]+[1]*(len(shape)-1))
                        expected[ps]=before[ps]+(proposed[ps].reshape(shape)*mult).reshape(-1)
        np.testing.assert_array_equal(expected.astype(np.float32),a['actual_parameters'].numpy())
        count+=1
    assert count==3,(folder,count)
    saved=torch.load(folder/'end.pt',map_location='cpu',weights_only=False)
    assert saved['state']==max(boundaries)
    assert all(int(s['step'])==saved['state'] for s in saved['optimizer']['state'].values())
    for v in saved['model'].values():assert bool(torch.isfinite(v).all())
    if arm.get('embedding_lr_multiplier',1)!=1:
        for names,group in zip(saved['param_group_names'],saved['optimizer']['param_groups']):
            if 'encoder.weight' in names:assert names==['encoder.weight'];check(group['lr'],.003*arm['embedding_lr_multiplier'])
            else:check(group['lr'],.003)
    return count


def reduce_worker(run,worker):
    cfg=read(run/'config.json');manifest=read(worker/'manifest.json');assert manifest['status']=='completed' and not manifest['mechanics_only']
    replay=read(worker/'representation_audit.json');assert replay['status']=='passed' and replay['source_sha256']==sha(HERE/'18_verify_representations.py')
    assert {(r['arm'],r['state']) for r in replay['checks']}=={(a,t) for a in manifest['arms'] for t in [cfg['fork'],cfg['fork']+32,cfg['end']]}
    qualification=read(worker/'shared/qualification.json');assert qualification['status']=='passed' and qualification['direct2_vs_resume_exact']
    if cfg.get('baseline_reference'):
        reuse=read(worker/'control_reuse.json');assert reuse['status']=='passed' and reuse['reference']==cfg['baseline_reference'] and reuse['native_updates_executed_this_run']==0
        source=ROOT/reuse['reference']['worker'];same_state(torch.load(worker/'shared/fork.pt',map_location='cpu',weights_only=False),torch.load(source/'shared/fork.pt',map_location='cpu',weights_only=False))
        for f in reuse['files']:
            left=worker/'control'/f['path'];right=source/'control'/f['path']
            assert left.stat().st_size==right.stat().st_size==f['bytes'] and sha(left)==sha(right)==f['sha256']
        assert qualification['native_two_step_fingerprint']==reuse['reference']['native_two_step_fingerprint']
    calibration=read(worker/'shared/thresholds.json');layout=read(worker/'layout.json');cal=np.load(worker/'shared/calibration.npz')
    for i,n in enumerate(calibration['module_order']):
        for kind in ['coordinate','functional']:
            for q,tau in calibration['modules'][n][kind].items():check(np.quantile(cal[f'{i}_{kind}'],float(q),method='linear'),tau)
    if (worker/'shared/functional_budget.json').exists():
        budget=read(worker/'shared/functional_budget.json')
        assert budget['native_outgoing_updates']==list(range(cfg['fork']-cfg['calibration_updates'],cfg['fork']))
        for n,v in budget['rms'].items():
            assert len(v)==cfg['calibration_updates']
            for q,tau in budget['thresholds'][n].items():check(np.quantile(v,float(q),method='linear'),tau)
        budget_replay=read(worker/'functional_budget_audit.json');assert budget_replay['status']=='passed'
        assert budget_replay['source_sha256']==sha(ROOT/'runs/278-2026-11-07-mlp-functional-step-budget/02_audit_budget.py')
    start,end=cfg['fork'],cfg['end'];states=list(range(start,end+1));curves={};metrics=[];checks=0
    for name in manifest['arms']:
        folder=worker/name;arm=next(a for a in cfg['arms'] if a['id']==name);status=read(folder/'manifest.json')
        assert status['status']=='completed' and status['completed_state']==end+1
        br=rows(folder/'boundaries.jsonl');ur=rows(folder/'updates.jsonl')
        assert [q['state'] for q in br]==list(range(start,end+2));assert [q['state'] for q in ur]==states
        b={q['state']:q for q in br};u={q['state']:q for q in ur};ema=dict(calibration['ema']);previous_factor=1.
        if arm['kind']=='loss_guard':
            probe_audit=read(worker/'probe_loss_audit.json');assert probe_audit['status']=='passed' and probe_audit['source_sha256']==sha(HERE/'28_audit_probe_loss.py')
        for t in states:
            for n,m in b[t]['blocks'].items():
                check(ema[n],m['previous_ema']);ema[n]=.9*ema[n]+.1*m['norm']
            if arm['kind']=='budget':
                gates=u[t]['gates'];factors=[g['factor'] for g in gates.values()]
                assert len(factors)==len(layout['groups']) and len(set(factors))==1 and 0<=factors[0]<=1
                detail=next(g['functional_budget'] for g in gates.values() if 'functional_budget' in g)
                expected={n:v[str(arm['quantile'])] for n,v in budget['thresholds'].items() if (arm['scope']=='representations' or n=='output') and v[str(arm['quantile'])]>0}
                assert detail['budgets']==expected
                maximum_factor=detail.get('maximum_factor',1.)
                if arm.get('recovery_multiplier'):
                    check(detail['previous_factor'],previous_factor);check(maximum_factor,min(1.,previous_factor*arm['recovery_multiplier']))
                else:check(maximum_factor,1.)
                trials=detail['trials'];assert 1<=len(trials)<=10 and trials[0]['factor']==1
                check(max(detail['native_ratios'].values(),default=0),trials[0]['max_ratio'])
                for j,trial in enumerate(trials[1:]):
                    prior=trials[j];assert prior['max_ratio']>1+1e-6 or prior['factor']>maximum_factor
                    proposed=min(maximum_factor,min(.999999,1/prior['max_ratio']) if prior['max_ratio']>1 else 1.) if j==0 else prior['factor']*.5
                    if j==8:proposed=0.
                    check(proposed,trial['factor'])
                check(trials[-1]['factor'],factors[0]);check(max(detail['actual_ratios'].values(),default=0),trials[-1]['max_ratio'])
                assert trials[-1]['max_ratio']<=1+1e-6
                assert factors[0]<=maximum_factor;previous_factor=factors[0]
            if arm['kind']=='loss_guard':
                gates=u[t]['gates'];factors=[g['factor'] for g in gates.values()]
                assert len(factors)==len(layout['groups']) and len(set(factors))==1
                detail=next(g['loss_guard'] for g in gates.values() if 'loss_guard' in g);trials=detail['trials']
                assert 1<=len(trials)<=13
                for j,trial in enumerate(trials):
                    check(trial['factor'],2.**(-j) if j<12 else 0.)
                    if j<len(trials)-1:assert trial['loss']>=detail['before_loss']
                check(factors[0],trials[-1]['factor'])
                assert trials[-1]['loss']<detail['before_loss'] or (factors[0]==0 and trials[-1]['loss']==detail['before_loss'])
            for n,gate in u[t]['gates'].items():
                if 'projection_kind' in gate:
                    beta=gate['coefficient'];kept=beta;kind=gate['projection_kind'];tau=gate['threshold'];rms=gate['previous_response_rms']
                    check(tau,calibration['modules'][n]['functional'][str(arm['quantile'])])
                    if beta<0:kept=0. if kind=='remove' else beta/2 if kind=='half' else max(beta,-tau/rms)
                    check(kept,gate['retained_coefficient']);check(beta-kept,gate['removed_coefficient']);check(float(beta!=kept),gate['affected_fraction'])
                    if rms:check(beta,gate['response_cosine']*gate['proposal_response_rms']/rms)
                    if arm.get('norm_cap_native'):
                        assert gate['actual_norm']<=gate['proposed_norm']+gate['rounding_norm_bound']+1e-10
                        check(gate['normalization_factor'],min(1.,gate['native_parameter_norm']/gate['candidate_parameter_norm']) if gate['candidate_parameter_norm'] else 1.)
                elif 'factor' in gate:
                    if arm['kind'] in ['budget','loss_guard']:check(float(gate['factor']<1),gate['affected_fraction'])
                    elif arm['kind']=='paired_norm':
                        target=read(run/'norm_schedule.json')['states'][str(t)][n];check(target,gate['target_norm'])
                        check(gate['factor'],target/gate['proposed_norm'] if gate['proposed_norm'] else 0.)
                        assert abs(gate['actual_norm']-target)<=gate['rounding_norm_bound']+1e-10
                    else:check(gate['factor'],factor(b[t]['blocks']['__global__' if arm.get('scope')=='global' else n],arm))
                else:
                    key='functional' if arm['kind']=='functional' else 'coordinate'
                    check(gate['threshold'],calibration['modules'][n][key][str(arm['quantile'])])
                    if arm['kind']=='functional' and gate['threshold']:
                        scores=np.array(gate['proposed_scores']);actual=np.array(gate['actual_scores']);tau=gate['threshold']
                        check(np.minimum(1.,tau/np.maximum(scores,np.finfo(float).tiny)),gate['factors'])
                        assert np.all(actual<=tau+np.array(gate['rounding_bound'])+1e-10)
                        check(np.mean(scores>tau),gate['affected_fraction'])
            assert u[t]['all_optimizer_steps']==t+1
        checks+=audit(folder,arm,layout,b,u)
        curve={key:[b[t][key] for t in states] for key in ['loss','probe_loss','rho','eta_kappa_hat','raw_gradient_norm']}
        curve.update({key:[u[t][key] for t in states] for key in ['proposed_norm','actual_norm','retained_norm_ratio','actual_update_rho','raw_gradient_actual_cosine','embedding_lr']})
        curve['representations']={n:{key:[b[t+1]['representations'][n][key] for t in states] for key in ['rho','change_rms','activation_rms']} for n in br[0]['representations']}
        probe_units=replay['checks'][0]['representations']['output']['shape'][0]
        # An intact original normalization group makes the ResNet subtraction valid.
        total_units=status['training_units'];assert probe_units<=total_units
        curve['outside_probe_loss']=None if probe_units==total_units else ((total_units*np.asarray(curve['loss'])-probe_units*np.asarray(curve['probe_loss']))/(total_units-probe_units)).tolist()
        # This fraction describes parameter-owning blocks, not a fraction of coordinates.
        curve['fraction_blocks_damped']=[sum(g.get('factor',1)<1 for g in u[t]['gates'].values())/len(layout['groups']) for t in states]
        curve['mean_target_fraction_modified']=[float(np.mean([g['affected_fraction'] for g in u[t]['gates'].values()])) if u[t]['gates'] else 0. for t in states]
        loss=np.array(curve['loss']);assert np.isfinite(loss).all()
        metric=dict(case=cfg['case'],worker=worker.parent.name,arm=name,begin=start,end=end,mean_loss=float(loss.mean()),
            tail32_mean=float(loss[-32:].mean()),final_loss=float(loss[-1]),max_loss=float(loss.max()),peak_state=states[int(loss.argmax())],
            positive_variation=float(np.maximum(np.diff(loss),0).sum()),late_half_mean=float(loss[len(loss)//2:].mean()),
            mean_retained_norm=float(np.mean(curve['retained_norm_ratio'])),seconds=status['elapsed_seconds'],
            output_reversals=int(np.sum(np.array(curve['representations']['output']['rho'],dtype=float)<0)),
            probe_loss_units=probe_units,full_loss_units=total_units,mean_probe_loss=float(np.mean(curve['probe_loss'])),
            mean_outside_probe_loss=None if curve['outside_probe_loss'] is None else float(np.mean(curve['outside_probe_loss'])))
        if (folder/'evaluation.json').exists():metric['evaluation']=read(folder/'evaluation.json')
        metrics.append(metric);curves[name]=curve
    baseline=metrics[0];assert baseline['arm']=='control'
    native=torch.load(worker/'control'/f"audit_{cfg['fork']}.pt",map_location='cpu',weights_only=False)
    for m in metrics:
        for k in ['mean_loss','tail32_mean','max_loss','positive_variation','late_half_mean','final_loss']:
            m[k+'_pct']=100*(m[k]/baseline[k]-1) if baseline[k] else None
        check(curves[m['arm']]['loss'][0],curves['control']['loss'][0],rtol=0,atol=0)
        first=torch.load(worker/m['arm']/f"audit_{cfg['fork']}.pt",map_location='cpu',weights_only=False)
        for k in ['before_parameters','raw_gradient','previous_gradient']:assert torch.equal(first[k],native[k])
        arm=next(a for a in cfg['arms'] if a['id']==m['arm'])
        if arm.get('embedding_lr_multiplier',1)==1:assert torch.equal(first['proposed_parameters'],native['proposed_parameters'])
        else:
            # The first native proposal may differ only on the embedding coordinates.
            span=layout['groups']['encoder']
            assert span['names']==['encoder.weight']
            for sl in [slice(0,span['start']),slice(span['end'],None)]:
                assert torch.equal(first['proposed_parameters'][sl],native['proposed_parameters'][sl])
    return dict(case=cfg['case'],seed=cfg['seed'],run_id=cfg['run_id'],environment=read(worker/'control/manifest.json')['environment'],config_sha256=sha(run/'config.json'),worker=worker.relative_to(ROOT).as_posix(),states=states,curves=curves,metrics=metrics,
        qualification=qualification,audit_states_checked=checks,layout=layout,source_manifest_sha256=sha(worker/'manifest.json'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=list(RUNS));parser.add_argument('--run',type=Path);args=parser.parse_args();results=[]
    selected={read(args.run/'config.json')['case']:args.run.resolve()} if args.run else RUNS
    for case,run in selected.items():
        if args.case and case!=args.case:continue
        for item in read(run/'input_manifest.json')['files']:
            path=ROOT/item.get('local_path',item['path']);assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],str(path)
        for receipt in run.glob('*retrieval.json'):
            rec=read(receipt)
            if not rec.get('verified'):continue
            root=ROOT/rec['local'];inventory=read(root/'transfer_inventory.json')
            for f in inventory['files']:
                p=root/f['path'];assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256']
        for p in (run/'artifacts').rglob('manifest.json'):
            m=read(p)
            if 'results' in m and m.get('status')=='completed' and not m.get('mechanics_only',True):results.append(reduce_worker(run,p.parent))
    assert results
    out=dict(status='verified',generator_sha256=sha(Path(__file__)),workers=results)
    suffix='_run'+str(read(args.run/'config.json')['run_id']) if args.run else '_'+args.case if args.case else ''
    (HERE/('transfer_comparison'+suffix+'.json')).write_text(json.dumps(out,indent=2)+'\n')
    metrics=[{k:v for k,v in m.items() if k!='evaluation'} for r in results for m in r['metrics']]
    with (HERE/('transfer_metrics'+suffix+'.csv')).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    print(json.dumps(dict(audit_states=sum(r['audit_states_checked'] for r in results),metrics=metrics),indent=2))


if __name__=='__main__':main()
