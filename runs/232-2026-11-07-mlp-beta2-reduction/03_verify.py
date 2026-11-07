"""Independent local checkpoint, first-update mathematics and artifact audit."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT/'runs/223-2026-11-04-multispike-interventions/artifacts/spark/payload/worker'
NEW = HERE/'artifacts/spark/payload/worker'


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def rows(p):
    return [json.loads(s) for s in p.read_text().splitlines()]


def load(p):
    return torch.load(p, map_location='cpu', weights_only=False)


def eq(a,b):
    if isinstance(a,torch.Tensor):
        assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):
        assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:
            eq(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):
            eq(x,y)
    else:
        assert a==b


def flat(xs):
    return torch.cat([x.reshape(-1).double() for x in xs])


def main():
    receipt=json.loads((HERE/'retrieval.json').read_text())
    assert receipt['verified']
    assert sha(HERE/'artifacts/spark/output.tar.gz')==receipt['sha256']
    inventory=json.loads((NEW.parent/'transfer_inventory.json').read_text())['files']
    for item in inventory:
        p=NEW.parent/item['path']
        assert p.resolve().is_relative_to(NEW.parent.resolve())
        assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
    inputs=json.loads((HERE/'input_manifest.json').read_text())['files']
    for item in inputs:
        p=ROOT/item.get('local_path',item['path'])
        assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
    reference=load(BASE/'qualification/direct/end.pt')
    for folder in ['qualification2','instrumentation2']:
        q=load(NEW/folder/'end.pt')
        for k in ['model','optimizer','previous_gradient','rng','current_gradient']:
            eq(reference[k],q[k])
    parent=load(BASE/'prefix/end.pt')
    assert parent['completed_updates']==512
    manifest=json.loads((NEW/'manifest.json').read_text())
    assert manifest['status']=='completed'
    control=rows(BASE/'control/boundaries.jsonl')
    report={}
    for name,result in manifest['results'].items():
        beta=result['beta2']
        path=NEW/name
        b,m=rows(path/'boundaries.jsonl'),rows(path/'metrics.jsonl')
        assert [x['state'] for x in b]==list(range(512,801))
        assert [x['state'] for x in m]==list(range(512,800))
        assert [x['update'] for x in m]==list(range(513,801))
        assert result['status']=='completed' and result['updates']==288
        assert result['evaluation']['test']['loss_units']==10000
        for key in ['loss','raw_gradient_norm','rho','parameter_norm']:
            assert b[0][key]==control[0][key],(name,key)
        # New observer sums squared entries; the parent uses a dot-product reduction.
        np.testing.assert_allclose(b[0]['eta_kappa_hat'],control[0]['eta_kappa_hat'],rtol=1e-13,atol=1e-13)
        for index,r in enumerate(m):
            assert r['beta2']==beta and r['beta1']==.9 and r['applied_lr']==.001
            assert r['raw_gradient_norm']==r['optimizer_input_norm']==b[index]['raw_gradient_norm']
            assert r['bias_correction2']==1-beta**r['update']
            assert r['baseline_bias_correction2']==1-.999**r['update']
            np.testing.assert_allclose(r['moment_over_gradient_norm'],r['moment_norm']/r['raw_gradient_norm'],rtol=1e-13)
            for key in ['moment_rho','adaptive_rho','gradient_current_moment_cosine','gradient_adaptive_cosine']:
                assert r[key] is None or -1<=r[key]<=1
        for previous,current in zip(b,b[1:]):
            np.testing.assert_allclose(current['eta_kappa_hat'],1-current['rho']*current['raw_gradient_norm']/previous['raw_gradient_norm'],rtol=1e-11,atol=1e-11)
        first=load(path/'first_step.pt')
        last=load(path/'end.pt')
        assert first['completed_updates']==513 and last['completed_updates']==800
        assert last['config_sha256']==sha(HERE/'config.json')
        assert last['input_manifest_sha256']==sha(HERE/'input_manifest.json')
        assert last['parent_sha256']==sha(BASE/'prefix/end.pt')
        for checkpoint in [first,last]:
            group=checkpoint['optimizer']['param_groups'][0]
            assert group['betas']==(.9,beta) and group['lr']==.001 and group['eps']==1e-8
            assert group['weight_decay']==0 and not group['foreach'] and not group['fused']
            assert {int(v['step']) for v in checkpoint['optimizer']['state'].values()}=={checkpoint['completed_updates']}
        # Independent FP64 recurrence from the saved warm moments and exact fork gradient.
        offset=0
        error_m=error_v=error_parameter=0.
        reconstructed_d=[]
        for pid,old in parent['optimizer']['state'].items():
            actual=first['optimizer']['state'][pid]
            n=old['exp_avg'].numel()
            g=parent['current_gradient'][offset:offset+n].double().reshape(old['exp_avg'].shape)
            offset+=n
            mm=.9*old['exp_avg'].double()+.1*g
            vv=beta*old['exp_avg_sq'].double()+(1-beta)*g.square()
            np.testing.assert_allclose(actual['exp_avg'].numpy(),mm.numpy(),rtol=3e-6,atol=2e-9)
            np.testing.assert_allclose(actual['exp_avg_sq'].numpy(),vv.numpy(),rtol=3e-6,atol=1e-12)
            error_m=max(error_m,float((actual['exp_avg'].double()-mm).abs().max()))
            error_v=max(error_v,float((actual['exp_avg_sq'].double()-vv).abs().max()))
            # Use stored post-update moments to separate recurrence rounding from update rounding.
            direction=(actual['exp_avg'].double()/(1-.9**513))/((actual['exp_avg_sq'].double()/(1-beta**513)).sqrt()+1e-8)
            reconstructed_d.append(direction)
        d=flat(reconstructed_d)
        before,after=flat(parent['model'].values()),flat(first['model'].values())
        error_parameter=float((after-(before-.001*d)).abs().max())
        assert error_parameter<2e-7,error_parameter
        g=parent['current_gradient'].double()
        cosine=float(g.dot(d)/(g.norm()*d.norm()))
        np.testing.assert_allclose(cosine,m[0]['gradient_adaptive_cosine'],rtol=1e-11,atol=1e-12)
        np.testing.assert_allclose(float(d.norm()),m[0]['adaptive_norm'],rtol=1e-11)
        np.testing.assert_allclose(float((after-before).norm()),m[0]['actual_increment_norm'],rtol=1e-11)
        np.testing.assert_allclose(float(last['current_gradient'].double().norm()),b[-1]['raw_gradient_norm'],rtol=1e-11)
        np.testing.assert_allclose(float(last['previous_moment'].norm()),m[-1]['moment_norm'],rtol=1e-11)
        np.testing.assert_allclose(float(last['previous_direction'].norm()),m[-1]['adaptive_norm'],rtol=1e-11)
        report[name]=dict(updates=len(m),boundaries=len(b),first_moment_max_abs_error=error_m,
            second_moment_max_abs_error=error_v,parameter_step_max_abs_rounding=error_parameter,
            first_adaptive_cosine=cosine,final_loss=b[-1]['loss'],
            fork_projected_change_roundoff=b[0]['eta_kappa_hat']-control[0]['eta_kappa_hat'],
            first_bias_correction2=m[0]['bias_correction2'],old_bias_correction2=m[0]['baseline_bias_correction2'])
    result=dict(status='passed',source_files=len(inputs),artifact_files=len(inventory),
        exact_control_replay=True,exact_added_instrumentation_replay=True,
        first_step_recurrence_and_direction_audit=True,arms=report,
        verifier_sha256=sha(Path(__file__)))
    (HERE/'verification_final.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
