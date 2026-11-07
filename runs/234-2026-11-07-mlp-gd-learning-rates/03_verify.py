"""Independent CPU checks of GD placement, retained control and exact resume."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=ROOT/'runs/223-2026-11-04-multispike-interventions/artifacts/spark/payload/worker'
NEW=HERE/'artifacts/spark/payload/worker'
spec=importlib.util.spec_from_file_location('frozen223',ROOT/'runs/223-2026-11-04-multispike-interventions/01_worker.py')
w=importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def load(p):
    return torch.load(p,map_location='cpu',weights_only=False)


def main():
    receipt=json.loads((HERE/'retrieval.json').read_text())
    assert receipt['verified'] and w.sha(HERE/'artifacts/spark/output.tar.gz')==receipt['sha256']
    inventory=json.loads((NEW.parent/'transfer_inventory.json').read_text())['files']
    for item in inventory:
        p=NEW.parent/item['path']
        assert p.resolve().is_relative_to(NEW.parent.resolve())
        assert p.stat().st_size==item['bytes'] and w.sha(p)==item['sha256']
    inputs=json.loads((HERE/'input_manifest.json').read_text())['files']
    for item in inputs:
        p=ROOT/item.get('local_path',item['path'])
        assert p.stat().st_size==item['bytes'] and w.sha(p)==item['sha256']
    x,y=load(BASE/'qualification/direct/end.pt'),load(NEW/'qualification2/end.pt')
    for key in ['model','optimizer','rng','current_gradient','previous_gradient']:
        w.equal(x[key],y[key])
    results={}
    for lr in [.04,.08]:
        name='gd_lr'+str(lr).replace('.','p')
        q=NEW/(name+'_qualification')
        x,y=load(q/'direct2/end.pt'),load(q/'resumed/end.pt')
        for key in ['model','optimizer','rng','current_gradient','previous_gradient']:
            w.equal(x[key],y[key])
        results[name]=verify_branch(lr)
    result=dict(status='passed',source_files=len(inputs),artifact_files=len(inventory),
        exact_control_replay=True,results=results,verifier_sha256=w.sha(Path(__file__)))
    (HERE/'verification_final.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))


def verify_branch(lr):
    folder=NEW/('gd_lr'+str(lr).replace('.','p'))
    boundaries,metrics=w.rows(folder/'boundaries.jsonl'),w.rows(folder/'metrics.jsonl')
    control=w.rows(BASE/'control/boundaries.jsonl')
    status=json.loads((folder/'status.json').read_text())
    complete=lr==.04
    end=800 if complete else 547
    assert status['status']==('completed' if complete else 'failed')
    assert status['completed_state']==end and status['updates']==end-512
    assert [r['state'] for r in boundaries]==list(range(512,801 if complete else 547))
    assert [r['state'] for r in metrics]==list(range(512,end))
    for key in ['loss','raw_gradient_norm','rho','eta_kappa_hat','parameter_norm']:
        assert boundaries[0][key]==control[0][key]
    for before,after in zip(boundaries,boundaries[1:]):
        np.testing.assert_allclose(after['eta_kappa_hat'],1-after['rho']*after['raw_gradient_norm']/before['raw_gradient_norm'],rtol=1e-10,atol=1e-12)
    for boundary,r in zip(boundaries,metrics):
        assert r['optimizer']=='GD' and r['applied_lr']==lr and r['momentum']==r['weight_decay']==0
        assert r['raw_gradient_norm']==r['optimizer_input_norm']
        assert r['gradient_dot_actual_step']<0
        # Absolute rounding grows with the exploding .08 trajectory. Use a
        # scale-aware FP32 bound; do not mistake overflow for a GD-rule error.
        assert r['gd_step_rounding_residual_norm']<=4*np.finfo(np.float32).eps*(
            boundary['parameter_norm']+lr*r['raw_gradient_norm'])
        # Tiny GD updates are rounded in FP32 parameter space; bound their geometry
        # using the recorded vector residual rather than assuming exact -1 cosine.
        gn=r['raw_gradient_norm']
        residual=r['gd_step_rounding_residual_norm']
        assert abs(r['actual_increment_norm']-lr*gn)<=residual+1e-12
        assert abs(r['gradient_dot_actual_step']+lr*gn*gn)<=gn*residual+1e-12
        np.testing.assert_allclose(r['raw_gradient_actual_step_cosine'],
            r['gradient_dot_actual_step']/(gn*r['actual_increment_norm']),rtol=1e-12)
    parent=load(BASE/'prefix/end.pt')
    first=load(folder/'first_step.pt')
    last=load(folder/('end.pt' if complete else 'recovery.pt'))
    for p,state in [(first,513),(last,800 if complete else 544)]:
        assert p['completed_updates']==state and not p['optimizer']['state']
        g=p['optimizer']['param_groups'][0]
        assert g['lr']==lr and g['momentum']==g['weight_decay']==0 and not g['nesterov']
        assert p['config_sha256']==w.sha(HERE/'config.json')
        assert p['input_manifest_sha256']==w.sha(HERE/'input_manifest.json')
        assert p['parent_sha256']==w.sha(BASE/'prefix/end.pt')
    before=w.flat(parent['model'].values()).double()
    after=w.flat(first['model'].values()).double()
    gradient=parent['current_gradient'].double()
    err=float((after-before+lr*gradient).abs().max())
    assert err<2e-8,err
    np.testing.assert_allclose(float((after-before).norm()),metrics[0]['actual_increment_norm'],rtol=1e-11)
    np.testing.assert_allclose(float(gradient.dot(after-before)),metrics[0]['gradient_dot_actual_step'],rtol=1e-11)
    if complete:
        np.testing.assert_allclose(float(last['current_gradient'].double().norm()),boundaries[-1]['raw_gradient_norm'],rtol=1e-11)
        evaluation=status['evaluation']['test']
        assert evaluation['loss_units']==evaluation['objective_units']==10000
    else:
        assert last['current_gradient'] is None and not (folder/'end.pt').exists()
        np.testing.assert_allclose(float(last['previous_gradient'].double().norm()),boundaries[543-512]['raw_gradient_norm'],rtol=1e-11)
        audit=json.loads((HERE/'failure_audit.json').read_text())
        assert audit['state']==547 and audit['status']=='verified_nonfinite_boundary' and audit['loss']=='inf'
        assert audit['checkpoint_sha256']==w.sha(folder/'recovery.pt')
        assert audit['audit_script_sha256']==w.sha(HERE/'04_audit_failure.py')
        assert audit['all_parameters_finite'] and audit['gradient_nan_count']==audit['gradient_inf_count']==0
    result=dict(status='passed',lr=lr,execution_status=status['status'],updates=len(metrics),boundaries=len(boundaries),
        exact_control_replay=True,exact_gd_resume=True,gd_first_step_max_abs_rounding=err,
        all_updates_locally_descending=True,max_step_rounding_residual=max(r['gd_step_rounding_residual_norm'] for r in metrics),
        maximum_actual_step_cosine=max(r['raw_gradient_actual_step_cosine'] for r in metrics),
        maximum_relative_rounding_residual=max(r['gd_step_rounding_residual_norm']/(lr*r['raw_gradient_norm']) for r in metrics),
        final_loss=boundaries[-1]['loss'] if complete else None,
        last_finite_loss=boundaries[-1]['loss'],last_finite_state=boundaries[-1]['state'],
        failed_state=None if complete else 547,verifier_sha256=w.sha(Path(__file__)))
    return result


if __name__=='__main__':
    main()
