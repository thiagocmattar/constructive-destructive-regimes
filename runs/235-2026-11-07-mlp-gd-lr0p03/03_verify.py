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
    for a,b in [(BASE/'qualification/direct/end.pt',NEW/'qualification2/end.pt'),
                (NEW/'gd_lr0p03_qualification/direct2/end.pt',NEW/'gd_lr0p03_qualification/resumed/end.pt')]:
        x,y=load(a),load(b)
        for key in ['model','optimizer','rng','current_gradient','previous_gradient']:
            w.equal(x[key],y[key])
    folder=NEW/'gd_lr0p03'
    boundaries,metrics=w.rows(folder/'boundaries.jsonl'),w.rows(folder/'metrics.jsonl')
    control=w.rows(BASE/'control/boundaries.jsonl')
    assert [r['state'] for r in boundaries]==list(range(512,801))
    assert [r['state'] for r in metrics]==list(range(512,800))
    for key in ['loss','raw_gradient_norm','rho','eta_kappa_hat','parameter_norm']:
        assert boundaries[0][key]==control[0][key]
    for before,after in zip(boundaries,boundaries[1:]):
        np.testing.assert_allclose(after['eta_kappa_hat'],1-after['rho']*after['raw_gradient_norm']/before['raw_gradient_norm'],rtol=1e-10,atol=1e-12)
    for r in metrics:
        assert r['optimizer']=='GD' and r['applied_lr']==.03 and r['momentum']==r['weight_decay']==0
        assert r['raw_gradient_norm']==r['optimizer_input_norm']
        assert r['gradient_dot_actual_step']<0
        assert r['gd_step_rounding_residual_norm']<2e-6
        # Tiny GD updates are rounded in FP32 parameter space; bound their geometry
        # using the recorded vector residual rather than assuming exact -1 cosine.
        gn=r['raw_gradient_norm']
        residual=r['gd_step_rounding_residual_norm']
        assert abs(r['actual_increment_norm']-.03*gn)<=residual+1e-12
        assert abs(r['gradient_dot_actual_step']+.03*gn*gn)<=gn*residual+1e-12
        np.testing.assert_allclose(r['raw_gradient_actual_step_cosine'],
            r['gradient_dot_actual_step']/(gn*r['actual_increment_norm']),rtol=1e-12)
    parent=load(BASE/'prefix/end.pt')
    first,last=load(folder/'first_step.pt'),load(folder/'end.pt')
    for p,state in [(first,513),(last,800)]:
        assert p['completed_updates']==state and not p['optimizer']['state']
        g=p['optimizer']['param_groups'][0]
        assert g['lr']==.03 and g['momentum']==g['weight_decay']==0 and not g['nesterov']
        assert p['config_sha256']==w.sha(HERE/'config.json')
        assert p['input_manifest_sha256']==w.sha(HERE/'input_manifest.json')
        assert p['parent_sha256']==w.sha(BASE/'prefix/end.pt')
    before=w.flat(parent['model'].values()).double()
    after=w.flat(first['model'].values()).double()
    gradient=parent['current_gradient'].double()
    err=float((after-before+.03*gradient).abs().max())
    assert err<2e-8,err
    np.testing.assert_allclose(float((after-before).norm()),metrics[0]['actual_increment_norm'],rtol=1e-11)
    np.testing.assert_allclose(float(gradient.dot(after-before)),metrics[0]['gradient_dot_actual_step'],rtol=1e-11)
    np.testing.assert_allclose(float(last['current_gradient'].double().norm()),boundaries[-1]['raw_gradient_norm'],rtol=1e-11)
    evaluation=json.loads((folder/'status.json').read_text())['evaluation']['test']
    assert evaluation['loss_units']==evaluation['objective_units']==10000
    result=dict(status='passed',source_files=len(inputs),artifact_files=len(inventory),updates=288,boundaries=289,
        exact_control_replay=True,exact_gd_resume=True,gd_first_step_max_abs_rounding=err,
        all_updates_locally_descending=True,max_step_rounding_residual=max(r['gd_step_rounding_residual_norm'] for r in metrics),
        maximum_actual_step_cosine=max(r['raw_gradient_actual_step_cosine'] for r in metrics),
        maximum_relative_rounding_residual=max(r['gd_step_rounding_residual_norm']/(.03*r['raw_gradient_norm']) for r in metrics),
        final_loss=boundaries[-1]['loss'],verifier_sha256=w.sha(Path(__file__)))
    (HERE/'verification_final.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
