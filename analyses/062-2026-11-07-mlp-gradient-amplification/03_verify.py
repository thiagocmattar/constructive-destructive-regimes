"""Independent reductions from raw retained arrays and exact control checkpoints."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT/'runs/237-2026-11-07-mlp-gradient-amplification'
RAW = RUN/'artifacts/spark/payload/worker'
OLD = ROOT/'runs/236-2026-11-07-mlp-adam-spike-internals/artifacts/spark/payload/worker'
BASE = ROOT/'runs/223-2026-11-04-multispike-interventions/artifacts/spark/payload/worker'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def equal(a,b):
    if isinstance(a,torch.Tensor): assert torch.equal(a,b)
    elif isinstance(a,np.ndarray): assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal(x,y)
    else: assert a==b


def near(a,b,rtol=2e-6,atol=1e-10):
    assert np.allclose(a,b,rtol=rtol,atol=atol),(a,b)


def main():
    torch.set_num_threads(4)
    inputs=json.loads((RUN/'input_manifest.json').read_text())['files']
    for item in inputs:
        path=ROOT/item.get('local_path',item['path'])
        assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
    files=json.loads((RAW.parent/'transfer_inventory.json').read_text())['files']
    for item in files:
        path=RAW.parent/item['path']
        assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
    reference=rows(BASE/'prefix/boundaries.jsonl')[:-1]+rows(BASE/'control/boundaries.jsonl')
    boundaries=rows(RAW/'boundaries.jsonl')
    assert len(boundaries)==536
    for index,row in enumerate(boundaries):
        assert row['state']==index
        for k in ['loss','raw_gradient_norm']: assert row[k]==reference[index][k]
    final=torch.load(RAW/'end.pt',weights_only=False)
    old=torch.load(OLD/'snapshots/state535.pt',weights_only=False)
    for key in ['model','optimizer','rng','previous_gradient','current_gradient']: equal(final[key],old[key])
    backward={r['state']:r for r in rows(RAW/'backward.jsonl')}
    adam={r['state']:r for r in rows(RAW/'adam.jsonl')}
    directional={r['state']:r for r in rows(RAW/'directional.jsonl')}
    assert list(backward)==list(range(448,536))
    assert list(adam)==list(directional)==list(range(448,535))
    vectors=torch.load(RAW/'output_vectors.pt',weights_only=False)
    residuals=vectors['residuals'].double()
    for i,state in enumerate(vectors['states']):
        row=backward[state]
        near(.5*float(residuals[i].square().sum())/5000,row['loss']['mean'],rtol=1e-12)
        near(row['gradient_coherence']*row['unit_gradient_mean_squared_norm'],row['gradient_norm']**2,rtol=1e-12)
        near(sum(g['norm']**2 for g in row['gradient_groups'].values()),row['gradient_norm']**2,rtol=1e-12)
        if state==535: continue
        t=vectors['tangents'][i].double();tt=vectors['second_tangents'][i].double();dr=directional[state]
        near(float((residuals[i]*t).sum())/5000,dr['tangent_linear'],rtol=1e-12)
        near(.5*float(t.square().sum())/5000,dr['gauss_newton_quadratic'],rtol=1e-12)
        near(.5*float((residuals[i]*tt).sum())/5000,dr['residual_curvature_quadratic'],rtol=1e-12)
        assert dr['actual_loss_change']==boundaries[state+1]['loss']-boundaries[state]['loss']
        ar=adam[state]
        near(sum(g['step_norm']**2 for g in ar['groups'].values()),ar['actual_step_norm']**2,rtol=1e-12)
        assert ar['minimum_denominator_ratio']>=ar['denominator_ratio_lower_bound']-1e-7
    x=torch.load(ROOT/'cache/cifar10-eos/data.pt',weights_only=True)['x_train'].double().flatten(1)
    anchors=json.loads((RUN/'config.json').read_text())['anchors']
    energy_error=0.
    for state in anchors:
        q=torch.load(RAW/f'backward{state}.pt',weights_only=False)
        acts=torch.load(OLD/f'snapshots/activations{state}.pt',weights_only=False)['arrays']
        cp=torch.load(OLD/f'snapshots/state{state}.pt',weights_only=False)
        offset=0
        for name,param in cp['model'].items():
            layer={'1':(q['dz1'],x),'3':(q['dz2'],acts['h1']),'5':(q['residual'],acts['h2'])}[name[0]]
            dy,features=[v.double() for v in layer]
            example_energy=dy.square().sum(1)
            if name.endswith('weight'): example_energy=example_energy*features.square().sum(1)
            energy=float(example_energy.mean());record=backward[state]['gradient_groups'][name]
            near(energy,record['unit_mean_squared_norm'],rtol=1e-12)
            energy_error=max(energy_error,abs(energy/record['unit_mean_squared_norm']-1))
            g=cp['current_gradient'][offset:offset+param.numel()].double();offset+=param.numel()
            near(float(g.norm()),record['norm'],rtol=1e-12)
        # Independently check the actual increment between consecutive retained states.
        if state in [473,492,502,521]:
            nxt=torch.load(OLD/f'snapshots/state{state+1}.pt',weights_only=False)
            delta=torch.cat([(nxt['model'][k].double()-p.double()).flatten() for k,p in cp['model'].items()])
            near(float(delta.norm()),adam[state]['actual_step_norm'],rtol=1e-12)
            near(float(delta@cp['current_gradient'].double()),directional[state]['linear'],rtol=1e-12)
    gpu=json.loads((HERE/'cross_jacobian_probe.json').read_text())
    cpu=json.loads((HERE/'cross_jacobian_cpu.json').read_text())
    cross_error=0.
    assert gpu['generator_sha256']==sha(HERE/'02_cross_jacobian_probe.py')
    for ge,ce in zip(gpu['events'],cpu['events']):
        for g,c in zip(ge['crosses'],ce['crosses']):
            for k in ['step_norm','mean_tangent_energy','directional_gain']:
                near(g[k],c[k],rtol=1e-5)
                cross_error=max(cross_error,abs(g[k]/c[k]-1))
            if g['jacobian_state']==g['direction_state']:
                near(g['directional_gain'],directional[g['direction_state']]['directional_parameter_gain'],rtol=1e-9)
    result=dict(status='passed',input_hashes=len(inputs),artifact_hashes=len(files),exact_loss_gradient_pairs=len(boundaries),
        exact_final_state=535,remote_exact_anchors=json.loads((RAW/'qualification.json').read_text())['exact_anchor_states'],
        independent_per_example_energy_anchors=len(anchors),independent_per_example_energy_max_relative_error=energy_error,
        gradient_coherence_identities=88,full_example_directional_reductions=87,independent_actual_increments=4,
        cpu_gpu_cross_jacobian_evaluations=10,cpu_gpu_max_relative_difference=cross_error,
        generator_sha256=sha(Path(__file__)))
    (HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result))


if __name__=='__main__': main()
