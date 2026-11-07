"""Independent CPU integrity, parity and decomposition checks."""
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=HERE/'artifacts/spark/payload/worker'
BASE=ROOT/'runs/223-2026-11-04-multispike-interventions/artifacts/spark/payload/worker'
spec=importlib.util.spec_from_file_location('w223',ROOT/'runs/223-2026-11-04-multispike-interventions/01_worker.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
torch.set_num_threads(4)
def read(p):return json.loads(p.read_text())
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def near(a,b,rtol=1e-10,atol=1e-12):np.testing.assert_allclose(a,b,rtol=rtol,atol=atol)


def main():
    receipt=read(HERE/'retrieval.json')
    assert receipt['verified'] and w.sha(HERE/'artifacts/spark/output.tar.gz')==receipt['sha256']
    inventory=read(OUT.parent/'transfer_inventory.json')['files']
    for item in inventory:
        p=OUT.parent/item['path'];assert p.resolve().is_relative_to(OUT.parent.resolve())
        assert p.stat().st_size==item['bytes'] and w.sha(p)==item['sha256']
    sources=read(HERE/'input_manifest.json')['files']
    for item in sources:
        p=ROOT/item.get('local_path',item['path']);assert p.stat().st_size==item['bytes'] and w.sha(p)==item['sha256']
    manifest=read(OUT/'manifest.json');assert manifest['status']=='completed' and manifest['completed_state']==535
    reference=w.rows(BASE/'prefix/boundaries.jsonl')[:-1]+w.rows(BASE/'control/boundaries.jsonl')
    boundaries=w.rows(OUT/'boundaries.jsonl');assert [r['state'] for r in boundaries]==list(range(536))
    for row in boundaries:
        for key in ['loss','raw_gradient_norm']:assert row[key]==reference[row['state']][key]
    old=load(BASE/'prefix/end.pt');own=load(OUT/'snapshots/state512.pt')
    for key in ['model','optimizer','rng','previous_gradient','current_gradient']:w.equal(old[key],own[key])
    records={r['state']:r for r in w.rows(OUT/'internals.jsonl')}
    assert list(records)==list(range(448,536))
    groups=read(HERE/'config.json')['groups']
    snapshots={};arrays={}
    for state in manifest['anchors']:
        p=load(OUT/f'snapshots/state{state}.pt');a=load(OUT/f'snapshots/activations{state}.pt')
        assert p['completed_updates']==state==a['state']
        assert p['config_sha256']==w.sha(HERE/'config.json') and p['input_manifest_sha256']==w.sha(HERE/'input_manifest.json')
        assert len(a['labels'])==5000 and list(p['model'])==groups
        for key,tensor in p['model'].items():
            v=tensor.double();stats=records[state]['parameters'][key]
            near(float(v.norm()),stats['norm']);near(float(v.square().mean().sqrt()),stats['rms'])
            if v.ndim==2:
                sv=np.array(stats['singular_values']);near(float(np.sum(sv**2)),float(v.square().sum()),rtol=1e-9)
                assert np.all(np.diff(sv)<=0) and stats['spectral_norm']==sv[0]
        for key,tensor in a['arrays'].items():
            v=tensor.double();stats=records[state]['activations'][key]
            near(float(v.mean()),stats['mean']);near(float(v.square().mean().sqrt()),stats['rms'])
            near(float(v.std(unbiased=False)),stats['std'])
            if key.startswith('h'):
                assert float(v.abs().max())<=1
                # Thresholds are compared in native FP32, as in the observer;
                # casting to FP64 first changes exact threshold ties.
                near(float((tensor.abs()>.95).double().mean()),stats['saturated95'])
                near(float((tensor.abs()>.99).double().mean()),stats['saturated99'])
                near(float((1-v.square()).mean()),stats['mean_tanh_slope'])
        residual=a['arrays']['output'].double()-torch.nn.functional.one_hot(a['labels'],10).double()
        loss=float(.5*residual.square().sum(1).mean());mean=residual.mean(0)
        near(loss,records[state]['output']['loss_fp64']);assert abs(loss-reference[state]['loss'])<2e-7
        near(float(.5*mean.square().sum()),records[state]['output']['mean_residual_loss'])
        near(loss,records[state]['output']['mean_residual_loss']+records[state]['output']['centered_residual_loss'])
        snapshots[state]=p;arrays[state]=a
    counter=w.rows(OUT/'counterfactuals.jsonl');assert len(counter)==8
    maximum_allocation_error=0.
    for item in counter:
        h=item['hybrid'];values=h['loss_by_mask'];assert len(values)==64 and h['groups']==groups
        first,last=item['from_state'],item['to_state']
        assert values[0]==reference[first]['loss'] and values[-1]==reference[last]['loss']
        # Independent permutation definition, not the worker's subset weights.
        sums=np.zeros(6)
        for permutation in itertools.permutations(range(6)):
            mask=0
            for i in permutation:
                next_mask=mask|(1<<i);sums[i]+=values[next_mask]-values[mask];mask=next_mask
        near(sums/720,[h['shapley'][k] for k in groups])
        maximum_allocation_error=max(maximum_allocation_error,abs(sum(h['shapley'].values())-(values[-1]-values[0])))
        a,b=snapshots[first]['model'],snapshots[last]['model'];aa,bb=arrays[first]['arrays'],arrays[last]['arrays']
        h0,h1=aa['h2'].double(),bb['h2'].double();w0,w1=a['5.weight'].double(),b['5.weight'].double()
        b0,b1=a['5.bias'].double(),b['5.bias'].double();y=arrays[first]['labels']
        f0=h0@w0.T+b0;f1=h1@w1.T+b1;delta=f1-f0;r=f0-torch.nn.functional.one_hot(y,10).double()
        terms=[(h1-h0)@w0.T,h0@(w1-w0).T,(h1-h0)@(w1-w0).T,(b1-b0).expand(5000,-1)]
        near(sum(terms),delta)
        allocations=[float((r*t+.5*delta*t).sum(1).mean()) for t in terms]
        near(allocations,[q['loss_allocation'] for q in item['output']['terms'].values()])
        near(sum(allocations),item['output']['loss_change_fp64_head'])
        assert abs(sum(allocations)-h['total_loss_change'])<2e-7
    steps=w.rows(OUT/'steps.jsonl');assert [q['state'] for q in steps]==list(range(448,535))
    curves=w.rows(OUT/'step_curves.jsonl');assert len(curves)==4
    for item in curves:
        a,b=snapshots[item['from_state']],snapshots[item['to_state']]
        delta=w.flat(b['model'].values()).double()-w.flat(a['model'].values()).double()
        near(float(a['current_gradient'].double().dot(delta)),item['linear_prediction'])
        assert item['curve'][0]['loss']==reference[item['from_state']]['loss']
        assert item['curve'][16]['loss']==reference[item['to_state']]['loss']
        near(item['actual_loss_change'],item['linear_prediction']+item['nonlinear_remainder'])
    result=dict(status='passed',source_files=len(sources),artifact_files=len(inventory),replayed_updates=535,
        exact_loss_norm_boundaries=536,exact_state512=True,internal_states=88,anchor_snapshots=len(snapshots),
        full_training_examples=5000,hybrid_evaluations=512,step_curve_evaluations=100,
        maximum_shapley_sum_error=maximum_allocation_error,verifier_sha256=w.sha(Path(__file__)))
    (HERE/'verification_final.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
