"""Independently verify transferred checkpoints and the LR-only optimizer change."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];RAW=HERE/'artifacts/spark/payload/worker'
PARENT=ROOT/'runs/244-2026-11-07-mlp-freeze-first-linear/artifacts/spark/payload/worker'


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


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


def main():
    inputs=json.loads((HERE/'input_manifest.json').read_text())['files']
    for item in inputs:
        p=ROOT/item.get('local_path',item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],str(p)
    files=json.loads((HERE/'artifacts/spark/payload/transfer_inventory.json').read_text())['files']
    for item in files:
        p=HERE/'artifacts/spark/payload'/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],str(p)
    def load(name):return torch.load(RAW/name,map_location='cpu',weights_only=False)
    fork=load('fork500.pt');start=load('intervention_start500.pt');end=load('end.pt');qualified=load('qualification501.pt')
    assert fork['completed_updates']==start['completed_updates']==500 and end['completed_updates']==801
    for k in ['model','rng','previous_gradient','current_gradient']:equal(fork[k],start[k])
    equal(fork['optimizer']['state'],start['optimizer']['state'])
    for a,b in zip(fork['optimizer']['param_groups'],start['optimizer']['param_groups']):
        assert a['lr']==.001 and b['lr']==0.005
        equal({k:v for k,v in a.items() if k!='lr'},{k:v for k,v in b.items() if k!='lr'})
    equal(start['optimizer']['param_groups'],end['optimizer']['param_groups'])
    origin=torch.load(PARENT/'fork.pt',map_location='cpu',weights_only=False)
    for n in ['1.weight','1.bias']:
        equal(origin['model'][n],fork['model'][n]);equal(fork['model'][n],end['model'][n])
    active=[n for n in fork['model'] if not n.startswith('1.')]
    assert all(not torch.equal(fork['model'][n],end['model'][n]) for n in active)
    assert all(int(s['step'])==500 for s in start['optimizer']['state'].values())
    assert all(int(s['step'])==801 for s in end['optimizer']['state'].values())
    assert all(torch.isfinite(p).all() for p in end['model'].values())
    original_g=np.load(PARENT/'raw_gradient.npy',mmap_mode='r')
    np.testing.assert_array_equal(fork['current_gradient'].numpy(),original_g[500-447])
    np.testing.assert_array_equal(qualified['current_gradient'].numpy(),original_g[501-447])
    q=json.loads((RAW/'qualification.json').read_text());assert q['status']=='passed'
    assert q['exact_boundary_states']==q['exact_raw_gradient_states']==list(range(448,502))
    assert q['exact_representation_states']==[499,500,501] and q['first_changed_step_moments_and_clocks_exact']
    result=dict(status='passed',source_files=len(inputs),retrieved_files=len(files),exact_parent_boundary_and_gradient_states=[448,501],
        complete_start500_differs_only_in_lr=True,starting_moments_and_clocks_unchanged=True,first_layer_fixed_from448=True,
        remaining_four_parameter_tensors_changed=True,final_lr=0.005,all_final_clocks=801,
        first_changed_step_moments_match_control=True,first_step_scaled_increment_max_error=q['first_changed_step_scaled_increment_max_abs_error'],first_step_lr_ratio=q['first_step_lr_ratio'],
        final_parameters_finite=True,checkpoints={n:sha(RAW/n) for n in ['fork500.pt','intervention_start500.pt','qualification501.pt','end.pt']},

        generator_sha256=sha(Path(__file__)))
    (HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n',newline='\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
