"""Post-retrieval checks of saved state, immutable inputs and control qualification."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
RAW=HERE/'artifacts/spark/payload/worker'


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
        p=ROOT/item.get('local_path',item['path'])
        assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],str(p)
    inventory=json.loads((HERE/'artifacts/spark/payload/transfer_inventory.json').read_text())['files']
    for item in inventory:
        p=HERE/'artifacts/spark/payload'/item['path']
        assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],str(p)
    fork=torch.load(RAW/'fork.pt',map_location='cpu',weights_only=False)
    end=torch.load(RAW/'end.pt',map_location='cpu',weights_only=False)
    qualified=torch.load(RAW/'qualification512.pt',map_location='cpu',weights_only=False)
    parent=ROOT/'runs/223-2026-11-04-multispike-interventions/artifacts/spark/payload/worker'
    historical=torch.load(parent/'prefix/end.pt',map_location='cpu',weights_only=False)
    for key in ['model','optimizer','rng','previous_gradient','current_gradient']:equal(qualified[key],historical[key])
    assert fork['completed_updates']==448 and end['completed_updates']==801
    for n in ['1.weight','1.bias']:equal(fork['model'][n],end['model'][n])
    active=[n for n in fork['model'] if not n.startswith('1.')]
    assert all(not torch.equal(fork['model'][n],end['model'][n]) for n in active)
    equal(fork['optimizer']['param_groups'],end['optimizer']['param_groups'])
    assert all(int(s['step'])==801 for s in end['optimizer']['state'].values())
    assert all(torch.isfinite(p).all() for p in end['model'].values())
    first=ROOT/'runs/242-2026-11-07-mlp-all-layer-oscillation/artifacts/spark/payload/worker'
    for n,f in [('1.weight','first_weights.npy'),('1.bias','first_biases.npy')]:
        np.testing.assert_array_equal(fork['model'][n].numpy(),np.load(first/f,mmap_mode='r')[1])
    actual=[json.loads(x) for x in (RAW/'boundaries.jsonl').read_text().splitlines()]
    refs=[json.loads(x) for x in (parent/'prefix/boundaries.jsonl').read_text().splitlines()]
    for a,b in zip(actual[:449],refs[:449]):
        for k in ['state','loss','raw_gradient_norm']:assert a[k]==b[k]
    result=dict(status='passed',source_files=len(inputs),retrieved_files=len(inventory),
        complete_state512_matches_historical=True,prefix_pairs_exact=449,
        first_layer_fork_matches_retained_state448=True,frozen_weight_bias_bitwise_equal=True,
        all_four_later_parameter_tensors_changed=True,all_adam_groups_unchanged=True,all_final_clocks=801,
        final_parameters_finite=True,checkpoints={f:sha(RAW/f) for f in ['fork.pt','qualification512.pt','end.pt']},
        generator_sha256=sha(Path(__file__)))
    (HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n',newline='\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
