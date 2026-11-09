"""Read-only audit of retained Run223 fork versus each first update's input state."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import numpy as np


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
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--report-name',default='prefix_audit.json')
    args=parser.parse_args();cfg=json.loads((args.run/'config.json').read_text());torch.set_num_threads(2)
    parent=torch.load(args.run/'inputs/prefix.pt',map_location='cpu',weights_only=False)
    results=[]
    for path in args.out.glob(f'*/audit-{cfg["start"]:04d}.pt'):
        audit=torch.load(path,map_location='cpu',weights_only=False);before=audit['before']
        for k in ['model','rng','previous_gradient']:equal(before[k],parent[k])
        equal(before['optimizer']['state'],parent['optimizer']['state'])
        for a,b in zip(before['optimizer']['param_groups'],parent['optimizer']['param_groups']):
            assert {k:v for k,v in a.items() if k!='betas'}=={k:v for k,v in b.items() if k!='betas'}
        old=parent.get('current_gradient');new=audit['raw_gradient']
        result=dict(arm=path.parent.name,initial_model_moments_clock_rng_previous_gradient_exact=True,
            historical_current_gradient_available=old is not None)
        if old is not None:
            result.update(historical_current_gradient_exact=torch.equal(old,new),
                historical_current_gradient_relative_l2_error=float((old.double()-new.double()).norm()/old.double().norm()))
        results.append(result)
    assert results,'First-update audit not yet available'
    result=dict(status='passed',results=results,
        qualification_note='qualification.json used a recomputed-gradient fork snapshot with current_gradient=None. Its historical_comparison.fork_gradient_exact=False is not a measured mismatch; this direct original-checkpoint comparison is authoritative.')
    (args.out/args.report_name).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
