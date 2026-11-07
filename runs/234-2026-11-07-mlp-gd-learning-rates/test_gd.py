"""Plain GD placement, exact FP32 formula and optimizer resume mechanics."""
import copy
import importlib.util
import json
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('gd_worker',HERE/'01_worker.py')
w=importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
for lr in [.04,.08]:
    p=torch.nn.Parameter(torch.tensor([1.,-.3,.2]))
    adam=torch.optim.Adam([p],lr=.001)
    p.grad=torch.tensor([.2,.5,-.1])
    adam.step()
    before=p.detach().clone()
    gd=w.gd_optimizer([p],lr)
    assert torch.equal(before,p) and not gd.state
    assert gd.param_groups[0]['momentum']==0 and gd.param_groups[0]['weight_decay']==0
    expected=before.clone().add_(p.grad,alpha=-lr)
    gd.step()
    assert torch.equal(p,expected) and not gd.state
    q=torch.nn.Parameter(p.detach().clone())
    other=w.gd_optimizer([q],lr)
    other.load_state_dict(copy.deepcopy(gd.state_dict()))
    p.grad=torch.tensor([.4,.1,-.9]);q.grad=p.grad.clone()
    gd.step();other.step()
    assert torch.equal(p,q)
    assert gd.state_dict()==other.state_dict()
print(json.dumps(dict(status='passed',rates=[.04,.08],checks=['switch preserves weights','zero momentum/decay','exact FP32 GD formula','exact optimizer resume'])))
