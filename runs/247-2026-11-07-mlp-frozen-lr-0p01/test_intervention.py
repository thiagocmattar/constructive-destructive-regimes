import copy
import json
import torch
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.eos_tasks.frozen_first_linear import freeze_first_step, cosine

torch.manual_seed(7)
m = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(6,5), torch.nn.Tanh(),
                        torch.nn.Linear(5,5), torch.nn.Tanh(), torch.nn.Linear(5,2))
o = torch.optim.Adam(m.parameters(), lr=.001, foreach=False, fused=False)
x, y = torch.randn(9,2,3), torch.randn(9,2)
for _ in range(3):
    o.zero_grad(); ((m(x)-y).square().sum()/18).backward(); o.step()
base = copy.deepcopy(m); bo = torch.optim.Adam(base.parameters(), lr=.001, foreach=False, fused=False)
bo.load_state_dict(copy.deepcopy(o.state_dict()))
fixed = {n: p.detach().clone() for n,p in m[1].named_parameters()}
low=copy.deepcopy(m);lo=torch.optim.Adam(low.parameters(),lr=.001,foreach=False,fused=False)
lo.load_state_dict(copy.deepcopy(o.state_dict()))
before={n:p.detach().clone() for n,p in m.named_parameters()}
for opt in [o,bo]:
    for group in opt.param_groups:group['lr']=.01
for opt, model in [(o,m),(bo,base),(lo,low)]:
    opt.zero_grad(); ((model(x)-y).square().sum()/18).backward()
rng = torch.get_rng_state().clone()
freeze_first_step(m,o,fixed); bo.step();lo.step()
for n,p in m.named_parameters():
    if not n.startswith("1."):
        expected=before[n].double()+10*(dict(low.named_parameters())[n].detach().double()-before[n].double())
        torch.testing.assert_close(p.double(),expected,rtol=0,atol=1e-6)
for p,q in zip(m.parameters(),low.parameters()):
    for k,v in o.state[p].items():assert torch.equal(v,lo.state[q][k])
assert torch.equal(rng,torch.get_rng_state())
assert all(torch.equal(p,fixed[n]) for n,p in m[1].named_parameters())
for (name,p),(_,q) in zip(m.named_parameters(),base.named_parameters()):
    if not name.startswith('1.'): assert torch.equal(p,q),name
    for k,v in o.state[p].items(): assert torch.equal(v,bo.state[q][k]),(name,k)
assert all(int(o.state[p]['step'])==4 for p in m.parameters())
assert all(torch.equal(p.grad,q.grad) for p,q in zip(m.parameters(),base.parameters()))
assert cosine(torch.zeros(4),torch.ones(4)) is None
assert cosine(torch.ones(4),-torch.ones(4)) == -1
for _ in range(2):
    o.zero_grad(); ((m(x)-y).square().sum()/18).backward(); freeze_first_step(m,o,fixed)
    assert all(torch.equal(p,fixed[n]) for n,p in m[1].named_parameters())
print(json.dumps(dict(status='passed',checks=['10x baseline LR has expected FP32 increments and preserves first-step moments versus low LR','warm Adam moments and clocks preserved',
    'all later-layer parameters exactly native on first step','first weight and bias exactly fixed across steps',
    'RNG and raw gradients unchanged by suppression','undefined zero cosine and negative cosine'])))
