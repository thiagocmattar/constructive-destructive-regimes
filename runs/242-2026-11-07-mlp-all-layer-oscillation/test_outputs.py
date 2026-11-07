import json
import torch
from diagnostics import capture

torch.manual_seed(7)
m=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(6,5),torch.nn.Tanh(),torch.nn.Linear(5,5),torch.nn.Tanh(),torch.nn.Linear(5,2))
x=torch.randn(9,2,3);m(x).sum().backward();rng=torch.get_rng_state().clone();grads=[p.grad.clone() for p in m.parameters()]
z1,h1,z2,h2,f,n=capture(m,x,4,x.clone())
assert n==9 and torch.equal(rng,torch.get_rng_state())
assert all(torch.equal(p.grad,g) for p,g in zip(m.parameters(),grads))
torch.testing.assert_close(z1,m[1](x.flatten(1)));torch.testing.assert_close(h1,torch.tanh(z1))
torch.testing.assert_close(z2,m[3](h1));torch.testing.assert_close(h2,torch.tanh(z2));torch.testing.assert_close(f,m[-1](h2))
failed=False
try:capture(m,x.flip(0),4,x)
except AssertionError:failed=True
assert failed,'Changed sample order must be rejected'
assert all(not q._forward_hooks and not q._forward_pre_hooks for q in m.modules())
print(json.dumps(dict(status='passed',checks=['all layer placements','image flattening and partial tail','exact input/order check rejects changed inputs','preserved RNG/gradients','hooks removed on success and failure'])))
