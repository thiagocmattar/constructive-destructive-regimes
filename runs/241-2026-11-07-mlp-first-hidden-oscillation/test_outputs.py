import json
import torch
from diagnostics import capture

torch.manual_seed(7)
m=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3,5),torch.nn.Tanh(),torch.nn.Linear(5,5),torch.nn.Tanh(),torch.nn.Linear(5,2))
x=torch.randn(9,3);m(x).sum().backward()
rng=torch.get_rng_state().clone();grads=[p.grad.clone() for p in m.parameters()]
h1,z2,h2,f=capture(m,x,4)
assert torch.equal(rng,torch.get_rng_state())
assert all(torch.equal(p.grad,g) for p,g in zip(m.parameters(),grads))
assert not m[3]._forward_hooks and not m[3]._forward_pre_hooks and not m[-1]._forward_pre_hooks
torch.testing.assert_close(h1,m[:3](x));torch.testing.assert_close(z2,m[3](h1))
torch.testing.assert_close(h2,torch.tanh(z2));torch.testing.assert_close(f,m[-1](h2))
assert h1.shape==z2.shape==h2.shape==(9,5)
print(json.dumps(dict(status='passed',checks=['full samples including partial tail','H1-Z2-H2 placement','linear and tanh reconstruction','observer preserves RNG and native gradients','hooks removed'])))
