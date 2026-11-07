import json
import torch
from diagnostics import capture

torch.manual_seed(7)
model=torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.Tanh(),torch.nn.Linear(5,2))
x=torch.randn(9,3);model(x).sum().backward()
rng=torch.get_rng_state().clone();grads=[p.grad.clone() for p in model.parameters()]
h,f=capture(model,x,4)
assert torch.equal(rng,torch.get_rng_state())
assert all(torch.equal(p.grad,g) for p,g in zip(model.parameters(),grads))
assert len(model[-1]._forward_pre_hooks)==0
torch.testing.assert_close(h,model[:-1](x));torch.testing.assert_close(f,model[-1](h))
assert h.shape==(9,5) and f.shape==(9,2)
print(json.dumps(dict(status='passed',checks=['all samples including batch tail','correct final hidden layer','observer preserves RNG and gradients','hook removed','head reconstruction'])))
