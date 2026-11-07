"""Test exact quadratic decomposition, group interactions and observation purity."""
import copy
import json
import torch
import diagnostics as d
torch.manual_seed(23)
model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(4,3),torch.nn.Tanh(),torch.nn.Linear(3,3),torch.nn.Tanh(),torch.nn.Linear(3,10))
x=torch.randn(12,4);y=torch.arange(12)%10
opt=torch.optim.Adam(model.parameters(),lr=.001)
target=torch.nn.functional.one_hot(y,10).float()
(.5*(model(x)-target).square().sum()/len(x)).backward();opt.step()
a=copy.deepcopy(model.state_dict());aa=d.forward_arrays(model,x,4)
rng=torch.get_rng_state().clone();grads=[p.grad.clone() for p in model.parameters()]
d.observe(model,opt,x,y,4,True)
assert torch.equal(rng,torch.get_rng_state())
for p,g in zip(model.parameters(),grads):assert torch.equal(p.grad,g)
with torch.no_grad():
    for p in model.parameters():p.add_(.01*torch.randn_like(p))
b=copy.deepcopy(model.state_dict());bb=d.forward_arrays(model,x,4)
decomp=d.output_decomposition(a,b,aa,bb,y)
hybrid=d.hybrids(model,a,b,x,y,4)
assert abs(sum(hybrid['shapley'].values())-hybrid['total_loss_change'])<1e-12
assert abs(decomp['loss_change_fp64_head']-hybrid['total_loss_change'])<1e-6
assert d.shapley([0,1,2,6],2)==[2.5,3.5]
assert d.cosine(torch.zeros(3),torch.ones(3)) is None
print(json.dumps(dict(status='passed',checks=['observer preserves RNG and gradients','output/loss decomposition','native/full output consistency','Shapley interaction allocation','zero denominator'])))
