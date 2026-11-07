"""Check the per-example norm reduction against independently computed autograd."""
import json
import torch
import diagnostics as d

torch.manual_seed(38)
m=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(4,3),torch.nn.Tanh(),torch.nn.Linear(3,3),torch.nn.Tanh(),torch.nn.Linear(3,10)).double()
x=torch.randn(5,4,dtype=torch.float64);y=torch.arange(5)
vectors=[]
for i in range(5):
    loss=.5*(m(x[i:i+1])-torch.nn.functional.one_hot(y[i:i+1],10)).square().sum()
    vectors.append(torch.cat([v.flatten() for v in torch.autograd.grad(loss,tuple(m.parameters()))]))
g=torch.stack(vectors);energies=d.example_energies(m,x,y,physical=2)
torch.testing.assert_close(energies,g.square().sum(1),rtol=1e-12,atol=1e-12)
c=d.coherence(float(g.mean(0).norm()),energies)
torch.testing.assert_close(torch.tensor(c['intrabatch_coherence']),g.sum(0).norm().div(g.norm(dim=1).sum()).float())
assert abs(c['intrabatch_coherence']-c['energy_coherence'])>1e-3
assert d.coherence(0.,torch.zeros(3))['intrabatch_coherence'] is None
print(json.dumps(dict(status='passed',checks=['whole-example energy against autograd including biases','non-divisible physical batching','norm of sum / sum of norms','distinct squared-energy ratio','zero-denominator null'])))
