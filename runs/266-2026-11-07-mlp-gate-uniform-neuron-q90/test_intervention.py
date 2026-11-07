"""Mathematical and native-state tests for all selective policies and controls."""
import copy
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.eos_tasks.selective_first_linear import transform,apply_after_adam,neuron_rms

torch.manual_seed(19)
x=torch.randn(9,6);y=torch.randn(9,2)
m=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(6,5),torch.nn.Tanh(),torch.nn.Linear(5,5),torch.nn.Tanh(),torch.nn.Linear(5,2))
o=torch.optim.Adam(m.parameters(),lr=.001,foreach=False,fused=False)
for _ in range(4):
    o.zero_grad();((m(x)-y).square().sum()/18).backward();o.step()
for kind in ['discard','cap','neuron','uniform']:
    test=copy.deepcopy(m);native=copy.deepcopy(m)
    opts=[]
    for model in [test,native]:
        opt=torch.optim.Adam(model.parameters(),lr=.001,foreach=False,fused=False)
        opt.load_state_dict(copy.deepcopy(o.state_dict()));opts.append(opt)
        opt.zero_grad();((model(x)-y).square().sum()/18).backward()
    before={n:p.detach().clone() for n,p in test.named_parameters()}
    for opt in opts:opt.step()
    dw=native[1].weight.detach().double()-before['1.weight'].double();db=native[1].bias.detach().double()-before['1.bias'].double()
    z=x.double().numpy()@dw.numpy().T+db.numpy()
    np.testing.assert_allclose(neuron_rms(x,dw,db).numpy(),np.sqrt((z*z).mean(0)),rtol=1e-12,atol=1e-14)
    threshold=float(neuron_rms(x,dw,db).median()) if kind=='neuron' else .0004
    target=float(torch.cat([dw.flatten(),db.flatten()]).norm())*.37
    rng=torch.get_rng_state().clone()
    stats,audit=apply_after_adam(test,before,kind,threshold,x,target,capture_audit=True)
    assert torch.equal(rng,torch.get_rng_state())
    a=np.concatenate([dw.numpy().ravel(),db.numpy()])
    if kind=='discard':want=np.where(np.abs(a)<=threshold,a,0.)
    elif kind=='cap':want=np.clip(a,-threshold,threshold)
    elif kind=='uniform':want=a*(target/np.linalg.norm(a))
    else:
        factors=np.minimum(1,threshold/np.sqrt((z*z).mean(0)))
        want=np.concatenate([(dw.numpy()*factors[:,None]).ravel(),db.numpy()*factors])
    applied=torch.cat([audit['actual_weight'].flatten(),audit['actual_bias']]).numpy()
    assert np.linalg.norm(applied-want)<=stats['rounding_error_norm']+1e-12
    for (name,p),(_,q) in zip(test.named_parameters(),native.named_parameters()):
        assert torch.equal(p.grad,q.grad)
        for key in opts[0].state[p]:assert torch.equal(opts[0].state[p][key],opts[1].state[q][key])
        if not name.startswith('1.'):assert torch.equal(p,q)
zero=torch.zeros(5,6);bias=torch.zeros(5)
for kind in ['discard','cap','neuron','uniform']:
    a,b,_=transform(zero,bias,kind,1.,x,target_norm=0.)
    assert not a.any() and not b.any()
print(json.dumps(dict(status='passed',checks=['NumPy independent formulas for all four policies','neuron RMS keeps sample structure','FP32 rounding bounds','native moments clocks raw gradients RNG and later layers preserved','zero-motion behavior','weights and biases included'])))
