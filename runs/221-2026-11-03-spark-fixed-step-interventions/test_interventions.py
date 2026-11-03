"""Mathematical and native-optimizer contract checks for the fixed-step treatments."""
import importlib.util
from pathlib import Path
import copy
import pytest
import torch
spec=importlib.util.spec_from_file_location('fixed_interventions',Path(__file__).with_name('01_worker.py'))
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)

@pytest.mark.parametrize('floor',[.5,.25,.1])
@pytest.mark.parametrize('duration',[8,32,64])
def test_cosine_endpoints_monotonic_and_hold(floor,duration):
 arm=dict(kind='cosine',ratio=floor,duration=duration)
 values=[w.lr_at(arm,k,.003) for k in range(70)]
 assert values[0]==.003 and values[duration]==.003*floor
 assert all(a>=b for a,b in zip(values,values[1:]))
 assert all(x==.003*floor for x in values[duration:])


def test_full_gradient_clipping_preserves_raw_and_handles_zero():
 p=torch.nn.Parameter(torch.tensor([1.,2.]));model=torch.nn.ParameterList([p]);p.grad=torch.tensor([3.,4.]);raw=p.grad.clone()
 factor,before,after=w.modify_gradients_and_decay(model,raw,dict(kind='clip',factor=1.),2.,.001)
 assert factor==.4 and torch.equal(raw,torch.tensor([3.,4.]))
 assert torch.allclose(p.grad,torch.tensor([1.2,1.6])) and torch.equal(before,after)
 p.grad=torch.zeros(2);raw=p.grad.clone()
 factor,_,_=w.modify_gradients_and_decay(model,raw,dict(kind='clip',factor=1.),2.,.001)
 assert factor==1 and torch.equal(p.grad,raw)


@pytest.mark.parametrize('lr',[.001,.003])
@pytest.mark.parametrize('decay',[.01,.1])
def test_warm_decoupled_decay_exact_native_adamw(lr,decay):
 a=torch.nn.Parameter(torch.tensor([[1.,-2.],[.3,-.4]]));b=torch.nn.Parameter(torch.tensor([1.,-2.]))
 c=torch.nn.Parameter(a.detach().clone());d=torch.nn.Parameter(b.detach().clone())
 model=torch.nn.ParameterList([a,b]);refmodel=torch.nn.ParameterList([c,d])
 opt=torch.optim.Adam(model.parameters(),lr=lr,betas=(.9,.999),eps=1e-8,foreach=False,fused=False)
 ref=torch.optim.AdamW([dict(params=[c],weight_decay=0),dict(params=[d],weight_decay=0)],lr=lr,betas=(.9,.999),eps=1e-8,foreach=False,fused=False)
 for step in range(8):
  for x,y in [(a,c),(b,d)]:x.grad=torch.full_like(x,(-1.)**step*(step+1)*.1);y.grad=x.grad.clone()
  if step>=2:
   ref.param_groups[0]['weight_decay']=decay
   raw=w.flat([x.grad for x in model.parameters()]).clone()
   w.modify_gradients_and_decay(model,raw,dict(kind='decay',weight_decay=decay),1.,lr)
  opt.step();ref.step()
  for x,y in [(a,c),(b,d)]:
   assert torch.equal(x,y)
   for key in ['step','exp_avg','exp_avg_sq']:assert torch.equal(opt.state[x][key],ref.state[y][key])


def test_lr_reduction_preserves_moments_for_identical_input():
 a=torch.nn.Parameter(torch.ones(2));b=torch.nn.Parameter(a.detach().clone())
 opt=torch.optim.Adam([a],lr=.003,foreach=False,fused=False);ref=torch.optim.Adam([b],lr=.00075,foreach=False,fused=False)
 for i in range(5):
  a.grad=torch.tensor([1.,-2.])*(i+1);b.grad=a.grad.clone();opt.step();ref.step()
  for key in ['step','exp_avg','exp_avg_sq']:assert torch.equal(opt.state[a][key],ref.state[b][key])
 assert not torch.equal(a,b)
