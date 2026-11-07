import copy, json, sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.eos_tasks.layer_lr_continuation import layer_step
torch.manual_seed(11)
m=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(6,5),torch.nn.Tanh(),torch.nn.Linear(5,5),torch.nn.Tanh(),torch.nn.Linear(5,2))
o=torch.optim.Adam(m.parameters(),lr=.001,foreach=False,fused=False)
x,y=torch.randn(9,6),torch.randn(9,2)
for _ in range(4):
 o.zero_grad();((m(x)-y).square().sum()/18).backward();o.step()
for a,b in [(1,1),(.5,1),(.25,1),(.5,.5),(.25,.25),(1,.5),(1,.25)]:
 test=copy.deepcopy(m);ref=copy.deepcopy(m)
 testopt=torch.optim.Adam(test.parameters(),lr=.001,foreach=False,fused=False)
 refopt=torch.optim.Adam(ref.parameters(),lr=.001,foreach=False,fused=False)
 frozen=copy.deepcopy(o.state_dict())
 for opt in [testopt,refopt]:opt.load_state_dict(copy.deepcopy(frozen))
 # Independent native Adam parameter groups with copied warm moments.
 old=refopt.param_groups[0]
 refopt.param_groups=[dict(old,params=list(ref[layer].parameters()),lr=.001*(a if layer==1 else b)) for layer in [1,3,5]]
 for _ in range(3):
  for opt,model in [(testopt,test),(refopt,ref)]:
   opt.zero_grad();((model(x)-y).square().sum()/18).backward()
  rng=torch.get_rng_state().clone()
  layer_step(test,testopt,.001*a,.001*b);refopt.step()
  assert torch.equal(rng,torch.get_rng_state()) and len(testopt.param_groups)==1
  for p,q in zip(test.parameters(),ref.parameters()):
   assert torch.equal(p,q) and torch.equal(p.grad,q.grad)
   for key in testopt.state[p]:assert torch.equal(testopt.state[p][key],refopt.state[q][key])
 assert all(int(v['step'])==4 for v in frozen['state'].values())
print(json.dumps(dict(status='passed',checks=['seven LR combinations including identity','three warm steps exactly match independently grouped native Adam','biases included','moments clocks raw gradients RNG preserved','checkpoint parameter ordering preserved'])))
