"""Focused mechanics: Adam LR versus memory, and selective local-cap geometry."""
import copy
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.eos_tasks import representation_policy as p

torch.manual_seed(12)
model=torch.nn.Sequential(torch.nn.Linear(4,3),torch.nn.Tanh(),torch.nn.Linear(3,2))
x=torch.randn(7,4)
opt=torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,foreach=False,fused=False)
for _ in range(4):
 opt.zero_grad();model(x).square().sum().backward();opt.step()
weights=copy.deepcopy(model.state_dict());state=copy.deepcopy(opt.state_dict())
proposals=[];moments=[];checks=[]
for factor in [1,2,4]:
 model.load_state_dict(weights);opt.load_state_dict(copy.deepcopy(state))
 opt.param_groups[0]['lr']=.001*factor
 opt.zero_grad();model(x).square().sum().backward()
 before={n:v.detach().clone() for n,v in model.named_parameters()}
 opt.step();native=copy.deepcopy(opt.state_dict())
 u=torch.cat([(v.detach()-before[n]).flatten() for n,v in model.named_parameters()])
 proposals.append(u);moments.append([native['state'][j][key].clone() for j in native['state'] for key in ['exp_avg','exp_avg_sq']])
 h=x.double().numpy();dw=(model[0].weight.detach().double()-before['0.weight'].double()).numpy()
 db=(model[0].bias.detach().double()-before['0.bias'].double()).numpy()
 expected=np.sqrt(np.mean((h@dw.T+db)**2,axis=0))
 measured=p.affine_scores(model[0],x,torch.from_numpy(dw),torch.from_numpy(db)).numpy()
 np.testing.assert_allclose(expected,measured,rtol=1e-12,atol=1e-14)
 # One fixed threshold for all LR multipliers, and only the first affine targeted.
 tau=.002
 record=p.apply_policy(model,before,dict(kind='functional',quantile=.5),p.group_layout(model),{}, {'0':x},
                       {'0':{'functional':{'0.5':tau}}},['0'])['0']
 np.testing.assert_allclose(record['factors'],np.minimum(1,tau/expected),rtol=1e-12)
 a=np.asarray(record['factors'])
 np.testing.assert_array_equal(model[0].weight.detach().numpy(),(before['0.weight'].double().numpy()+a[:,None]*dw).astype('float32'))
 np.testing.assert_array_equal(model[0].bias.detach().numpy(),(before['0.bias'].double().numpy()+a*db).astype('float32'))
 for key in native['state']:
  for n,v in native['state'][key].items(): assert torch.equal(v,opt.state_dict()['state'][key][n])
 expected_output=(before['2.weight'].double()+u[-8:-2].reshape(2,3).double()).float()
 assert torch.equal(model[2].weight,expected_output)
 checks.append(dict(lr_multiplier=factor,proposed_norm=float(u.norm()),max_local_score=max(record['actual_scores'])))
for i,factor in [(1,2),(2,4)]:
 for a,b in zip(moments[0],moments[i]): assert torch.equal(a,b)
 rounding_bound=(factor+1)*torch.finfo(torch.float32).eps*max(float(w.abs().max()) for w in weights.values())
 assert float((proposals[i]-proposals[0]*factor).abs().max()) <= rounding_bound
print(json.dumps(dict(status='passed',checks=['independent NumPy scores','fixed budget across LRs','weight row and bias scaling',
 'untargeted output native','moments unchanged by cap','moments independent of LR for identical gradient','LR scales native increment'],measurements=checks)))
