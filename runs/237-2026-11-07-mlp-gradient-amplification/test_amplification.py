"""Small exact autograd checks of factored gradient/Jacobian/curvature formulas."""
import json
import torch
import diagnostics as d
torch.manual_seed(37)
m=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(4,3),torch.nn.Tanh(),torch.nn.Linear(3,3),torch.nn.Tanh(),torch.nn.Linear(3,10)).double()
x=torch.randn(4,4,dtype=torch.float64);y=torch.arange(4)
params=dict(m.named_parameters());delta={k:torch.randn_like(p)*.02 for k,p in params.items()}
a=d.forward(m,x);b=d.unit_backward(m,a,y);jp,jx=d.jacobian_traces(m,a);df,ddf=d.directional(m,a,delta)
target=torch.nn.functional.one_hot(y,10).double();loss=.5*(a['output']-target).square().sum()/len(x)
g=torch.autograd.grad(loss,list(params.values()),create_graph=True)
gd=sum((v*delta[k]).sum() for k,v in zip(params,g));hv=torch.autograd.grad(gd,list(params.values()))
quadratic=sum((v*delta[k]).sum() for k,v in zip(params,hv))
torch.testing.assert_close(gd,(b['residual']*df).sum()/len(x),rtol=1e-10,atol=1e-12)
torch.testing.assert_close(quadratic,(df.square()+b['residual']*ddf).sum()/len(x),rtol=1e-10,atol=1e-12)
for i in range(len(x)):
    one=lambda *v:torch.func.functional_call(m,dict(zip(params,v)),x[i:i+1]).flatten()
    jac=torch.autograd.functional.jacobian(one,tuple(params.values()))
    trace=sum(q.square().sum() for q in jac)
    torch.testing.assert_close(trace,jp[i],rtol=1e-10,atol=1e-12)
    ji=torch.autograd.functional.jacobian(lambda v:m(v[None]).flatten(),x[i])
    torch.testing.assert_close(ji.square().sum(),jx[i],rtol=1e-10,atol=1e-12)
    grads=torch.autograd.grad(.5*(m(x[i:i+1])-target[i]).square().sum(),list(params.values()))
    manual=[b['dz1'][i][:,None]*x[i][None,:],b['dz1'][i],b['dz2'][i][:,None]*a['h1'][i][None,:],b['dz2'][i],b['residual'][i][:,None]*a['h2'][i][None,:],b['residual'][i]]
    for actual,wanted in zip(grads,manual):torch.testing.assert_close(actual,wanted,rtol=1e-10,atol=1e-12)
small=torch.nn.Sequential(torch.nn.Linear(3,2))
opt=torch.optim.Adam(small.parameters(),lr=.001,foreach=False,fused=False)
for p in small.parameters():p.grad=torch.ones_like(p)*.2
opt.step();old=d.adam_vectors(opt);before={k:p.detach().clone() for k,p in small.named_parameters()}
for p in small.parameters():p.grad=torch.ones_like(p)*-.1
opt.step();record,_=d.adam_observe(small,opt,before,old)
assert max(q['predicted_step_residual_norm'] for q in record['groups'].values())<1e-7
print(json.dumps(dict(status='passed',checks=['per-example outer-product gradients','parameter Jacobian trace','input Jacobian trace','directional gradient','GGN plus residual curvature equals Hessian form','Adam denominator and actual increment'])))
