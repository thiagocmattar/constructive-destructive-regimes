"""Independent small-matrix checks before any scientific execution."""
import copy
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.eos_tasks import representation_policy as p


def main():
    torch.manual_seed(11)
    x=torch.randn(2,2,4,5,dtype=torch.float64);m=nn.Conv2d(2,3,3,stride=2,padding=1).double()
    dw=torch.randn_like(m.weight);db=torch.randn_like(m.bias)
    xp=np.pad(x.numpy(),((0,0),(0,0),(1,1),(1,1)));v=[]
    for n in range(2):
        for i in range(0,4,2):
            for j in range(0,5,2):v.append((dw.numpy()*xp[n,:,i:i+3,j:j+3]).sum((1,2,3))+db.numpy())
    np.testing.assert_allclose(p.affine_scores(m,x,dw,db).numpy(),np.sqrt(np.mean(np.array(v)**2,axis=0)),rtol=1e-12)
    q=nn.MultiheadAttention(4,2).double();xq=torch.randn(5,3,4,dtype=torch.float64)
    dw=torch.randn_like(q.in_proj_weight);db=torch.randn_like(q.in_proj_bias)
    reference=xq.numpy().reshape(-1,4)@dw.numpy().T+db.numpy()
    np.testing.assert_allclose(p.affine_scores(q,xq,dw,db).numpy(),np.sqrt(np.mean(reference**2,axis=0)),rtol=1e-12)
    assert p.scalar_factor(dict(rho=-.5,norm=4.,previous_ema=2.),'shock')==.5
    assert p.scalar_factor(dict(rho=.5,norm=4.,previous_ema=2.),'shock')==1
    assert p.scalar_factor(dict(rho=-.5,norm=1.,previous_ema=2.),'shock')==1
    assert p.scalar_factor(dict(eta_kappa_hat=4.),'secant')==.5
    assert p.scalar_factor(dict(eta_kappa_hat=None),'secant')==1
    for kind in ['discard','cap','functional','shock','secant']:
        model=nn.Sequential(nn.Flatten(),nn.Linear(4,3),nn.Tanh(),nn.Linear(3,2));opt=torch.optim.Adam(model.parameters(),lr=.01,foreach=False,fused=False)
        x=torch.randn(7,4);model(x).square().sum().backward();before={n:a.detach().clone() for n,a in model.named_parameters()}
        g=torch.cat([a.grad.flatten() for a in model.parameters()]);groups=p.group_layout(model)
        metrics=p.block_diagnostics(g,-g,groups,{n:.5*float(g[v['start']:v['end']].norm()) for n,v in groups.items()})
        opt.step();native={n:a.detach().clone() for n,a in model.named_parameters()};moments=copy.deepcopy(opt.state_dict());raw=[a.grad.clone() for a in model.parameters()]
        arm=dict(kind=kind,scope='entry',quantile=.75);cal={'1':{'coordinate':{'0.75':.001},'functional':{'0.75':.001}}}
        records=p.apply_policy(model,before,arm,groups,metrics,{'1':x},cal,['1'])
        for name,a in model.named_parameters():
            if not name.startswith('1.'):assert torch.equal(a,native[name])
        for a,b in zip(model.parameters(),raw):assert torch.equal(a.grad,b)
        after=opt.state_dict()
        for idx,state in moments['state'].items():
            for key,value in state.items():assert torch.equal(value,after['state'][idx][key])
        if kind=='functional':
            scores=p.affine_scores(model[1],x,model[1].weight.double()-before['1.weight'].double(),model[1].bias.double()-before['1.bias'].double())
            assert bool((scores<=.001+torch.tensor(records['1']['rounding_bound'])+1e-10).all())
    z=torch.zeros(3);assert p.cosine(z,z) is None
    print(json.dumps(dict(status='passed',checks=['Independent NumPy convolution patch RMS','Independent QKV RMS','reversal-plus-growth and secant rules','post-Adam placement','native moments clocks raw gradients and untargeted parameters preserved','storage-rounding bound','zero norm cosine'])))


if __name__=='__main__':main()
