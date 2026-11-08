"""Native Adam recurrence and clipping placement, including retained history."""
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.eos_tasks.conventional_continuation import configure_arm, update, vectors


def main():
    checked=[]
    for beta,cap in [(.999,None),(.95,None),(.9,None),(.5,None),(.999,.5),(.999,1.)]:
        model=torch.nn.Linear(3,2,bias=True); opt=torch.optim.Adam(model.parameters(),lr=.003,foreach=False,fused=False)
        for p in model.parameters():
            p.grad=torch.ones_like(p)*.2
        for _ in range(7):opt.step()
        before={p:(p.detach().clone(),opt.state[p]['exp_avg'].clone(),opt.state[p]['exp_avg_sq'].clone()) for p in model.parameters()}
        c=dict(opt=opt,base=.003,state=7,task=type('Task',(),dict(model=model))())
        arm=dict(id='test',beta2=beta)
        if cap:arm['clip']=cap
        configure_arm(c,arm)
        for p in model.parameters():
            assert torch.equal(opt.state[p]['exp_avg'],before[p][1]) and torch.equal(opt.state[p]['exp_avg_sq'],before[p][2])
            assert int(opt.state[p]['step'])==7
            p.grad=torch.arange(p.numel(),dtype=p.dtype).reshape(p.shape)+.1
        raw=torch.cat([p.grad.flatten() for p in model.parameters()]).clone(); oldraw=raw.clone()
        factor,actual=update(c,arm,raw);assert torch.equal(raw,oldraw)
        assert factor==min(1.,cap/float(raw.double().norm())) if cap else factor==1.
        assert torch.equal(actual,raw*factor)
        offset=0
        for p in model.parameters():
            old,m,v=before[p];g=actual[offset:offset+p.numel()].reshape(p.shape).double();offset+=p.numel()
            em=.9*m.double()+.1*g;ev=beta*v.double()+(1-beta)*g*g
            expected=old.double()-.003*(em/(1-.9**8))/((ev/(1-beta**8)).sqrt()+1e-8)
            torch.testing.assert_close(opt.state[p]['exp_avg'].double(),em,rtol=2e-6,atol=2e-7)
            torch.testing.assert_close(opt.state[p]['exp_avg_sq'].double(),ev,rtol=2e-6,atol=2e-7)
            torch.testing.assert_close(p.double(),expected,rtol=2e-6,atol=2e-7)
        assert c['state']==8 and torch.equal(c['previous_gradient'],raw)
        assert torch.isfinite(vectors(c)[2]).all();checked.append([beta,cap])
    print(json.dumps(dict(status='passed',cases=checked,checks=['retained moments and clock','clip before moments','raw history preserved','independent FP64 recurrence','native bias correction','actual update'])))


if __name__=='__main__':main()
