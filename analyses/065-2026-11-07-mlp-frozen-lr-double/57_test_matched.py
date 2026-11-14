"""Focused optimizer placement, local-cap and observation checks for H051."""
import copy
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.eos_tasks import matched_interventions as m


def main():
    checked=[]
    arms=[dict(id='control',kind='control')]
    arms += [dict(id=f'lr{q}',kind='lr',lr_multiplier=q) for q in [.75,.5,.1]]
    arms += [dict(id=f'beta{q}',kind='beta2',beta2=q) for q in [.95,.9,.5]]
    arms += [dict(id=f'clip{q}',kind='clip',clip=q) for q in [.5,1.]]
    arms += [dict(id=f'cap{q}',kind='functional',quantile=q) for q in [.25,.5,.75]]
    for arm in arms:
        torch.manual_seed(7)
        model=torch.nn.Sequential(torch.nn.Linear(3,2))
        opt=torch.optim.Adam(model.parameters(),lr=.003,foreach=False,fused=False)
        for p in model.parameters(): p.grad=torch.ones_like(p)*.2
        for _ in range(7): opt.step()
        c=dict(opt=opt,base=.003,state=7,task=type('Task',(),dict(model=model))())
        m.configure_arm(c,arm)
        reference=copy.deepcopy(model)
        refopt=torch.optim.Adam(reference.parameters(),lr=.003,foreach=False,fused=False)
        refopt.load_state_dict(copy.deepcopy(opt.state_dict()))
        for p,rp in zip(model.parameters(),reference.parameters()):
            p.grad=torch.arange(p.numel(),dtype=p.dtype).reshape(p.shape)+.1
            rp.grad=p.grad.clone()
        raw=m.w.flat([p.grad for p in model.parameters()]).clone()
        factor=min(1.,arm['clip']/float(raw.double().norm())) if 'clip' in arm else 1.
        for p in reference.parameters(): p.grad.mul_(factor)
        refopt.step()
        inputs={'0':torch.tensor([[1.,2.,3.],[-2.,1.,.5]])}
        thresholds={'0':{'functional':{str(q):.0003+q*.0001 for q in [.25,.5,.75]}}}
        result=m.apply_update(c,arm,raw,inputs,thresholds,m.policy.group_layout(model),['0'])
        assert c['state']==8 and torch.equal(raw,c['previous_gradient'])
        assert torch.equal(result['optimizer_input'],raw*factor)
        assert torch.equal(result['proposed_parameters'],m.w.flat(reference.parameters()))
        for p,rp in zip(model.parameters(),reference.parameters()):
            for key in ['step','exp_avg','exp_avg_sq']:
                assert torch.equal(opt.state[p][key],refopt.state[rp][key])
        if arm['kind']!='functional': assert torch.equal(m.w.flat(model.parameters()),m.w.flat(reference.parameters()))
        else:
            gate=result['gates']['0']; assert gate['affected_fraction']>0
            assert max(gate['actual_scores']) <= gate['threshold']+max(gate['rounding_bound'])+1e-10
        checked.append(arm['id'])
    # Incoming-change indexing and undefined zero denominator.
    c=dict(case='resnet');history={}
    for v,expected in [(torch.tensor([0.,0.]),None),(torch.tensor([1.,0.]),None),(torch.tensor([0.,0.]),-1.)]:
        rows,_=m.representation_rows({'stage':v},history,c)
        assert rows['stage']['rho']==expected
    assert m.policy.cosine(torch.zeros(2),torch.ones(2)) is None
    print(json.dumps(dict(status='passed',arms=checked,checks=['native LR/beta2 recurrence and preserved state','global clipping before Adam','raw history','cap stored bound and unchanged moments','incoming representation timing','zero denominator'])))


if __name__=='__main__': main()
