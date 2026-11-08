"""Run275 scalar-per-layer control, preserving native Adam moment evolution."""
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.eos_tasks import representation_continuation as continuation
from src.eos_tasks import representation_policy as policy


@torch.no_grad()
def match(model,before,groups,targets):
    params=dict(model.named_parameters());records={}
    for n,target in targets.items():
        names=groups[n]['names'];native=torch.cat([(params[k].double()-before[k].double()).flatten() for k in names]);pn=float(native.norm())
        assert pn or not target,'A zero native direction cannot match a positive target norm'
        scale=target/pn if pn else 0.
        for k in names:params[k].copy_((before[k].double()+scale*(params[k].double()-before[k].double())).to(params[k].dtype))
        actual=torch.cat([(params[k].double()-before[k].double()).flatten() for k in names]);error=float((actual-scale*native).norm())
        assert abs(float(actual.norm())-target)<=error+1e-10
        records[n]=dict(factor=scale,affected_fraction=float(scale!=1),proposed_norm=pn,target_norm=target,actual_norm=float(actual.norm()),rounding_norm_bound=error)
    return records


def test():
    model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Tanh(),torch.nn.Linear(4,2));opt=torch.optim.Adam(model.parameters(),lr=.01,foreach=False)
    model(torch.ones(5,3)).square().mean().backward();before={n:p.detach().clone() for n,p in model.named_parameters()};opt.step()
    state=continuation.fingerprint(opt.state_dict());layout=policy.group_layout(model);original={n:p.detach().clone() for n,p in model.named_parameters()}
    records=match(model,before,layout,{'0':.0123})
    assert continuation.fingerprint(opt.state_dict())==state
    for n,p in model.named_parameters():
        if n.startswith('2.'):assert torch.equal(p,original[n])
    assert abs(records['0']['actual_norm']-.0123)<2e-7
    print(json.dumps(dict(status='passed',checks=['per-layer target norm within storage bound','native moments and clocks unchanged','untargeted output increment unchanged'])))


def main():
    if '--selftest' in sys.argv:test();return
    here=Path(__file__).resolve().parent;cfg=json.loads((here/'config.json').read_text());schedule=json.loads((here/'norm_schedule.json').read_text())['states']
    original=policy.apply_policy;position={}
    def wrapped(model,before,arm,groups,metrics,inputs,calibration,targets):
        if arm['kind']!='paired_norm':return original(model,before,arm,groups,metrics,inputs,calibration,targets)
        t=position.get(arm['id'],cfg['fork']);position[arm['id']]=t+1
        return match(model,before,groups,schedule[str(t)])
    policy.apply_policy=wrapped;continuation.main()


if __name__=='__main__':main()
