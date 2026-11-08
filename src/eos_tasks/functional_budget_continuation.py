"""Budget actual probe movement by scaling a complete native Adam proposal."""
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.eos_tasks import representation_continuation as c
from src.eos_tasks import representation_policy as policy


def rms(a,b):return float((a.double()-b.double()).square().mean().sqrt())


@torch.no_grad()
def constrain(model,before,current,budgets,evaluate,maximum_factor=1.):
    params=dict(model.named_parameters());delta={n:p.detach().double()-before[n].double() for n,p in params.items()}
    active={n:v for n,v in budgets.items() if v>0};trials=[];alpha=1.;native_ratios=None
    for attempt in range(10):
        for n,p in params.items():p.copy_((before[n].double()+alpha*delta[n]).to(p.dtype))
        candidate=evaluate();ratios={n:rms(candidate[n],current[n])/tau for n,tau in active.items()}
        maximum=max(ratios.values(),default=0.);maximum=maximum if math.isfinite(maximum) else float('inf')
        trials.append(dict(factor=alpha,max_ratio=maximum))
        if native_ratios is None:native_ratios=ratios
        if maximum<=1+1e-6 and alpha<=maximum_factor:break
        alpha=min(maximum_factor,min(.999999,1/maximum) if maximum>1 and math.isfinite(maximum) else .5 if not math.isfinite(maximum) else 1.) if attempt==0 else alpha*.5
        if attempt==8:alpha=0.
    assert maximum<=1+1e-6
    return alpha,dict(trials=trials,native_ratios=native_ratios,actual_ratios=ratios,budgets=active,maximum_factor=maximum_factor)


def test():
    model=torch.nn.Linear(2,1,bias=True);x=torch.tensor([[1.,2.],[-3.,1.]])
    with torch.no_grad():model.weight.fill_(.2);model.bias.zero_()
    before={n:p.detach().clone() for n,p in model.named_parameters()};old={'output':model(x).detach()}
    with torch.no_grad():model.weight.add_(1.)
    factor,record=constrain(model,before,old,{'output':.1},lambda:{'output':model(x).detach()})
    measured=float(np.sqrt(np.mean((model(x).detach().numpy().astype(float)-old['output'].numpy())**2)))
    assert measured<=.1*(1+1e-6) and 0<factor<1 and record['actual_ratios']['output']<=1+1e-6
    with torch.no_grad():
        for n,p in model.named_parameters():p.copy_(before[n]+1.)
    factor,record=constrain(model,before,old,{'output':100.},lambda:{'output':model(x).detach()},maximum_factor=.1)
    assert factor==.1 and len(record['trials'])==2 and record['maximum_factor']==.1
    print(json.dumps(dict(status='passed',checks=['Independent NumPy actual response budget','global factor in0..1','finite backtracking to a verified candidate'])))


def main():
    if '--selftest' in sys.argv:test();return
    original_capture=policy.capture;original_prepare=c.prepare;original_arm=c.run_arm;original_apply=policy.apply_policy
    state=dict(calibrating=False,previous=None,pools={})
    def record(values):
        if state['previous'] is not None:
            for n,v in values.items():state['pools'].setdefault(n,[]).append(rms(v,state['previous'][n]))
        state['previous']=values
    def capture(task,case,probe,targets):
        values,inputs,loss=original_capture(task,case,probe,targets)
        if state['calibrating']:record(values)
        inputs['__budget_values__']=values
        return values,inputs,loss
    def prepare(run,cfg,out):
        state.update(calibrating=True,previous=None,pools={});result=original_prepare(run,cfg,out)
        context=result[0];rng=c.w.rng();values,_,_=original_capture(context['task'],cfg['case'],cfg['probe'],[]);c.w.equal(rng,c.w.rng());record(values)
        assert all(len(v)==cfg['calibration_updates'] for v in state['pools'].values())
        state['budgets']={n:{str(q):float(np.quantile(v,q,method='linear')) for q in cfg['quantiles']} for n,v in state['pools'].items()}
        c.w.write(out/'shared/functional_budget.json',dict(native_outgoing_updates=list(range(cfg['fork']-cfg['calibration_updates'],cfg['fork'])),rms=state['pools'],thresholds=state['budgets']))
        state.update(calibrating=False,previous=None);return result
    def run_arm(context,fork,cfg,arm,folder,thresholds,ema,groups,entry,trunk,probe_history,parameter_history,mechanics=False):
        state.update(task=context['task'],case=cfg['case'],probe=cfg['probe'],previous_factor=1.)
        return original_arm(context,fork,cfg,arm,folder,thresholds,ema,groups,entry,trunk,probe_history,parameter_history,mechanics)
    @torch.no_grad()
    def apply(model,before,arm,groups,metrics,inputs,thresholds,targets):
        if arm['kind']!='budget':return original_apply(model,before,arm,groups,metrics,inputs,thresholds,targets)
        budgets={n:v[str(arm['quantile'])] for n,v in state['budgets'].items() if arm['scope']=='representations' or n=='output'}
        rng=c.w.rng()
        factor,detail=constrain(model,before,inputs['__budget_values__'],budgets,lambda:original_capture(state['task'],state['case'],state['probe'],[])[0],maximum_factor=min(1.,state['previous_factor']*arm.get('recovery_multiplier',float('inf'))))
        detail['previous_factor']=state['previous_factor'];state['previous_factor']=factor
        c.w.equal(rng,c.w.rng())
        records={n:dict(factor=factor,affected_fraction=float(factor<1)) for n in groups}
        records[next(iter(groups))]['functional_budget']=detail
        return records
    policy.capture=capture;policy.apply_policy=apply;c.prepare=prepare;c.run_arm=run_arm;c.main()


if __name__=='__main__':main()
