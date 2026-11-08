"""A probe-loss backtracking comparator; no claim of a novel SVD optimizer."""
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.eos_tasks import representation_continuation as c
from src.eos_tasks import representation_policy as policy


@torch.no_grad()
def backtrack(model,before,old_loss,evaluate):
    params=dict(model.named_parameters());delta={n:p.detach().double()-before[n].double() for n,p in params.items()};trials=[]
    for attempt in range(13):
        factor=2.**(-attempt) if attempt<12 else 0.
        for n,p in params.items():p.copy_((before[n].double()+factor*delta[n]).to(p.dtype))
        loss=evaluate();trials.append(dict(factor=factor,loss=loss))
        if loss<old_loss or (factor==0 and loss==old_loss):break
    assert loss<=old_loss
    return factor,dict(before_loss=old_loss,trials=trials)


def main():
    if '--selftest' in sys.argv:
        model=torch.nn.Linear(1,1,bias=False)
        with torch.no_grad():model.weight.fill_(1.)
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        with torch.no_grad():model.weight.add_(10.)
        factor,_=backtrack(model,before,1.,lambda:float(model.weight.double().square().sum()))
        assert factor==0 and torch.equal(model.weight,before['weight'])
        with torch.no_grad():model.weight.sub_(4.)
        factor,_=backtrack(model,before,1.,lambda:float(model.weight.double().square().sum()))
        assert factor==.25 and model.weight.item()==0
        print('{"status":"passed","checks":["reject all ascent candidates without moving parameters","accept first strictly decreasing dyadic candidate"]}');return
    original_capture=policy.capture;original_arm=c.run_arm;original_apply=policy.apply_policy;state={}
    def loss(values,task,case,probe):
        f=values['output'].double();y=task.data['y_train'][:probe]
        if case=='mlp':return float(.5*(f-torch.nn.functional.one_hot(y,10)).square().sum(1).mean())
        if case=='transformer':y=y[y!=-100]
        return float(-f[torch.arange(len(y),device=f.device),y].mean())
    def capture(task,case,probe,targets):
        values,inputs,native_loss=original_capture(task,case,probe,targets)
        inputs['__guard_loss__']=loss(values,task,case,probe);return values,inputs,native_loss
    def run_arm(context,fork,cfg,arm,folder,thresholds,ema,groups,entry,trunk,probe_history,parameter_history,mechanics=False):
        state.update(task=context['task'],case=cfg['case'],probe=cfg['probe'])
        return original_arm(context,fork,cfg,arm,folder,thresholds,ema,groups,entry,trunk,probe_history,parameter_history,mechanics)
    @torch.no_grad()
    def apply(model,before,arm,groups,metrics,inputs,thresholds,targets):
        if arm['kind']!='loss_guard':return original_apply(model,before,arm,groups,metrics,inputs,thresholds,targets)
        def evaluate():
            values,_,_=original_capture(state['task'],state['case'],state['probe'],[])
            return loss(values,state['task'],state['case'],state['probe'])
        rng=c.w.rng();factor,detail=backtrack(model,before,inputs['__guard_loss__'],evaluate);c.w.equal(rng,c.w.rng())
        gates={n:dict(factor=factor,affected_fraction=float(factor<1)) for n in groups};gates[next(iter(groups))]['loss_guard']=detail;return gates
    policy.capture=capture;policy.apply_policy=apply;c.run_arm=run_arm;c.main()


if __name__=='__main__':main()
