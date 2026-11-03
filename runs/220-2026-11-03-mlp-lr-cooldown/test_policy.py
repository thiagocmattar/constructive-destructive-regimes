"""Contract checks for persistence, restart, recovery and unchanged Adam arithmetic."""
import copy
import importlib.util
import math
from pathlib import Path

import pytest
import torch

spec=importlib.util.spec_from_file_location('cooldown_policy',Path(__file__).with_name('01_cooldown.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def sequence(condition,values):
    policy=m.initial_policy();result=[]
    for value in values:
        before=copy.deepcopy(policy)
        lr,phase,trigger,next_policy=m.schedule(condition,value,policy)
        assert policy==before
        result.append((lr,phase,trigger));policy=next_policy
    return result,policy


def test_hold_exactly_eight_then_restore():
    rows,p=sequence('hold8',[1.8]+[0.]*9)
    assert [x[0] for x in rows]==[.0005]*8+[.001]*2
    assert p['half_updates']==8 and p['triggers']==1 and p['switches_within']==1


def test_cosine_eight_explicit_endpoints_monotone_and_no_overshoot():
    rows,p=sequence('hold8_cosine8',[1.8]+[0.]*17)
    expected=[.0005]*8+[.0005+.00025*(1-math.cos(math.pi*k/8)) for k in range(1,9)]+[.001]*2
    assert [x[0] for x in rows]==expected
    assert all(.0005<=x[0]<=.001 for x in rows)
    assert all(a[0]<b[0] for a,b in zip(rows[7:15],rows[8:16]))
    assert rows[7][0]==.0005 and rows[15][0]==.001 and p['below_base_updates']==15


@pytest.mark.parametrize('condition',['hold8','hold8_cosine8'])
@pytest.mark.parametrize('retrigger_at',[2,8,11])
def test_retrigger_resets_hold_without_compounding(condition,retrigger_at):
    values=[0.]*30;values[0]=1.8;values[retrigger_at]=2.
    rows,p=sequence(condition,values)
    assert [x[0] for x in rows[retrigger_at:retrigger_at+8]]==[.0005]*8
    assert min(x[0] for x in rows)==.0005 and p['triggers']==2
    if condition=='hold8_cosine8':assert p['retriggers']==1
    else:assert p['retriggers']==int(retrigger_at<8)


def test_undefined_does_not_cancel_active_schedule_and_nonfinite_fails():
    rows,p=sequence('hold8_cosine8',[None,1.8]+[None]*17)
    assert rows[0][0]==.001
    assert [x[0] for x in rows[1:9]]==[.0005]*8
    assert rows[16][0]==.001 and p['undefined_updates']==18
    with pytest.raises(ValueError):m.schedule('hold8',float('nan'),m.initial_policy())


@pytest.mark.parametrize('condition',m.CONDITIONS)
def test_native_weights_moments_clocks_and_raw_gradient_preserved(condition):
    parameter=torch.nn.Parameter(torch.tensor([1.,-2.]));model=torch.nn.ParameterList([parameter])
    reference=torch.nn.Parameter(parameter.detach().clone());refmodel=torch.nn.ParameterList([reference])
    opt=m.b.make_optimizer(model);ref=m.b.make_optimizer(refmodel);obs=m.b.Observer(2,'cpu');policy=m.initial_policy()
    for state in range(22):
        # One initial reversal triggers, then a constant gradient exercises the full hold and ramp.
        gradient=torch.tensor([1.,1.]) if state==0 else torch.tensor([-1.,-1.])
        parameter.grad=gradient.clone();reference.grad=gradient.clone();pre=obs.observe(state,gradient)
        expected,_,_,_=m.schedule(condition,pre['raw_projected_change'],policy)
        ref.param_groups[0]['lr']=expected;ref.step()
        decision,post=m.step(opt,obs,gradient,condition,policy,pre)
        assert torch.equal(parameter,reference) and decision['applied_lr']==expected
        assert decision['available_before_update']==state+1 and post['update']==state+1
        for key in ['step','exp_avg','exp_avg_sq']:assert torch.equal(opt.state[parameter][key],ref.state[reference][key])
        assert torch.equal(parameter.grad,gradient) and torch.equal(obs.previous_gradient,gradient)
        assert opt.param_groups[0]['lr']==.001 and policy['updates']==state+1


def test_original_conditions_match_frozen_lr_choices():
    for condition in ['unchanged','always_half','temporary']:
        values=[None,1.799,1.8,3.,-2.,0.]
        rows,_=sequence(condition,values)
        assert [x[0] for x in rows]==[m.r.choose_lr(condition,v)[0] for v in values]


def test_windows_retry_keeps_value_and_propagates_other_failures(monkeypatch):
    calls=[]
    def writer(path,value):
        calls.append(copy.deepcopy(value))
        if len(calls)<3:raise PermissionError('transient lock')
        return 'written'
    monkeypatch.setattr(m,'ORIGINAL_WRITE',writer);monkeypatch.setattr(m.time,'sleep',lambda seconds:None)
    assert m.write_json(None,{'state':8})=='written' and calls==[{'state':8}]*3
