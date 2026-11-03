r"""Focused synthetic checks for the available \eta\widehat\kappa_t rule."""
import importlib.util
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('temporary_lr',HERE/'01_temporary_lr.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_direct_pair_formula_and_undefined():
    previous=torch.tensor([1.,2.]);current=torch.tensor([-1.,.5])
    value,reason=m.eta_kappa_hat(current,previous)
    x=previous.double().numpy();y=current.double().numpy()
    assert value==float(x@(x-y)/(x@x)) and reason is None
    assert m.eta_kappa_hat(current,None)==(None,'missing_previous_gradient')
    assert m.eta_kappa_hat(current,torch.zeros(2))==(None,'zero_previous_gradient_norm')


def test_absolute_values_threshold_equality_and_recovery():
    assert [m.choose_lr('temporary',v)[0] for v in [1.8,1.79,3.,0.,1.8]]==[.0005,.001,.0005,.001,.0005]
    assert m.choose_lr('temporary',None,'zero_previous_gradient_norm')==(.001,'zero_previous_gradient_norm')
    assert m.choose_lr('always_half',None)[0]==.0005
    assert m.choose_lr('unchanged',100.)[0]==.001


def test_nonfinite_is_not_an_undefined_fallback():
    import pytest
    with pytest.raises(ValueError):m.eta_kappa_hat(torch.tensor([float('nan')]),torch.ones(1))
    with pytest.raises(ValueError):m.choose_lr('temporary',float('inf'))


def test_native_moments_clocks_and_current_decision_inputs():
    parameter=torch.nn.Parameter(torch.tensor([1.,-2.]));model=torch.nn.ParameterList([parameter])
    reference=torch.nn.Parameter(torch.tensor([1.,-2.]));refmodel=torch.nn.ParameterList([reference])
    opt=m.b.make_optimizer(model);ref=m.b.make_optimizer(refmodel)
    obs=m.b.Observer(2,'cpu');policy=m.initial_policy()
    previous=None
    for state,values in enumerate([[1.,1.],[-1.,-1.],[.2,.2],[-.4,-.4],[.1,.1]]):
        gradient=torch.tensor(values);parameter.grad=gradient.clone();reference.grad=gradient.clone()
        pre=obs.observe(state,gradient)
        value,reason=m.eta_kappa_hat(gradient,previous);expected=m.choose_lr('temporary',value,reason)[0]
        ref.param_groups[0]['lr']=expected;ref.step()
        decision,post=m.step(opt,obs,gradient,'temporary',policy,pre)
        assert decision['state']==state and decision['available_before_update']==state+1
        assert decision['applied_lr']==expected and post['update']==state+1
        assert torch.equal(parameter,reference)
        for key in ['step','exp_avg','exp_avg_sq']:assert torch.equal(opt.state[parameter][key],ref.state[reference][key])
        assert torch.equal(parameter.grad,gradient) and torch.equal(obs.previous_gradient,gradient)
        assert opt.param_groups[0]['lr']==.001
        previous=gradient.clone()
