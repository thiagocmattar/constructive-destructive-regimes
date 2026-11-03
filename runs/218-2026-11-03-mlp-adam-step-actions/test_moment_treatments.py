"""Independent NumPy/native controls for A3/A5/A6; generated gradients only."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('run218_action_helpers',HERE/'test_actions.py')
h=importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
a=h.a
parent_file=h.parent_file


def vector_state(optimizer,key):
    return np.concatenate([s[key].detach().cpu().numpy().ravel() for s in optimizer.state.values()]).astype(np.float64)


def reference_direction(optimizer,f,k,raw=None,switched=False):
    m=vector_state(optimizer,'exp_avg'); v=vector_state(optimizer,'exp_avg_sq')
    age=f+k
    b2=1-.999**f*.99**k if switched else 1-.999**age
    numerator=m/(1-.9**age) if raw is None else raw.double().numpy()
    return numerator/(np.sqrt(v/b2)+1e-8)


def step(task,opt,obs,branch,g):
    h.set_gradient(task.model,g)
    return a.update(opt,obs,g,branch)


def test_a3_global_clip_changes_both_moments_keeps_raw_stream(parent_file):
    task,opt,obs,identity,branch=h.new_branch(parent_file,'A3')
    plain,ref,unused,_,_=h.new_branch(parent_file,'A0')
    cap=branch['cap']; g=torch.tensor([100.,-50.,30.])*cap
    expected_factor=cap/np.linalg.norm(g.double().numpy())
    expected_input=(g.numpy()*np.float32(expected_factor)).astype(np.float32)
    before_m=vector_state(opt,'exp_avg'); before_v=vector_state(opt,'exp_avg_sq')
    h.set_gradient(plain.model,torch.from_numpy(expected_input.copy())); ref.step()
    pre,post=step(task,opt,obs,branch,g)
    h.same_moments(opt,ref); h.h.same(task.model.state_dict(),plain.model.state_dict())
    assert torch.equal(g,obs.previous_gradient)
    assert torch.equal(g,a.base.flat([p.grad for p in task.model.parameters()]))
    assert torch.equal(branch['last_optimizer_input'],torch.from_numpy(expected_input))
    assert pre['gradient_norm']==pytest.approx(np.linalg.norm(g.double().numpy()))
    assert post['streams']['raw']['norm']==pre['gradient_norm']
    assert post['optimizer_input']['norm']==pytest.approx(np.linalg.norm(expected_input.astype(np.float64)))
    assert post['action']['input_clip_factor']==pytest.approx(expected_factor)
    np.testing.assert_allclose(vector_state(opt,'exp_avg'),.9*before_m+.1*expected_input,rtol=2e-7,atol=3e-9)
    np.testing.assert_allclose(vector_state(opt,'exp_avg_sq'),.999*before_v+.001*expected_input.astype(np.float64)**2,rtol=2e-7,atol=2e-9)
    assert branch['input_clip_count']==1
    assert identity['transformation']['input_transform']=='one_global_clip_after_complete_accumulation'


def test_a3_one_complete_vector_not_individual_chunks(parent_file,monkeypatch):
    task,opt,obs,_,branch=h.new_branch(parent_file,'A3')
    # Two hypothetical physical contributions cancel in one coordinate.
    cap=branch['cap']; first=torch.tensor([10.,0.,0.])*cap; second=torch.tensor([-9.,4.,0.])*cap
    raw=first+second
    global_h=raw*min(1.,cap/float(raw.double().norm()))
    per_chunk=first*min(1.,cap/float(first.double().norm()))+second*min(1.,cap/float(second.double().norm()))
    calls=[]; native=opt.step
    def inspect():
        incoming=a.base.flat([p.grad for p in task.model.parameters()]).clone()
        assert torch.equal(incoming,global_h) and not torch.allclose(incoming,per_chunk)
        assert opt.param_groups[0]['lr']==0.
        calls.append(incoming); return native()
    monkeypatch.setattr(opt,'step',inspect)
    step(task,opt,obs,branch,raw)
    assert len(calls)==1


def test_a3_zero_input_advances_decaying_moments_without_division(parent_file):
    task,opt,obs,_,branch=h.new_branch(parent_file,'A3')
    m=vector_state(opt,'exp_avg');v=vector_state(opt,'exp_avg_sq')
    _,post=step(task,opt,obs,branch,torch.zeros(3))
    assert post['action']['input_clip_factor']==1 and post['optimizer_input']['norm']==0
    assert torch.equal(branch['last_optimizer_input'],torch.zeros(3))
    np.testing.assert_allclose(vector_state(opt,'exp_avg'),.9*m,rtol=2e-7,atol=1e-9)
    np.testing.assert_allclose(vector_state(opt,'exp_avg_sq'),.999*v,rtol=2e-7,atol=1e-9)
    assert int(opt.state[task.model.first]['step'])==65


def test_a3_prior64_raw_threshold_ignores_other_streams(parent_file):
    task,opt,obs,_,branch=h.new_branch(parent_file,'A3')
    for i,row in enumerate(obs.history):
        row['pre']['gradient_norm']=float(i+1)
        row['post']['streams']['adaptive']['norm']=1e8
        row['post']['streams']['actual']['norm']=2e8
        row['post']['streams']['raw']['norm']=3e8
    value=a.treatments.raw_cap(obs)
    assert value['cap']==32.5 and value['prior_states']==[0,63]
    old=branch['cap']; step(task,opt,obs,branch,torch.ones(3))
    assert branch['cap']==old


@pytest.mark.parametrize('case',['missing','zero','nonfinite','clock'])
def test_a3_invalid_prior_threshold_rejected(parent_file,case):
    task,opt,obs,_,_=h.new_branch(parent_file,'A3')
    if case=='missing': obs.history.pop()
    if case=='zero':
        for row in obs.history: row['pre']['gradient_norm']=0.
    if case=='nonfinite': obs.history[7]['pre']['gradient_norm']=float('nan')
    if case=='clock': obs.history[2]['post']['update']=99
    with pytest.raises(ValueError): a.treatments.raw_cap(obs)


@pytest.mark.parametrize('raw',[[.8,-.4,1.2],[0.,0.,0.]])
def test_a5_raw_numerator_ordinary_stored_moments_and_valid_rounding(parent_file,raw):
    task,opt,obs,identity,branch=h.new_branch(parent_file,'A5')
    plain,ref,_,_,_=h.new_branch(parent_file,'A0')
    g=torch.tensor(raw); before=a.base.flat(task.model.parameters()).detach().double().numpy()
    h.set_gradient(plain.model,g); ref.step()
    _,post=step(task,opt,obs,branch,g)
    h.same_moments(opt,ref)
    applied=reference_direction(opt,64,1,raw=g)
    ordinary=reference_direction(opt,64,1)
    actual=a.base.flat(task.model.parameters()).detach().double().numpy()-before
    np.testing.assert_allclose(branch['last_applied_direction'].numpy(),applied,rtol=3e-15,atol=1e-15)
    assert post['streams']['adaptive']['norm']==pytest.approx(np.linalg.norm(ordinary),rel=2e-15)
    assert post['applied_direction']['norm']==pytest.approx(np.linalg.norm(applied),rel=2e-15)
    assert post['action']['ordinary_ideal_proposal_norm']==pytest.approx(.001*np.linalg.norm(ordinary))
    assert post['action']['chosen_ideal_proposal_norm']==pytest.approx(.001*np.linalg.norm(applied))
    assert post['action_audit']['ideal_actual_rounding_residual_norm']==pytest.approx(np.linalg.norm(actual+.001*applied),abs=1e-15)
    assert 'adam_step_rounding_residual_norm' not in post
    assert 'adaptive_actual_step_cosine' not in post
    assert not np.allclose(applied,ordinary)
    assert post['adaptive_definition']=='ordinary_adam_from_actual_stored_moments'
    assert identity['transformation']['numerator']=='current_raw_gradient'
    assert torch.equal(branch['last_optimizer_input'],g) and torch.equal(obs.previous_gradient,g)
    if raw==[0.,0.,0.]:
        assert np.array_equal(actual,np.zeros(3))
        assert post['branch_actual_retention'] is None
        assert post['branch_actual_retention_undefined_reason']=='zero_path_length'


def test_a6_mixed_history_recursion_and_correct_direction_every_changed_step(parent_file,monkeypatch):
    task,opt,obs,identity,branch=h.new_branch(parent_file,'A6')
    old_m=vector_state(opt,'exp_avg');old_v=vector_state(opt,'exp_avg_sq')
    assert opt.param_groups[0]['betas']==(.9,.99)
    assert branch['beta2_history']['changed_updates']==0
    def forbidden(*args,**kwargs): raise AssertionError('Ordinary adaptive helper is invalid for switched beta2 history')
    monkeypatch.setattr(a,'adam_streams',forbidden)
    for k,values in enumerate([[.3,-.8,.4],[-1.,0.,2.],[0.,0.,0.],[4.,-3.,1.]],1):
        g=torch.tensor(values); before=a.base.flat(task.model.parameters()).detach().double().numpy()
        expected_m=.9*old_m+.1*g.double().numpy()
        expected_v=.99*old_v+.01*g.double().numpy()**2
        pre,post=step(task,opt,obs,branch,g)
        np.testing.assert_allclose(vector_state(opt,'exp_avg'),expected_m,rtol=3e-7,atol=2e-8)
        np.testing.assert_allclose(vector_state(opt,'exp_avg_sq'),expected_v,rtol=2e-7,atol=1e-9)
        correct=1-.999**64*.99**k
        assert post['action']['second_moment_bias_denominator']==pytest.approx(correct,abs=3e-16)
        assert post['action']['second_moment_history']['changed_updates']==k
        assert post['action']['first_moment_global_age']==64+k
        assert post['action']['direction_moment_age']==64+k
        expected=reference_direction(opt,64,k,switched=True)
        np.testing.assert_allclose(branch['last_applied_direction'].numpy(),expected,rtol=5e-15,atol=2e-15)
        assert post['streams']['adaptive']['norm']==pytest.approx(np.linalg.norm(expected),rel=5e-15)
        wrong=vector_state(opt,'exp_avg')/(1-.9**(64+k))/(np.sqrt(vector_state(opt,'exp_avg_sq')/(1-.99**(64+k)))+1e-8)
        assert np.linalg.norm(wrong-expected)>1e-3
        actual=a.base.flat(task.model.parameters()).detach().double().numpy()-before
        assert post['action_audit']['ideal_actual_rounding_residual_norm']==pytest.approx(np.linalg.norm(actual+.001*expected),abs=2e-15)
        assert 'ordinary_ideal_proposal_norm' not in post['action']
        assert post['adaptive_definition']=='history_corrected_adam'
        assert pre['state']==63+k and post['applied_direction']['available_before_parameter_update']==64+k
        assert post['action_audit']['available_after_update']==64+k
        old_m=vector_state(opt,'exp_avg');old_v=vector_state(opt,'exp_avg_sq')
    with pytest.raises(ValueError,match='recipe'): a.base.check_optimizer(opt,68)


def test_a6_constant_gradient_unbiased_across_switch(tmp_path):
    task,opt,obs=h.h.setup(); g=torch.tensor([1.,-2.,.5]); f=11
    for _ in range(f):
        h.set_gradient(task.model,g); pre=obs.observe(obs.completed_updates,g)
        before=a.base.flat(task.model.parameters()).detach().double(); opt.step()
        obs.accept(g,opt,before,a.base.flat(task.model.parameters()).detach().double(),pre)
    parent=tmp_path/'constant.pt'
    a.base.save_checkpoint(parent,a.base.checkpoint(task.model,opt,obs,h.PARENT_ID,h.h.data_state(),'fork'),immutable=True)
    identity,branch=a.branch_from_parent(parent,a.base.file_hash(parent),h.PARENT_ID,task.model,opt,obs,h.h.data_state(),'A6',h.IMPLEMENTATION,f)
    for k in range(1,6):
        _,post=step(task,opt,obs,branch,g)
        vhat=vector_state(opt,'exp_avg_sq')/(1-.999**f*.99**k)
        mhat=vector_state(opt,'exp_avg')/(1-.9**(f+k))
        np.testing.assert_allclose(vhat,g.double().numpy()**2,rtol=5e-7,atol=1e-8)
        np.testing.assert_allclose(mhat,g.double().numpy(),rtol=5e-7,atol=1e-8)
        assert branch['beta2_history']['changed_updates']==k
        assert post['action']['second_moment_history']['fork_age']==f


@pytest.mark.parametrize('f,k',[(0,1),(657,1),(657,128),(1_000_000,128)])
def test_a6_bias_product_is_stable_and_has_declared_history(f,k):
    history=a.treatments.beta2_history(f,k)
    assert history['bias_denominator']==pytest.approx(1-.999**f*.99**k,abs=3e-16)
    assert 0 < history['bias_denominator'] <= 1


@pytest.mark.parametrize('field',['fork_age','changed_updates','beta2_before','beta2_after','bias_denominator','log_beta2_product'])
def test_a6_wrong_saved_history_rejected_even_with_known_file_hash(parent_file,tmp_path,field):
    task,opt,obs,identity,branch=h.new_branch(parent_file,'A6')
    step(task,opt,obs,branch,torch.ones(3))
    path=tmp_path/'good.pt';a.save_branch(path,task.model,opt,obs,identity,h.h.data_state(),branch,'fork',True)
    payload=torch.load(path,weights_only=False)
    payload['branch_state']['beta2_history'][field]+=1
    bad=tmp_path/'wrong-history.pt';a.base.save_checkpoint(bad,payload,immutable=True)
    other,op,ob,id2,origin=h.new_branch(parent_file,'A6')
    with pytest.raises(ValueError,match='history'):
        a.load_branch(bad,a.base.file_hash(bad),other.model,op,ob,id2,h.h.data_state(),origin)


@pytest.mark.parametrize('arm',['A3','A5','A6'])
@pytest.mark.parametrize('wrong',['arm','numerator','cap_or_history','input_clock'])
def test_treatment_identity_and_saved_state_rejection(parent_file,tmp_path,arm,wrong):
    task,opt,obs,identity,branch=h.new_branch(parent_file,arm)
    step(task,opt,obs,branch,torch.ones(3))
    path=tmp_path/'good.pt';a.save_branch(path,task.model,opt,obs,identity,h.h.data_state(),branch,'fork',True)
    payload=torch.load(path,weights_only=False)
    expected=copy.deepcopy(identity)
    if wrong=='arm': expected['arm']='A0'
    if wrong=='numerator': payload['branch_state']['numerator']='incorrect'
    if wrong=='cap_or_history':
        if arm=='A3': payload['branch_state']['cap']*=2
        elif arm=='A6': payload['branch_state']['beta2_history']['changed_updates']+=1
        else: payload['branch_state']['fork_update']+=1
    if wrong=='input_clock': payload['branch_state']['input_geometry']['count']+=1
    bad=tmp_path/'wrong.pt';a.base.save_checkpoint(bad,payload,immutable=True)
    other,op,ob,id2,origin=h.new_branch(parent_file,arm)
    with pytest.raises(ValueError):
        a.load_branch(bad,a.base.file_hash(bad),other.model,op,ob,expected,h.h.data_state(),origin)


@pytest.mark.parametrize('wrong',['normalization','position','observer_age','optimizer_recipe'])
def test_a6_restore_checks_data_and_recipe_without_relaxing_a0(parent_file,tmp_path,wrong):
    task,opt,obs,identity,branch=h.new_branch(parent_file,'A6')
    path=tmp_path/'good.pt';a.save_branch(path,task.model,opt,obs,identity,h.h.data_state(),branch,'fork',True)
    payload=torch.load(path,weights_only=False)
    if wrong=='normalization': payload['data_state']['normalization_std']+=1
    if wrong=='position': payload['data_state']['position']=1
    if wrong=='observer_age': payload['observer']['completed_updates']+=1
    if wrong=='optimizer_recipe': payload['optimizer']['param_groups'][0]['betas']=(.9,.999)
    bad=tmp_path/'bad.pt';a.base.save_checkpoint(bad,payload,immutable=True)
    other,op,ob,id2,origin=h.new_branch(parent_file,'A6')
    with pytest.raises(ValueError):
        a.load_branch(bad,a.base.file_hash(bad),other.model,op,ob,id2,h.h.data_state(),origin)


@pytest.mark.parametrize('arm',['A3','A5','A6'])
def test_treatment_input_and_direction_clocks_have_no_outgoing_decision_data(parent_file,arm,monkeypatch):
    task,opt,obs,identity,branch=h.new_branch(parent_file,arm)
    before=a.base.flat(task.model.parameters()).detach().clone(); calls=[]; native=opt.step
    def stage_inspect():
        assert torch.equal(a.base.flat(task.model.parameters()).detach(),before)
        assert obs.completed_updates==64 and branch['updates']==0
        calls.append(True); return native()
    monkeypatch.setattr(opt,'step',stage_inspect)
    pre,post=step(task,opt,obs,branch,torch.tensor([4.,-3.,1.]))
    assert calls==[True] and pre['state']==64
    assert post['optimizer_input']['available_before_parameter_update']==65
    assert post['action']['alpha_available_before_parameter_update']==65
    assert not any(k in post['action'] for k in ['actual_increment_norm','ideal_actual_rounding_residual_norm'])
    assert post['action_audit']['available_after_update']==65
    assert branch['last_optimizer_input'].dtype==torch.float32 and branch['last_applied_direction'].dtype==torch.float64

@pytest.mark.parametrize('arm',['A5','A6'])
def test_nonfinite_staged_moment_fails_without_overwriting_recovery(parent_file,tmp_path,arm):
    task,opt,obs,identity,branch=h.new_branch(parent_file,arm)
    original=a.base.flat(task.model.parameters()).detach().clone()
    def huge():
        raw=torch.tensor([1e30,-1e30,1e30]);h.set_gradient(task.model,raw)
        return 1.,raw
    task.gradient=huge
    output=tmp_path/'nonfinite'
    with pytest.raises(ValueError,match='treatment moment'):
        a.run_branch(task,opt,obs,branch,identity,h.h.data_state(),output,2)
    assert torch.equal(a.base.flat(task.model.parameters()).detach(),original)
    saved=torch.load(output/'recovery.pt',weights_only=False)
    assert saved['completed_updates']==64 and saved['branch_state']['updates']==0
    assert all(bool(torch.isfinite(state['exp_avg_sq']).all()) for state in saved['optimizer']['state'].values())
    status=json.loads((output/'status.json').read_text())
    assert status['status']=='failed' and 'treatment moment' in status['traceback']

@pytest.mark.parametrize('arm',['A3','A5','A6'])
def test_wrong_arm_save_is_rejected_before_treatment_metadata(parent_file,tmp_path,arm):
    task,opt,obs,identity,branch=h.new_branch(parent_file,arm)
    wrong=copy.deepcopy(identity);wrong['arm']='A0'
    path=tmp_path/'must-not-exist.pt'
    with pytest.raises(ValueError,match='identity'):
        a.save_branch(path,task.model,opt,obs,wrong,h.h.data_state(),branch,'fork',True)
    assert not path.exists()
