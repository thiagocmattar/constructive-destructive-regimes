"""Run218 generated-vector/gradient-oracle contracts. No production MLP."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('run218_actions', HERE/'01_actions.py')
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
spec = importlib.util.spec_from_file_location('run217_helpers', a.BASE_HERE/'test_control.py')
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)

PARENT_ID = dict(arm='A0', source='generated_contract_fixture', runtime={'device': 'cpu'})
IMPLEMENTATION = dict(source='generated_contract_fixture', runtime={'device': 'cpu'})


def blank_branch(arm, model, cap=None):
    origin = a.base.flat(model.parameters()).detach().double().cpu()
    return dict(arm=arm, fork_update=0, fork_parameters=origin.clone(), updates=0, cap=cap,
                path_length=0., actual_total=torch.zeros_like(origin), ideal_total=torch.zeros_like(origin),
                actions=[], fallback_count=0, floor_count=0, upper_cap_count=0)


def set_gradient(model, g):
    model.first.grad = g[:2].clone(); model.second.grad = g[2:].clone()


def same_moments(opt, reference):
    for x, y in zip(opt.state.values(), reference.state.values()):
        for key in ['step', 'exp_avg', 'exp_avg_sq']:
            assert torch.equal(x[key], y[key])
            assert x[key].dtype == torch.float32


@pytest.fixture
def parent_file(tmp_path):
    task, opt, obs = h.setup(stochastic=True)
    h.consume(task, opt, obs, 64)
    path = tmp_path/'common-parent.pt'
    a.base.save_checkpoint(path, a.base.checkpoint(task.model, opt, obs, PARENT_ID, h.data_state(), 'common_fork'), immutable=True)
    return path


def new_branch(parent, arm):
    task, opt, obs = h.setup(stochastic=True)
    identity, branch = a.branch_from_parent(parent, a.base.file_hash(parent), PARENT_ID,
                    task.model, opt, obs, h.data_state(), arm, IMPLEMENTATION, 64)
    return task, opt, obs, identity, branch


@pytest.mark.parametrize('arm', a.MOMENT_PRESERVING)
def test_native_fp32_moments_and_age_identical_under_common_gradients(arm):
    task, opt, obs = h.setup(); plain, ref, _ = h.setup()
    branch = blank_branch(arm, task.model, cap=.0001 if arm == 'A4' else None)
    gradients = [torch.tensor(x) for x in [[1.,-2.,.3],[-2.,1.,0.],[0.,0.,0.],[3.,-1.,-.5]]]
    for age, g in enumerate(gradients, 1):
        set_gradient(task.model, g); set_gradient(plain.model, g)
        ref.step()
        pre, post = a.update(opt, obs, g, branch)
        same_moments(opt, ref)
        assert obs.completed_updates == age
        assert opt.param_groups[0]['lr'] == .001
        assert post['action']['direction_moment_age'] == age
        assert pre['state'] == age-1
        assert not post['action']['decision_requires_outgoing_increment']
        if arm == 'A0':
            h.same(task.model.state_dict(), plain.model.state_dict()); h.same(opt.state_dict(), ref.state_dict())
        assert 'cumulative_step_rounding_residual_norm' not in post
        assert branch['updates'] == age


@pytest.mark.parametrize('arm,factor', [('A1',.5),('A2',.25)])
def test_half_quarter_native_lr_weights_match_with_inherited_age(parent_file, arm, factor):
    task,opt,obs,identity,branch = new_branch(parent_file,arm)
    plain,ref,unused,_,_ = new_branch(parent_file,'A0')
    ref.param_groups[0]['lr'] = .001*factor
    for values in [[.3,-.8,.4],[1.,2.,-.1],[0.,0.,0.]]:
        g=torch.tensor(values)
        set_gradient(task.model,g); set_gradient(plain.model,g)
        ref.step(); _, post = a.update(opt,obs,g,branch)
        h.same(task.model.state_dict(),plain.model.state_dict()); same_moments(opt,ref)
        assert post['action']['alpha'] == .001*factor
        assert int(opt.state[task.model.first]['step']) > 64


def test_staged_direction_is_same_step_once_before_any_outgoing_motion(monkeypatch):
    task,opt,obs = h.setup(); h.consume(task,opt,obs,3)
    branch=blank_branch('A8',task.model); branch['fork_update']=3
    before=a.base.flat(task.model.parameters()).detach().double()
    previous_g=obs.previous_gradient.clone(); previous_s=obs.previous_increment.clone()
    _,g=task.gradient()
    plain,ref,_=h.setup(); plain.model.load_state_dict(task.model.state_dict()); ref.load_state_dict(copy.deepcopy(opt.state_dict()))
    set_gradient(plain.model,g); ref.step()
    expected=a.adam_streams(ref,4)['adaptive']
    original_choose=a.choose_action; calls=[]
    def inspect(arm, gradient, direction, old_g=None, old_s=None, cap=None):
        assert torch.equal(a.base.flat(task.model.parameters()).detach().double(),before)
        assert torch.equal(direction,expected)
        assert torch.equal(old_g,previous_g) and torch.equal(old_s,previous_s)
        assert int(opt.state[task.model.first]['step']) == 4
        calls.append(arm)
        return original_choose(arm,gradient,direction,old_g,old_s,cap)
    monkeypatch.setattr(a,'choose_action',inspect)
    real_step=opt.step; step_calls=[]
    def counted():
        step_calls.append(opt.param_groups[0]['lr']); return real_step()
    monkeypatch.setattr(opt,'step',counted)
    a.update(opt,obs,g,branch)
    assert calls == ['A8'] and step_calls == [0.]
    same_moments(opt,ref)


@pytest.mark.parametrize('arm', ['A4','A8'])
def test_uncapped_or_fallback_application_is_bitwise_native(arm):
    task,opt,obs = h.setup(); plain,ref,_ = h.setup()
    branch=blank_branch(arm,task.model,cap=1. if arm=='A4' else None)
    g=torch.tensor([1.,-2.,.3]); set_gradient(task.model,g); set_gradient(plain.model,g)
    ref.step(); _,post=a.update(opt,obs,g,branch)
    h.same(task.model.state_dict(),plain.model.state_dict()); h.same(opt.state_dict(),ref.state_dict())
    assert post['action']['alpha'] == .001
    if arm=='A8':
        assert post['action']['fallback_reason'] == 'missing_previous_gradient_or_actual_increment'


def test_a4_cap_is_from_prior64_ideal_proposals_only(parent_file):
    task,opt,obs,identity,branch=new_branch(parent_file,'A4')
    for state,row in enumerate(obs.history):
        row['post']['streams']['adaptive']['norm']=float(state+1)
        row['post']['streams']['actual']['norm']=1e6
        row['post']['streams']['raw']['norm']=2e6
    derived=a.fixed_proposal_cap(obs)
    assert derived['prior_states']==[0,63] and derived['count']==64
    assert derived['cap']==pytest.approx(.0325)
    assert derived['proposal_norms']==pytest.approx([.001*(i+1) for i in range(64)])
    # No incoming/outgoing fork gradient is provided to the threshold function.
    assert branch['cap']==identity['transformation']['threshold']['cap']
    old=branch['cap']; g=torch.tensor([1.,2.,3.]); set_gradient(task.model,g)
    a.update(opt,obs,g,branch)
    assert branch['cap']==old


@pytest.mark.parametrize('case', ['missing','zero','nonfinite','clock'])
def test_a4_unusable_common_prefix_is_explicit(case):
    task,opt,obs=h.setup(); h.consume(task,opt,obs,64 if case!='missing' else 63)
    if case=='zero':
        for row in obs.history:
            row['post']['streams']['adaptive']['norm']=0.
    if case=='nonfinite':
        obs.history[5]['post']['streams']['adaptive']['norm']=float('nan')
    if case=='clock':
        obs.history[-1]['pre']['state']=64
    with pytest.raises(ValueError):
        a.fixed_proposal_cap(obs)


def test_a4_caps_ideal_proposal_and_reports_actual_rounding_excess():
    task,opt,obs=h.setup()
    with torch.no_grad(): task.model.first[0]=1.
    branch=blank_branch('A4',task.model,cap=4e-8)
    g=torch.tensor([1.,0.,0.]); set_gradient(task.model,g)
    _,post=a.update(opt,obs,g,branch); action=post['action']; audit=post['action_audit']
    assert action['ordinary_ideal_proposal_norm'] > branch['cap']
    assert action['chosen_ideal_proposal_norm'] == pytest.approx(branch['cap'],abs=1e-20)
    assert audit['actual_increment_norm'] > branch['cap']
    assert audit['actual_cap_excess_due_to_rounding'] > 0
    assert post['adam_step_rounding_residual_norm'] == audit['ideal_actual_rounding_residual_norm']
    assert not any(k in action for k in ['actual_increment_norm','actual_cap_excess_due_to_rounding','ideal_actual_rounding_residual_norm'])
    assert action['alpha_available_before_parameter_update']==1
    assert audit['available_after_update']==1
    assert action['proposal_clip_factor'] < 1


def test_zero_a4_proposal_is_zero_without_division():
    zero=torch.zeros(3)
    row=a.choose_action('A4',zero,zero,cap=.1)
    assert row['proposal_clip_factor']==1. and row['alpha']==.001
    assert row['ordinary_ideal_proposal_norm']==0


@pytest.mark.parametrize('raw,expected,floor,cap', [(.0005,.0005,False,False),(1e-6,5e-5,True,False),(.01,.001,False,True)])
def test_a8_formula_floor_and_upper_cap(raw,expected,floor,cap):
    g=torch.tensor([2.,0.,0.],dtype=torch.float64); d=torch.tensor([1.,0.,0.],dtype=torch.float64)
    s=torch.tensor([1.,0.,0.],dtype=torch.float64); y=torch.tensor([2/raw,0.,0.],dtype=torch.float64)
    row=a.choose_action('A8',g,d,g-y,s)
    assert not row['fallback']
    assert row['kappa_sec']==pytest.approx(2/raw)
    assert row['alpha_hat']==pytest.approx(raw)
    assert row['alpha']==pytest.approx(expected)
    assert row['floor_active']==floor and row['upper_cap_active']==cap
    assert row['loss_decrease_guarantee'] is False


@pytest.mark.parametrize('case,reason', [
 ('missing_g','missing_previous_gradient_or_actual_increment'),('missing_s','missing_previous_gradient_or_actual_increment'),
 ('zero_s','zero_previous_actual_increment'),('zero_d','zero_current_direction'),
 ('uphill','non_descending_current_direction'),('orthogonal_direction','non_descending_current_direction'),
 ('negative_curvature','insufficient_positive_secant_alignment'),('zero_y','insufficient_positive_secant_alignment'),
 ('threshold_equality','insufficient_positive_secant_alignment'),('wrong_shape','invalid_previous_gradient_shape')])
def test_a8_invalid_history_and_geometry_fall_back_exactly(case,reason):
    g=torch.tensor([0.,1.,0.],dtype=torch.float64); d=g.clone()
    prev=torch.tensor([-1.,0.,0.],dtype=torch.float64); s=torch.tensor([1.,0.,0.],dtype=torch.float64)
    if case=='missing_g': prev=None
    if case=='missing_s': s=None
    if case=='zero_s': s.zero_()
    if case=='zero_d': d.zero_()
    if case=='uphill': d=-g
    if case=='orthogonal_direction': d=torch.tensor([1.,0.,0.],dtype=torch.float64)
    if case=='negative_curvature': prev=torch.tensor([1.,0.,0.],dtype=torch.float64)
    if case=='zero_y': prev=g.clone()
    if case=='threshold_equality': prev=torch.tensor([-1e-8,0.,0.],dtype=torch.float64)
    if case=='wrong_shape': prev=torch.ones(2)
    row=a.choose_action('A8',g,d,prev,s)
    assert row['fallback_reason']==reason and row['fallback']
    assert row['alpha']==.001 and row['alpha_hat'] is None
    json.dumps(row,allow_nan=False)


@pytest.mark.parametrize('field', ['current_gradient','current_direction','previous_gradient','previous_actual_increment'])
@pytest.mark.parametrize('bad', [float('nan'),float('inf'),-float('inf')])
def test_a8_nonfinite_inputs_fallback_with_json_safe_reason(field,bad):
    values={k:torch.ones(3,dtype=torch.float64) for k in ['current_gradient','current_direction','previous_gradient','previous_actual_increment']}
    values[field][0]=bad
    row=a.choose_action('A8',values['current_gradient'],values['current_direction'],values['previous_gradient'],values['previous_actual_increment'])
    assert row['fallback_reason']=='nonfinite_'+field and row['alpha']==.001
    json.dumps(row,allow_nan=False)


@pytest.mark.parametrize('case,reason', [('difference','nonfinite_gradient_difference'),('reduction','nonfinite_reduction'),('alpha','nonfinite_alpha_hat'),('denominator','invalid_proposal_denominator')])
def test_a8_nonfinite_derived_quantities_fallback(case,reason):
    g=torch.tensor([1.,1.,0.],dtype=torch.float64); d=g.clone(); prev=torch.zeros(3,dtype=torch.float64); s=torch.ones(3,dtype=torch.float64)
    if case=='difference': g[0]=1e308; prev[0]=-1e308
    if case=='reduction': d[0]=1e200
    if case=='alpha':
        g=torch.tensor([1e150,0.,0.],dtype=torch.float64); d=torch.tensor([1.,0.,0.],dtype=torch.float64)
        prev=torch.tensor([1e150,-1e-200,0.],dtype=torch.float64); s=torch.tensor([0.,1.,0.],dtype=torch.float64)
    if case=='denominator':
        g=torch.tensor([1e100,0.,0.],dtype=torch.float64); d=g.clone(); prev=torch.zeros(3,dtype=torch.float64); s=torch.tensor([1e-100,0.,0.],dtype=torch.float64)
    row=a.choose_action('A8',g,d,prev,s)
    assert row['fallback_reason']==reason and row['alpha']==.001
    json.dumps(row,allow_nan=False)


def test_zero_previous_gradient_can_be_eligible_and_tolerance_is_strict():
    g=torch.tensor([1.,0.,0.],dtype=torch.float64)
    assert not a.choose_action('A8',g,g,torch.zeros(3),g)['fallback']
    g=torch.tensor([0.,1.,0.],dtype=torch.float64); s=torch.tensor([1.,0.,0.],dtype=torch.float64)
    assert not a.choose_action('A8',g,g,torch.tensor([-2e-8,0.,0.],dtype=torch.float64),s)['fallback']


@pytest.mark.parametrize('arm', a.SUPPORTED)
def test_explicit_common_branch_roundtrip_and_stochastic_suffix(parent_file,tmp_path,arm):
    parent_hash=a.base.file_hash(parent_file)
    task,opt,obs,identity,branch=new_branch(parent_file,arm)
    a.run_branch(task,opt,obs,branch,identity,h.data_state(),tmp_path/'full',5,checkpoint_every=1)
    expected=(copy.deepcopy(task.model.state_dict()),copy.deepcopy(opt.state_dict()),obs.state_dict(),copy.deepcopy(branch))
    task2,opt2,obs2,id2,branch2=new_branch(parent_file,arm)
    assert identity==id2
    a.run_branch(task2,opt2,obs2,branch2,id2,h.data_state(),tmp_path/'part',2,checkpoint_every=1)
    recovery=tmp_path/'part/recovery.pt'; frozen=a.base.file_hash(tmp_path/'part/checkpoints/branch_start-state-000064.pt')
    task3,opt3,obs3,id3,origin=new_branch(parent_file,arm)
    restored=a.load_branch(recovery,a.base.file_hash(recovery),task3.model,opt3,obs3,id3,h.data_state(),origin)
    a.run_branch(task3,opt3,obs3,restored,id3,h.data_state(),tmp_path/'resumed',5,checkpoint_every=1)
    h.same(expected[0],task3.model.state_dict()); h.same(expected[1],opt3.state_dict()); h.same(expected[3],restored)
    x,y=copy.deepcopy(expected[2]),obs3.state_dict()
    for history in [x['history'],y['history']]:
        for row in history: row['pre'].pop('utc',None)
    h.same(x,y)
    assert a.base.file_hash(parent_file)==parent_hash
    assert a.base.file_hash(tmp_path/'part/checkpoints/branch_start-state-000064.pt')==frozen
    assert id3['parent']['scientific_identity']==PARENT_ID and id3['parent']['sha256']==parent_hash
    assert restored['updates']==5 and obs3.completed_updates==69
    assert (tmp_path/'resumed/checkpoints/endpoint-state-000069.pt').is_file()


@pytest.mark.parametrize('wrong', ['arm','parent','transformation','implementation','file_hash','origin'])
def test_resume_rejects_arm_parent_action_or_file_identity(parent_file,tmp_path,wrong):
    task,opt,obs,identity,branch=new_branch(parent_file,'A8')
    path=tmp_path/'branch.pt'
    a.save_branch(path,task.model,opt,obs,identity,h.data_state(),branch,'fork',True)
    expected=copy.deepcopy(identity); origin=copy.deepcopy(branch); sha=a.base.file_hash(path)
    if wrong=='arm': expected['arm']='A1'
    if wrong=='parent': expected['parent']['sha256']='changed'
    if wrong=='transformation': expected['transformation']['alpha_floor_multiple']=.1
    if wrong=='implementation': expected['implementation']['source']='changed'
    if wrong=='file_hash': sha='incorrect'
    if wrong=='origin': origin['fork_parameters'][0]+=1
    with pytest.raises(ValueError):
        a.load_branch(path,sha,task.model,opt,obs,expected,h.data_state(),origin)


def test_parent_validation_is_strict_and_never_relaxes_a0(parent_file,tmp_path):
    task,opt,obs=h.setup(); old=copy.deepcopy(PARENT_ID); old['source']='changed'
    with pytest.raises(ValueError,match='identity'):
        a.branch_from_parent(parent_file,a.base.file_hash(parent_file),old,task.model,opt,obs,h.data_state(),'A1',IMPLEMENTATION,64)
    with pytest.raises(ValueError,match='SHA'):
        a.branch_from_parent(parent_file,'changed',PARENT_ID,task.model,opt,obs,h.data_state(),'A1',IMPLEMENTATION,64)
    with pytest.raises(ValueError,match='fork'):
        a.branch_from_parent(parent_file,a.base.file_hash(parent_file),PARENT_ID,task.model,opt,obs,h.data_state(),'A1',IMPLEMENTATION,63)
    identity,branch=a.branch_from_parent(parent_file,a.base.file_hash(parent_file),PARENT_ID,task.model,opt,obs,h.data_state(),'A1',IMPLEMENTATION,64)
    path=tmp_path/'branch.pt'; a.save_branch(path,task.model,opt,obs,identity,h.data_state(),branch,'fork',True)
    with pytest.raises(ValueError,match='identity'):
        a.base.load_checkpoint(path,task.model,opt,obs,PARENT_ID,h.data_state())
    opt.param_groups[0]['lr']=.0005
    with pytest.raises(ValueError,match='recipe'):
        a.base.checkpoint(task.model,opt,obs,identity,h.data_state(),'invalid')


@pytest.mark.parametrize('arm', ['A7','unknown'])
def test_unsupported_arms_fail_explicitly_before_any_update(arm):
    task,opt,obs=h.setup(); before=copy.deepcopy(task.model.state_dict())
    with pytest.raises(NotImplementedError,match='unsupported'):
        a.update(opt,obs,torch.ones(3),blank_branch(arm,task.model))
    h.same(before,task.model.state_dict()); assert not opt.state


def test_failure_and_interruption_preserve_branch_recovery(parent_file,tmp_path):
    task,opt,obs,identity,branch=new_branch(parent_file,'A8')
    real=task.gradient
    def failure():
        if branch['updates']==2: raise RuntimeError('synthetic branch failure')
        return real()
    task.gradient=failure
    with pytest.raises(RuntimeError,match='synthetic'):
        a.run_branch(task,opt,obs,branch,identity,h.data_state(),tmp_path/'failed',4,checkpoint_every=1)
    status=json.loads((tmp_path/'failed/status.json').read_text())
    assert status['status']=='failed' and status['branch_updates']==2 and 'synthetic' in status['traceback']
    assert torch.load(tmp_path/'failed/recovery.pt',weights_only=False)['branch_state']['updates']==2
    other,op,ob,id2,b=new_branch(parent_file,'A1')
    status=a.run_branch(other,op,ob,b,id2,h.data_state(),tmp_path/'stop',3,stop_requested=lambda: b['updates']==1)
    assert status['status']=='interrupted' and b['updates']==1 and other.calls==1


def test_foundation_is_pinned_and_native_staging_reference_is_verified():
    assert a.base.file_hash(a.BASE_HERE/'01_control.py')==a.BASE_SHA256
    cfg=json.loads((HERE/'config.json').read_text())
    assert cfg['native_adam_source_sha256']==a.native_source_hash()
    assert cfg['implemented_arms']==list(a.SUPPORTED)

def test_branch_cap_and_clock_cannot_change_silently(parent_file,tmp_path):
    task,opt,obs,identity,branch=new_branch(parent_file,'A4')
    branch['cap'] *= 2
    with pytest.raises(ValueError,match='cap differs'):
        a.save_branch(tmp_path/'bad.pt',task.model,opt,obs,identity,h.data_state(),branch,'fork',True)
    branch['updates'] += 1
    with pytest.raises(ValueError,match='clock'):
        a.update(opt,obs,torch.ones(3),branch)
    assert not (tmp_path/'bad.pt').exists()


def test_a0_reference_keeps_native_application_path(monkeypatch):
    task,opt,obs=h.setup(); branch=blank_branch('A0',task.model)
    def forbidden(*args): raise AssertionError('A0 must not use custom parameter application')
    monkeypatch.setattr(a,'apply_staged_alpha',forbidden)
    g=torch.ones(3); set_gradient(task.model,g)
    _,post=a.update(opt,obs,g,branch)
    assert post['action']['alpha']==.001 and obs.completed_updates==1
