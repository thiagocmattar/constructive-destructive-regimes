"""Run218 A3/A5/A6 only: explicit input/numerator/beta2-history contracts."""
import copy
import math
import torch
from src.cifar_eos.core import Geometry, cosine, flat

ARMS = ('A3', 'A5', 'A6')


def raw_cap(observer):
    f = observer.completed_updates
    if f < 64 or len(observer.history) != f:
        raise ValueError('A3 requires64 complete preceding raw full-objective gradients.')
    values=[]
    for state,row in zip(range(f-64,f),observer.history[-64:]):
        if row['pre']['state'] != state or row['post']['gradient_state'] != state or row['post']['update'] != state+1:
            raise ValueError('A3 prior raw-history clock differs.')
        value=row['pre']['gradient_norm']
        if not math.isfinite(value) or value < 0:
            raise ValueError('A3 prior raw norm is invalid.')
        values.append(value)
    ordered=sorted(values); cap=(ordered[31]+ordered[32])/2
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError('A3 median is zero/invalid: threshold rule unusable for this fork.')
    return dict(cap=cap,prior_states=[f-64,f-1],count=64,raw_gradient_norms=values,
                definition='norm of complete raw full-objective gradient before any transform')


def beta2_history(f,k):
    if not isinstance(f,int) or not isinstance(k,int) or f < 0 or k < 0:
        raise ValueError('Invalid beta2 history ages.')
    log_product=f*math.log1p(.999-1.)+k*math.log1p(.99-1.)
    return dict(fork_age=f,changed_updates=k,beta2_before=.999,beta2_after=.99,
                log_beta2_product=log_product,bias_denominator=-math.expm1(log_product))


def validate_branch(branch, identity=None):
    arm=branch['arm']
    if arm not in ARMS:
        return
    if branch.get('method_schema') != 1:
        raise ValueError('Missing/unsupported treatment-state schema.')
    expected_numerator='current_raw_gradient' if arm=='A5' else 'bias_corrected_first_moment'
    if branch.get('numerator') != expected_numerator:
        raise ValueError('Treatment numerator differs.')
    if identity is not None:
        spec=identity['transformation']
        if spec.get('numerator') != expected_numerator:
            raise ValueError('Declared numerator differs.')
        if arm=='A6' and spec.get('beta2_history') != dict(beta2_before=.999,beta2_after=.99,fork_age=branch['fork_update'],correction='1-0.999^f*0.99^k'):
            raise ValueError('Declared beta2 history differs.')
    if arm=='A6' and branch.get('beta2_history') != beta2_history(branch['fork_update'],branch['updates']):
        raise ValueError('Stored beta2 history/changed age/correction differs.')
    for key in ['input_geometry','applied_direction_geometry']:
        if branch[key]['count'] != branch['updates']:
            raise ValueError('Treatment stream clock differs.')
    for key in ['last_optimizer_input','last_applied_direction']:
        if (branch[key] is None) != (branch['updates']==0):
            raise ValueError('Treatment previous-vector history differs.')
    if branch['input_clip_count'] != sum(int(row.get('input_clip_factor',1.)<1.) for row in branch['actions']):
        raise ValueError('Treatment clipping counter differs.')


def initialize(branch, optimizer):
    arm=branch['arm']; count=branch['fork_parameters'].numel()
    branch.update(method_schema=1,numerator='current_raw_gradient' if arm=='A5' else 'bias_corrected_first_moment',
                  input_geometry=Geometry(count,'cpu').state_dict(),applied_direction_geometry=Geometry(count,'cpu').state_dict(),
                  last_optimizer_input=None,last_applied_direction=None,input_clip_count=0)
    if arm=='A6':
        branch['beta2_history']=beta2_history(branch['fork_update'],0)
        optimizer.param_groups[0]['betas']=(.9,.99)


def check_optimizer(optimizer,completed,branch):
    """A6 validator only. The unchanged A0 validator is never relaxed/masked."""
    validate_branch(branch)
    if completed != branch['fork_update']+branch['updates'] or len(optimizer.param_groups)!=1:
        raise ValueError('A6 global/changed clocks differ.')
    group=optimizer.param_groups[0]
    expected=dict(lr=.001,betas=(.9,.99),eps=1e-8,weight_decay=0,amsgrad=False,foreach=False,fused=False,maximize=False)
    if any(group[k]!=v for k,v in expected.items()) or any(group.get(k,False) for k in ['capturable','differentiable','decoupled_weight_decay']):
        raise ValueError('A6 optimizer recipe differs.')
    for p in group['params']:
        state=optimizer.state.get(p,{})
        if p.dtype!=torch.float32 or (completed==0 and state):
            raise ValueError('A6 FP32/fresh optimizer state differs.')
        if completed and (int(state['step'])!=completed or any(state[k].dtype!=torch.float32 or state[k].shape!=p.shape for k in ['exp_avg','exp_avg_sq'])):
            raise ValueError('A6 moment precision/global age differs.')
        if completed and (any(not bool(torch.isfinite(state[k]).all()) for k in ['exp_avg','exp_avg_sq']) or bool((state['exp_avg_sq']<0).any())):
            raise ValueError('A6 nonfinite/negative stored moment.')


@torch.no_grad()
def stage_apply(optimizer,gradient,branch):
    arm=branch['arm']; group=optimizer.param_groups[0]; params=group['params']
    raw_norm=float(gradient.double().norm())
    factor=1.
    if arm=='A3':
        cap=branch['cap']
        if cap is None or not math.isfinite(cap) or cap<=0:
            raise ValueError('A3 requires a fixed positive raw-gradient cap.')
        factor=1. if raw_norm==0 else min(1.,cap/raw_norm)
    h=gradient.detach().clone()
    h.mul_(factor)  # One full-vector FP32 input transform, after complete accumulation.
    old_grads=[p.grad for p in params]; offset=0
    before=flat(params).detach().double()
    try:
        for p in params:
            p.grad=h[offset:offset+p.numel()].view_as(p).clone(); offset+=p.numel()
        group['lr']=0.
        optimizer.step()  # Exactly one native FP32 recurrence; A6 group has beta2=.99.
    finally:
        group['lr']=.001
        for p,g in zip(params,old_grads): p.grad=g
    if not torch.equal(before,flat(params).detach().double()):
        raise ValueError('Treatment moment staging moved parameters.')
    age=branch['fork_update']+branch['updates']+1
    history=beta2_history(branch['fork_update'],branch['updates']+1) if arm=='A6' else None
    b1=1-.9**age; b2=history['bias_denominator'] if history else 1-.999**age
    moments=[]; corrected=[]; references=[]; applied=[]; offset=0
    for p in params:
        state=optimizer.state[p]
        if int(state['step'])!=age:
            raise ValueError('Treatment moment staging age differs.')
        if any(not bool(torch.isfinite(state[k]).all()) for k in ['exp_avg','exp_avg_sq']) or bool((state['exp_avg_sq']<0).any()):
            raise ValueError('Nonfinite/negative treatment moment after staging.')
        m=state['exp_avg'].detach().double(); mhat=m/b1
        denominator=(state['exp_avg_sq'].detach().double()/b2).sqrt()+group['eps']
        reference=mhat/denominator
        numerator=gradient[offset:offset+p.numel()].view_as(p).double() if arm=='A5' else mhat
        direction=numerator/denominator
        moments.append(m); corrected.append(mhat); references.append(reference); applied.append(direction)
        offset+=p.numel()
    vectors=dict(moment=flat(moments),unbiased_moment=flat(corrected),reference=flat(references),applied=flat(applied))
    if not all(bool(torch.isfinite(v).all()) for v in vectors.values()):
        raise ValueError('Nonfinite treatment moment/direction.')
    offset=0
    for p in params:
        state=optimizer.state[p]
        denominator=(state['exp_avg_sq'].sqrt()/(b2**.5)).add_(group['eps'])
        numerator=gradient[offset:offset+p.numel()].view_as(p) if arm=='A5' else state['exp_avg']
        value=-.001 if arm=='A5' else -.001/b1
        p.addcdiv_(numerator,denominator,value=value)
        offset+=p.numel()
    return h,vectors,dict(input_clip_factor=factor,optimizer_input_norm=float(h.double().norm()),
                         raw_gradient_norm=raw_norm,numerator=branch['numerator'],
                         first_moment_global_age=age,second_moment_bias_denominator=b2,
                         second_moment_history=history,
                         reference_direction_definition='history_corrected_adam' if arm=='A6' else 'ordinary_adam_from_actual_stored_moments')


def consume_branch_stream(branch,key,vector):
    stream=Geometry(vector.numel(),vector.device); stream.load_state_dict(branch[key])
    row=stream.consume(vector); branch[key]=stream.state_dict()
    return row


@torch.no_grad()
def accept(observer,optimizer,gradient,h,vectors,details,before,after,pre,branch):
    """Raw stream stays raw. Reference adaptive and applied direction are explicit."""
    age=observer.completed_updates+1; raw=gradient.detach().double(); delta=after-before
    previous_moment=observer.post.streams['moment'].previous
    streams=dict(raw=raw,moment=vectors['moment'],adaptive=vectors['reference'],actual=delta)
    stats={k:observer.post.streams[k].consume(v) for k,v in streams.items()}
    row=dict(update=age,gradient_state=age-1,available_after_update=age,streams=stats,
             adaptive_definition=details['reference_direction_definition'],
             projected_gradient_change=pre['raw_projected_change'],
             unbiased_moment_norm=float(vectors['unbiased_moment'].norm()),
             raw_gradient_prior_moment_cosine=None if previous_moment is None else cosine(raw,previous_moment),
             optimizer_input_prior_moment_cosine=None if previous_moment is None else cosine(h,previous_moment),
             gradient_current_moment_cosine=cosine(raw,vectors['moment']),
             gradient_reference_adaptive_cosine=cosine(raw,vectors['reference']),
             gradient_applied_direction_cosine=cosine(raw,vectors['applied']),
             applied_direction_actual_step_cosine=cosine(-vectors['applied'],delta),
             gradient_dot_actual_step=float(raw.dot(delta)),
             optimizer_input=dict(available_before_parameter_update=age,clip_factor=details['input_clip_factor'],
                                  **consume_branch_stream(branch,'input_geometry',h)),
             applied_direction=dict(moment_age=age,available_before_parameter_update=age,
                                    **consume_branch_stream(branch,'applied_direction_geometry',vectors['applied'])))
    observer.history.append(dict(pre=copy.deepcopy(pre),post=copy.deepcopy(row)))
    observer.previous_gradient=gradient.detach().clone()
    observer.previous_increment=delta.detach().clone()
    observer.completed_updates=age
    branch['last_optimizer_input']=h.detach().cpu().clone()
    branch['last_applied_direction']=vectors['applied'].detach().cpu().clone()
    branch['input_clip_count']+=int(details['input_clip_factor']<1.)
    if branch['arm']=='A6': branch['beta2_history']=details['second_moment_history']
    return row


def checkpoint(base,model,optimizer,observer,identity,data_state,branch,label):
    check_optimizer(optimizer,observer.completed_updates,branch)
    return dict(schema=1,label=label,boundary='before_current_gradient',completed_updates=observer.completed_updates,
                identity=copy.deepcopy(identity),model=copy.deepcopy(model.state_dict()),model_training=model.training,
                optimizer=copy.deepcopy(optimizer.state_dict()),observer=observer.state_dict(),rng=base.rng_state(),
                data_state=copy.deepcopy(data_state),examples_seen=observer.completed_updates*data_state['train_examples'])


def restore(base,saved,model,optimizer,observer,identity,data_state,branch):
    if saved['schema']!=1 or saved['boundary']!='before_current_gradient' or saved['identity']!=identity:
        raise ValueError('A6 source/config/input identity or boundary differs.')
    if saved['data_state']['identity']!=data_state['identity']:
        raise ValueError('A6 data/normalization identity differs.')
    for key in ['train_examples','physical','sampling','position']:
        if saved['data_state'][key]!=data_state[key]:
            raise ValueError('A6 data coverage/position differs.')
    if saved['data_state']['position']!=0:
        raise ValueError('A6 checkpoint is not a complete-objective boundary.')
    for key in ['normalization_mean','normalization_std']:
        if not torch.equal(saved['data_state'][key],data_state[key]):
            raise ValueError('A6 normalization differs.')
    if saved['completed_updates']!=saved['observer']['completed_updates']:
        raise ValueError('A6 observer/global age differs.')
    model.load_state_dict(saved['model']); model.train(saved['model_training'])
    optimizer.load_state_dict(saved['optimizer']); check_optimizer(optimizer,saved['completed_updates'],branch)
    observer.load_state_dict(saved['observer']); model.zero_grad(set_to_none=True)
    base.restore_rng(saved['rng'])
