"""Run218: A0/A1/A2/A4/A8 branches; no training on import or --help."""
import argparse
import copy
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE_HERE = ROOT/'runs/217-2026-11-03-mlp-adam-control-foundation'
BASE_SHA256 = '01a434fc1bb42090d1ccedcf7cd86b4c022b117e2d5dd78bf06e9bd71ba7bdd0'
if hashlib.sha256((BASE_HERE/'01_control.py').read_bytes()).hexdigest() != BASE_SHA256:
    raise ValueError('Accepted Run217 foundation source changed.')
spec = importlib.util.spec_from_file_location('run217_foundation', BASE_HERE/'01_control.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
from src.eos_tasks.adam_diagnostics import adam_streams

SUPPORTED = ('A0', 'A1', 'A2', 'A4', 'A8')
BASE_LR = .001


def require_arm(arm):
    if arm not in SUPPORTED:
        raise NotImplementedError(f'Arm {arm} is unsupported; implemented: {SUPPORTED}.')


def native_source_hash():
    source = inspect.getsource(importlib.import_module('torch.optim.adam')._single_tensor_adam)
    return hashlib.sha256(source.encode()).hexdigest()


def fixed_proposal_cap(observer, window=64):
    """Strictly preceding gradient states [f-64,f-1], ideal -base_lr*d norms."""
    f = observer.completed_updates
    if f < window or len(observer.history) != f:
        raise ValueError('A4 requires 64 complete common-prefix proposal norms.')
    values = []
    for state, row in zip(range(f-window, f), observer.history[-window:]):
        if (row['pre']['state'] != state or row['post']['gradient_state'] != state or
                row['post']['update'] != state+1):
            raise ValueError('A4 prior-history clock differs.')
        value = BASE_LR*row['post']['streams']['adaptive']['norm']
        if not math.isfinite(value) or value < 0:
            raise ValueError('A4 prior ideal proposal norm is invalid.')
        values.append(value)
    cap = float(np.median(np.asarray(values, dtype=np.float64)))
    if cap <= 0:
        raise ValueError('A4 median is zero: threshold rule unusable for this fork.')
    return dict(cap=cap, prior_states=[f-window, f-1], count=window, proposal_norms=values,
                definition='norm(-base_lr*d), d reconstructed in FP64 from ordinary FP32 moments')


@torch.no_grad()
def choose_action(arm, gradient, direction, previous_gradient=None, previous_increment=None, cap=None):
    """Only current raw/staged direction and PRIOR history; no outgoing increment."""
    require_arm(arm)
    row = dict(arm=arm, alpha=BASE_LR, alpha_hat=None, kappa_sec=None,
               fallback=False, fallback_reason=None, floor_active=False, upper_cap_active=False,
               proposal_clip_factor=1., secant_tolerance=1e-8, loss_decrease_guarantee=False)
    if arm in ('A0', 'A1', 'A2'):
        row['alpha'] = BASE_LR*{'A0': 1., 'A1': .5, 'A2': .25}[arm]
        return row
    g, d = gradient.detach().double(), direction.detach().double()
    if arm == 'A4':
        if cap is None or not math.isfinite(cap) or cap <= 0:
            raise ValueError('A4 requires a fixed positive finite common-prefix proposal cap.')
        proposal_norm = float((BASE_LR*d).norm())
        if not math.isfinite(proposal_norm):
            raise ValueError('Nonfinite A4 ideal proposal.')
        factor = 1. if proposal_norm == 0 else min(1., cap/proposal_norm)
        row.update(alpha=BASE_LR*factor, proposal_clip_factor=factor, cap=cap,
                   ordinary_ideal_proposal_norm=proposal_norm)
        return row

    def fallback(reason):
        row.update(fallback=True, fallback_reason=reason)
        return row

    if previous_gradient is None or previous_increment is None:
        return fallback('missing_previous_gradient_or_actual_increment')
    previous, s = previous_gradient.detach().double(), previous_increment.detach().double()
    for name, value in [('current_gradient', g), ('current_direction', d), ('previous_gradient', previous), ('previous_actual_increment', s)]:
        if value.shape != g.shape or value.ndim != 1:
            return fallback('invalid_' + name + '_shape')
        if not bool(torch.isfinite(value).all()):
            return fallback('nonfinite_' + name)
    y = g-previous
    if not bool(torch.isfinite(y).all()):
        return fallback('nonfinite_gradient_difference')
    reductions = dict(s_dot_s=float(s.dot(s)), d_dot_d=float(d.dot(d)), s_dot_y=float(s.dot(y)),
                      g_dot_d=float(g.dot(d)), s_norm=float(s.norm()), y_norm=float(y.norm()))
    row.update({k: v if math.isfinite(v) else None for k, v in reductions.items()})
    if not all(math.isfinite(v) for v in reductions.values()):
        return fallback('nonfinite_reduction')
    ss, dd, sy, gd = [reductions[k] for k in ['s_dot_s', 'd_dot_d', 's_dot_y', 'g_dot_d']]
    if ss == 0:
        return fallback('zero_previous_actual_increment')
    if dd == 0:
        return fallback('zero_current_direction')
    if gd <= 0:
        return fallback('non_descending_current_direction')
    tolerance = 1e-8*reductions['s_norm']*reductions['y_norm']
    if not math.isfinite(tolerance):
        return fallback('nonfinite_secant_tolerance')
    row['secant_alignment_threshold'] = tolerance
    if sy <= tolerance:
        return fallback('insufficient_positive_secant_alignment')
    kappa = sy/ss
    denominator = kappa*dd
    if not math.isfinite(kappa) or not math.isfinite(denominator) or denominator <= 0:
        return fallback('invalid_proposal_denominator')
    alpha = gd/denominator
    if not math.isfinite(alpha):
        return fallback('nonfinite_alpha_hat')
    row.update(kappa_sec=kappa, alpha_hat=alpha, alpha=min(BASE_LR, max(.05*BASE_LR, alpha)),
               floor_active=alpha < .05*BASE_LR, upper_cap_active=alpha > BASE_LR)
    return row


def transformation(arm, observer):
    require_arm(arm)
    common = dict(arm=arm, base_lr=BASE_LR, ordinary_moments=True, reset_optimizer_age=False)
    if arm in ('A0', 'A1', 'A2'):
        return dict(common, action='fixed_lr_multiplier', multiplier={'A0': 1., 'A1': .5, 'A2': .25}[arm])
    if arm == 'A4':
        return dict(common, action='complete_ideal_adam_proposal_clip', threshold=fixed_proposal_cap(observer))
    return dict(common, action='cap_only_transported_secant', secant_alignment_tolerance=1e-8,
                alpha_floor_multiple=.05, alpha_cap_multiple=1., invalid_action='base_lr_with_reason')


def implementation_identity():
    cfg = json.loads((HERE/'config.json').read_text())
    manifest = json.loads((HERE/'input_manifest.json').read_text())
    base.verify_inputs(ROOT, manifest)
    if native_source_hash() != cfg['native_adam_source_sha256']:
        raise ValueError('Native Adam source differs from the verified staging/application reference.')
    return dict(config_sha256=base.file_hash(HERE/'config.json'), manifest_sha256=base.file_hash(HERE/'input_manifest.json'),
                files=manifest['files'], foundation_commit='9af24113bebd3d76a3bad5cabda78690fa69b7c9',
                native_adam_source_sha256=native_source_hash())


def branch_from_parent(parent, parent_sha256, expected_parent_identity, model, optimizer, observer,
                       data_state, arm, implementation, expected_fork):
    require_arm(arm)
    if expected_parent_identity.get('arm') != 'A0':
        raise ValueError('A common parent must have the validated untreated A0 identity.')
    if base.file_hash(parent) != parent_sha256:
        raise ValueError('Parent file SHA-256 differs.')
    # Exact foundation identity/recipe checks; never relax them for a branch.
    saved = base.load_checkpoint(parent, model, optimizer, observer, expected_parent_identity, data_state)
    if saved.get('checkpoint_kind') == 'branch' or saved['completed_updates'] != expected_fork:
        raise ValueError('Parent is not the declared common-prefix fork.')
    if base.file_hash(parent) != parent_sha256:
        raise ValueError('Parent changed during load.')
    spec = transformation(arm, observer)
    identity = dict(run_id='218', checkpoint_kind='branch', arm=arm,
                    parent=dict(sha256=parent_sha256, bytes=Path(parent).stat().st_size,
                                completed_updates=expected_fork, scientific_identity=copy.deepcopy(expected_parent_identity)),
                    transformation=spec, implementation=copy.deepcopy(implementation))
    origin = base.flat(model.parameters()).detach().double().cpu()
    branch = dict(arm=arm, fork_update=expected_fork, fork_parameters=origin.clone(), updates=0,
                  cap=spec['threshold']['cap'] if arm == 'A4' else None, path_length=0.,
                  actual_total=torch.zeros_like(origin), ideal_total=torch.zeros_like(origin),
                  actions=[], fallback_count=0, floor_count=0, upper_cap_count=0)
    return identity, branch


@torch.no_grad()
def apply_staged_alpha(optimizer, alpha):
    """Only the FP32 noncapturable Adam parameter op; moments are already staged.

    Mirrors the pinned torch.optim.adam single-tensor parameter expression.
    Uses stored FP32 m/v and native denominator rounding, with NO moment update.
    """
    for group in optimizer.param_groups:
        beta1, beta2 = group['betas']
        for p in group['params']:
            state = optimizer.state[p]
            age = float(state['step'])
            denominator = (state['exp_avg_sq'].sqrt() / ((1-beta2**age)**.5)).add_(group['eps'])
            p.addcdiv_(state['exp_avg'], denominator, value=-alpha/(1-beta1**age))


@torch.no_grad()
def update(optimizer, observer, gradient, branch, observation=None):
    arm = branch['arm']; require_arm(arm)
    completed = observer.completed_updates
    if branch['updates'] != completed-branch['fork_update']:
        raise ValueError('Branch/observer update clock differs.')
    base.check_optimizer(optimizer, completed)
    if len(optimizer.param_groups) != 1 or any(optimizer.param_groups[0].get(k, False) for k in ['capturable', 'differentiable', 'decoupled_weight_decay']):
        raise ValueError('Requires one ordinary noncapturable/nondifferentiable Adam group.')
    parameters = optimizer.param_groups[0]['params']
    if gradient.dtype != torch.float32 or not bool(torch.isfinite(gradient).all()):
        raise ValueError('Requires a finite FP32 raw gradient.')
    if not torch.equal(gradient, base.flat([p.grad.detach() for p in parameters])):
        raise ValueError('Captured raw gradient differs from optimizer input.')
    pre = observer.observe(completed, gradient) if observation is None else observation
    if pre['state'] != completed:
        raise ValueError('Supplied pre-update observation clock differs.')
    before = base.flat(parameters).detach().double()
    group = optimizer.param_groups[0]
    if arm == 'A0':
        # The accepted reference remains a native, unmodified step.
        optimizer.step()
        direction = adam_streams(optimizer, completed+1)['adaptive']
        action = choose_action(arm, gradient, direction)
    else:
        # Exactly one native recurrence, before the scalar decision. No outgoing motion.
        group['lr'] = 0.
        try:
            optimizer.step()
        finally:
            group['lr'] = BASE_LR
        if not torch.equal(before, base.flat(parameters).detach().double()):
            raise ValueError('Zero-LR moment staging moved parameters.')
        direction = adam_streams(optimizer, completed+1)['adaptive']
        action = choose_action(arm, gradient, direction, observer.previous_gradient,
                               observer.previous_increment, branch['cap'])
        apply_staged_alpha(optimizer, action['alpha'])
    after = base.flat(parameters).detach().double()
    if not bool(torch.isfinite(after).all()):
        raise ValueError('Nonfinite parameters after action.')
    # The foundation's current-step rounding audit needs this step's chosen LR.
    # Restore the nominal A0 group recipe at every checkpoint/step boundary.
    group['lr'] = action['alpha']
    try:
        post = observer.accept(gradient, optimizer, before, after, pre)
    finally:
        group['lr'] = BASE_LR
    # The inherited constant-LR cumulative expression is invalid for varying alpha.
    post.pop('cumulative_step_rounding_residual_norm')
    observer.history[-1]['post'].pop('cumulative_step_rounding_residual_norm')
    delta, ideal = after-before, -action['alpha']*direction
    action.update(gradient_state=completed, direction_moment_age=completed+1,
                  alpha_available_before_parameter_update=completed+1,
                  direction_capture_phase='after_native_A0_apply' if arm == 'A0' else 'after_moment_staging_before_apply',
                  decision_requires_outgoing_increment=False,
                  ordinary_ideal_proposal_norm=float((BASE_LR*direction).norm()),
                  chosen_ideal_proposal_norm=float(ideal.norm()))
    audit = dict(available_after_update=completed+1, actual_increment_norm=float(delta.norm()),
                 ideal_actual_rounding_residual_norm=float((delta-ideal).norm()))
    if arm == 'A4':
        audit['actual_cap_excess_due_to_rounding'] = max(0., float(delta.norm())-branch['cap'])
    branch['updates'] += 1
    branch['path_length'] += float(delta.norm())
    branch['actual_total'] += delta.cpu()
    branch['ideal_total'] += ideal.cpu()
    branch['actions'].append(copy.deepcopy(action))
    for counter, flag in [('fallback_count', 'fallback'), ('floor_count', 'floor_active'), ('upper_cap_count', 'upper_cap_active')]:
        branch[counter] += int(action[flag])
    displacement = float((after.cpu()-branch['fork_parameters']).norm())
    retention = displacement/branch['path_length'] if branch['path_length'] > 0 else None
    post.update(action=copy.deepcopy(action), action_audit=audit, branch_updates=branch['updates'],
                branch_displacement_norm=displacement, branch_path_length=branch['path_length'],
                branch_actual_retention=retention, branch_actual_retention_undefined_reason='zero_path_length' if retention is None else None,
                branch_cumulative_rounding_residual_norm=float((branch['actual_total']-branch['ideal_total']).norm()),
                fallback_fraction=branch['fallback_count']/branch['updates'],
                clamp_fraction=(branch['floor_count']+branch['upper_cap_count'])/branch['updates'],
                floor_fraction=branch['floor_count']/branch['updates'], upper_cap_fraction=branch['upper_cap_count']/branch['updates'])
    observer.history[-1]['post'] = copy.deepcopy(post)
    return pre, post


def save_branch(path, model, optimizer, observer, identity, data_state, branch, label, immutable):
    require_arm(identity['arm'])
    if identity['arm'] != branch['arm'] or branch['updates'] != observer.completed_updates-branch['fork_update']:
        raise ValueError('Branch identity/clock differs.')
    expected_cap = identity['transformation']['threshold']['cap'] if identity['arm'] == 'A4' else None
    if branch['cap'] != expected_cap:
        raise ValueError('Fixed branch cap differs from its declared transformation.')
    payload = base.checkpoint(model, optimizer, observer, identity, data_state, label)
    payload.update(checkpoint_kind='branch', branch_state=copy.deepcopy(branch))
    base.save_checkpoint(path, payload, immutable=immutable)


def load_branch(path, sha256, model, optimizer, observer, identity, data_state, expected_origin):
    require_arm(identity['arm'])
    if base.file_hash(path) != sha256:
        raise ValueError('Branch file SHA-256 differs.')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('checkpoint_kind') != 'branch' or saved['identity'] != identity:
        raise ValueError('Branch arm/parent/transformation/implementation identity differs.')
    state = saved['branch_state']
    if (state['arm'] != identity['arm'] or state['fork_update'] != identity['parent']['completed_updates'] or
            state['updates'] != saved['completed_updates']-state['fork_update'] or len(state['actions']) != state['updates'] or
            not torch.equal(state['fork_parameters'], expected_origin['fork_parameters']) or state['cap'] != expected_origin['cap']):
        raise ValueError('Branch state/parent origin/clock differs.')
    if any(row['arm'] != identity['arm'] or row['gradient_state'] != state['fork_update']+i for i, row in enumerate(state['actions'])):
        raise ValueError('Branch action history differs.')
    base.load_checkpoint(path, model, optimizer, observer, identity, data_state)
    if base.file_hash(path) != sha256:
        raise ValueError('Branch changed during load.')
    return copy.deepcopy(state)


def run_branch(task, optimizer, observer, branch, identity, data_state, output, updates,
               checkpoint_every=32, stop_requested=lambda: False):
    output = Path(output)
    if checkpoint_every < 1:
        raise ValueError('Invalid checkpoint cadence.')
    if not 0 <= branch['updates'] <= updates <= 128:
        raise ValueError('Branch horizon must be within [current_updates,128].')
    output.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    status = dict(status='running', run_id='218', arm=identity['arm'], requested_branch_updates=updates,
                  scientific_status='implementation_control_unqualified', started_utc=base.stamp())
    base.write_json(output/'resolved_identity.json', identity)
    base.write_json(output/'status.json', status)

    def save(label=None):
        target = output/'recovery.pt' if label is None else output/'checkpoints'/f'{label}-state-{observer.completed_updates:06d}.pt'
        save_branch(target, task.model, optimizer, observer, identity, data_state, branch, label or 'rolling_recovery', label is not None)
        if label is not None:
            base.append_json(output/'snapshots.jsonl', dict(label=label, path=target.relative_to(output).as_posix(),
                            state=observer.completed_updates, bytes=target.stat().st_size, sha256=base.file_hash(target)))
    try:
        require_arm(identity['arm'])
        save('branch_start'); save()
        while True:
            if branch['updates'] == updates:
                save('endpoint'); save()
            if stop_requested():
                save(); status['status'] = 'interrupted'; break
            loss, gradient = task.gradient()
            if not math.isfinite(loss) or gradient.dtype != torch.float32 or not bool(torch.isfinite(gradient).all()):
                raise ValueError('Requires finite full-objective loss and FP32 raw gradient.')
            pre = observer.observe(observer.completed_updates, gradient)
            pre.update(loss=loss, utc=base.stamp())
            base.append_json(output/'boundaries.jsonl', pre)
            if branch['updates'] == updates:
                status['status'] = 'completed'; break
            _, post = update(optimizer, observer, gradient, branch, pre)
            base.append_json(output/'actions.jsonl', post['action'])
            base.append_json(output/'metrics.jsonl', post)
            if branch['updates'] % checkpoint_every == 0:
                save()
            progress = dict(utc=base.stamp(), completed_updates=observer.completed_updates, branch_updates=branch['updates'],
                            loss=loss, loss_state=observer.completed_updates-1,
                            examples_seen=observer.completed_updates*task.units, elapsed_seconds=time.perf_counter()-begin)
            base.append_json(output/'events.jsonl', progress)
            base.write_json(output/'progress.json', progress)
            status.update(completed_updates=observer.completed_updates, branch_updates=branch['updates'])
            base.write_json(output/'status.json', status)
    except BaseException as error:
        status.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      exception=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        status.update(completed_updates=observer.completed_updates, branch_updates=branch['updates'],
                      finished_utc=base.stamp(), elapsed_seconds=time.perf_counter()-begin)
        base.write_json(output/'status.json', status)
        base.write_json(output/'manifest.json', dict(status, identity=identity))
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True, choices=SUPPORTED)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--parent-sha256', required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--resume-sha256')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=128)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if (args.resume is None) != (args.resume_sha256 is None):
        parser.error('--resume and --resume-sha256 must be supplied together.')
    if not 1 <= args.updates <= 128:
        parser.error('--updates must be within [1,128].')
    args.output.mkdir(parents=True, exist_ok=False)
    base.write_json(args.output/'status.json', dict(status='preparing', neural_training_started=False))
    try:
        implementation = implementation_identity()
        cfg = json.loads((HERE/'config.json').read_text())
        _, task, optimizer, observer, parent_identity, data_state = base.prepare(ROOT, BASE_HERE, torch.device(args.device))
        identity, branch = branch_from_parent(args.parent, args.parent_sha256, parent_identity,
                    task.model, optimizer, observer, data_state, args.arm, implementation, cfg['fork_update'])
        if args.resume:
            branch = load_branch(args.resume, args.resume_sha256, task.model, optimizer, observer, identity, data_state, branch)
        base.write_json(args.output/'resolved_config.json', cfg)
        base.write_json(args.output/'parent_file.json', dict(path=str(args.parent.resolve()), sha256=args.parent_sha256))
        stopping = [False]
        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, lambda *_: stopping.__setitem__(0, True))
        run_branch(task, optimizer, observer, branch, identity, data_state, args.output, args.updates,
                   checkpoint_every=cfg['checkpoint_every'], stop_requested=lambda: stopping[0] or (HERE/'STOP').exists())
    except BaseException as error:
        status = json.loads((args.output/'status.json').read_text())
        if status['status'] not in ['failed', 'interrupted']:
            status.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                          exception=repr(error), traceback=traceback.format_exc(), finished_utc=base.stamp())
            base.write_json(args.output/'status.json', status)
            base.write_json(args.output/'manifest.json', dict(status, run_id='218'))
        raise


if __name__ == '__main__':
    main()
