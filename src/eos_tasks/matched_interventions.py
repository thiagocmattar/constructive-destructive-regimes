"""H051: shared past-calibrated forks and fully matched intervention histories.

Scientific inputs are frozen separately in Runs293--295. Reuses the verified
native objectives, affine cap and snapshot semantics from H048/H050.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch

from src.eos_tasks import representation_continuation as r
from src.eos_tasks import representation_policy as policy
from src.eos_tasks.conventional_continuation import vectors, qualify

w = r.w
ROOT = r.ROOT


def timed(fn):
    if torch.cuda.is_available(): torch.cuda.synchronize()
    start = time.perf_counter()
    value = fn()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return value, time.perf_counter() - start


def observe(c, cfg, targets):
    rng = w.rng()
    values, inputs, loss = policy.capture(c['task'], cfg['case'], cfg['probe'], targets)
    w.equal(rng, w.rng())
    if cfg['case'] != 'mlp':
        y = c['task'].data['y_train'][:cfg['probe']]
        y = y[y != -100] if cfg['case'] == 'transformer' else y
        readout = values['output'].gather(1, y.reshape(-1, 1)).squeeze(1)
        assert math.isclose(float(-readout.double().mean()), loss, rel_tol=3e-6, abs_tol=2e-6)
        values['correct_label_log_probability'] = readout
    return values, inputs, loss


def representation_rows(values, history, c):
    rows = {}
    old_output = history.get('output', (None, None))[0]
    terms = None
    for name, value in values.items():
        old, previous_delta = history.get(name, (None, None))
        delta = None if old is None else value.double() - old.double()
        rows[name] = dict(activation_rms=float(value.double().square().mean().sqrt()),
            change_rms=None if delta is None else float(delta.square().mean().sqrt()),
            rho=None if delta is None else policy.cosine(delta, previous_delta), coordinates=value.numel())
        history[name] = (value, delta)
    if c['case'] == 'mlp' and old_output is not None:
        y = torch.nn.functional.one_hot(c['task'].data['y_train'], 10).double()
        delta = values['output'].double() - old_output.double()
        linear = float(((old_output.double() - y) * delta).sum() / len(y))
        quadratic = float(delta.square().sum() / (2 * len(y)))
        direct = float(((values['output'].double() - y).square().sum() -
                        (old_output.double() - y).square().sum()) / (2 * len(y)))
        terms = dict(linear=linear, quadratic=quadratic, delta_loss=direct,
                     residual=direct - linear - quadratic)
    return rows, terms


def evaluation(c, state):
    rng = w.rng()
    splits = {name: c['task'].evaluate(name) for name in ['valid', 'test'] if 'x_' + name in c['task'].data}
    w.equal(rng, w.rng())
    return dict(state=state, splits=splits, selection='Descriptive only; no held-out selection')


def check_stop(stop):
    if stop():
        raise RuntimeError('Prespecified time limit or stop request; recovery retained')


def make_prefix(c, cfg, out, stop):
    out.mkdir(parents=True, exist_ok=False)
    model = c['task'].model
    _, targets = policy.affine_targets(model, cfg['case'])
    groups = policy.group_layout(model)
    pools = {n: [] for n in targets}
    start = time.monotonic()
    assert c['state'] == 0
    for s in range(cfg['fork']):
        check_stop(stop)
        (loss, g), seconds = timed(c['task'].gradient)
        c['current_gradient'] = g.clone()
        w.append(out/'prefix.jsonl', dict(state=s, loss=loss, raw_gradient_norm=float(g.double().norm()),
                                        gradient_seconds=seconds))
        if s == cfg['fork'] - 2:
            r.save(c, out/'predecessor.pt')
        capture = s >= cfg['fork'] - 8
        if capture:
            values, inputs, _ = observe(c, cfg, targets)
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
        c['opt'].step()
        if capture:
            for n in targets:
                module, weight, bias = policy.affine_parameters(model, n)
                wn = n + ('.in_proj_weight' if isinstance(module, torch.nn.MultiheadAttention) else '.weight')
                bn = n + ('.in_proj_bias' if isinstance(module, torch.nn.MultiheadAttention) else '.bias')
                dw = weight.detach().double() - before[wn].double()
                db = None if bias is None else bias.detach().double() - before[bn].double()
                pools[n].append(policy.affine_scores(module, inputs[n], dw, db).cpu().numpy())
            del before, values, inputs
        c['previous_gradient'] = g.clone()
        c['current_gradient'] = None
        c['state'] += 1
        if c['state'] % cfg['checkpoint_every'] == 0:
            r.save(c, out/'recovery.pt')
        progress = dict(phase='prefix', case=cfg['case'], state=c['state'], loss=loss,
            loss_state=s, elapsed_seconds=time.monotonic()-start,
            processed_units=c['state']*c['task'].units, training_units=c['task'].units,
            peak_cuda_bytes=torch.cuda.max_memory_allocated())
        w.write(out/'progress.json', progress)
        if c['state'] % 16 == 0:
            print(json.dumps(progress), flush=True)
    arrays = {str(i): np.stack(pools[n]) for i, n in enumerate(targets)}
    np.savez(out/'calibration.npz', **arrays)
    thresholds = {n: {'functional': {str(q): float(np.quantile(arrays[str(i)], q, method='linear'))
                    for q in cfg['quantiles']}} for i, n in enumerate(targets)}
    w.write(out/'thresholds.json', dict(modules=thresholds, module_order=targets,
        outgoing_states=list(range(cfg['fork']-8, cfg['fork'])), quantile_method='linear',
        fixed_input_scope='same predeclared probe; units/channels pooled across eight preceding untreated transitions'))
    loss, g = c['task'].gradient()
    c['current_gradient'] = g.clone()
    w.append(out/'prefix.jsonl', dict(state=c['state'], loss=loss, raw_gradient_norm=float(g.double().norm())))
    r.save(c, out/'fork.pt')
    w.write(out/'fork_evaluation.json', evaluation(c, c['state']))
    w.write(out/'layout.json', dict(targets=targets, groups=groups, probe=cfg['probe'],
        training_units=c['task'].units, parameters=sum(p.numel() for p in model.parameters())))
    w.write(out/'ready.json', dict(status='completed', fork=cfg['fork'], fork_loss=loss,
        fork_fingerprint=r.fingerprint(r.snapshot(c)), elapsed_seconds=time.monotonic()-start,
        environment=environment(), config_sha256=w.sha(cfg['_config_path'])))


def environment():
    return dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                numpy=np.__version__, deterministic=torch.are_deterministic_algorithms_enabled(),
                tf32=torch.backends.cuda.matmul.allow_tf32)


def load_fork(c, cfg, shared, out):
    ready = json.loads((shared/'ready.json').read_text())
    assert ready['environment'] == environment(), (ready['environment'], environment())
    fork = torch.load(shared/'fork.pt', map_location='cpu', weights_only=False)
    assert r.fingerprint(fork) == ready['fork_fingerprint']
    r.restore(c, torch.load(shared/'predecessor.pt', map_location='cpu', weights_only=False))
    history = {}
    previous_actual = None
    for s in range(cfg['fork']-2, cfg['fork']):
        assert c['state'] == s
        loss, g = c['task'].gradient()
        values, _, _ = observe(c, cfg, [])
        representation_rows(values, history, c)
        before = w.flat(c['task'].model.parameters()).detach().clone()
        c['opt'].step()
        previous_actual = w.flat(c['task'].model.parameters()).detach().double() - before.double()
        c['previous_gradient'] = g.clone()
        c['current_gradient'] = None
        c['state'] += 1
    _, g = c['task'].gradient()
    c['current_gradient'] = g.clone()
    w.equal(r.snapshot(c), fork)
    # Original complete fork and retained predecessor reproduce on this worker.
    q = qualify(c, fork, out, dict(start=cfg['fork']))
    w.write(out/'shared_replay.json', dict(status='passed', predecessor_to_fork_exact=True,
        fork_fingerprint=ready['fork_fingerprint'], eight_step_fingerprint=q['eight_step_fingerprint'],
        shared_ready_sha256=w.sha(shared/'ready.json')))
    return fork, history, previous_actual


def configure_arm(c, arm):
    for group in c['opt'].param_groups:
        group['lr'] = c['base'] * arm.get('lr_multiplier', 1.)
        group['betas'] = (.9, arm.get('beta2', .999))


def apply_update(c, arm, raw, inputs, thresholds, groups, targets):
    model = c['task'].model
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    before_flat = w.flat(before.values()).double()
    gn = float(raw.double().norm())
    factor = min(1., arm['clip']/gn) if arm.get('clip') and gn else 1.
    def clip():
        if factor < 1:
            for p in model.parameters(): p.grad.mul_(factor)
    _, clip_seconds = timed(clip)
    input_gradient = w.flat([p.grad for p in model.parameters()]).detach().clone()
    _, optimizer_seconds = timed(c['opt'].step)
    proposed_parameters = w.flat(model.parameters()).detach().clone()
    proposed = proposed_parameters.double() - before_flat
    cap = arm['kind'] == 'functional'
    if cap:
        gates, cap_seconds = timed(lambda: policy.apply_policy(model, before, arm, groups, {}, inputs, thresholds, targets))
    else:
        gates, cap_seconds = {}, 0.
    after = w.flat(model.parameters()).detach().clone()
    actual = after.double() - before_flat
    assert bool(torch.isfinite(after).all())
    c['previous_gradient'] = raw.clone()
    c['current_gradient'] = None
    c['state'] += 1
    return dict(before_parameters=before_flat.float(), proposed_parameters=proposed_parameters,
        actual_parameters=after, proposed=proposed, actual=actual, optimizer_input=input_gradient,
        clip_factor=factor, gates=gates, optimizer_seconds=optimizer_seconds,
        controller_seconds=clip_seconds+cap_seconds)


def branch(c, fork, original_history, previous_actual, shared, cfg, arm, folder, stop, mechanics):
    folder.mkdir(parents=True, exist_ok=False)
    r.restore(c, fork)
    configure_arm(c, arm)
    history = {n: (v.clone(), None if d is None else d.clone()) for n, (v, d) in original_history.items()}
    model = c['task'].model
    _, targets = policy.affine_targets(model, cfg['case'])
    groups = policy.group_layout(model)
    thresholds = json.loads((shared/'thresholds.json').read_text())['modules']
    start = cfg['fork']; end = start+2 if mechanics else cfg['end']
    evaluation_states = {start, start+32, (start+cfg['end'])//2, cfg['end']-32, cfg['end']}
    begin = time.monotonic()
    status = dict(status='running', arm=arm, case=cfg['case'], start=start, end=end,
                  mechanics_only=mechanics, started_utc=w.stamp(), environment=environment())
    w.write(folder/'status.json', status)
    try:
        while c['state'] <= end:
            check_stop(stop)
            s = c['state']; tick = time.monotonic()
            (loss, g), gradient_seconds = timed(c['task'].gradient)
            assert math.isfinite(loss) and bool(torch.isfinite(g).all())
            c['current_gradient'] = g.clone()
            (values, inputs, probe_loss), probe_seconds = timed(lambda: observe(c, cfg, targets if arm['kind']=='functional' else []))
            assert torch.equal(w.flat([p.grad for p in model.parameters()]), g)
            representations, terms = representation_rows(values, history, c)
            gd = g.double(); previous = c['previous_gradient'].double()
            norm = float(gd.norm()); den = float(previous.square().sum())
            w.append(folder/'boundaries.jsonl', dict(state=s, loss=loss, probe_loss=probe_loss,
                raw_gradient_norm=norm, rho=policy.cosine(gd, previous),
                eta_kappa_hat=float(previous.dot(previous-gd))/den if den else None,
                representations=representations, mse_identity=terms,
                gradient_seconds=gradient_seconds, probe_seconds=probe_seconds))
            if s in evaluation_states:
                if s == start:
                    result = json.loads((shared/'fork_evaluation.json').read_text()); eval_seconds=0.
                else:
                    result, eval_seconds = timed(lambda: evaluation(c, s))
                w.append(folder/'evaluations.jsonl', dict(**result, seconds=eval_seconds, shared=s==start))
            if s == end:
                r.save(c, folder/'end.pt')
                break
            audit = s in {start, start+32, end-1}
            before_state = r.snapshot(c) if audit else None
            result = apply_update(c, arm, g, inputs, thresholds, groups, targets)
            m, v, d = vectors(c)
            applied = result['actual']; proposed = result['proposed']
            pn, an = float(proposed.norm()), float(applied.norm())
            w.append(folder/'updates.jsonl', dict(state=s, update=s+1, lr=c['opt'].param_groups[0]['lr'],
                beta2=arm.get('beta2', .999), raw_gradient_norm=norm,
                optimizer_input_norm=float(result['optimizer_input'].double().norm()), clip_factor=result['clip_factor'],
                moment_norm=float(m.norm()), second_moment_norm=float(v.norm()), adaptive_norm=float(d.norm()),
                proposed_norm=pn, actual_norm=an, retained_norm_ratio=an/pn if pn else None,
                raw_gradient_dot_actual=float(gd.dot(applied)), raw_gradient_actual_cosine=policy.cosine(gd, applied),
                actual_update_rho=policy.cosine(applied, previous_actual), gates=result['gates'],
                gradient_seconds=gradient_seconds, probe_seconds=probe_seconds,
                optimizer_seconds=result['optimizer_seconds'], controller_seconds=result['controller_seconds'],
                step_seconds=time.monotonic()-tick))
            if audit:
                torch.save(dict(state=s, before=before_state, after=r.snapshot(c), raw_gradient=g.cpu(),
                    optimizer_input=result['optimizer_input'].cpu(), proposed_parameters=result['proposed_parameters'].cpu(),
                    gates=result['gates'], arm=arm), folder/f'audit-{s:04d}.pt')
            previous_actual = applied
            if c['state'] % cfg['checkpoint_every'] == 0:
                r.save(c, folder/'recovery.pt')
                w.write(folder/'recovery_contract.json', dict(state=c['state'],
                    diagnostic_resume='Replay this arm from the immutable shared fork; rolling representation buffers are not retained in the recovery checkpoint.'))
            elapsed = time.monotonic()-begin
            event = dict(phase='arm', case=cfg['case'], arm=arm['id'], state=c['state'], loss=loss,
                loss_state=s, updates=c['state']-start, elapsed_seconds=elapsed,
                remaining_seconds=(end-c['state'])*elapsed/(c['state']-start), training_units=c['task'].units,
                processed_units=(c['state']-start)*c['task'].units, peak_cuda_bytes=torch.cuda.max_memory_allocated(), utc=w.stamp())
            w.append(folder/'events.jsonl', event)
            w.write(folder.parent/'progress.json', event)
            if c['state'] % 16 == 0: print(json.dumps(event), flush=True)
            del result, before_state, m, v, d, inputs, values
        status.update(status='completed', completed_state=c['state'], updates=c['state']-start)
    except BaseException as error:
        r.save(c, folder/'recovery.pt')
        status.update(status='failed', completed_state=c['state'], error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        status.update(elapsed_seconds=time.monotonic()-begin, peak_cuda_bytes=torch.cuda.max_memory_allocated(), finished_utc=w.stamp())
        w.write(folder/'status.json', status)
    return status


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', choices=['prefix', 'arms'], required=True)
    p.add_argument('--shared', type=Path)
    p.add_argument('--arms', nargs='+')
    p.add_argument('--mechanics', action='store_true')
    a = p.parse_args()
    run = a.run.resolve(); out = a.out.resolve()
    cfg = json.loads((run/'config.json').read_text())
    cfg['_config_path'] = str(run/'config.json')
    for item in json.loads((run/'input_manifest.json').read_text())['files']:
        path = ROOT/item['path']
        assert path.stat().st_size == item['bytes'] and w.sha(path) == item['sha256'], str(path)
    w.configure(); torch.use_deterministic_algorithms(True); torch.backends.cudnn.deterministic=True
    assert torch.__version__.split('+')[0] == cfg['torch_version'] and torch.version.cuda == cfg['cuda_version']
    stopped = [False]; begin = time.monotonic()
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    stop = lambda: stopped[0] or time.monotonic()-begin > cfg['maximum_worker_seconds'] or (run/'STOP').exists()
    c = w.prepare(cfg['case'], json.loads((ROOT/'runs/223-2026-11-04-multispike-interventions/config.json').read_text()))
    assert c['base'] == cfg['base_lr']
    if cfg['case']=='transformer': assert bool((c['task'].data['y_train'][:cfg['probe']] != -100).all())
    if a.mode == 'prefix':
        make_prefix(c, cfg, out, stop)
        return
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='running', run=run.name, environment=environment(), config_sha256=w.sha(run/'config.json'),
        input_manifest_sha256=w.sha(run/'input_manifest.json'), arms=a.arms, results=[], started_utc=w.stamp())
    w.write(out/'manifest.json', manifest)
    try:
        shared = a.shared.resolve()
        fork, history, previous_actual = load_fork(c, cfg, shared, out)
        for name in a.arms:
            arm = next(arm for arm in cfg['arms'] if arm['id']==name)
            manifest['results'].append(branch(c, fork, history, previous_actual, shared, cfg, arm, out/name, stop, a.mechanics))
            w.write(out/'manifest.json', manifest)
        manifest['status'] = 'completed'
    except BaseException as error:
        manifest.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        manifest.update(elapsed_seconds=time.monotonic()-begin, finished_utc=w.stamp())
        w.write(out/'manifest.json', manifest)


if __name__ == '__main__':
    main()
