"""H050: frozen-fork native Adam beta2 and global-gradient clipping continuations."""
import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import time
import traceback

import torch

from src.eos_tasks.representation_continuation import snapshot, restore, fingerprint

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('parent223', ROOT/'runs/223-2026-11-04-multispike-interventions/01_worker.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def cosine(a, b):
    denominator = float(a.norm()*b.norm())
    return float(a.dot(b))/denominator if denominator else None


def configure_arm(c, arm):
    for group in c['opt'].param_groups:
        assert group['lr'] == c['base'] and group['eps'] == 1e-8
        assert group['weight_decay'] == 0 and group['foreach'] is False and group['fused'] is False
        group['betas'] = (.9, arm.get('beta2', .999))


def vectors(c):
    ms, vs, ds = [], [], []
    for group in c['opt'].param_groups:
        b1, b2 = group['betas']
        for p in group['params']:
            s = c['opt'].state[p]
            step = int(s['step'])
            assert step == c['state']
            m, v = s['exp_avg'].double(), s['exp_avg_sq'].double()
            ms.append(m); vs.append(v)
            ds.append((m/(1-b1**step))/((v/(1-b2**step)).sqrt()+group['eps']))
    return tuple(w.flat(values) for values in (ms, vs, ds))


def update(c, arm, raw):
    factor = min(1., arm['clip']/float(raw.double().norm())) if arm.get('clip') and bool(raw.norm()) else 1.
    for p in c['task'].model.parameters():
        if factor < 1:
            p.grad.mul_(factor)
    actual_input = w.flat([p.grad for p in c['task'].model.parameters()]).detach().clone()
    c['opt'].step()
    c['previous_gradient'] = raw.clone()
    c['current_gradient'] = None
    c['state'] += 1
    return factor, actual_input


def checkpoint(c, path):
    tmp = path.with_suffix('.tmp')
    torch.save(snapshot(c), tmp)
    os.replace(tmp, path)


def qualify(c, saved, out, cfg):
    """Exact local 2 vs 1+1 replay; eight-step fingerprint gates cross-worker reuse."""
    restore(c, saved)
    loss, g = c['task'].gradient()
    previous = saved.get('current_gradient')
    historical = dict(fork_loss=loss, fork_gradient_exact=previous is not None and torch.equal(g.cpu(), previous))
    if previous is not None:
        historical['fork_gradient_relative_error'] = float((g.cpu().double()-previous.double()).norm()/previous.double().norm())
    raw0 = g.clone()
    start = c['state']; arm = dict(id='control')
    configure_arm(c, arm)
    begin = time.monotonic(); traces = []; direct = None
    for i in range(8):
        loss, g = c['task'].gradient()
        traces.append(dict(state=c['state'], loss=loss, norm=float(g.double().norm())))
        update(c, arm, g)
        if i == 1:
            direct = snapshot(c)
    eight = fingerprint(snapshot(c))
    restore(c, saved)
    _, g = c['task'].gradient(); update(c, arm, g)
    one = snapshot(c); restore(c, one)
    _, g = c['task'].gradient(); update(c, arm, g)
    w.equal(direct, snapshot(c))
    # The instrumented first update must match plain native Adam, including RNG.
    restore(c, saved); _, g = c['task'].gradient(); c['opt'].step()
    c['previous_gradient'] = g.clone(); c['current_gradient'] = None; c['state'] += 1
    w.equal(one, snapshot(c))
    restore(c, saved)
    result = dict(status='passed', exact_two_vs_one_plus_one=True, instrumentation_native_exact=True,
        fork_fingerprint=fingerprint(saved), eight_step_fingerprint=eight, eight_step_trace=traces,
        historical_comparison=historical, measured_seconds_per_update=(time.monotonic()-begin)/11,
        start_state=start, environment=dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda))
    w.write(out/'qualification.json', result)
    return result


def branch(c, saved, arm, folder, cfg, stop, mechanics=False):
    folder.mkdir(parents=True, exist_ok=False)
    restore(c, saved)
    previous_m, _, previous_d = vectors(c)  # Previous update used beta2=.999.
    configure_arm(c, arm)
    end = cfg['start']+2 if mechanics else cfg['end']
    start = cfg['start']; begin = time.monotonic()
    status = dict(status='running', arm=arm, start_state=start, end_state=end, mechanics_only=mechanics)
    try:
        while c['state'] <= end:
            if stop():
                checkpoint(c, folder/'recovery.pt')
                raise RuntimeError('Prespecified duration limit or stop request')
            t = c['state']; loss, g = c['task'].gradient(); gd = g.double(); oldg = c['previous_gradient'].double()
            assert math.isfinite(loss) and bool(torch.isfinite(g).all())
            gn = float(gd.norm()); den = float(oldg.dot(oldg)); c['current_gradient'] = g.clone()
            w.append(folder/'boundaries.jsonl', dict(state=t, loss=loss, raw_gradient_norm=gn,
                rho=cosine(oldg, gd), eta_kappa_hat=float(oldg.dot(oldg-gd))/den if den else None,
                parameter_norm=float(w.flat(c['task'].model.parameters()).double().norm()),
                applied_lr=c['base'] if t < end else None, beta2=arm.get('beta2', .999) if t < end else None))
            if t == end:
                checkpoint(c, folder/'recovery.pt')
                (folder/'end.pt').hardlink_to(folder/'recovery.pt')
                break
            audit = t in {start, end-1}
            before = w.flat(c['task'].model.parameters()).detach().clone()
            before_state = snapshot(c) if audit else None
            factor, actual_input = update(c, arm, g)
            after = w.flat(c['task'].model.parameters()).detach()
            delta = after.double()-before.double(); m, v, d = vectors(c)
            assert bool(torch.isfinite(after).all())
            b2 = arm.get('beta2', .999)
            w.append(folder/'metrics.jsonl', dict(state=t, update=t+1, beta1=.9, beta2=b2,
                applied_lr=c['base'], raw_gradient_norm=gn, optimizer_input_norm=float(actual_input.double().norm()),
                clip_factor=factor, moment_norm=float(m.norm()), second_moment_norm=float(v.norm()),
                adaptive_norm=float(d.norm()), moment_rho=cosine(previous_m,m), adaptive_rho=cosine(previous_d,d),
                gradient_current_moment_cosine=cosine(gd,m), gradient_adaptive_cosine=cosine(gd,d),
                incoming_gradient_prior_moment_cosine=cosine(gd,previous_m),
                bias_correction2=1-b2**(t+1), baseline_bias_correction2=1-.999**(t+1),
                actual_increment_norm=float(delta.norm()), gradient_dot_actual_step=float(gd.dot(delta)),
                raw_gradient_actual_step_cosine=cosine(gd,delta),
                adam_step_rounding_residual_norm=float((delta+c['base']*d).norm())))
            if audit:
                torch.save(dict(before=before_state, after=snapshot(c), raw_gradient=g.cpu(),
                    optimizer_input=actual_input.cpu(), arm=arm, state=t), folder/f'audit-{t:04d}.pt')
            previous_m, previous_d = m, d
            if c['state'] % cfg['checkpoint_every'] == 0:
                checkpoint(c, folder/'recovery.pt')
            elapsed = time.monotonic()-begin; completed = c['state']-start
            event = dict(arm=arm['id'], state=c['state'], loss=loss, loss_state=t,
                elapsed_seconds=elapsed, updates=completed, remaining_seconds=(end-c['state'])*elapsed/completed,
                training_units=c['task'].units, processed_units=completed*c['task'].units,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(), utc=w.stamp())
            w.append(folder/'events.jsonl', event)
            w.write(folder.parent/'progress.json', event)
            if completed % 16 == 0: print(json.dumps(event), flush=True)
        status.update(status='completed', evaluation={s:c['task'].evaluate(s) for s in ['valid','test'] if 'x_'+s in c['task'].data})
    except BaseException as error:
        status.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        status.update(completed_state=c['state'], updates=c['state']-start, elapsed_seconds=time.monotonic()-begin,
            training_units=c['task'].units, peak_cuda_bytes=torch.cuda.max_memory_allocated())
        w.write(folder/'status.json', status)
    return status


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True); parser.add_argument('--arms', nargs='+', required=True)
    parser.add_argument('--mechanics', action='store_true'); args = parser.parse_args()
    run=args.run.resolve(); cfg=json.loads((run/'config.json').read_text()); out=args.out.resolve()
    for item in json.loads((run/'input_manifest.json').read_text())['files']:
        p=ROOT/item['path']; assert p.stat().st_size==item['bytes'] and w.sha(p)==item['sha256'],str(p)
    out.mkdir(parents=True, exist_ok=False); w.configure(); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    start=time.monotonic(); stopped=[False]
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    stop=lambda:stopped[0] or time.monotonic()-start>cfg['maximum_worker_seconds'] or (run/'STOP').exists()
    manifest=dict(status='running', run=run.name, config_sha256=w.sha(run/'config.json'),
        manifest_sha256=w.sha(run/'input_manifest.json'), started_utc=w.stamp(), results=[],
        environment=dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda))
    w.write(out/'manifest.json',manifest)
    try:
        c=w.prepare(cfg['case'],json.loads((ROOT/'runs/223-2026-11-04-multispike-interventions/config.json').read_text()))
        w.load(c,run/'inputs/prefix.pt'); assert c['state']==cfg['start']
        saved=snapshot(c); checkpoint(c,out/'fork.pt')
        qualify(c,saved,out,cfg)
        for arm in cfg['arms']:
            if arm['id'] in args.arms:
                manifest['results'].append(branch(c,saved,arm,out/arm['id'],cfg,stop,args.mechanics))
                w.write(out/'manifest.json',manifest)
        assert len(manifest['results'])==len(args.arms)
        manifest['status']='completed'
    except BaseException as error:
        manifest.update(status='failed',error=repr(error),traceback=traceback.format_exc()); raise
    finally:
        manifest.update(elapsed_seconds=time.monotonic()-start,finished_utc=w.stamp())
        w.write(out/'manifest.json',manifest)
        inventory=[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=w.sha(p))
            for p in sorted(out.rglob('*')) if p.is_file() and p.name!='inventory.json']
        w.write(out/'inventory.json',dict(status=manifest['status'],files=inventory))


if __name__=='__main__': main()
