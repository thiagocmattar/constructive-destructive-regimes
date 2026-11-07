"""Three native-Adam beta2 changes from one authenticated MLP warm state."""
import importlib.util
import json
import signal
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CFG = json.loads((HERE / 'config.json').read_text())
PARENT = ROOT / CFG['parent_run']
SOURCE = ROOT / CFG['parent_worker']
spec = importlib.util.spec_from_file_location('frozen223', PARENT / '01_worker.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
torch = w.torch


def change_beta2(optimizer, beta2):
    for group in optimizer.param_groups:
        assert group['betas'][0] == .9 and group['lr'] == .001
        assert group['eps'] == 1e-8 and group['weight_decay'] == 0
        assert group['foreach'] is False and group['fused'] is False
        group['betas'] = (.9, beta2)


def cosine(a, b):
    den = float(a.norm() * b.norm())
    return float(a.dot(b)) / den if den > 0 else None


def vectors(opt, step):
    ms, vs, ds = [], [], []
    for group in opt.param_groups:
        b1, b2 = group['betas']
        for p in group['params']:
            s = opt.state[p]
            assert int(s['step']) == step
            m, v = s['exp_avg'].double(), s['exp_avg_sq'].double()
            ms.append(m)
            vs.append(v)
            ds.append((m / (1-b1**step)) / ((v / (1-b2**step)).sqrt()+group['eps']))
    return tuple(w.flat(x) for x in [ms, vs, ds])


def save(c, folder, name='recovery.pt'):
    payload = dict(run='232', config_sha256=w.sha(HERE/'config.json'),
        input_manifest_sha256=w.sha(HERE/'input_manifest.json'),
        parent_sha256=w.sha(SOURCE/'prefix/end.pt'), completed_updates=c['state'],
        beta2=c['opt'].param_groups[0]['betas'][1], model=w.cpu(c['task'].model.state_dict()),
        optimizer=w.cpu(c['opt'].state_dict()), rng=w.rng(),
        previous_gradient=w.cpu(c['previous_gradient']), current_gradient=w.cpu(c.get('current_gradient')),
        previous_moment=w.cpu(c['previous_moment']), previous_direction=w.cpu(c['previous_direction']))
    tmp = folder/(name+'.tmp')
    torch.save(payload, tmp)
    tmp.replace(folder/name)


def branch(c, beta2, folder, stop, end=None):
    end = CFG['end'] if end is None else end
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    m, _, d = vectors(c['opt'], c['state'])
    c['previous_moment'], c['previous_direction'] = m, d
    change_beta2(c['opt'], beta2)
    status = dict(status='running', beta2=beta2, start_state=c['state'], end_state=end, started_utc=w.stamp())
    w.write(folder/'status.json', status)
    try:
        while True:
            if stop():
                save(c, folder)
                raise RuntimeError('Bounded stop before next update')
            state = c['state']
            loss, g = c['task'].gradient()
            gd, previous = g.double(), c['previous_gradient'].double()
            assert w.math.isfinite(loss) and bool(torch.isfinite(g).all())
            gn, pn = float(gd.norm()), float(previous.norm())
            den = float(previous.square().sum())
            row = dict(state=state, loss=loss, raw_gradient_norm=gn, rho=cosine(previous, gd),
                eta_kappa_hat=float(previous.dot(previous-gd))/den if den else None,
                parameter_norm=float(w.flat(c['task'].model.parameters()).double().norm()),
                applied_lr=CFG['base_lr'] if state < end else None,
                beta2=beta2 if state < end else None, utc=w.stamp())
            w.append(folder/'boundaries.jsonl', row)
            c['current_gradient'] = g.clone()
            if state == CFG['start']+1:
                save(c, folder, 'first_step.pt')
            if state == end:
                save(c, folder)
                (folder/'end.pt').hardlink_to(folder/'recovery.pt')
                break
            before = w.flat(c['task'].model.parameters()).detach().double()
            c['opt'].step()
            after = w.flat(c['task'].model.parameters()).detach().double()
            assert bool(torch.isfinite(after).all())
            m, v, d = vectors(c['opt'], state+1)
            delta = after-before
            mn, dn, sn = float(m.norm()), float(d.norm()), float(delta.norm())
            bc1, bc2 = 1-.9**(state+1), 1-beta2**(state+1)
            record = dict(state=state, update=state+1, beta1=.9, beta2=beta2,
                applied_lr=CFG['base_lr'], raw_gradient_norm=gn, optimizer_input_norm=gn,
                moment_norm=mn, unbiased_moment_norm=mn/bc1, adaptive_norm=dn,
                moment_rho=cosine(c['previous_moment'], m), adaptive_rho=cosine(c['previous_direction'], d),
                gradient_current_moment_cosine=cosine(gd, m), gradient_adaptive_cosine=cosine(gd, d),
                incoming_gradient_prior_moment_cosine=cosine(gd, c['previous_moment']),
                moment_over_gradient_norm=mn/gn if gn else None,
                unbiased_moment_over_gradient_norm=mn/bc1/gn if gn else None,
                second_moment_norm=float(v.norm()), bias_correction2=bc2,
                baseline_bias_correction2=1-.999**(state+1),
                actual_increment_norm=sn, gradient_dot_actual_step=float(gd.dot(delta)),
                raw_gradient_actual_step_cosine=cosine(gd, delta),
                adam_step_rounding_residual_norm=float((delta+CFG['base_lr']*d).norm()))
            w.append(folder/'metrics.jsonl', record)
            c['previous_gradient'], c['previous_moment'], c['previous_direction'] = g.clone(), m, d
            c['state'] = state+1
            c['current_gradient'] = None
            if c['state'] % CFG['checkpoint_every'] == 0:
                save(c, folder)
            elapsed = time.monotonic()-started
            event = dict(arm=folder.name, state=c['state'], completed_updates=c['state']-CFG['start'],
                loss=loss, loss_state=state, elapsed_seconds=elapsed,
                updates_per_second=(c['state']-CFG['start'])/elapsed,
                remaining_seconds=(end-c['state'])*elapsed/(c['state']-CFG['start']),
                examples_seen=(c['state']-CFG['start'])*5000,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), utc=w.stamp())
            w.append(folder/'events.jsonl', event)
            if c['state'] % 32 == 0:
                w.write(folder/'progress.json', event)
                print(json.dumps(event), flush=True)
        status.update(status='completed', evaluation={'test': c['task'].evaluate('test')})
    except BaseException as error:
        status.update(status='failed', exception=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        status.update(completed_state=c['state'], updates=c['state']-CFG['start'],
            elapsed_seconds=time.monotonic()-started, finished_utc=w.stamp())
        w.write(folder/'status.json', status)
    return status


def main():
    for item in json.loads((HERE/'input_manifest.json').read_text())['files']:
        p = ROOT/item['path']
        assert p.stat().st_size == item['bytes'] and w.sha(p) == item['sha256'], str(p)
    w.configure()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    out = HERE/'artifacts/attempt-01/worker'
    out.mkdir(parents=True, exist_ok=False)
    started, stopped = time.monotonic(), [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    stop = lambda: stopped[0] or time.monotonic()-started > CFG['runtime_limit_seconds'] or (HERE/'STOP').exists()
    status = dict(status='running', started_utc=w.stamp(), gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, cuda=torch.version.cuda, results={})
    w.write(out/'manifest.json', status)
    try:
        parent_cfg = json.loads((PARENT/'config.json').read_text())
        c = w.prepare('mlp', parent_cfg)
        saved = w.load(c, SOURCE/'prefix/end.pt')
        _, g = c['task'].gradient()
        w.equal(g.cpu(), saved['current_gradient'])
        r = w.run(c, parent_cfg['control'], out/'qualification2', CFG['start']+2, stop)
        assert r['status'] == 'completed'
        old = torch.load(SOURCE/'qualification/direct/end.pt', map_location='cpu', weights_only=False)
        new = torch.load(out/'qualification2/end.pt', map_location='cpu', weights_only=False)
        for k in ['model','optimizer','previous_gradient','rng','current_gradient']:
            w.equal(old[k], new[k])
        c = w.prepare('mlp', parent_cfg)
        w.load(c, SOURCE/'prefix/end.pt')
        branch(c, .999, out/'instrumentation2', stop, end=CFG['start']+2)
        instrumented = torch.load(out/'instrumentation2/end.pt', map_location='cpu', weights_only=False)
        for k in ['model','optimizer','previous_gradient','rng','current_gradient']:
            w.equal(old[k], instrumented[k])
        w.write(out/'qualification.json', dict(status='passed', fork_gradient_exact=True,
            two_update_tensor_replay_exact=True, added_instrumentation_replay_exact=True,
            parent_sha256=w.sha(SOURCE/'prefix/end.pt')))
        for beta2 in CFG['beta2_values']:
            c = w.prepare('mlp', parent_cfg)
            w.load(c, SOURCE/'prefix/end.pt')
            name = 'beta2_'+str(beta2).replace('.', 'p')
            status['results'][name] = branch(c, beta2, out/name, stop)
            w.write(out/'manifest.json', status)
        status['status'] = 'completed'
    except BaseException as error:
        status.update(status='failed', exception=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        status.update(finished_utc=w.stamp(), elapsed_seconds=time.monotonic()-started)
        w.write(out/'manifest.json', status)
        inventory = [dict(path=p.relative_to(out).as_posix(), bytes=p.stat().st_size, sha256=w.sha(p))
                     for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'inventory.json']
        w.write(out/'inventory.json', dict(status=status['status'], files=inventory))


if __name__ == '__main__':
    main()
