"""Reconstruct the native ResNet control; capture only the six displayed stages."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.eos_tasks import representation_continuation as c

spec = importlib.util.spec_from_file_location('trace_diagnostics', ROOT / 'runs/243-2026-11-07-architecture-instability-tracing/diagnostics.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
CFG = json.loads((HERE / 'config.json').read_text())


def main():
    out = HERE / 'artifacts/attempt-01'
    out.mkdir(parents=True, exist_ok=False)
    (out / 'raw').mkdir(); (out / 'checkpoints').mkdir()
    started = time.monotonic()
    status = dict(status='running', started_utc=c.w.stamp(), gpu=torch.cuda.get_device_name(),
                  torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                  config_sha256=c.w.sha(HERE / 'config.json'))
    c.w.write(out / 'manifest.json', status)
    try:
        for item in json.loads((HERE / 'input_manifest.json').read_text())['files']:
            p = ROOT / item['path']
            assert p.stat().st_size == item['bytes'] and c.w.sha(p) == item['sha256'], str(p)
        c.w.configure(); torch.use_deterministic_algorithms(True); torch.backends.cudnn.deterministic = True
        context = c.w.prepare('resnet', json.loads((ROOT / 'runs/223-2026-11-04-multispike-interventions/config.json').read_text()))
        saved = torch.load(HERE / 'inputs/fork.pt', map_location='cpu', weights_only=False)
        c.restore(context, saved)
        assert context['state'] == CFG['source_state']
        reference = {r['state']: r for r in c.w.rows(HERE / 'inputs/boundaries.jsonl')}
        audit = torch.load(HERE / 'inputs/audit_296.pt', map_location='cpu', weights_only=False)
        model = context['task'].model
        arrays = {}; metadata = {}; exact_losses = []; differences = []; times = []
        begin, end = CFG['capture_states']
        np.save(out / 'raw/fixed_inputs.npy', context['task'].data['x_train'][:CFG['probe']].cpu().numpy())
        np.save(out / 'raw/fixed_targets.npy', context['task'].data['y_train'][:CFG['probe']].cpu().numpy())
        while context['state'] <= end:
            assert time.monotonic() - started < CFG['maximum_worker_seconds']
            assert not (HERE / 'STOP').exists()
            state = context['state']; tick = time.monotonic()
            loss, gradient = context['task'].gradient()
            assert loss == reference[state]['loss'], ('loss replay mismatch', state, loss, reference[state]['loss'])
            exact_losses.append(state)
            if state == CFG['source_state']:
                assert torch.equal(gradient.cpu(), saved['current_gradient']), 'Initial gradient differs'
            if state == 296:
                assert torch.equal(c.flat(model.parameters()).detach().cpu(), audit['before_parameters'])
                assert torch.equal(gradient.cpu(), audit['raw_gradient'])
            context['current_gradient'] = gradient.clone()
            norm = float(gradient.double().norm())
            np.testing.assert_allclose(norm, reference[state]['raw_gradient_norm'], rtol=2e-13)
            previous = context['previous_gradient']
            rho = float(gradient.double() @ previous.double()) / (norm * float(previous.double().norm()))
            probe = None
            if state >= begin:
                rng = c.w.rng(); grads = [p.grad.clone() for p in model.parameters()]
                buffers = {n: b.clone() for n, b in model.named_buffers()}
                values, _, probe = c.policy.capture(context['task'], 'resnet', CFG['probe'], [])
                c.w.equal(rng, c.w.rng())
                for p, g in zip(model.parameters(), grads): assert torch.equal(p.grad, g)
                for n, b in model.named_buffers(): assert torch.equal(b, buffers[n])
                np.testing.assert_allclose(probe, reference[state]['probe_loss'], rtol=0, atol=0)
                if state == begin:
                    with torch.no_grad():
                        value, output = context['task'].loss(context['task'].data['x_train'][:CFG['probe']], context['task'].data['y_train'][:CFG['probe']])
                    assert torch.equal(values['output'], output.log_softmax(-1))
                    assert probe == float(value.double() / CFG['probe'])
                torch.save(c.w.cpu(model.state_dict()), out / 'checkpoints' / f'{state}.pt')
                values['raw_gradient'] = gradient
                for name, value in values.items():
                    assert name in [*CFG['stages'], 'raw_gradient']
                    a = value.detach().cpu().numpy()
                    if name not in arrays:
                        metadata[name] = dict(path=name + '.npy', shape=list(a.shape), coordinates=int(a.size))
                        arrays[name] = np.lib.format.open_memmap(out / 'raw' / (name + '.npy'), mode='w+', dtype=np.float32, shape=(end-begin+1, *a.shape))
                    arrays[name][state-begin] = a
                del values
            row = dict(state=state, loss=loss, raw_gradient_norm=norm, rho=rho, probe_loss=probe,
                       elapsed_seconds=time.monotonic()-started, seconds=time.monotonic()-tick)
            times.append(row['seconds']); c.w.append(out / 'boundaries.jsonl', row)
            c.w.write(out / 'progress.json', dict(phase='capture' if state >= begin else 'replay', **row,
                remaining_seconds=float(np.median(times[-16:])) * (end-state), peak_cuda_bytes=torch.cuda.max_memory_allocated()))
            if state % 8 == 0 or state == end: print(json.dumps(row), flush=True)
            if state == end: break
            context['opt'].step(); context['previous_gradient'] = gradient.clone(); context['state'] += 1
            if state % 32 == 0: c.save(context, out / 'recovery.pt')
        c.save(context, out / 'end.pt')
        for array in arrays.values(): array.flush()
        c.w.write(out / 'stages.json', metadata)
        grams = {}; levels = {}
        with threadpool_limits(limits=4):
            for name, meta in metadata.items():
                path = out / 'raw' / meta['path']
                if name == 'raw_gradient':
                    # g_389..g_449 caused the increments observed at s=390..450.
                    # g_388 is only the predecessor for adjacent cosine.
                    fit_path = out / 'raw/gradient_fit.npy'
                    np.save(fit_path, arrays[name][:-1]); path = fit_path
                gram, level = d.gram_reduce(path, raw_gradient=name == 'raw_gradient')
                grams[name] = gram; levels[name] = level
                p = meta['coordinates']; norms = np.sqrt(np.diag(gram))
                cosines = np.diag(gram, 1) / (norms[1:] * norms[:-1])
                states = range(389, 450) if name == 'raw_gradient' else range(390, 451)
                if name == 'raw_gradient':
                    expected_rho = [reference[s]['rho'] for s in states]
                    expected_rms = [reference[s]['raw_gradient_norm'] / np.sqrt(p) for s in states]
                else:
                    expected_rho = [reference[s]['representations'][name]['rho'] for s in states]
                    expected_rms = [reference[s]['representations'][name]['change_rms'] for s in states]
                np.testing.assert_allclose(cosines, expected_rho, rtol=2e-10, atol=2e-12)
                np.testing.assert_allclose(norms[1:] / np.sqrt(p), expected_rms, rtol=2e-10, atol=2e-12)
                differences.append(dict(stage=name, cosine_max_abs_error=float(np.max(np.abs(cosines-expected_rho))), rms_max_abs_error=float(np.max(np.abs(norms[1:]/np.sqrt(p)-expected_rms)))))
                c.w.write(out / 'progress.json', dict(phase='gram', stage=name, completed=len(grams), total=len(metadata), elapsed_seconds=time.monotonic()-started))
                print(json.dumps(dict(phase='gram', stage=name)), flush=True)
        np.savez(out / 'grams.npz', **grams); np.savez(out / 'levels.npz', **levels)
        c.w.write(out / 'qualification.json', dict(status='passed', exact_loss_states=exact_losses,
            initial_gradient_exact=True, audit296_parameters_and_gradient_exact=True,
            observer_rng_gradients_buffers_unchanged=True, plain_output_parity=True,
            fixed_original_probe=True, source_scalar_agreement=differences))
        status.update(status='completed', completed_state=context['state'], replay_updates=context['state']-CFG['source_state'],
            capture_states=CFG['capture_states'], completed_steps=CFG['completed_steps'], gradient_states=CFG['gradient_states'],
            peak_cuda_bytes=torch.cuda.max_memory_allocated(), stages=len(metadata), training_examples=50000, probe_examples=1000)
    except BaseException as error:
        status.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        status.update(elapsed_seconds=time.monotonic()-started, finished_utc=c.w.stamp())
        c.w.write(out / 'manifest.json', status)


if __name__ == '__main__':
    main()
