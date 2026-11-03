"""Run217 A0 only: pre-update observation and durable boundary checkpoints.

No training occurs on import. The CLI requires an explicit output and horizon.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import sys
import tempfile
import time
import traceback

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import append_json, file_hash, flat, stamp, tensor_hash, write_json
from src.eos_tasks.adam_diagnostics import Diagnostics
from src.eos_tasks.production import ProductionTask
from src.eos_tasks.task import configure, model_for


def quotient(row, key, numerator, denominator, reason):
    """Signed denominator allowed. No epsilon substitution or hidden clamping."""
    row[key] = None if denominator == 0 else numerator / denominator
    row[key + '_undefined_reason'] = reason if denominator == 0 else None


class Observer:
    """At boundary t retain only g_(t-1), s_(t-1), and completed history.

    s is the FP64 subtraction of actual stored FP32 endpoints, not -lr*d.
    observe() is read-only; accept() is called only after ordinary Adam.step().
    """
    def __init__(self, count, device):
        self.completed_updates = 0
        self.previous_gradient = None
        self.previous_increment = None
        self.history = []
        self.post = Diagnostics(count, device)

    @torch.no_grad()
    def observe(self, state, gradient):
        if state != self.completed_updates:
            raise ValueError('Pre-update observer clock differs.')
        g = gradient.detach().double()
        if g.ndim != 1 or g.numel() != self.post.streams['raw'].total.numel():
            raise ValueError('Gradient scope differs.')
        if not bool(torch.isfinite(g).all()):
            raise ValueError('Nonfinite raw gradient.')
        row = dict(state=state, available_before_update=state + 1,
                   previous_gradient_state=None if state == 0 else state - 1,
                   completed_step=None if state == 0 else [state - 1, state],
                   gradient_norm=float(g.norm()))
        keys = ['previous_gradient_norm', 'previous_increment_norm', 'gradient_difference_norm',
                'previous_increment_dot_y', 'previous_increment_dot_previous_gradient',
                'previous_increment_squared_norm', 'raw_adjacent_cosine',
                'raw_projected_change', 'completed_step_q', 'completed_step_kappa_sec']
        if self.previous_gradient is None:
            for key in keys:
                row[key] = None
                row[key + '_undefined_reason'] = 'no_completed_step'
            return row
        previous = self.previous_gradient.double()
        s = self.previous_increment.double()
        # Subtract rounded raw gradients in FP64, then reduce in FP64.
        y = g - previous
        gg, ss = float(previous.dot(previous)), float(s.dot(s))
        sy, sg = float(s.dot(y)), float(s.dot(previous))
        row.update(previous_gradient_norm=float(previous.norm()),
                   previous_increment_norm=float(s.norm()), gradient_difference_norm=float(y.norm()),
                   previous_increment_dot_y=sy, previous_increment_dot_previous_gradient=sg,
                   previous_increment_squared_norm=ss)
        quotient(row, 'raw_adjacent_cosine', float(previous.dot(g)), float(previous.norm()*g.norm()), 'zero_gradient_norm')
        quotient(row, 'raw_projected_change', -float(previous.dot(y)), gg, 'zero_previous_gradient')
        quotient(row, 'completed_step_q', -sy, sg, 'zero_previous_step_slope')
        quotient(row, 'completed_step_kappa_sec', sy, ss, 'zero_previous_increment')
        return row

    @torch.no_grad()
    def accept(self, gradient, optimizer, before, after, observation):
        if observation['state'] != self.completed_updates:
            raise ValueError('Post-update observer clock differs.')
        step = self.completed_updates + 1
        stats = self.post.consume(gradient, optimizer, step, before, after)
        row = dict(update=step, gradient_state=step - 1, available_after_update=step,
                   actual_increment_definition='FP64(theta_after_FP32)-FP64(theta_before_FP32)', **stats)
        # History owns the rows; callers cannot mutate retained diagnostics.
        self.history.append(dict(pre=copy.deepcopy(observation), post=copy.deepcopy(row)))
        self.previous_gradient = gradient.detach().clone()
        self.previous_increment = (after.detach().double() - before.detach().double()).clone()
        self.completed_updates = step
        return row

    def state_dict(self):
        return dict(completed_updates=self.completed_updates,
                    previous_gradient=None if self.previous_gradient is None else self.previous_gradient.cpu().clone(),
                    previous_increment=None if self.previous_increment is None else self.previous_increment.cpu().clone(),
                    history=copy.deepcopy(self.history), post=self.post.state_dict())

    def load_state_dict(self, saved):
        self.completed_updates = saved['completed_updates']
        device = self.post.streams['raw'].total.device
        for name in ['previous_gradient', 'previous_increment']:
            value = saved[name]
            setattr(self, name, None if value is None else value.to(device).clone())
        self.history = copy.deepcopy(saved['history'])
        self.post.load_state_dict(saved['post'])
        if (len(self.history) != self.completed_updates or
                any(stream.count != self.completed_updates for stream in self.post.streams.values())):
            raise ValueError('Checkpoint diagnostic clocks differ.')
        if (self.previous_gradient is None) != (self.completed_updates == 0) or (self.previous_increment is None) != (self.completed_updates == 0):
            raise ValueError('Checkpoint missing previous raw gradient/increment.')


def make_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8,
                            weight_decay=0, amsgrad=False, foreach=False, fused=False)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(saved):
    random.setstate(saved['python'])
    np.random.set_state(saved['numpy'])
    torch.set_rng_state(saved['torch'])
    if saved['cuda']:
        torch.cuda.set_rng_state_all(saved['cuda'])


def check_optimizer(optimizer, completed):
    for group in optimizer.param_groups:
        expected = dict(lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0,
                        amsgrad=False, foreach=False, fused=False, maximize=False)
        if any(group[k] != value for k, value in expected.items()):
            raise ValueError('A0 optimizer recipe changed.')
        for p in group['params']:
            if p.dtype != torch.float32:
                raise ValueError('A0 requires FP32 parameters.')
            state = optimizer.state.get(p, {})
            if completed == 0 and state:
                raise ValueError('Fresh Adam must have no moments.')
            if completed and (int(state['step']) != completed or
                              any(state[k].dtype != torch.float32 for k in ['exp_avg', 'exp_avg_sq'])):
                raise ValueError('Adam age/precision differs.')


def checkpoint(model, optimizer, observer, identity, data_state, label):
    completed = observer.completed_updates
    check_optimizer(optimizer, completed)
    return dict(schema=1, label=label, boundary='before_current_gradient', completed_updates=completed,
                identity=copy.deepcopy(identity), model=copy.deepcopy(model.state_dict()),
                model_training=model.training, optimizer=copy.deepcopy(optimizer.state_dict()),
                observer=observer.state_dict(), rng=rng_state(), data_state=copy.deepcopy(data_state),
                examples_seen=completed*data_state['train_examples'])


def save_checkpoint(path, payload, *, immutable):
    """Atomic publish. Exclusive hard-link creation forbids replacing labeled states.

    Write/fsync a sibling first. No fallback to replacement for immutable files.
    Rolling recovery alone uses os.replace. A failed write preserves old bytes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path, model, optimizer, observer, identity, data_state):
    # Only load trusted, locally created checkpoints; RNG includes Python/NumPy state.
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['schema'] != 1 or saved['boundary'] != 'before_current_gradient':
        raise ValueError('Checkpoint schema/boundary differs.')
    if saved['identity'] != identity:
        raise ValueError('Checkpoint source/config/input/runtime identity differs.')
    if saved['data_state']['identity'] != data_state['identity']:
        raise ValueError('Checkpoint data/normalization identity differs.')
    for key in ['train_examples', 'physical', 'sampling', 'position']:
        if saved['data_state'][key] != data_state[key]:
            raise ValueError('Checkpoint data coverage/position differs.')
    if saved['data_state']['position'] != 0:
        raise ValueError('Checkpoint is not a complete-objective boundary.')
    for key in ['normalization_mean', 'normalization_std']:
        if not torch.equal(saved['data_state'][key], data_state[key]):
            raise ValueError('Checkpoint normalization state differs.')
    if saved['completed_updates'] != saved['observer']['completed_updates']:
        raise ValueError('Checkpoint clocks differ.')
    model.load_state_dict(saved['model'])
    model.train(saved['model_training'])
    optimizer.load_state_dict(saved['optimizer'])
    check_optimizer(optimizer, saved['completed_updates'])
    observer.load_state_dict(saved['observer'])
    # Gradients are intentionally recomputed at this boundary, never replayed/applied twice.
    model.zero_grad(set_to_none=True)
    restore_rng(saved['rng'])
    return saved


def run_control(task, optimizer, observer, output, identity, data_state, until,
                snapshot_states, checkpoint_every=32, stop_requested=lambda: False):
    """Shared loop tested with a generated-gradient mock, not a training workload."""
    output = Path(output)
    start = observer.completed_updates
    if until < start or checkpoint_every < 1:
        raise ValueError('Invalid horizon/checkpoint cadence.')
    write_json(output/'resolved_identity.json', identity)
    begin = time.perf_counter()
    status = dict(status='running', started_utc=stamp(), start_update=start,
                  requested_updates=until, completed_updates=start, scientific_status='untreated_control_unqualified')
    write_json(output/'status.json', status)
    write_json(output/'manifest.json', dict(run_id='217', identity=identity, status='running'))

    def save(label=None):
        state = observer.completed_updates
        payload = checkpoint(task.model, optimizer, observer, identity, data_state, label or 'rolling_recovery')
        target = output/'recovery.pt' if label is None else output/'checkpoints'/f'{label}-state-{state:06d}.pt'
        save_checkpoint(target, payload, immutable=label is not None)
        if label is not None:
            append_json(output/'snapshots.jsonl', dict(label=label, state=state, path=target.relative_to(output).as_posix(),
                        bytes=target.stat().st_size, sha256=file_hash(target)))

    try:
        check_optimizer(optimizer, start)
        save()
        for state in range(start, until + 1):
            # Preserve the boundary BEFORE asking the task for g_state.
            if state in snapshot_states:
                label = snapshot_states[state]
                if re.fullmatch(r'[a-z][a-z0-9_-]*', label) is None:
                    raise ValueError('Invalid snapshot label.')
                save(label)
            if state == until:
                save('endpoint')
                save()
            if stop_requested():
                save()
                status['status'] = 'interrupted'
                break
            loss, gradient = task.gradient()
            if not math.isfinite(loss) or not bool(torch.isfinite(gradient).all()):
                raise ValueError(f'Nonfinite objective/gradient at state {state}.')
            if gradient.dtype != torch.float32:
                raise ValueError('A0 requires FP32 raw gradients.')
            # Check that the captured raw vector really is the optimizer input.
            if not torch.equal(gradient, flat([p.grad.detach() for p in task.model.parameters()])):
                raise ValueError('Captured raw gradient differs from optimizer input.')
            row = observer.observe(state, gradient)
            row.update(loss=loss, objective_images=task.units, utc=stamp())
            append_json(output/'boundaries.jsonl', row)
            if state in snapshot_states or state in [0, 1, until]:
                # Selected dense audit only: all other boundaries retain scalar reductions.
                previous = observer.previous_gradient
                audit = dict(state=state, available_before_update=state + 1,
                             raw_gradient=gradient.detach().cpu().clone(),
                             previous_gradient=None if previous is None else previous.cpu().clone(),
                             y=None if previous is None else (gradient.double()-previous.double()).cpu(),
                             previous_actual_increment=None if observer.previous_increment is None else observer.previous_increment.cpu().clone())
                target = output/'audits'/f'pre-state-{state:06d}.pt'
                save_checkpoint(target, audit, immutable=True)
                append_json(output/'audits.jsonl', dict(state=state, path=target.relative_to(output).as_posix(),
                            bytes=target.stat().st_size, sha256=file_hash(target)))
            if state == until:
                status['status'] = 'completed'
                break
            before = flat(task.model.parameters()).detach().double()
            optimizer.step()  # No staging/reimplementation, capture changes no Adam inputs.
            after = flat(task.model.parameters()).detach().double()
            if not bool(torch.isfinite(after).all()):
                raise ValueError(f'Nonfinite parameters after update {state + 1}.')
            post = observer.accept(gradient, optimizer, before, after, row)
            append_json(output/'metrics.jsonl', post)
            if observer.completed_updates % checkpoint_every == 0:
                save()
            event = dict(utc=stamp(), completed_updates=observer.completed_updates,
                         loss=loss, loss_state=state, examples_seen=observer.completed_updates*task.units,
                         elapsed_seconds=time.perf_counter()-begin)
            append_json(output/'events.jsonl', event)
            write_json(output/'progress.json', event)
            status['completed_updates'] = observer.completed_updates
            write_json(output/'status.json', status)
    except BaseException as error:
        status.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      exception=repr(error), traceback=traceback.format_exc())
        # Never overwrite a good recovery with a potentially partial failed update.
        raise
    finally:
        status.update(completed_updates=observer.completed_updates, finished_utc=stamp(),
                      elapsed_seconds=time.perf_counter()-begin,
                      recovery_policy='last_successful_boundary; failed updates never saved')
        write_json(output/'status.json', status)
        write_json(output/'manifest.json', dict(run_id='217', identity=identity, **status))
    return status


def verify_inputs(root, manifest):
    for item in manifest['files']:
        path = root/item['path']
        if path.stat().st_size != item['bytes'] or file_hash(path) != item['sha256']:
            raise ValueError('Input/source changed: ' + item['path'])


def runtime_identity(device):
    return dict(torch=str(torch.__version__), numpy=np.__version__, cuda=torch.version.cuda,
                device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
                threads=torch.get_num_threads(), tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                tf32_cudnn=torch.backends.cudnn.allow_tf32, cudnn_benchmark=torch.backends.cudnn.benchmark,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


def prepare(root, here, device):
    """Verify pinned bytes before deserialization/model construction; never downloads."""
    cfg = json.loads((here/'config.json').read_text())
    manifest = json.loads((here/'input_manifest.json').read_text())
    verify_inputs(root, manifest)
    configure()
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    prepared = torch.load(root/cfg['data'], weights_only=True, map_location='cpu')
    metadata = json.loads((root/cfg['data_metadata']).read_text())
    for key, expected in metadata['data_tensor_sha256'].items():
        if tensor_hash([prepared[key]]) != expected:
            raise ValueError('Prepared tensor changed: ' + key)
    if len(prepared['x_train']) != 5000 or len(prepared['y_train']) != 5000:
        raise ValueError('Expected the same 5000 training images.')
    data = {k: v.to(device) for k, v in prepared.items() if k.startswith(('x_', 'y_'))}
    model = model_for('mlp', root/'cache/eos-multitask').to(device)
    model.load_state_dict(torch.load(root/cfg['initial'], weights_only=True, map_location='cpu'))
    if tensor_hash(model.parameters()) != cfg['initial_tensor_sha256']:
        raise ValueError('Frozen seed0 initialization changed.')
    task = ProductionTask('mlp', model, data, cfg['physical'])
    optimizer = make_optimizer(model)
    observer = Observer(sum(p.numel() for p in model.parameters()), device)
    identity = dict(config_sha256=file_hash(here/'config.json'), manifest_sha256=file_hash(here/'input_manifest.json'),
                    files=manifest['files'], runtime=runtime_identity(device), arm='A0')
    data_state = dict(identity={k: metadata[k] for k in ['data_tensor_sha256', 'order', 'source_revision']},
                      normalization_mean=prepared['normalization_mean'].cpu().clone(),
                      normalization_std=prepared['normalization_std'].cpu().clone(),
                      position=0, train_examples=5000, physical=cfg['physical'], sampling='fixed_complete_objective')
    return cfg, task, optimizer, observer, identity, data_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New attempt directory; existing outputs are rejected.')
    parser.add_argument('--until', type=int, required=True, help='Total completed updates, at most 785; requires separate run authorization.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if not 1 <= args.until <= 785:
        parser.error('--until must be within [1,785].')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'status.json', dict(status='preparing', utc=stamp(), neural_training_started=False))
    try:
        cfg, task, optimizer, observer, identity, data_state = prepare(ROOT, HERE, torch.device(args.device))
        if args.resume:
            load_checkpoint(args.resume, task.model, optimizer, observer, identity, data_state)
            write_json(args.output/'resume_identity.json', dict(path=str(args.resume.resolve()),
                       bytes=args.resume.stat().st_size, sha256=file_hash(args.resume)))
        write_json(args.output/'resolved_config.json', cfg)
        stopping = [False]
        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, lambda *_: stopping.__setitem__(0, True))
        run_control(task, optimizer, observer, args.output, identity, data_state, args.until,
                    {int(k): v for k, v in cfg['snapshots'].items()}, cfg['checkpoint_every'],
                    stop_requested=lambda: stopping[0] or (HERE/'STOP').exists())
    except BaseException as error:
        # Include preparation/identity/resume failures, before the loop exists.
        status = json.loads((args.output/'status.json').read_text())
        if status['status'] not in ['failed', 'interrupted']:
            status.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                          exception=repr(error), traceback=traceback.format_exc(), finished_utc=stamp())
            write_json(args.output/'status.json', status)
            write_json(args.output/'manifest.json', dict(run_id='217', **status))
        raise


if __name__ == '__main__':
    main()
