"""Generated tensors / gradient mocks only. Never construct or train the CIFAR MLP."""
import copy
import importlib.util
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('run217_control', HERE/'01_control.py')
a0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a0)


class ParameterOnly(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Parameter(torch.tensor([2., -3.]))
        self.second = torch.nn.Parameter(torch.tensor([.125]))


class GradientMock:
    """Algebraic full-vector oracle. No forward/backward or neural model."""
    units = 7

    def __init__(self, stochastic=False):
        self.model = ParameterOnly()
        self.calls = 0
        self.stochastic = stochastic
        self.on_gradient = lambda: None

    def gradient(self):
        self.on_gradient()
        self.calls += 1
        theta = a0.flat(self.model.parameters()).detach()
        gradient = theta*torch.tensor([.5, 2., 3.]) + torch.tensor([.1, -.2, .3])
        if self.stochastic:
            gradient += torch.rand(3)*.01 + random.random()*.01 + float(np.random.random())*.01
        self.model.first.grad = gradient[:2].clone()
        self.model.second.grad = gradient[2:].clone()
        return float(theta.double().square().sum()), gradient


def setup(stochastic=False):
    task = GradientMock(stochastic)
    optimizer = a0.make_optimizer(task.model)
    observer = a0.Observer(3, 'cpu')
    return task, optimizer, observer


def data_state():
    return dict(identity={'data': 'generated', 'normalization': 'unit'},
                normalization_mean=torch.tensor([.1]), normalization_std=torch.tensor([.5]),
                position=0, train_examples=7, physical=3, sampling='fixed_complete_objective')


def same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for k in left:
            same(left[k], right[k])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for x, y in zip(left, right):
            same(x, y)
    else:
        assert left == right


def consume(task, optimizer, observer, count):
    for _ in range(count):
        _, g = task.gradient()
        row = observer.observe(observer.completed_updates, g)
        before = a0.flat(task.model.parameters()).detach().double()
        optimizer.step()
        after = a0.flat(task.model.parameters()).detach().double()
        observer.accept(g, optimizer, before, after, row)


def test_capture_enabled_is_bitwise_unchanged_adam():
    task, opt, obs = setup()
    plain, reference, _ = setup()
    for state in range(12):
        _, g = task.gradient()
        plain.gradient()
        retained = [p.grad.clone() for p in task.model.parameters()]
        row = obs.observe(state, g)
        before = a0.flat(task.model.parameters()).detach().double()
        opt.step(); reference.step()
        obs.accept(g, opt, before, a0.flat(task.model.parameters()).detach().double(), row)
        same(task.model.state_dict(), plain.model.state_dict())
        same(opt.state_dict(), reference.state_dict())
        same(retained, [p.grad for p in task.model.parameters()])
        same(g, a0.flat(retained))


def test_preupdate_clock_products_and_no_future_data_leakage():
    task, opt, obs = setup()
    before_observer = copy.deepcopy(obs.state_dict())
    initial = obs.observe(0, torch.tensor([1., 2., 3.]))
    assert initial['completed_step_q'] is None
    assert initial['completed_step_q_undefined_reason'] == 'no_completed_step'
    same(before_observer, obs.state_dict())
    consume(task, opt, obs, 1)
    previous, s = obs.previous_gradient.double().numpy(), obs.previous_increment.numpy()
    # The incoming gradient is known, while the outgoing direction/update has not occurred.
    current = torch.tensor([-2., 3., 0.])
    old = copy.deepcopy(obs.state_dict())
    row = obs.observe(1, current)
    y = current.double().numpy() - previous
    assert row['state'] == 1 and row['available_before_update'] == 2
    assert row['completed_step'] == [0, 1]
    assert row['previous_gradient_state'] == 0
    assert row['previous_increment_dot_y'] == pytest.approx(s@y, abs=1e-16)
    assert row['previous_increment_dot_previous_gradient'] == pytest.approx(s@previous, abs=1e-16)
    assert row['gradient_difference_norm'] == pytest.approx(np.linalg.norm(y))
    assert row['raw_projected_change'] == pytest.approx(-previous@y/(previous@previous))
    assert row['completed_step_q'] == pytest.approx(-(s@y)/(s@previous))
    assert row['completed_step_kappa_sec'] == pytest.approx((s@y)/(s@s))
    assert not any(key in row for key in ['gradient_dot_actual_step', 'streams', 'outgoing_increment'])
    same(old, obs.state_dict())
    assert int(opt.state[task.model.first]['step']) == 1
    with pytest.raises(ValueError, match='clock'):
        obs.observe(2, current)


@pytest.mark.parametrize('previous,current,increment,q_reason,k_reason,raw_reason', [
    ([0., 0., 0.], [1., 2., 3.], [0., 0., 0.], 'zero_previous_step_slope', 'zero_previous_increment', 'zero_previous_gradient'),
    ([1., 0., 0.], [0., 0., 0.], [0., 1., 0.], 'zero_previous_step_slope', None, None),
    ([1., 0., 0.], [1., 0., 0.], [-1., 0., 0.], None, None, None),
    ([1., 0., 0.], [2., 0., 0.], [1., 0., 0.], None, None, None),
])
def test_zero_undefined_and_signed_cases(previous, current, increment, q_reason, k_reason, raw_reason):
    obs = a0.Observer(3, 'cpu')
    obs.completed_updates = 1
    obs.previous_gradient = torch.tensor(previous)
    obs.previous_increment = torch.tensor(increment, dtype=torch.float64)
    row = obs.observe(1, torch.tensor(current))
    for key, reason in [('completed_step_q', q_reason), ('completed_step_kappa_sec', k_reason), ('raw_projected_change', raw_reason)]:
        assert row[key + '_undefined_reason'] == reason
        assert (row[key] is None) == (reason is not None)
    # Zero y is a defined zero kappa for a nonzero step. Signed slopes aren't discarded.
    if previous == current:
        assert row['completed_step_kappa_sec'] == 0
    if increment == [1., 0., 0.]:
        assert row['completed_step_q'] == -1
    json.dumps(row, allow_nan=False)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_observation_rejected_without_mutation(bad):
    obs = a0.Observer(3, 'cpu')
    old = copy.deepcopy(obs.state_dict())
    with pytest.raises(ValueError, match='Nonfinite'):
        obs.observe(0, torch.tensor([bad, 0., 0.]))
    same(old, obs.state_dict())


def test_actual_increment_includes_fp32_rounding():
    task, opt, obs = setup()
    # Large FP32 coordinate absorbs the proposed small Adam step completely.
    with torch.no_grad():
        task.model.first[0] = 2**24
    consume(task, opt, obs, 1)
    assert obs.previous_increment.dtype == torch.float64
    assert obs.previous_gradient.dtype == torch.float32
    assert obs.previous_increment[0] == 0
    assert obs.history[0]['post']['adam_step_rounding_residual_norm'] > .0009


def test_checkpoint_roundtrip_restores_optimizer_observer_rng_and_suffix(tmp_path):
    random.seed(3); np.random.seed(4); torch.manual_seed(5)
    task, opt, obs = setup(stochastic=True)
    consume(task, opt, obs, 3)
    payload = a0.checkpoint(task.model, opt, obs, {'source': 'generated'}, data_state(), 'fork')
    path = tmp_path/'fork.pt'
    a0.save_checkpoint(path, payload, immutable=True)
    expected_rng = (random.random(), np.random.random(), torch.rand(3))
    consume(task, opt, obs, 4)
    expected = (copy.deepcopy(task.model.state_dict()), copy.deepcopy(opt.state_dict()), obs.state_dict())
    other, optimizer, observer = setup(stochastic=True)
    saved = a0.load_checkpoint(path, other.model, optimizer, observer, {'source': 'generated'}, data_state())
    assert saved['completed_updates'] == 3 and saved['examples_seen'] == 21
    assert saved['boundary'] == 'before_current_gradient'
    assert all(p.grad is None for p in other.model.parameters())
    same((random.random(), np.random.random(), torch.rand(3)), expected_rng)
    consume(other, optimizer, observer, 4)
    same(other.model.state_dict(), expected[0]); same(optimizer.state_dict(), expected[1]); same(observer.state_dict(), expected[2])


def test_checkpoint_restores_cuda_rng_through_mock(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [torch.tensor([7], dtype=torch.uint8)])
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', lambda x: calls.append(x))
    saved = a0.rng_state()
    a0.restore_rng(saved)
    same(calls, [[torch.tensor([7], dtype=torch.uint8)]])


def test_immutable_forks_survive_rolling_recovery_and_failed_rewrite(tmp_path, monkeypatch):
    task, opt, obs = setup()
    for state in range(3):
        payload = a0.checkpoint(task.model, opt, obs, {}, data_state(), 'fork')
        a0.save_checkpoint(tmp_path/f'fork-{state}.pt', payload, immutable=True)
        a0.save_checkpoint(tmp_path/'recovery.pt', payload, immutable=False)
        consume(task, opt, obs, 1)
    hashes = [a0.file_hash(tmp_path/f'fork-{state}.pt') for state in range(3)]
    with pytest.raises(FileExistsError):
        a0.save_checkpoint(tmp_path/'fork-0.pt', payload, immutable=True)
    original = a0.file_hash(tmp_path/'recovery.pt')
    def fail(*args, **kwargs):
        raise OSError('synthetic disk error')
    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(OSError, match='disk error'):
        a0.save_checkpoint(tmp_path/'recovery.pt', payload, immutable=False)
    assert original == a0.file_hash(tmp_path/'recovery.pt')
    assert hashes == [a0.file_hash(tmp_path/f'fork-{state}.pt') for state in range(3)]
    assert not list(tmp_path.glob('*.tmp'))


def test_loop_fork_is_saved_before_current_gradient_and_resume_matches(tmp_path):
    task, opt, obs = setup()
    output = tmp_path/'first'
    checked = []
    def inspect_boundary():
        if obs.completed_updates == 2:
            fork = torch.load(output/'checkpoints/main_fork-state-000002.pt', weights_only=False)
            assert fork['completed_updates'] == 2
            assert len(fork['observer']['history']) == 2
            assert fork['observer']['history'][-1]['pre']['state'] == 1
            checked.append(True)
    task.on_gradient = inspect_boundary
    status = a0.run_control(task, opt, obs, output, {}, data_state(), 5, {2: 'main_fork'}, checkpoint_every=2)
    assert status['status'] == 'completed' and task.calls == 6 and checked == [True]
    endpoint = output/'checkpoints/endpoint-state-000005.pt'
    assert endpoint.is_file() and (output/'recovery.pt').is_file()
    first_hash = a0.file_hash(output/'checkpoints/main_fork-state-000002.pt')
    other, optimizer, observer = setup()
    a0.load_checkpoint(output/'checkpoints/main_fork-state-000002.pt', other.model, optimizer, observer, {}, data_state())
    a0.run_control(other, optimizer, observer, tmp_path/'second', {}, data_state(), 5, {}, checkpoint_every=1)
    same(task.model.state_dict(), other.model.state_dict()); same(opt.state_dict(), optimizer.state_dict())
    # UTC/timing are metadata; all numerical and availability fields agree exactly.
    for x, y in zip(obs.history, observer.history):
        x, y = copy.deepcopy(x), copy.deepcopy(y)
        x['pre'].pop('utc'); y['pre'].pop('utc')
        same(x, y)
    assert first_hash == a0.file_hash(output/'checkpoints/main_fork-state-000002.pt')
    boundaries = [json.loads(s) for s in (output/'boundaries.jsonl').read_text().splitlines()]
    posts = [json.loads(s) for s in (output/'metrics.jsonl').read_text().splitlines()]
    assert [s['state'] for s in boundaries] == list(range(6))
    assert [s['update'] for s in posts] == list(range(1, 6))


@pytest.mark.parametrize('change', ['identity', 'normalization', 'position', 'age', 'history'])
def test_restore_rejects_changed_identity_or_inconsistent_state(tmp_path, change):
    task, opt, obs = setup(); consume(task, opt, obs, 2)
    payload = a0.checkpoint(task.model, opt, obs, {'source': 'one'}, data_state(), 'fork')
    identity = {'source': 'one'}
    if change == 'identity':
        identity['source'] = 'two'
    if change == 'normalization':
        payload['data_state']['normalization_std'] += 1
    if change == 'position':
        payload['data_state']['position'] = 1
    if change == 'age':
        payload['optimizer']['state'][0]['step'] += 1
    if change == 'history':
        payload['observer']['history'].pop()
    a0.save_checkpoint(tmp_path/'bad.pt', payload, immutable=True)
    other, optimizer, observer = setup()
    with pytest.raises(ValueError):
        a0.load_checkpoint(tmp_path/'bad.pt', other.model, optimizer, observer, identity, data_state())


def test_failure_retains_last_good_recovery_and_traceback(tmp_path):
    task, opt, obs = setup()
    base_gradient = task.gradient
    def fail():
        if obs.completed_updates == 2:
            raise RuntimeError('synthetic objective failure')
        return base_gradient()
    task.gradient = fail
    with pytest.raises(RuntimeError, match='synthetic'):
        a0.run_control(task, opt, obs, tmp_path, {}, data_state(), 5, {1: 'fork'}, checkpoint_every=1)
    status = json.loads((tmp_path/'status.json').read_text())
    assert status['status'] == 'failed' and status['completed_updates'] == 2
    assert 'synthetic objective failure' in status['traceback']
    assert torch.load(tmp_path/'recovery.pt', weights_only=False)['completed_updates'] == 2
    assert (tmp_path/'checkpoints/fork-state-000001.pt').is_file()
    assert not list((tmp_path/'checkpoints').glob('endpoint*'))


def test_graceful_stop_saves_boundary_without_consuming_gradient(tmp_path):
    task, opt, obs = setup()
    status = a0.run_control(task, opt, obs, tmp_path, {}, data_state(), 3, {}, stop_requested=lambda: obs.completed_updates == 1)
    assert task.calls == 1 and status['status'] == 'interrupted'
    assert torch.load(tmp_path/'recovery.pt', weights_only=False)['completed_updates'] == 1


def test_input_byte_identity_and_preparation_failure_status(tmp_path, monkeypatch):
    item = tmp_path/'generated.txt'; item.write_text('original')
    manifest = dict(files=[dict(path=item.name, bytes=item.stat().st_size, sha256=a0.file_hash(item))])
    a0.verify_inputs(tmp_path, manifest)
    item.write_text('modified')
    with pytest.raises(ValueError, match='changed'):
        a0.verify_inputs(tmp_path, manifest)
    def refuse(*args):
        raise ValueError('synthetic input failure')
    monkeypatch.setattr(a0, 'prepare', refuse)
    output = tmp_path/'attempt'
    monkeypatch.setattr(sys, 'argv', ['01_control.py', '--output', str(output), '--until', '657'])
    with pytest.raises(ValueError, match='synthetic input'):
        a0.main()
    status = json.loads((output/'status.json').read_text())
    assert status['status'] == 'failed' and status['neural_training_started'] is False
    assert 'synthetic input failure' in status['traceback']


def test_frozen_plan_is_a0_only_and_planned_fork_precedes_gradient():
    cfg = json.loads((HERE/'config.json').read_text())
    assert cfg['arm'] == 'A0' and cfg['seed'] == 0 and cfg['physical'] == 1000
    assert cfg['snapshots'] == {'657': 'main_fork', '689': 'pre_rise', '690': 'onset'}
    assert cfg['initial_tensor_sha256'] == '8de1e7c880539f122aa840d5b7c28027310e82bd3ee4e753afef87fe0876a2da'

def test_endpoint_snapshot_is_also_before_endpoint_rng_consumption(tmp_path):
    random.seed(7); np.random.seed(8); torch.manual_seed(9)
    task, opt, obs = setup(stochastic=True)
    a0.run_control(task, opt, obs, tmp_path, {}, data_state(), 2, {}, checkpoint_every=1)
    audit = torch.load(tmp_path/'audits/pre-state-000002.pt', weights_only=False)
    other, optimizer, observer = setup(stochastic=True)
    a0.load_checkpoint(tmp_path/'checkpoints/endpoint-state-000002.pt', other.model, optimizer, observer, {}, data_state())
    _, g = other.gradient()
    assert torch.equal(audit['raw_gradient'], g)
    assert torch.equal(audit['y'], g.double()-observer.previous_gradient.double())
    assert torch.equal(audit['previous_actual_increment'], observer.previous_increment)
    recovery = torch.load(tmp_path/'recovery.pt', weights_only=False)
    endpoint = torch.load(tmp_path/'checkpoints/endpoint-state-000002.pt', weights_only=False)
    same(recovery['rng'], endpoint['rng'])


def test_bad_optimizer_recipe_rejected_before_any_gradient(tmp_path):
    task, opt, obs = setup()
    opt.param_groups[0]['lr'] = .002
    with pytest.raises(ValueError, match='recipe'):
        a0.run_control(task, opt, obs, tmp_path, {}, data_state(), 2, {})
    assert task.calls == 0
    assert json.loads((tmp_path/'status.json').read_text())['status'] == 'failed'
