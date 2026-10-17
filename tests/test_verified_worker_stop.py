"""A worker can stop only after its entire assigned cohort is locally accepted."""
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / 'analyses/025-2026-10-17-pythia-descent-mechanism/09_stop_verified_workers.py'


@pytest.fixture
def module(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('verified_worker_stop', PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    return module


def receipt(root, name, valid=True):
    path = root / 'runs' / name / 'prelaunch/attempt-01-retrieval.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(verified=True, scientific_verification=dict(verified=valid))))


def test_missing_or_scientifically_unverified_branch_prevents_stop(module, monkeypatch):
    receipt(module.ROOT, 'baseline')
    monkeypatch.setattr(module.control, 'assert_owned', lambda pod: pytest.fail('Premature infrastructure read/action'))
    assert module.stop_verified({'id':'owned'}, ['baseline', 'branch']) is None
    receipt(module.ROOT, 'branch', valid=False)
    assert module.stop_verified({'id':'owned'}, ['baseline', 'branch']) is None


def test_all_assigned_runs_are_required_and_owned_stop_is_read_back(module, monkeypatch):
    names = module.assigned_runs([dict(baseline='base', branches=['fork175','fork225'])])
    for name in names:
        receipt(module.ROOT, name)
    states = iter([dict(runtimeStatus='running'), dict(runtimeStatus='stopped', desiredStatus='EXITED')])
    reads, actions = [], []
    def owned(pod):
        reads.append(pod['id'])
        return next(states)
    monkeypatch.setattr(module.control, 'assert_owned', owned)
    monkeypatch.setattr(module.control, 'cli', lambda *args: actions.append(args))
    result = module.stop_verified(dict(id='own-pod', worker=0), names)
    assert reads == ['own-pod','own-pod']
    assert actions == [('pod','stop','own-pod')]
    assert result['runtimeStatus'] == 'stopped'


def test_empty_assignment_cannot_stop_a_worker(module):
    assert not module.all_verified(module.ROOT, [])


def test_budget_includes_storage_and_uses_the_earlier_cutoff(module):
    lease = dict(pods=[dict(hourly_price=.89),dict(hourly_price=.89)],
                 storage_hourly_price=20/730, created_at_epoch=1000, hard_deadline_epoch=44200)
    result = module.budget_snapshot(lease, 25, 4600)
    assert result['estimated_task_cost_usd'] == pytest.approx(1.78+20/730)
    assert result['estimated_budget_remaining_usd'] == pytest.approx(25-1.78-20/730)
    assert result['automatic_cutoff_epoch'] == 44200
    assert module.budget_snapshot(lease, 1, 1000)['automatic_cutoff_epoch'] < 44200
