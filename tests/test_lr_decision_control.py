"""Cloud-control attribution and cleanup checks, with no infrastructure actions."""
import importlib.util
import ast
import json
from pathlib import Path
import subprocess

import pytest

PATH = Path(__file__).resolve().parents[1] / "analyses/030-2026-10-30-adam-lr-decisions/04_control.py"
spec = importlib.util.spec_from_file_location("causal_control", PATH)
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


def test_attribution_rejects_foreign_id_and_name(monkeypatch):
    pod = dict(id="owned-id", name="cdr-analysis030-worker0")
    def answer(record):
        monkeypatch.setattr(control, "cli", lambda *a, **k: subprocess.CompletedProcess(a, 0, json.dumps(record), ""))
    answer(dict(id="owned-id", name="cdr-analysis030-worker0"))
    assert control.assert_owned(pod)["id"] == "owned-id"
    for record in (dict(id="foreign", name=pod["name"]), dict(id=pod["id"], name="unrelated-pod")):
        answer(record)
        with pytest.raises(ValueError, match="unattributed"):
            control.assert_owned(pod)


def test_absence_requires_not_found_not_network_failure(monkeypatch):
    def answer(code, returncode=1):
        monkeypatch.setattr(control, "cli", lambda *a, **k: subprocess.CompletedProcess(a, returncode, "", json.dumps(dict(code=code))))
    answer("not_found")
    control.confirm_absent("pod", "get", "owned")
    answer("network_error")
    with pytest.raises(RuntimeError, match="Deletion read failed"):
        control.confirm_absent("pod", "get", "owned")
    answer("", returncode=0)
    with pytest.raises(RuntimeError, match="still exists"):
        control.confirm_absent("pod", "get", "owned")


def test_inventory_escape_is_rejected_before_file_transfer(monkeypatch, tmp_path):
    monkeypatch.setattr(control, "ROOT", tmp_path)
    def inventory_only(pod, lease, relative, destination):
        assert relative.endswith("inventory.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(dict(files=[dict(path="../outside", bytes=1, sha256="0"*64)])))
    monkeypatch.setattr(control, "copy_file", inventory_only)
    with pytest.raises(ValueError, match="Unsafe"):
        control.retrieve({}, {}, "127-task", [])


def test_remote_status_program_compiles_and_has_bounded_transport(monkeypatch, tmp_path):
    def inspect(command, **kwargs):
        ast.parse(kwargs["input"])
        assert "WORKER" not in kwargs["input"]
        assert kwargs["timeout"] == 45
        assert kwargs["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)
        assert command[0] == "ssh"
        return subprocess.CompletedProcess(command, 0, '{"status":"running"}', "")
    monkeypatch.setattr(control.subprocess, "run", inspect)
    result = control.remote_status(dict(worker=0, user="root", host="127.0.0.1", port=22),
                                   dict(ssh_key=str(tmp_path / "key")))
    assert result["status"] == "running"


def test_local_cli_and_transfers_do_not_create_windows(monkeypatch, tmp_path):
    calls=[]
    def inspect(command, **kwargs):
        assert kwargs['creationflags'] == getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        calls.append(command[0])
        return subprocess.CompletedProcess(command, 0, b'' if isinstance(kwargs.get('input'), bytes) else '', '')
    monkeypatch.setattr(control.subprocess, 'run', inspect)
    pod=dict(worker=0, user='root', host='127.0.0.1', port=22)
    lease=dict(ssh_key=str(tmp_path/'key'))
    control.cli('pod', 'list')
    control.copy_file(pod, lease, 'test.txt', tmp_path/'test.txt')
    launch_spec=importlib.util.spec_from_file_location('hidden_launch', PATH.with_name('05_launch.py'))
    launch=importlib.util.module_from_spec(launch_spec);launch_spec.loader.exec_module(launch)
    launch.ssh(pod, lease, 'true\n')
    launch.upload(pod, lease, tmp_path/'test.txt', '/tmp/test.txt')
    assert len(calls)==4


def test_remote_guard_uses_only_matching_container_identity():
    path = PATH.with_name('08_remote_guard.py')
    guard_spec = importlib.util.spec_from_file_location('remote_deadline', path)
    guard = importlib.util.module_from_spec(guard_spec)
    guard_spec.loader.exec_module(guard)
    env = b'RUNPOD_POD_ID=owned\0RUNPOD_API_KEY=example-only\0'
    assert guard.own_identity('owned', env) == 'example-only'
    with pytest.raises(AssertionError, match='Foreign'):
        guard.own_identity('foreign', env)
    with pytest.raises(AssertionError, match='Missing'):
        guard.own_identity('owned', b'RUNPOD_POD_ID=owned\0')


def test_monitor_accepts_manifest_before_first_update(monkeypatch, tmp_path):
    (tmp_path/'scratch').mkdir()
    rows=[dict(run='187-test',worker=0,size='14M')]
    lease=dict(pods=[dict(id='owned',worker=0)],created_at_epoch=0,maximum_hourly_price=1)
    (tmp_path/'scratch/lease.json').write_text(json.dumps(lease))
    (tmp_path/'cohort.json').write_text(json.dumps(rows))
    monkeypatch.setattr(control,'HERE',tmp_path)
    monkeypatch.setattr(control,'remote_status',lambda *_:dict(
        status='running',completed=[],manifest=dict(run_id='187',scope='scientific'),
        calibration=dict(by_size={'14M':dict(seconds=[10,9,9])})))
    control.monitor(lease,once=True)
    snapshot=json.loads((tmp_path/'scratch/latest-status.json').read_text())['pods'][0]
    assert snapshot['status']=='running' and 'error' not in snapshot
    assert snapshot['cohort_remaining_seconds_with_ancillary_allowance']==128*10+90


def test_retrieval_receipt_creates_prelaunch_directory(tmp_path):
    receipt=tmp_path/'runs/187-test/prelaunch/attempt-01-retrieval.json'
    control.save(receipt,dict(verified=True))
    assert json.loads(receipt.read_text())==dict(verified=True)
    assert not receipt.with_suffix('.json.tmp').exists()
