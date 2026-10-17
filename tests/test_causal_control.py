"""Cloud-control attribution and cleanup checks, with no infrastructure actions."""
import importlib.util
import ast
import json
from pathlib import Path
import subprocess

import pytest

PATH = Path(__file__).resolve().parents[1] / "analyses/025-2026-10-17-pythia-descent-mechanism/04_control.py"
spec = importlib.util.spec_from_file_location("causal_control", PATH)
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


def test_attribution_rejects_foreign_id_and_name(monkeypatch):
    pod = dict(id="owned-id", name="cdr-analysis025-worker0")
    def answer(record):
        monkeypatch.setattr(control, "cli", lambda *a, **k: subprocess.CompletedProcess(a, 0, json.dumps(record), ""))
    answer(dict(id="owned-id", name="cdr-analysis025-worker0"))
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
        assert command[0] == "ssh"
        return subprocess.CompletedProcess(command, 0, '{"status":"running"}', "")
    monkeypatch.setattr(control.subprocess, "run", inspect)
    result = control.remote_status(dict(worker=0, user="root", host="127.0.0.1", port=22),
                                   dict(ssh_key=str(tmp_path / "key")))
    assert result["status"] == "running"
