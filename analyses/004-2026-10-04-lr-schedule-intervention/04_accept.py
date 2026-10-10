"""Accept one immutable continuation archive before deleting its own worker."""
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import hashlib
import importlib.util
import json
import sys
import tarfile

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def accept(run_name):
    row = next(r for r in json.loads((HERE/"cohort.json").read_text()) if r["run"] == run_name)
    run = ROOT/"runs"/run_name
    archive = ROOT/"cache/runpod"/f"run{run_name[:3]}-attempt-01.tar.gz"
    expected_bytes = int(archive.with_name(archive.name+".bytes").read_text())
    expected_sha = archive.with_name(archive.name+".sha256").read_text().split()[0]
    assert archive.stat().st_size == expected_bytes
    actual = sha(archive)
    assert actual == expected_sha
    target = (run/"artifacts").resolve()
    assert target.is_relative_to(ROOT.resolve())
    base = target/"attempt-01"
    assert not base.exists(), "Preserve any previously extracted evidence; investigate before retrying"
    with tarfile.open(archive, "r:gz") as incoming:
        seen = set()
        members = incoming.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            assert path.parts and path.parts[0] == "attempt-01"
            assert not path.is_absolute() and ".." not in path.parts
            assert "\\" not in member.name and ":" not in member.name and member.name not in seen
            assert member.isdir() or member.isfile()
            seen.add(member.name)
        incoming.extractall(target, members=members, filter="data")
    sys.path.insert(0, str(ROOT/"tools"))
    from artifact_inventory import verify
    accepted = verify(base, base/"inventory.json", strict=True)
    assert accepted["verified"], accepted
    source_matches = {}
    for path in (base/"source").glob("*"):
        if path.is_file():
            assert path.read_bytes() == (run/path.name).read_bytes(), path
            source_matches[path.relative_to(base).as_posix()] = True
    for path in (base/"source/baseline").glob("*.py"):
        assert path.read_bytes() == (ROOT/"src/baseline"/path.name).read_bytes(), path
        source_matches[path.relative_to(base).as_posix()] = True
    for name in ("source-inventory.json", "lease.json"):
        assert (base/"source/prelaunch"/name).read_bytes() == (run/"prelaunch"/name).read_bytes()
    cfg = json.loads((run/"config.json").read_text())
    identity = json.loads((base/"full-pass/resume_identity.json").read_text())
    assert identity["parent_checkpoint_sha256"] == cfg["resume"]["checkpoint_sha256"]
    assert identity["parent_config_sha256"] == cfg["resume"]["parent_config_sha256"]
    assert identity["order_sha256"] == cfg["resume"]["order_sha256"]
    assert cfg["resume"]["parent_run"] == row["parent_run"]
    assert sha(base/"full-pass/resolved_config.json") == sha(run/"config.json")
    spec = importlib.util.spec_from_file_location("verify_continuation_"+run_name[:3], run/"03_verify.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(2)
    scientific = module.verify(base/"full-pass")
    assert scientific["verified"]
    remote_verification = json.loads((base/"full-pass/verification.json").read_text())
    assert remote_verification["verified"]
    transport = json.loads((run/"prelaunch/transport.json").read_text())
    lease = json.loads((run/"prelaunch/lease.json").read_text())
    assert transport["pod_id"] == lease["pod_id"]
    receipt = {"pod_id": transport["pod_id"], "attempt": "attempt-01",
        "archive_bytes": expected_bytes, "archive_sha256": actual, **accepted,
        "scientific_verification": scientific, "frozen_source_copies_match": source_matches,
        "parent_identity_verified": True, "accepted_at": datetime.now(timezone.utc).isoformat()}
    (run/"prelaunch/attempt1-retrieval.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({k:v for k,v in receipt.items() if k != "frozen_source_copies_match"}, indent=2), flush=True)


if __name__ == "__main__":
    accept(sys.argv[1])
