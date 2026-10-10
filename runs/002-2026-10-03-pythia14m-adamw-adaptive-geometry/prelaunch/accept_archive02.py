"""Accept the owned worker's frozen archive before cloud deletion."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import sys
import tarfile

RUN = Path(__file__).resolve().parents[1]
REPO = RUN.parents[1]
ARCHIVE = REPO / "cache/runpod/run002-attempt-02.tar.gz"
expected_hash = ARCHIVE.with_name(ARCHIVE.name + ".sha256").read_text().split()[0]
expected_bytes = int(ARCHIVE.with_name(ARCHIVE.name + ".bytes").read_text().strip())
assert ARCHIVE.stat().st_size == expected_bytes
with ARCHIVE.open("rb") as stream:
    actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
assert actual_hash == expected_hash
destination = (RUN / "artifacts").resolve()
assert destination.is_relative_to(REPO.resolve())
assert not (destination / "attempt-02").exists()
with tarfile.open(ARCHIVE, "r:gz") as incoming:
    members = incoming.getmembers()
    seen = set()
    for member in members:
        name = PurePosixPath(member.name)
        assert name.parts and name.parts[0] == "attempt-02"
        assert not name.is_absolute() and ".." not in name.parts
        assert "\\" not in member.name and ":" not in member.name
        assert member.name not in seen
        seen.add(member.name)
        assert member.isdir() or member.isfile()
    incoming.extractall(destination, members=members, filter="data")
sys.path.insert(0, str(REPO / "tools"))
from artifact_inventory import verify
root = destination / "attempt-02"
accepted = verify(root, root / "inventory.json", strict=True)
assert accepted["verified"], accepted
receipt = {"pod_id":"9tw9vb6d8m4mg4", "attempt":"attempt-02",
           "archive_bytes":expected_bytes, "archive_sha256":actual_hash,
           **accepted}
(RUN / "prelaunch/attempt2-retrieval.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
