"""Freeze the compatibility fix without changing input tokens or recipes."""
from pathlib import Path
import hashlib
import json
import tarfile

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


inventory = json.loads((RUN / "source_inventory.json").read_text())
for item in inventory["files"]:
    path = ROOT / item["path"]
    item.update(bytes=path.stat().st_size, sha256=digest(path))
dest = RUN / "source_inventory-attempt-02.json"
dest.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
packet = ROOT / "cache/spark/run020-calibration-attempt02.tar.gz"
with tarfile.open(packet, "w:gz") as archive:
    for path in (RUN / "01_calibrate.py", RUN / "02_run.sh", dest):
        archive.add(path, arcname=path.relative_to(ROOT).as_posix())
receipt = {"bytes": packet.stat().st_size, "sha256": digest(packet)}
(RUN / "packet_identity-attempt-02.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps(receipt))
