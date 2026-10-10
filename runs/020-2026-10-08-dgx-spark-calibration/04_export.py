"""Export compact calibration evidence; retain task-owned checkpoints on the Spark."""
from pathlib import Path
import hashlib
import json
import tarfile

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
successful = RUN / "artifacts/attempt-02"
assert json.loads((successful / "manifest.json").read_text())["status"] == "completed"
files = [p for p in (RUN / "artifacts").rglob("*")
         if p.is_file() and "checkpoints" not in p.parts]
files += [ROOT / name for name in ("environment-lock.txt", "setup-torch.log",
                                   "setup-transformers.log", "calibration.log",
                                   "calibration-attempt02.log")]
inventory = []
for path in sorted(files):
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    inventory.append({"path": path.relative_to(ROOT).as_posix(),
                      "bytes": path.stat().st_size, "sha256": sha})
dest = ROOT / "calibration-transfer-inventory.json"
dest.write_text(json.dumps({"files": inventory}, indent=2) + "\n")
archive_path = ROOT / "calibration-results.tar.gz"
with tarfile.open(archive_path, "w:gz") as archive:
    for path in files + [dest]:
        archive.add(path, arcname=path.relative_to(ROOT).as_posix())
with archive_path.open("rb") as stream:
    sha = hashlib.file_digest(stream, "sha256").hexdigest()
print(json.dumps({"bytes": archive_path.stat().st_size, "sha256": sha,
                  "files": len(files), "checkpoint_bytes_retained": sum(
                      p.stat().st_size for p in (successful / "checkpoints").rglob("*") if p.is_file())}))
