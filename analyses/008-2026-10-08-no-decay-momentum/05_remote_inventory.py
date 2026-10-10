"""Make a byte inventory for a failed attempt without claiming scientific success."""
from pathlib import Path
import hashlib
import json
import sys

folder = Path(sys.argv[1]).resolve()
if (folder / "transfer_inventory.json").exists():
    print("Existing inventory preserved")
    raise SystemExit(0)
assert json.loads((folder / "manifest.json").read_text())["status"] in ("failed", "completed")
assert not (folder / "verification.json").exists()
files = []
for path in sorted(folder.rglob("*")):
    if path.is_file():
        with path.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        files.append({"path": path.relative_to(folder).as_posix(), "bytes": path.stat().st_size, "sha256": sha})
(folder / "transfer_inventory.json").write_text(json.dumps({"files": files, "scope": "failed_attempt_bytes_only"}, indent=2) + "\n")
