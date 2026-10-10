"""Read one bounded status snapshot; never mutates experiment state."""
import json
from pathlib import Path
import sys

run = Path(__file__).resolve().parents[1]
lease = json.loads((run / "prelaunch/lease.json").read_text())
root = run / "artifacts" / lease.get("attempt", "attempt-01") / (sys.argv[1] if len(sys.argv) > 1 else "full-pass")
result = {"exists": root.exists()}
if (root / "manifest.json").exists():
    result["manifest"] = json.loads((root / "manifest.json").read_text())
    events = root / "events.jsonl"
    if events.exists():
        with events.open("rb") as stream:
            stream.seek(max(0, events.stat().st_size - 16000))
            lines = stream.read().splitlines()
        for line in reversed(lines):
            try:
                event = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            if event.get("event") == "update":
                result["latest_update"] = event
                break
    if (root / "verification.json").exists():
        try:
            result["verification"] = json.loads((root / "verification.json").read_text())
        except ValueError:
            result["verification_pending"] = True
print(json.dumps(result))
