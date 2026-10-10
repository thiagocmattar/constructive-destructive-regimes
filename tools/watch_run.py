"""Read-only manifest/event polling; adapted from sparsity-spillover."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any


TERMINAL = {"completed", "failed", "cancelled"}


def _latest_complete_event(path: Path) -> Any:
    """Read the tail only; a newline commits a JSONL event for monitoring."""
    if not path.exists():
        return None
    with path.open("rb") as stream:
        stream.seek(0, 2)
        position = stream.tell()
        tail = b""
        while position:
            size = min(8192, position)
            position -= size
            stream.seek(position)
            tail = stream.read(size) + tail
            lines = tail.split(b"\n")[:-1]  # Exclude any uncommitted last line.
            if position:
                lines = lines[1:]  # The chunk's first line may be incomplete.
            complete = [line for line in lines if line.strip()]
            if complete:
                return json.loads(complete[-1].decode("utf-8"))
    return None


def read_status(manifest_path: Path, events_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {"manifest": manifest, "latest_event": _latest_complete_event(events_path)}


def watch(manifest: Path, events: Path, *, interval: float,
          max_polls: int | None) -> str:
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("Interval must be positive and finite.")
    if max_polls is not None and (isinstance(max_polls, bool)
                                or not isinstance(max_polls, int) or max_polls <= 0):
        raise ValueError("Max polls must be a positive integer.")
    polls = 0
    while True:
        state = read_status(manifest, events)
        print(json.dumps(state, sort_keys=True), flush=True)
        polls += 1
        status = str(state["manifest"].get("status"))
        if status in TERMINAL or (max_polls is not None and polls >= max_polls):
            return status
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--max-polls", type=int)
    args = parser.parse_args()
    try:
        status = watch(args.manifest, args.events, interval=args.interval, max_polls=args.max_polls)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    if status in {"failed", "cancelled"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
