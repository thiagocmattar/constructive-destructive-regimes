"""Independent coverage and terminal streaming-state verification."""
from pathlib import Path
import hashlib
import json
import math
import sys

import numpy as np
import torch


def verify(folder):
    folder = Path(folder)
    config = json.loads((folder / "resolved_config.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    identity = json.loads((folder / "resume_identity.json").read_text())
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    rows = [r for r in events if r["event"] == "update"]
    vals = [r for r in events if r["event"] == "validation"]
    assert [r["update"] for r in rows] == list(range(357, 713))
    assert [r["update"] for r in vals] == [356, 445, 534, 623, 712]
    assert all(r["supervised_tokens"] == 693329 and r["input_tokens"] == 693668 for r in vals)
    assert all(r["short_tail_included"] and math.isfinite(r["loss"]) for r in vals)
    assert identity["data_position"] == 356 * 1024
    assert summary["completed_sequences"] == summary["planned_sequences"] == 728375
    assert summary["all_training_tokens_covered"] and summary["repeated_sequences"] == 0
    assert summary["completed_input_tokens"] == 1491711416
    assert summary["completed_supervised_tokens"] == 1490983041
    assert sum(r["sequences"] for r in rows) == 363831
    assert sum(r["input_tokens"] for r in rows) == 745125304
    assert sum(r["supervised_tokens"] for r in rows) == 744761473
    assert [r["sequences"] for r in rows] == [1024] * 355 + [311]
    assert all(r["branch"] == config["resume"]["schedule"] for r in rows)
    assert all(math.isfinite(r["loss"]) and -1-1e-10 <= r["adjacent_cosine"] <= 1+1e-10 for r in rows)
    assert max(abs(r["gradient_norm"]-r["gradient_clip_returned_norm"])/r["gradient_norm"] for r in rows) < 1e-5
    for stream, prefix, norm in [("geometry", "raw_norm_sum_at_junction", "gradient_norm"),
                                  ("adaptive_geometry", "adaptive_norm_sum_at_junction", "direction_norm")]:
        state = torch.load(folder / "checkpoints/update-712/recovery.pt", map_location="cpu", weights_only=False)
        assert state["config_sha256"] == hashlib.sha256((folder / "resolved_config.json").read_bytes()).hexdigest()
        assert state["order_sha256"] == config["resume"]["order_sha256"]
        assert state["data_position"] == 728375
        geometry = state[stream]
        assert geometry["updates"] == 712 and geometry["names"] == identity["parameter_names"]
        source = rows if stream == "geometry" else [r["adaptive"] for r in rows]
        expected_A = identity[prefix] + sum(r[norm] for r in source)
        expected_B = math.sqrt(sum(float(v.double().square().sum()) for v in geometry["cumulative"]))
        expected_norm = math.sqrt(sum(float(v.double().square().sum()) for v in geometry["previous"]))
        last = source[-1]
        B_field = "cumulative_gradient_norm" if stream == "geometry" else "cumulative_direction_norm"
        assert math.isclose(geometry["norm_sum"], expected_A, rel_tol=1e-12)
        assert math.isclose(last["cumulative_norm_sum"], expected_A, rel_tol=1e-12)
        assert math.isclose(last[B_field], expected_B, rel_tol=1e-12)
        assert math.isclose(last[norm], expected_norm, rel_tol=1e-12)
        assert math.isclose(last["cumulative_ratio"], expected_B/expected_A, rel_tol=1e-12)
        assert all(int(v["step"]) == 712 for v in state["optimizer"]["state"].values())
        del state
    return {"verified": True, "updates": len(rows), "global_updates": [357, 712],
        "training_input_tokens": summary["completed_input_tokens"],
        "continuation_input_tokens": summary["continuation_input_tokens"],
        "final_validation_loss": vals[-1]["loss"], "validation_updates": [v["update"] for v in vals],
        "full_validation_supervised_tokens": 693329, "terminal_state_checked_independently": True}


if __name__ == "__main__":
    torch.set_num_threads(2)
    print(json.dumps(verify(sys.argv[1]), indent=2))
