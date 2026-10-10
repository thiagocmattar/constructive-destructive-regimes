"""Independent coverage, LR, first-moment and final streaming-buffer verification."""
from pathlib import Path
import hashlib
import json
import math
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from baseline.recipe import warmup_constant_learning_rate


def verify(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    config = json.loads((folder / "resolved_config.json").read_text())
    env = json.loads((folder / "environment.json").read_text())
    rows = [e for line in (folder / "events.jsonl").read_text().splitlines()
            if (e := json.loads(line)).get("event") == "update"]
    assert manifest["status"] == "completed" and len(rows) == summary["completed_updates"]
    assert [e["update"] for e in rows] == list(range(1, len(rows) + 1))
    assert env["initial_parameter_sha256"] == config["model"]["initial_parameter_sha256"]
    full = summary["scope"] == "full_pass"
    if full:
        assert len(rows) == 712 and rows[-1]["sequences"] == 311
        assert summary["completed_input_tokens"] == 1491711416
        assert summary["completed_supervised_tokens"] == 1490983041
        assert summary["completed_sequences"] == 728375 and summary["all_training_tokens_covered"]
    assert summary["repeated_sequences"] == 0
    max_norm_error = 0.
    for e in rows:
        expected_lr = warmup_constant_learning_rate(e["update"], peak=config["training"]["peak_learning_rate"],
                                                    max_steps=712, warmup_fraction=.01)
        assert e["learning_rate"] == expected_lr and math.isfinite(e["loss"])
        assert e["input_tokens"] == e["supervised_tokens"] + e["sequences"]
        error = abs(e["gradient_norm"] - e["gradient_clip_returned_norm"]) / max(e["gradient_norm"], 1e-300)
        assert error < 2e-6
        max_norm_error = max(error, max_norm_error)
        for stream in (e, e["adaptive"], e["momentum"]):
            ratio = stream["cumulative_ratio"]
            assert ratio is None or 0 <= ratio <= 1 + 1e-10
            cosine = stream["adjacent_cosine"]
            assert cosine is None or -1-1e-10 <= cosine <= 1+1e-10
            if e["update"] == 1:
                assert cosine is None
        for key in ("momentum_to_raw_norm_ratio", "adaptive_to_momentum_norm_ratio", "adaptive_to_raw_norm_ratio"):
            assert e[key] is None or math.isfinite(e[key])
    for label in ("initial_validation", "final_validation"):
        val = summary[label]
        assert val["input_tokens"] == 693668 and val["supervised_tokens"] == 693329
        assert val["sequences"] == 339 and val["short_tail_included"]
    state = torch.load(folder / "checkpoints/final/recovery.pt", map_location="cpu", weights_only=False)
    final = rows[-1]
    for name, metrics, norm_key in (("geometry", final, "cumulative_gradient_norm"),
                                   ("adaptive_geometry", final["adaptive"], "cumulative_direction_norm"),
                                   ("momentum_geometry", final["momentum"], "cumulative_momentum_norm")):
        stream = state[name]
        assert stream["updates"] == len(rows)
        b = math.sqrt(sum(float(v.double().square().sum()) for v in stream["cumulative"]))
        previous = math.sqrt(sum(float(v.double().square().sum()) for v in stream["previous"]))
        assert math.isclose(b, metrics[norm_key], rel_tol=1e-10, abs_tol=1e-10)
        assert stream["norm_sum"] == metrics["cumulative_norm_sum"]
        assert math.isclose(previous, stream["previous_norm"], rel_tol=1e-10, abs_tol=1e-10)
    assert state["data_position"] == summary["completed_sequences"]
    assert state["order_sha256"] == summary["order_sha256"]
    assert state["geometry"]["names"] == state["momentum_geometry"]["names"] == state["adaptive_geometry"]["names"]
    by_name = dict(zip(state["momentum_geometry"]["names"], state["momentum_geometry"]["previous"]))
    group_names = (env["optimizer"]["decay_parameter_names"], env["optimizer"]["no_decay_parameter_names"])
    for group, names in zip(state["optimizer"]["param_groups"], group_names):
        assert len(group["params"]) == len(names)
        for pid, name in zip(group["params"], names):
            moment = state["optimizer"]["state"][pid]
            assert int(moment["step"].item()) == len(rows)
            torch.testing.assert_close(by_name[name], moment["exp_avg"], rtol=0, atol=0)
    result = {"verified": True, "scope": summary["scope"], "completed_updates": len(rows),
              "all_three_streams_reconstructed": True, "stored_m_matches_actual_exp_avg": True,
              "initial_tensors_match_run002": True, "max_raw_norm_vs_clip_relative_error": max_norm_error,
              "overflow_retries": sum(e["overflow_retries"] for e in rows)}
    (folder / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    # Select all scientific artifacts; progress logs are outside the immutable payload.
    inventory = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.name != "transfer_inventory.json":
            with path.open("rb") as stream:
                sha = hashlib.file_digest(stream, "sha256").hexdigest()
            inventory.append({"path": path.relative_to(folder).as_posix(), "bytes": path.stat().st_size, "sha256": sha})
    (folder / "transfer_inventory.json").write_text(json.dumps({"files": inventory}, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    verify(sys.argv[1])
