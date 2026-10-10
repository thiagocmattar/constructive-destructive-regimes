"""Verify full data coverage, per-update scalar identities, and saved gradient state."""

import argparse
import json
import math
from pathlib import Path

import torch


def verify(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    config = json.loads((root / "resolved_config.json").read_text())
    identity = json.loads((root / "input_identity.json").read_text())
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    updates = [e for e in events if e["event"] == "update"]
    assert manifest["status"] == "completed"
    assert summary["scope"] == "full_pass"
    assert summary["all_training_tokens_covered"] and summary["repeated_sequences"] == 0
    assert len(updates) == summary["full_updates"] == summary["completed_updates"]
    assert sum(e["input_tokens"] for e in updates) == identity["train"]["tokens"]
    assert sum(e["sequences"] for e in updates) == identity["sequence_count"]
    assert sum(e["supervised_tokens"] for e in updates) == identity["train"]["tokens"] - identity["sequence_count"]
    running = 0.0
    for i, event in enumerate(updates):
        assert event["update"] == i + 1 and event["gradient_step"] == i
        running += event["gradient_norm"]
        assert math.isclose(event["cumulative_norm_sum"], running, rel_tol=1e-12, abs_tol=1e-12)
        assert math.isfinite(event["loss"]) and math.isfinite(event["cumulative_gradient_norm"])
        if event["cumulative_ratio"] is not None:
            assert 0 <= event["cumulative_ratio"] <= 1 + 1e-10
            assert math.isclose(event["cumulative_ratio"], event["cumulative_gradient_norm"] / running, rel_tol=1e-12)
        if event["adjacent_cosine"] is not None:
            assert -1-1e-10 <= event["adjacent_cosine"] <= 1+1e-10
        if i == 0:
            assert event["adjacent_cosine"] is None
    state = torch.load(root / "checkpoints/final/recovery.pt", map_location="cpu", weights_only=False)
    geometry = state["geometry"]
    assert geometry["updates"] == len(updates)
    assert state["data_position"] == identity["sequence_count"]
    assert state["order_sha256"] == identity["order_sha256"]
    assert state["config_sha256"] == manifest["config_sha256"]
    norm = float(torch.stack([s.square().sum() for s in geometry["cumulative"]]).sum().sqrt())
    assert math.isclose(norm, updates[-1]["cumulative_gradient_norm"], rel_tol=1e-10, abs_tol=1e-10)
    previous_norm = float(torch.stack([s.double().square().sum() for s in geometry["previous"]]).sum().sqrt())
    assert math.isclose(previous_norm, updates[-1]["gradient_norm"], rel_tol=1e-10, abs_tol=1e-10)
    validation = summary["final_validation"]
    assert validation["input_tokens"] == identity["validation"]["tokens"]
    assert validation["supervised_tokens"] == validation["input_tokens"] - validation["sequences"]
    assert config["training"]["optimizer"] == "adamw"
    return {"verified": True, "updates": len(updates), "training_input_tokens": identity["train"]["tokens"],
            "final_validation_loss": validation["loss"], "final_adjacent_cosine": updates[-1]["adjacent_cosine"],
            "final_cumulative_gradient_norm": norm,
            "final_cumulative_norm_sum": running, "final_cumulative_ratio": updates[-1]["cumulative_ratio"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
