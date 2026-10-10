"""Verify complete coverage and both streaming geometries against recovery state."""

import argparse
import json
import math
from pathlib import Path
import torch


def verify(root):
    root = Path(root)
    load = lambda name: json.loads((root / name).read_text())
    manifest, summary, config, identity = map(load, ["manifest.json", "summary.json",
                                                    "resolved_config.json", "input_identity.json"])
    updates = [e for line in (root / "events.jsonl").read_text().splitlines()
               if (e := json.loads(line))["event"] == "update"]
    assert manifest["status"] == "completed" and manifest["run_id"] == config["run_id"]
    assert summary["scope"] == "full_pass" and summary["all_training_tokens_covered"]
    assert summary["repeated_sequences"] == identity["repeated_sequences"] == 0
    assert len(updates) == summary["full_updates"] == summary["completed_updates"] == 712
    assert sum(e["input_tokens"] for e in updates) == identity["train"]["tokens"]
    assert sum(e["sequences"] for e in updates) == identity["sequence_count"]
    assert sum(e["supervised_tokens"] for e in updates) == identity["train"]["tokens"] - identity["sequence_count"]
    assert all(e["sequences"] == 1024 for e in updates[:-1]) and updates[-1]["sequences"] == 311
    raw_sum = adaptive_sum = 0.
    for i, e in enumerate(updates):
        assert e["update"] == i + 1 and e["gradient_step"] == i
        raw_sum += e["gradient_norm"]
        adaptive_sum += e["adaptive"]["direction_norm"]
        assert math.isfinite(e["loss"])
        for x, norm_key, cumulative_key, running in [
            (e, "gradient_norm", "cumulative_gradient_norm", raw_sum),
            (e["adaptive"], "direction_norm", "cumulative_direction_norm", adaptive_sum),
        ]:
            assert math.isfinite(x[norm_key]) and math.isfinite(x[cumulative_key])
            assert math.isclose(x["cumulative_norm_sum"], running, rel_tol=1e-12, abs_tol=1e-12)
            if running:
                assert 0 <= x["cumulative_ratio"] <= 1 + 1e-10
                assert math.isclose(x["cumulative_ratio"], x[cumulative_key] / running, rel_tol=1e-12)
            else:
                assert x["cumulative_ratio"] is None
            if x["adjacent_cosine"] is not None:
                assert -1 - 1e-10 <= x["adjacent_cosine"] <= 1 + 1e-10
            if i == 0:
                assert x["adjacent_cosine"] is None
        if e["gradient_norm"]:
            assert math.isclose(e["adaptive_to_raw_norm_ratio"],
                               e["adaptive"]["direction_norm"] / e["gradient_norm"], rel_tol=1e-12)
    state = torch.load(root / "checkpoints/final/recovery.pt", map_location="cpu", weights_only=False)
    assert state["data_position"] == identity["sequence_count"]
    assert state["order_sha256"] == identity["order_sha256"]
    assert state["config_sha256"] == manifest["config_sha256"]
    norms = {}
    for key, event, current, cumulative, dtype in [
        ("geometry", updates[-1], "gradient_norm", "cumulative_gradient_norm", torch.float32),
        ("adaptive_geometry", updates[-1]["adaptive"], "direction_norm", "cumulative_direction_norm", torch.float64),
    ]:
        g = state[key]
        assert g["updates"] == len(updates) and g["names"] == state["geometry"]["names"]
        assert all(x.dtype == torch.float64 for x in g["cumulative"])
        assert all(x.dtype == dtype for x in g["previous"])
        final_norm = float(torch.stack([s.square().sum() for s in g["cumulative"]]).sum().sqrt())
        previous_norm = float(torch.stack([s.double().square().sum() for s in g["previous"]]).sum().sqrt())
        assert math.isclose(final_norm, event[cumulative], rel_tol=1e-10, abs_tol=1e-10)
        assert math.isclose(previous_norm, event[current], rel_tol=1e-10, abs_tol=1e-10)
        norms[key] = final_norm
    # Reconstruct the final preconditioned direction from independent saved moments.
    environment = load("environment.json")
    name_groups = [environment["optimizer"]["decay_parameter_names"],
                   environment["optimizer"]["no_decay_parameter_names"]]
    saved_directions = dict(zip(state["adaptive_geometry"]["names"], state["adaptive_geometry"]["previous"]))
    for group, names in zip(state["optimizer"]["param_groups"], name_groups):
        assert len(group["params"]) == len(names)
        for pid, name in zip(group["params"], names):
            m = state["optimizer"]["state"][pid]
            t = int(m["step"].item())
            assert t == len(updates)
            b1, b2 = group["betas"]
            expected = (m["exp_avg"].double() / (1 - b1 ** t)) / (
                (m["exp_avg_sq"].double() / (1 - b2 ** t)).sqrt() + group["eps"])
            # FP64 CPU/CUDA sqrt/division can differ by a few ULPs.
            torch.testing.assert_close(expected, saved_directions[name], rtol=1e-12, atol=1e-12)
    assert environment["initial_parameter_sha256"] == config["model"]["initial_parameter_sha256"]
    for validation in [summary["initial_validation"], summary["final_validation"]]:
        assert validation["input_tokens"] == identity["validation"]["tokens"]
        assert validation["supervised_tokens"] == validation["input_tokens"] - validation["sequences"]
    assert config["measurements"]["adaptive_excludes"][:2] == ["learning_rate", "weight_decay"]
    return {"verified": True, "updates": len(updates), "training_input_tokens": identity["train"]["tokens"],
            "final_validation_loss": summary["final_validation"]["loss"],
            "final_raw_cumulative_norm": norms["geometry"],
            "final_adaptive_cumulative_norm": norms["adaptive_geometry"],
            "final_raw_ratio": updates[-1]["cumulative_ratio"],
            "final_adaptive_ratio": updates[-1]["adaptive"]["cumulative_ratio"],
            "final_adaptive_norm": updates[-1]["adaptive"]["direction_norm"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    print(json.dumps(verify(parser.parse_args().root), indent=2))
