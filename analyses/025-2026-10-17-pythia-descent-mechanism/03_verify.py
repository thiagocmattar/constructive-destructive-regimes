"""Verify terminal H021 coverage, descent identities and parent/checkpoint integrity."""
from pathlib import Path
import argparse
import hashlib
import json
import math


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert not manifest["smoke_calibration"], "A mechanics smoke is not scientific evidence"
    config = json.loads((output / "resolved_config.json").read_text())
    assert sha(output / "resolved_config.json") == manifest["config_sha256"]
    causal = config["causal"]
    start = causal.get("junction_update", 0)
    end = start + causal["branch_updates"] if start else causal["baseline_updates"]
    events = [json.loads(s) for s in (output / "events.jsonl").read_text().splitlines()]
    updates = [e for e in events if e["event"] == "update"]
    assert [e["update"] for e in updates] == list(range(start + 1, end + 1))
    for e in updates:
        assert math.isfinite(e["loss"]) and e["loss"] > 0
        assert e["sequences"] == config["training"]["effective_batch_sequences"]
        assert e["supervised_tokens"] == e["sequences"] * (config["data"]["sequence_length"] - 1)
        assert e["gradient_step"] == e["update"] - 1
        for stream, norm_key in ((e, "gradient_norm"), (e["adaptive"], "direction_norm"),
                                 (e["momentum"], "momentum_norm"), (e["actual_parameter_update"], "norm")):
            assert math.isfinite(stream[norm_key]) and stream[norm_key] >= 0
            if stream["adjacent_cosine"] is not None:
                assert abs(stream["adjacent_cosine"]) <= 1 + 1e-10
        if start:
            assert e["intervention"]["nominal_adaptive_step_norm"] >= 0
    probes = [json.loads(s) for s in (output / "probe_events.jsonl").read_text().splitlines()]
    selected = [u for u in causal["measurement_updates"] if start < u <= end]
    expected = {"applied_intervention"} if start else {
        "native_adamw", "adam", "momentum", "current_preconditioned", "gradient", "adam_scale_0.5", "adam_scale_2"}
    assert len(probes) == len(selected) * len(expected)
    undefined = 0
    for update in selected:
        rows = [r for r in probes if r["update"] == update]
        assert {r["candidate"] for r in rows} == expected
        for r in rows:
            if r.get("undefined_reason"):
                undefined += 1
                continue
            assert all(math.isfinite(r[key]) for key in (
                "loss_before", "loss_after", "actual_decrease", "first_order_decrease", "nonlinear_residual"))
            assert abs(r["actual_decrease"] - (r["first_order_decrease"] - r["nonlinear_residual"])) < 1e-12
            if r["descent_cosine"] is not None:
                assert abs(r["descent_cosine"]) <= 1 + 1e-10
            if r["candidate"] == "adam":
                assert abs(r["native_reconstruction_loss_gap"]) < 1e-5
                assert r["native_reconstruction_max_parameter_error"] < 1e-6
        if not start:
            target = next(r["nominal_adaptive_step_norm"] for r in rows if r["candidate"] == "adam")
            for r in rows:
                if r["candidate"] in {"momentum", "current_preconditioned", "gradient"} and not r.get("undefined_reason"):
                    assert math.isclose(r["nominal_adaptive_step_norm"], target, rel_tol=1e-10, abs_tol=1e-12)
    checkpoints = output / "checkpoints"
    required = ["final.pt"] + ([] if start else [f"update-{u}.pt" for u in causal["junctions"]])
    assert all((checkpoints / p).is_file() for p in required)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["completed_updates"] == end
    identity = json.loads((output / "input_identity.json").read_text())
    assert identity["order_sha256"] == causal["order_sha256"]
    assert identity["probe_sha256"] == causal["probe_sha256"]
    validation = summary["final_validation"]
    validation_tokens = identity["validation"]["tokens"]
    validation_sequences = math.ceil(validation_tokens / config["data"]["sequence_length"])
    assert validation["input_tokens"] == validation_tokens
    assert validation["sequences"] == validation_sequences
    assert validation["supervised_tokens"] == validation_tokens - validation_sequences
    assert summary["final_validation"]["short_tail_included"]
    assert math.isfinite(summary["final_probe_loss"])
    result = dict(verified=True, completed_updates=end, first_update=start + 1,
        accepted_updates=len(updates), probe_records=len(probes), undefined_probe_records=undefined,
        actual_descent_accounting=True, norm_controlled_one_step_directions=not bool(start),
        probe_sha256=summary["probe_indices_sha256"],
        checkpoints_sha256={p: sha(checkpoints / p) for p in required})
    if start:
        receipt = json.loads((output / "parent_receipt.json").read_text())
        parent = (output.parents[2] / causal["parent_output"]).resolve()
        assert sha(parent / "checkpoints" / f"update-{start}.pt") == receipt["checkpoint_sha256"]
        assert sha(parent / "events.jsonl") == receipt["parent_events_sha256"]
        assert sha(parent / "resolved_config.json") == receipt["parent_config_sha256"]
        assert sha(parent / "verification.json") == receipt["verification_sha256"]
        parent_updates = {e["update"]: e for e in
            (json.loads(s) for s in (parent / "events.jsonl").read_text().splitlines()) if e["event"] == "update"}
        for event in updates:
            reference = parent_updates[event["update"]]
            assert event["learning_rate"] == reference["learning_rate"]
            target = reference["learning_rate"] * reference["adaptive"]["direction_norm"]
            assert math.isclose(event["intervention"]["nominal_adaptive_step_norm"], target, rel_tol=1e-10)
        result["matched_parent_adaptive_step_schedule"] = True
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    args = parser.parse_args()
    record = verify(args.output)
    (Path(args.output) / "verification.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
