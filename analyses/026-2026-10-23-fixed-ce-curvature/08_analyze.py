"""Verify frozen H022 evidence and reduce the predeclared paired endpoints."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.fixed_ce_curvature.core import save_json, sha, stamp, tensor_hash

HERE = Path(__file__).resolve().parent
ARMS = ("native32", "fixed32", "ce64")
STATES = sorted(set(range(0, 10001, 500)) | {9250, 9750})
LATE = (9000, 9250, 9500, 9750, 10000)


def paired_summary(values):
    values = np.asarray(values, dtype=float)
    assert values.size == 10 and np.isfinite(values).all() and (values > 0).all()
    logs = np.log(values)
    mean, se = float(logs.mean()), float(stats.sem(logs))
    interval = mean + np.array([-1., 1.]) * stats.t.ppf(0.975, len(logs) - 1) * se
    independent = stats.ttest_1samp(logs, 0).confidence_interval(confidence_level=0.95)
    np.testing.assert_allclose(interval, [independent.low, independent.high], atol=1e-14)
    signs = np.array(list(itertools.product((-1, 1), repeat=10)))
    permuted = np.mean(signs * logs, axis=1)
    p = float(np.mean(np.abs(permuted) >= abs(mean) - 1e-15))
    return {"seed_ratios": values.tolist(), "geometric_ratio": float(np.exp(mean)),
            "ci95": np.exp(interval).tolist(), "sign_flip_two_sided_p": p,
            "log_mean": mean, "log_standard_error": se, "n": len(logs)}


def verify_run(entry, verify_hashes=True):
    directory = ROOT / "runs" / entry["run"]
    output = directory / "artifacts/attempt-01"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["updates"] == 10000
    config = json.loads((directory / "config.json").read_text())
    assert manifest["initial_tensor_sha256"] == config["initial_tensor_sha256"]
    assert manifest["data_sha256"] == config["data_sha256"]
    assert sha(output / "config.json") == manifest["config_sha256"]
    first = torch.load(output / "states/state-000000.pt", map_location="cpu", weights_only=False)
    final = torch.load(output / "states/state-010000.pt", map_location="cpu", weights_only=False)
    assert tensor_hash(first["parameters"].values()) == manifest["initial_tensor_sha256"]
    assert tensor_hash(final["parameters"].values()) == manifest["final_tensor_sha256"]
    assert final["geometry"]["count"] == 10000
    proof = json.loads((output / "inventory.json").read_text())
    if verify_hashes:
        for item in proof["files"]:
            path = output / item["path"]
            assert path.stat().st_size == item["bytes"], str(path)
            assert sha(path) == item["sha256"], str(path)
    trajectories = {k: [] for k in ("loss", "collapse", "accuracy", "raw_cosine", "momentum_cosine",
                                   "adaptive_cosine", "update_cosine", "gradient_norm", "update_norm")}
    exposure, ref_collapse = 0, 0
    max_gradient_error = 0.
    previous_A = {name: 0. for name in ("raw", "momentum", "adaptive", "update")}
    last = None
    with gzip.open(output / "metrics.jsonl.gz", "rt") as stream:
        for count, line in enumerate(stream, 1):
            row = json.loads(line)
            # The raw "update" key names the actual-displacement geometry stream.
            # Its integer clock is recovered exactly as gradient_state + 1.
            assert isinstance(row["update"], dict) and row["gradient_state"] == count - 1
            assert row["optimization_loss"] >= 0 and row["common_logits_loss"] >= 0
            for name in previous_A:
                g = row[name]
                assert g["A"] >= previous_A[name] - 1e-12
                assert g["B"] <= g["A"] * (1 + 1e-10)
                if g["ratio"] is not None:
                    assert -1e-12 <= g["ratio"] <= 1 + 1e-10
                if g["cosine"] is not None:
                    assert -1 - 1e-10 <= g["cosine"] <= 1 + 1e-10
                previous_A[name] = g["A"]
            trajectories["loss"].append(row["common_logits_loss"])
            trajectories["collapse"].append(row["native_collapse_fraction"])
            trajectories["accuracy"].append(row["train_accuracy"])
            trajectories["gradient_norm"].append(row["raw"]["norm"])
            trajectories["update_norm"].append(row["update"]["norm"])
            for name in ("raw", "momentum", "adaptive", "update"):
                trajectories[name + "_cosine"].append(row[name]["cosine"])
            if count > 9000:
                exposure += row["native_collapse_fraction"] >= 0.1
                ref_collapse += row["reference64_collapse_fraction"] > 0
            if row["fixed_reference_logit_gradient_relative_error"] is not None:
                max_gradient_error = max(max_gradient_error, row["fixed_reference_logit_gradient_relative_error"])
            last = row
    assert count == 10000
    for index, name in enumerate(("raw", "momentum", "adaptive", "update")):
        np.testing.assert_allclose(float(final["geometry"]["A"][index]), last[name]["A"], rtol=1e-12)
        np.testing.assert_allclose(float(final["geometry"]["total"][index].norm()), last[name]["B"], rtol=1e-10)
    curvature = [json.loads(line) for line in (output / "curvature.jsonl").read_text().splitlines()]
    regular = sorted([r for r in curvature if r["checkpoint"].startswith("state-")], key=lambda r: r["state"])
    assert [r["state"] for r in regular] == STATES
    assert len({r["checkpoint"] for r in curvature}) == len(curvature)
    for record in curvature:
        assert record["ordinary"]["relative_residual"] <= 0.005
        assert record["ordinary"]["which"] == "LA"
        assert record["ordinary"]["objective_examples"] == 480
        assert record["ordinary"]["precision"] == "whole-model-fp64"
        assert record["gradient_curvature"] <= record["ordinary"]["value"] * 1.01 + 1e-12
        if record["state"] > 0 and record["checkpoint"].startswith("state-"):
            assert record["adam_scaled"]["relative_residual"] <= 0.005
        if record["state"] < 10000 and record["checkpoint"].startswith("state-"):
            assert "update_curvature" in record
            assert record["update_curvature"] <= record["ordinary"]["value"] * 1.01 + 1e-12
            np.testing.assert_allclose(record["actual_loss_change"],
                record["linear_change"] + record["quadratic_term"] + record["taylor_remainder"],
                rtol=1e-12, atol=1e-15)
        if record["state"] == 10000:
            assert "update_curvature" not in record
    late = [r for r in regular if r["state"] in LATE]
    assert len(late) == 5
    largest = [r["ordinary"]["value"] for r in late]
    assert min(largest) > 0
    summary = {**entry, "initial_tensor_sha256": manifest["initial_tensor_sha256"],
        "data_sha256": manifest["data_sha256"], "files": len(proof["files"]), "bytes": proof["total_bytes"],
        "late_lambda_geometric_mean": float(np.exp(np.log(largest).mean())),
        "late_gradient_curvature_mean": float(np.mean([r["gradient_curvature"] for r in late])),
        "late_update_curvature_mean": float(np.mean([r["update_curvature"] for r in late if "update_curvature" in r])),
        "late_adam_lambda_geometric_mean": float(np.exp(np.log([r["adam_scaled"]["value"] for r in late]).mean())),
        "exposure_states_last1000": exposure, "collapse_qualified": exposure >= 100,
        "reference64_collapse_states_last1000": ref_collapse,
        "maximum_native_collapse_fraction": max(trajectories["collapse"]),
        "fixed_reference_gradient_relative_error_max": max_gradient_error,
        "final_common_loss": regular[-1]["train_loss"], "final_heldout_loss": regular[-1]["heldout_loss"],
        "final_heldout_accuracy": regular[-1]["heldout_accuracy"],
        "final_parameter_norm": regular[-1]["parameter_norm"],
        "last_update_loss": last["common_logits_loss"],
        "last_margin_median": last["margin_median"],
        "seconds_training": manifest["seconds"], "peak_memory_gb": manifest["peak_memory_gb"],
        "spectra": len(curvature), "regular_states": len(regular)}
    return summary, trajectories, regular, curvature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    cohort = json.loads((HERE / "cohort.json").read_text())
    summaries, plots, records, all_curvatures = [], {}, {}, {}
    for entry in cohort:
        output = ROOT / "runs" / entry["run"] / "artifacts/attempt-01"
        if not (output / "manifest.json").exists() or json.loads((output / "manifest.json").read_text())["status"] != "complete":
            if args.partial:
                continue
            raise RuntimeError("Missing complete condition: " + entry["run"])
        summary, trajectories, regular, curvature = verify_run(entry)
        summaries.append(summary)
        plots[(entry["seed"], entry["arm"])] = trajectories
        records[(entry["seed"], entry["arm"])] = regular
        all_curvatures[(entry["seed"], entry["arm"])] = curvature
    if args.partial:
        save_json(HERE / "partial_results.json", {"utc": stamp(), "runs": summaries})
        print(json.dumps({"completed": len(summaries), "run_summaries": summaries}, indent=2))
        return
    assert len(summaries) == 30
    by_key = {(r["seed"], r["arm"]): r for r in summaries}
    for seed in range(10):
        assert len({by_key[(seed, arm)]["initial_tensor_sha256"] for arm in ARMS}) == 1
    assert len({r["initial_tensor_sha256"] for r in summaries}) == 10
    assert len({r["data_sha256"] for r in summaries}) == 1
    primary = {}
    for arm in ("fixed32", "ce64"):
        ratios = [by_key[(s, arm)]["late_lambda_geometric_mean"] /
                  by_key[(s, "native32")]["late_lambda_geometric_mean"] for s in range(10)]
        primary[arm + "_over_native32"] = paired_summary(ratios)
    primary["fixed32_over_ce64"] = paired_summary([
        by_key[(s, "fixed32")]["late_lambda_geometric_mean"] /
        by_key[(s, "ce64")]["late_lambda_geometric_mean"] for s in range(10)])
    qualification = {
        "required_qualified_native_seeds": 8,
        "qualified_native_seeds": sum(by_key[(s, "native32")]["collapse_qualified"] for s in range(10)),
        "reference64_late_collapse_states": sum(by_key[(s, "ce64")]["reference64_collapse_states_last1000"] for s in range(10)),
        "fixed_reference_gradient_error_max": max(by_key[(s, "fixed32")]["fixed_reference_gradient_relative_error_max"] for s in range(10))}
    qualification["mechanism_qualified"] = qualification["qualified_native_seeds"] >= 8 and qualification["reference64_late_collapse_states"] == 0
    lower, upper = primary["fixed32_over_native32"]["ci95"]
    verdict = ("material_raw_hessian_flattening" if upper < .8 else
               "approximate_equivalence" if lower > .8 and upper < 1.25 else
               "rules_out_20_percent_average_reduction" if lower > .8 else "inconclusive")
    matched = []
    for level in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        prefix = f"loss-{level:.0e}-step-"
        for seed in range(10):
            for arm in ("fixed32", "ce64"):
                a = [r for r in all_curvatures[(seed, "native32")] if r["checkpoint"].startswith(prefix)]
                b = [r for r in all_curvatures[(seed, arm)] if r["checkpoint"].startswith(prefix)]
                if a and b:
                    loss_ratio = b[0]["train_loss"] / a[0]["train_loss"]
                    valid = 1 / 1.1 <= loss_ratio <= 1.1
                    matched.append({"level": level, "seed": seed, "arm": arm, "qualified": valid,
                        "loss_ratio": loss_ratio, "curvature_ratio": b[0]["ordinary"]["value"] / a[0]["ordinary"]["value"],
                        "native_state": a[0]["state"], "comparison_state": b[0]["state"]})
                else:
                    matched.append({"level": level, "seed": seed, "arm": arm, "qualified": False,
                                    "reason": "level_not_reached_by_both"})
    results = {"utc": stamp(), "hypothesis": "H022", "status": "complete",
               "coverage": {"runs": 30, "updates": 300000, "paired_seeds": 10,
                            "training_pairs": 480, "heldout_pairs": 481,
                            "regular_curvature_states": 690,
                            "ordinary_eigenpairs": sum(r["spectra"] for r in summaries),
                            "adam_scaled_eigenpairs": 660},
               "qualification": qualification, "primary": primary, "raw_endpoint_verdict": verdict,
               "interpretation_status": verdict if qualification["mechanism_qualified"] else
                   "collapse_mechanism_unqualified__" + verdict,
               "runs": summaries, "matched_loss": matched,
               "cost": {"gpu_rental_usd": 0, "cloud_resources_created": 0},
               "total_evidence_bytes": sum(r["bytes"] for r in summaries)}
    save_json(HERE / "results.json", results)
    plot_data = {"states": STATES, "late_states": list(LATE), "curves": {},
                 "primary": primary, "matched_loss": matched, "qualification": qualification,
                 "source": "verified complete owning-run metric and curvature histories"}
    for arm in ARMS:
        selected = [plots[(s, arm)] for s in range(10)]
        sample = np.arange(0, 10000, 20)
        plot_data["curves"][arm] = {
            "loss_steps": (sample + 1).tolist(),
            "loss_seed_curves": [[t["loss"][i] for i in sample] for t in selected],
            "collapse_seed_curves": [[t["collapse"][i] for i in sample] for t in selected],
            "ordinary_seed_curves": [[r["ordinary"]["value"] for r in records[(s, arm)]] for s in range(10)],
            "gradient_seed_curves": [[r["gradient_curvature"] for r in records[(s, arm)]] for s in range(10)],
            "update_seed_curves": [[r.get("update_curvature") for r in records[(s, arm)]] for s in range(10)],
            "adam_seed_curves": [[r.get("adam_scaled", {}).get("value") for r in records[(s, arm)]] for s in range(10)]}
    save_json(HERE / "figure_data.json", plot_data)
    save_json(HERE / "verification.json", {"utc": stamp(), "status": "passed",
        "checks": ["all retained per-run byte sizes and SHA256", "300k contiguous updates",
                   "exact initial tensors paired across arms, ten distinct seeds", "complete fixed split identity",
                   "streaming geometry bounds and indexing; saved prefix vectors checked on CPU",
                   "initial/final tensor hashes independently reconstructed from checkpoints",
                   "directional Rayleigh quotients bounded by maximum algebraic eigenvalues",
                   "690 regular states and all loss-crossing states",
                   "ordinary and Adam-scaled eigenpair residuals", "terminal actual-update curvature undefined",
                   "exact stored Taylor decomposition", "independent paired CI calculation"],
        "files": sum(r["files"] for r in summaries), "bytes": results["total_evidence_bytes"],
        "source_inventory_sha256": sha(HERE / "source_inventory.json")})
    print(json.dumps({"qualification": qualification, "primary": primary,
                      "verdict": results["interpretation_status"], "coverage": results["coverage"]}, indent=2))


if __name__ == "__main__":
    main()
