"""Prospective EOS budget from retained evidence; no training or cloud mutation."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def schedule(jobs, workers, speedup):
    """Independent whole trajectories, longest first; no data parallelism."""
    loads = [0.0] * workers
    for seconds in sorted(jobs, reverse=True):
        worker = min(range(workers), key=loads.__getitem__)
        loads[worker] += seconds / speedup / 3600
    return loads


def main():
    design = json.loads((HERE / "design.json").read_text())
    quote = json.loads((HERE / "pricing.json").read_text())
    paths = ["analyses/020-2026-10-14-eos-multitask/gd_overview_seed0_results.json",
             "runs/067-2026-10-14-eos-multitask-calibration/calibration_results.json"]
    legacy = []
    for rid in range(59, 63):
        path = f"analyses/019-2026-10-13-cifar10-eos-lr/results/Run{rid:03}.json"
        paths.append(path)
        row = read(path)
        legacy.append({"run_id": row["run_id"], "lr": row["lr"],
                       "first_negative_adjacent_update": row["streams"]["raw"]["first_negative_adjacent_update"],
                       "updates": row["updates"], "elapsed_seconds": row["elapsed_seconds"]})
    # Complete 500-update production durations, including recorded diagnostics.
    # Sources are the UTC queue start/end events, retained in timing_evidence.json.
    timings = json.loads((HERE / "timing_evidence.json").read_text())
    from datetime import datetime
    seconds = {}
    for task, rows in timings["completed_500_update_runs"].items():
        samples = [(datetime.fromisoformat(r["ended_utc"]) - datetime.fromisoformat(r["started_utc"])).total_seconds() for r in rows]
        seconds[task] = sum(samples) / len(samples)
    mlp_full = sum(r["elapsed_seconds"] for r in legacy) / len(legacy)
    jobs, by_task = [], {}
    for case in design["cases"]:
        count = len(case["lrs"]) * len(design["seeds"])
        new = count - case["reused_complete_runs"]
        duration = mlp_full if case["task"] == "mlp" else seconds[case["task"]] * case["updates"] / 500
        jobs.extend([duration] * new)
        by_task[case["task"]] = {"total_conditions": count, "new_conditions": new,
            "updates_per_condition": case["updates"], "spark_hours_per_new_condition": duration / 3600,
            "new_spark_gpu_hours": duration * new / 3600}
    assert sum(x["total_conditions"] for x in by_task.values()) == 30
    assert len(jobs) == 26
    storage = quote["storage"]
    network = storage["network_volume_gb"] * storage["network_usd_per_gb_month"] * storage["network_retention_hours"] / storage["month_hours"]
    scenarios = []
    for speedup in [1, 2, 3, 4]:
        loads = schedule(jobs, quote["gpu_count"], speedup)
        # Qualified planning allowance, not a confidence interval or measured cloud ETC.
        wall = max(loads) * 1.20 + 2
        compute = wall * quote["gpu_count"] * quote["usd_per_gpu_hour"]
        container = storage["container_total_gb"] * storage["container_usd_per_gb_month"] * wall / storage["month_hours"]
        scenarios.append({"assumed_speedup_over_spark": speedup,
            "cloud_speedup_measured": False, "workers": quote["gpu_count"],
            "scheduled_training_wall_hours": max(loads), "planning_wall_hours": wall,
            "planning_usd": compute + container + network,
            "fits_120_hour_220_usd_limits": wall <= quote["limits"]["maximum_fleet_wall_hours"] and compute + container + network <= quote["limits"]["maximum_total_usd"]})
    maxwall = quote["limits"]["maximum_fleet_wall_hours"]
    maximum = maxwall * quote["gpu_count"] * quote["usd_per_gpu_hour"] + network + storage["container_total_gb"] * storage["container_usd_per_gb_month"] * maxwall / storage["month_hours"]
    assert maximum < quote["limits"]["maximum_total_usd"]
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
    for name in ["design.json", "pricing.json", "timing_evidence.json"]:
        hashes[str((HERE / name).relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
    result = {"status": "prospective_plan_arithmetic_verified", "not_training_results": True,
        "by_task": by_task, "legacy_mlp_first_sign_changes": legacy,
        "new_spark_gpu_hours": sum(jobs) / 3600,
        "spark_planning_hours_with_20_percent_allowance": sum(jobs) / 3600 * 1.2,
        "cloud_scenarios": scenarios, "maximum_priced_usd": maximum,
        "initial_30_minute_two_gpu_compute_usd": .5 * quote["gpu_count"] * quote["usd_per_gpu_hour"],
        "input_sha256": hashes,
        "limitations": ["Long-horizon and late eigensolver costs are unmeasured for ResNet/Transformer.",
                        "Cloud speedup scenarios are assumptions, not measured throughput.",
                        "No paid compute or changed training inputs executed by this script."]}
    (HERE / "estimate.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["status", "new_spark_gpu_hours", "cloud_scenarios", "maximum_priced_usd"]}, indent=2))


if __name__ == "__main__":
    main()
