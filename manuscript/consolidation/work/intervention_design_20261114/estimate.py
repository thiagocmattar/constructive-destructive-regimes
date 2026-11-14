"""Planning arithmetic only. Reads local evidence; never launches experiments.

Run from any directory: python path/to/estimate.py
The output is a range of qualified estimates, not measured job completion times.
"""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
QUOTE = json.loads((HERE / "hardware_quote.json").read_text())
PRICES = {
    row["displayName"]: {
        "secure": row["securePricePerHr"],
        "community": row["communityPricePerHr"],
    }
    for row in QUOTE["gpus"]
}

# Each pair is lower/upper seconds per update. These deliberately exceed or
# bracket the recorded sustained rates. Spark Transformer has only a 3-step
# mechanics measurement; its projection is substantially less certain.
RATES = {
    "mlp": {"Spark": {"ordinary": [.015, .03], "cap": [.06, .10]}},
    "resnet": {
        "Spark": {"ordinary": [16., 20.], "cap": [18., 24.]},
        "RTX 4090": {"ordinary": [3.5, 4.2], "cap": [3.8, 4.8]},
    },
    "transformer": {
        "Spark": {"ordinary": [45., 60.], "cap": [60., 80.]},
        "RTX 4090": {"ordinary": [10., 17.], "cap": [11., 18.]},
        "RTX 5090": {"ordinary": [8., 10.5], "cap": [8.2, 11.]},
    },
}
SETUPS = {
    "mlp": {"fork": 512, "end": 800},
    "resnet": {"fork": 256, "end": 576},
    "transformer": {"fork": 544, "end": 928},
}
WORKLOADS = {
    "initialization_0": {"initializations": 1, "ordinary_per_init": 9, "caps_per_init": 3},
    "additional_initializations_1_2": {"initializations": 2, "ordinary_per_init": 9, "caps_per_init": 3},
    "deferred_cap_lr_gross": {"initializations": 3, "ordinary_per_init": 3, "caps_per_init": 15},
}
FLEETS = {
    "spark_only": {"mlp": ("Spark", 1), "resnet": ("Spark", 1), "transformer": ("Spark", 1)},
    "balanced": {"mlp": ("Spark", 1), "resnet": ("RTX 4090", 1), "transformer": ("RTX 5090", 3)},
    "fast": {"mlp": ("Spark", 1), "resnet": ("RTX 4090", 2), "transformer": ("RTX 5090", 6)},
    "4090_fallback": {"mlp": ("Spark", 1), "resnet": ("RTX 4090", 2), "transformer": ("RTX 4090", 4)},
}
# Diagnostics, full held-out evaluations, terminal gradient and audit margin;
# terminal gradient is also explicitly counted once per continuation below.
WORK_MARGIN = [1.10, 1.20]
# Combined staging, qualification, retrieval per active worker; billed as time.
WORKER_OVERHEAD_HOURS = [.5, 1.0]
SPARK_OVERHEAD_HOURS = [.25, .5]
STORAGE_GB_PER_POD = 60  # 30 GB container + 30 GB temporary volume
STORAGE_PER_GPU_HOUR = STORAGE_GB_PER_POD * .1 / 730


def assignment_hours(durations, workers):
    """Whole continuations, longest first; no idealized fractional branch split."""
    loads = [0.] * workers
    for duration in sorted(durations, reverse=True):
        i = min(range(workers), key=loads.__getitem__)
        loads[i] += duration
    return max(loads)


def model_cost(case, workload, hardware, workers, endpoint):
    setup = SETUPS[case]
    rate = RATES[case][hardware]
    ordinary = rate["ordinary"][endpoint]
    cap = rate["cap"][endpoint]
    count = workload["initializations"]
    margin = WORK_MARGIN[endpoint]
    horizon = setup["end"] - setup["fork"]
    # Full 0->fork prefix per initialization, plus eight local-score captures.
    prefix = (setup["fork"] * ordinary + 8 * cap) * margin / 3600
    branch_times = [
        (horizon + 1) * ordinary * margin / 3600
    ] * (count * workload["ordinary_per_init"]) + [
        (horizon + 1) * cap * margin / 3600
    ] * (count * workload["caps_per_init"])
    overhead = (SPARK_OVERHEAD_HOURS if hardware == "Spark" else WORKER_OVERHEAD_HOURS)[endpoint]
    # Prefixes run concurrently on available workers before branch dispatch.
    # Additional workers are acquired near readiness, not idled throughout.
    critical_hours = math.ceil(count / workers) * prefix + assignment_hours(branch_times, workers) + overhead
    aggregate_hours = count * prefix + sum(branch_times) + workers * overhead
    return {
        "wall_hours": critical_hours,
        "aggregate_device_hours": aggregate_hours,
        "prefix_hours_each": prefix,
        "continuations": len(branch_times),
        "updates_excluding_prefixes": horizon * len(branch_times),
        "secure_usd": 0. if hardware == "Spark" else aggregate_hours * (PRICES[hardware]["secure"] + STORAGE_PER_GPU_HOUR),
        "community_usd": 0. if hardware == "Spark" else aggregate_hours * (PRICES[hardware]["community"] + STORAGE_PER_GPU_HOUR),
    }


def build():
    result = {
        "status": "planning_only_no_execution",
        "quote_utc": QUOTE["queried_utc"],
        "rates_seconds_per_update": RATES,
        "assumptions": {
            "full_prefixes_included": True,
            "extra_work_multiplier": WORK_MARGIN,
            "worker_overhead_hours": WORKER_OVERHEAD_HOURS,
            "spark_overhead_hours_per_model": SPARK_OVERHEAD_HOURS,
            "storage_gb_per_pod": STORAGE_GB_PER_POD,
            "storage_usd_per_gpu_hour": STORAGE_PER_GPU_HOUR,
            "no_reuse_credit": True,
            "excluded": ["capacity queue", "code preparation", "final figure/table authoring", "tax", "owned Spark electricity"],
            "community_limit": "Same runtime assumed only after actual worker qualification; a conditional quote, not a speed guarantee.",
            "spark_transformer_limit": "Extrapolated from three-update mechanics records, not a sustained run.",
        },
        "workloads": {},
    }
    for name, workload in WORKLOADS.items():
        per_case = workload["initializations"] * (workload["ordinary_per_init"] + workload["caps_per_init"])
        entry = {"definition": workload, "continuations_per_model": per_case, "total_continuations": 3 * per_case, "fleets": {}}
        for fleet_name, fleet in FLEETS.items():
            entry["fleets"][fleet_name] = []
            for i in range(2):
                cases = {case: model_cost(case, workload, *fleet[case], i) for case in SETUPS}
                # Single owned Spark serializes all three models in Spark-only.
                wall = sum(c["wall_hours"] for c in cases.values()) if fleet_name == "spark_only" else max(c["wall_hours"] for c in cases.values())
                entry["fleets"][fleet_name].append({
                    "wall_hours": wall,
                    "secure_usd": sum(c["secure_usd"] for c in cases.values()),
                    "community_usd": sum(c["community_usd"] for c in cases.values()),
                    "models": cases,
                })
        result["workloads"][name] = entry
    assert [v["total_continuations"] for v in result["workloads"].values()] == [36, 72, 162]
    assert sum((s["end"] - s["fork"]) * 12 for s in SETUPS.values()) == 11904
    for item in json.loads((HERE / "source_manifest.json").read_text())["files"]:
        source = ROOT / item["path"]
        assert source.stat().st_size == item["bytes"], item["path"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == item["sha256"], item["path"]
    for entry in result["workloads"].values():
        for bounds in entry["fleets"].values():
            assert bounds[0]["wall_hours"] <= bounds[1]["wall_hours"]
            assert bounds[0]["secure_usd"] <= bounds[1]["secure_usd"]
    return result


if __name__ == "__main__":
    result = build()
    (HERE / "estimates.json").write_bytes((json.dumps(result, indent=2) + "\n").encode("utf-8"))
    for name, entry in result["workloads"].items():
        print(name, entry["total_continuations"], "continuations")
        for fleet, (lo, hi) in entry["fleets"].items():
            print(f"  {fleet}: {lo['wall_hours']:.1f}-{hi['wall_hours']:.1f} h; secure ${lo['secure_usd']:.2f}-${hi['secure_usd']:.2f}; community ${lo['community_usd']:.2f}-${hi['community_usd']:.2f}")
