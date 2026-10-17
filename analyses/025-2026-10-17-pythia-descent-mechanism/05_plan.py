"""Balance the fixed H021 families and qualify cost from recorded live prices."""
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent


def main():
    cohort = json.loads((HERE / "cohort.json").read_text())
    historical = json.loads((HERE / "historical_timing.json").read_text())
    timing = {r["size"]: r["mean_update_seconds"] for r in historical if r["seed"] == 1234}
    baselines = [r for r in cohort if r["condition"] == "baseline"]
    queues, loads = [[], []], [0., 0.]
    for baseline in sorted(baselines, key=lambda r: -timing[r["size"]]):
        worker = min(range(2), key=lambda i: loads[i])
        family = dict(baseline=baseline["run"], branches=[r["run"] for r in cohort if r.get("parent") == baseline["run"]],
                      size=baseline["size"], seed=baseline["seed"])
        queues[worker].append(family)
        loads[worker] += 450 * timing[baseline["size"]]
    (HERE / "worker_plan.json").write_text(json.dumps(queues, indent=2) + "\n", encoding="utf-8", newline="\n")
    prices = json.loads((HERE / "live_price.json").read_text())
    gpu_price = prices["rtx4090"]["price"]["secure"]
    storage_hourly = (200 * .07 + 2 * 30 * .10) / 730
    record = dict(status="proposal_not_spending_authorization", workers=2,
        gpu="NVIDIA GeForce RTX 4090", gpu_hourly_price=gpu_price,
        combined_gpu_hourly_price=2*gpu_price, network_volume_gb=200,
        container_disk_gb_per_worker=30, storage_hourly_estimate=storage_hourly,
        storage_source="https://docs.runpod.io/pods/pricing", monthly_hour_conversion=730,
        training_updates=4050, baseline_updates=3150, continuation_updates=900,
        historical_native_gpu_hours=sum(loads)/3600, historical_worker_hours=[s/3600 for s in loads],
        qualification="Native timing from seed1234 RTX4090; new FP32 probes and new runtime placement need paid calibration",
        assumed_new_diagnostic_allowance=1.5, setup_allowance_hours=.75, retrieval_reserve_hours=1.5,
        qualified_elapsed_estimate_hours=1.5*max(loads)/3600+.75+1.5,
        proposed_maximum_elapsed_hours=12, proposed_maximum_total_usd=25,
        maximum_gpu_and_storage_estimate_usd=12*(2*gpu_price+storage_hourly),
        checkpoint_retention="fork175/fork225 and terminal recovery states; rolling state until terminal verification",
        expected_retained_checkpoint_bytes_upper=120_000_000_000,
        retrieval="individual verified files, avoiding a second local archive copy",
        calibration_gate="native/probe timing and VRAM must fit remaining training deadline; otherwise stop that worker",
        training_deadline_hours=10.5, guard="independent local process stops only lease-attributed task pods by12h",
        cleanup="download and verify each inventory/scientific result; then delete only this task's pods and new volume")
    assert record["maximum_gpu_and_storage_estimate_usd"] < record["proposed_maximum_total_usd"]
    (HERE / "launch_plan.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
