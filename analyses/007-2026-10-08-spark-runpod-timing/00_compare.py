"""Compare recorded full-pass GPU runtimes with the Spark calibration."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
spark_run = ROOT / "runs/020-2026-10-08-dgx-spark-calibration"
spark_summary = json.loads((spark_run / "artifacts/attempt-02/summary.json").read_text())
records, sources = [], []


def read(path):
    raw = path.read_bytes()
    sources.append({"path": path.relative_to(ROOT).as_posix(), "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest()})
    return json.loads(raw)


read(spark_run / "artifacts/attempt-02/summary.json")
reference = read(spark_run / "recipes/14m/config.json")
for run in sorted((ROOT / "runs").iterdir()):
    if run.name[:3] not in {f"{n:03}" for n in range(2, 11)}:
        continue
    attempt = "attempt-02" if run.name.startswith("002-") else "attempt-01"
    folder = run / "artifacts" / attempt / "full-pass"
    summary = read(folder / "summary.json")
    environment = read(folder / "environment.json")
    manifest = read(folder / "manifest.json")
    config = read(run / "config.json")
    assert config["training"] == reference["training"]
    for field in ("train_sha256", "validation_sha256", "sequence_length", "loss_reduction"):
        assert config["data"][field] == reference["data"][field]
    assert manifest["status"] == "completed" and summary["completed_updates"] == 712
    assert summary["completed_input_tokens"] == 1491711416
    assert summary["completed_sequences"] == 728375 and summary["repeated_sequences"] == 0
    size = next(s for s in ("14m", "31m", "70m") if f"pythia{s}" in run.name)
    event_path = folder / "events.jsonl"
    raw_events = event_path.read_bytes()
    sources.append({"path": event_path.relative_to(ROOT).as_posix(), "bytes": len(raw_events),
                    "sha256": hashlib.sha256(raw_events).hexdigest()})
    events = [json.loads(line) for line in raw_events.decode("utf-8").splitlines()]
    updates = [e for e in events if e.get("event") == "update"]
    assert len(updates) == 712
    # Exclude first-update warmup and the final 311-sequence partial update.
    full = [e["update_seconds"] for e in updates if 2 <= e["update"] <= 711]
    assert len(full) == 710
    mean_seconds = statistics.mean(full)
    spark_seconds = statistics.mean(spark_summary["models"][size]["optimizer_step_seconds"])
    records.append({"run": run.name, "size": size, "gpu": environment["gpu"],
                    "seed": config["seeds"]["model"], "post_warmup_full_updates": len(full),
                    "mean_seconds_per_update": mean_seconds,
                    "actual_execution_minutes": (datetime.fromisoformat(manifest["finished_at"]) -
                                                 datetime.fromisoformat(manifest["started_at"])).total_seconds() / 60,
                    "spark_seconds_per_update": spark_seconds,
                    "spark_to_cloud_update_time_ratio": spark_seconds / mean_seconds})
output = {"scope": "operational_timing_comparison", "runs": records, "sources": sources,
          "spark_projected_hours": {s: v["projected_full_run_hours"] for s, v in spark_summary["models"].items()}}
(HERE / "results.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
for size in ("14m", "31m", "70m"):
    for gpu in ("NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 5090"):
        selected = [r for r in records if r["size"] == size and r["gpu"] == gpu]
        print(size, gpu, json.dumps({
            "mean_full_update_seconds": statistics.mean(r["mean_seconds_per_update"] for r in selected),
            "actual_execution_minutes": [r["actual_execution_minutes"] for r in selected],
            "spark_time_ratio": statistics.mean(r["spark_to_cloud_update_time_ratio"] for r in selected)}))
