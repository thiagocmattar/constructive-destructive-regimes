"""Verify retrieved bytes, accepted updates, token coverage, and norm agreement."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--bytes", type=int, required=True)
    args = parser.parse_args()
    archive_path = ROOT / "cache/spark/calibration-results.tar.gz"
    assert archive_path.stat().st_size == args.bytes and digest(archive_path) == args.sha256
    incoming = ROOT / "cache/spark/020-retrieved"
    incoming.mkdir(exist_ok=False)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(incoming, filter="data")
    inventory = json.loads((incoming / "calibration-transfer-inventory.json").read_text())
    for item in inventory["files"]:
        path = incoming / item["path"]
        assert path.resolve().is_relative_to(incoming.resolve())
        assert path.stat().st_size == item["bytes"] and digest(path) == item["sha256"]
    source_run = incoming / RUN.relative_to(ROOT)
    manifest = json.loads((source_run / "artifacts/attempt-02/manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["completed_updates"] == 9
    summary = json.loads((source_run / "artifacts/attempt-02/summary.json").read_text())
    max_norm_relative_error = 0.0
    for result in summary["models"].values():
        assert len(result["all_updates"]) == 3 and len(result["optimizer_step_seconds"]) == 2
        assert all(t > 0 and math.isfinite(t) for t in result["optimizer_step_seconds"])
        for event in result["all_updates"]:
            assert event["sequences"] == 1024
            assert event["input_tokens"] == 2097152 and event["supervised_tokens"] == 2096128
            assert event["overflow_retries"] == 0 and math.isfinite(event["loss"])
            assert 0 <= event["cumulative_ratio"] <= 1 + 1e-10
            assert 0 <= event["adaptive"]["cumulative_ratio"] <= 1 + 1e-10
            error = abs(event["gradient_norm"] - event["gradient_clip_returned_norm"]) / event["gradient_norm"]
            max_norm_relative_error = max(max_norm_relative_error, error)
            assert error < 1e-6
        for label in ("initial_validation", "final_validation"):
            validation = result[label]
            assert validation["input_tokens"] == 693668 and validation["supervised_tokens"] == 693329
            assert validation["sequences"] == 339 and validation["short_tail_included"]
    for item in inventory["files"]:
        if item["path"].startswith(RUN.relative_to(ROOT).as_posix() + "/artifacts/"):
            relative = Path(item["path"]).relative_to(RUN.relative_to(ROOT))
            target = RUN / relative
            assert not target.exists()
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(incoming / item["path"], target)
    for name in ("environment-lock.txt", "calibration-transfer-inventory.json"):
        shutil.copyfile(incoming / name, RUN / name)
    receipt = {"accepted": True, "files_verified": len(inventory["files"]),
               "archive_bytes": args.bytes, "archive_sha256": args.sha256,
               "completed_calibration_updates": 9,
               "max_raw_norm_vs_clip_relative_error": max_norm_relative_error,
               "checkpoints": "retained_on_Spark; compact inventories retrieved"}
    (RUN / "acceptance-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))
    for size, result in summary["models"].items():
        print(size, json.dumps({k: result[k] for k in (
            "optimizer_step_seconds", "initial_validation", "final_validation",
            "recovery_checkpoint_seconds", "projected_full_run_hours",
            "historical_initial_hash_matches", "estimated_three_checkpoint_bytes")}))
        print("peak_memory_GiB", max(e["peak_cuda_allocated_bytes"] for e in result["all_updates"]) / 2**30)


if __name__ == "__main__":
    main()
