"""Freeze four logarithmically spaced additions to the warmup-only LR cohort."""
from pathlib import Path
import copy
import hashlib
import json
import shutil
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REFERENCE = ROOT / "runs/021-2026-10-08-pythia14m-no-decay-base"
STAGE = ROOT / "cache/spark/no-decay-inputs"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    baseline = json.loads((REFERENCE / "config.json").read_text())
    records = []
    for rid, exponent in zip(range(24, 28), (-11/3, -10/3, -8/3, -7/3)):
        peak = 10 ** exponent
        name = f"{rid:03d}-2026-10-09-pythia14m-no-decay-intermediate-{rid-23}"
        run = ROOT / "runs" / name
        if (run / "artifacts").exists() or (run / "prelaunch/launch.json").exists():
            raise RuntimeError("Cannot rewrite executed scientific inputs.")
        run.mkdir(exist_ok=True)
        (run / "prelaunch").mkdir(exist_ok=True)
        config = copy.deepcopy(baseline)
        config.update(run_id=f"{rid:03d}", name=name.split("-", 4)[-1], hypothesis="H008",
                      matched_reference_run="021")
        config["training"]["peak_learning_rate"] = peak
        config["execution"] = {"provider": "runpod", "workers": 1, "cohort_workers": 1,
            "maximum_worker_hours": 3, "maximum_cohort_cost_usd": 15,
            "maximum_price_per_gpu_hour": 0.89, "retrieval_margin_seconds": 600,
            "network_storage_gb": 60, "gpu_count": 2, "independent_jobs_per_gpu": 2,
            "gpu_type": "NVIDIA GeForce RTX 4090"}
        save(run / "config.json", config)
        for filename in ("01_train.py", "model_config.json"):
            shutil.copyfile(REFERENCE / filename, run / filename)
        (run / "README.md").write_text(f"""# Run {rid:03d}: intermediate warmup-only LR {peak:.12g}

Status: prepared; user authorized Runpod execution of four intermediate LR points.
H008 exploratory comparison. One full MiniPile pass, 712 updates, seed1234;
frozen initialization, token bytes, order, batch and AdamW recipe match Run021.
Only peak LR varies within the new four-run cohort. Warmup-only, no LR decay.
Every accepted update records raw gradient g, actual uncorrected Adam moment m,
and adaptive direction d: adjacent cosine, norm, A, B, B/A and norm ratios.
Full initial/final token-weighted validation includes the tail. Recovery states
are retained initially, every100 updates, at356, and finally, with all streams,
optimizer/scaler/RNG/order position. Original Runs021-023 remain frozen.
Hardware/runtime differences from Spark must be disclosed in pooled comparisons.
See [Analysis009](../../analyses/{HERE.name}/README.md) for execution and limits.
""", encoding="utf-8")
        records.append({"run_id": f"{rid:03d}", "run": name, "peak_lr": peak,
                        "log10_lr": exponent})
    save(HERE / "cohort.json", {"runs": records, "scope": "four_intermediate_full_passes",
                               "spacing": "equal_log10_thirds_of_each_decade"})
    shutil.copyfile(ROOT / "analyses/008-2026-10-08-no-decay-momentum/02_verify.py", HERE / "02_verify.py")
    files = list((ROOT / "src/baseline").glob("*.py"))
    files += [HERE / filename for filename in ("cohort.json", "01_worker.py", "02_verify.py", "03_setup.sh")]
    files += [ROOT / "tests/test_adam_momentum.py"]
    for record in records:
        files += [ROOT / "runs" / record["run"] / filename for filename in
                  ("config.json", "01_train.py", "model_config.json")]
    inventory = {"files": [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size,
                             "sha256": digest(p)} for p in sorted(files)]}
    save(HERE / "source_inventory.json", inventory)
    packet = ROOT / "cache/runpod/intermediate-lr-source.tar.gz"
    with tarfile.open(packet, "w:gz") as archive:
        for p in files + [HERE / "source_inventory.json"]:
            archive.add(p, arcname=p.relative_to(ROOT).as_posix())
        archive.add(STAGE / "initial-model", arcname="inputs/initial-model")
        for split in ("train", "validation"):
            archive.add(STAGE / split, arcname="data/" + split)
    save(HERE / "packet_identity.json", {"bytes": packet.stat().st_size, "sha256": digest(packet)})
    print(json.dumps({"runs": records, "packet_bytes": packet.stat().st_size,
                      "packet_sha256": digest(packet)}))


if __name__ == "__main__":
    main()
