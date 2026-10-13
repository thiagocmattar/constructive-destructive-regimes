"""Freeze eight user-authorized Spark inputs and a minimal execution packet."""
import copy
import json
import tarfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import file_hash, stamp, write_json

HERE = Path(__file__).resolve().parent
REMOTE = "/home/tcml-spark/experiments/constructive-destructive-regimes/cifar10-eos-lr-2026-10-13"
INPUTS = "/home/tcml-spark/experiments/constructive-destructive-regimes/cifar-eos-calibration-2026-10-13/cache/cifar10-eos"
PYTHON = "/home/tcml-spark/experiments/constructive-destructive-regimes/spark-calibration-2026-10-08/venv/bin/python"
DEPENDENCIES = "/home/tcml-spark/experiments/constructive-destructive-regimes/cifar-eos-calibration-2026-10-13/dependencies"


def main():
    if (HERE / "cohort.json").exists():
        raise FileExistsError("Frozen cohort exists; preserve executed inputs.")
    baseline = json.loads((ROOT / "runs/058-2026-10-13-cifar10-eos-calibration/config.json").read_text())
    frozen_inputs = ROOT / "cache/cifar10-eos/inputs.json"
    inputs = json.loads(frozen_inputs.read_text())
    for name, identity in inputs["prepared_files"].items():
        assert file_hash(frozen_inputs.parent / name) == identity["sha256"]
    rates = [("gd", "4e-3", .004), ("gd", "2-over-300", 2/300), ("gd", "1e-2", .01), ("gd", "2e-2", .02),
             ("adam", "5e-5", 5e-5), ("adam", "1e-4", 1e-4), ("adam", "2e-4", 2e-4), ("adam", "4e-4", 4e-4)]
    records = []
    for number, (method, slug, lr) in enumerate(rates, 59):
        name = f"{number:03}-2026-10-13-cifar10-eos-{method}-lr-{slug}"
        run = ROOT / "runs" / name
        run.mkdir(exist_ok=False)
        config = copy.deepcopy(baseline)
        config.update(run_id=f"{number:03}", hypothesis="H018", purpose="User-authorized 20k fixed-update CIFAR-10 full-batch LR ablation", selected_optimizer=method)
        config["optimizers"][method]["lr"] = lr
        config["proposed_production"]["accuracy_goal"] = None
        config["eigenvalues"]["fallback"] = {"ncv": 60, "maxiter": 400}
        config["notes"] = ["Only optimizer/LR vary within this eight-run cohort; one frozen seed0 initialization.",
                           "Fixed 20000 updates, with no accuracy stopping; no clipping, decay, schedule, augmentation or injected noise.",
                           "Largest algebraic top2 Hessian eigenvalues, residual-checked. Predeclared larger-Krylov retry does not change parameters/objective.",
                           "Adam additionally records the released code's P-preconditioned Hessian, with conventional D scaling.",
                           "Raw/m/mhat/adaptive/actual displacement streams remain distinct. Dense raw history disabled."]
        write_json(run / "config.json", config)
        (run / "README.md").write_text(f"# Run{number:03}: CIFAR-10 full-batch {method.upper()}, LR{lr:.9g}\n\n"
            f"User-authorized H018 learning-rate ablation on DGX Spark; fixed20,000 updates.\n"
            f"Owning [Analysis019](../../analyses/019-2026-10-13-cifar10-eos-lr/README.md) defines\n"
            f"matched data, frozen initialization, diagnostics, spectrum coverage and runtime limits.\n"
            f"Executed config and attempt sources are immutable. Status is recorded in the attempt manifest.\n", encoding="utf-8")
        records.append({"run_id": f"{number:03}", "run": name, "optimizer": method, "lr": lr, "config_sha256": file_hash(run / "config.json")})
    cohort = {"schema_version": 1, "created_utc": stamp(), "analysis_id": "019", "hypothesis": "H018", "runs": records,
              "authorization": "Use DGX Spark; ablate GD/Adam LR; user confirmed20k fixed updates per run",
              "remote_root": REMOTE, "remote_inputs": INPUTS, "remote_python": PYTHON, "remote_dependencies": DEPENDENCIES,
              "input_manifest_sha256": file_hash(frozen_inputs), "initial_tensor_sha256": inputs["initial_tensor_sha256"],
              "limits": {"queue_seconds": 4*3600, "run_seconds": 45*60, "progress_seconds": 60, "retained_storage_allowance_bytes": 5*10**9},
              "calibration": {"run_id": "058", "projected_training_minutes": 58.36, "qualified_allowance_minutes": 111.6,
                              "notes": "early24update samples; later eigenvalue clustering and iterations unmeasured; allow10min retrieval/plotting"}}
    write_json(HERE / "cohort.json", cohort)
    sources = list((ROOT / "src/cifar_eos").glob("*.py")) + list(HERE.glob("*.py")) + [HERE / "cohort.json"]
    for record in records:
        sources += [ROOT / "runs" / record["run"] / "config.json"]
    inventory = {"files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": file_hash(p), "bytes": p.stat().st_size} for p in sources]}
    write_json(HERE / "source_inventory.json", inventory)
    packet = ROOT / "cache/cifar10-eos-lr-sources.tar.gz"
    with tarfile.open(packet, "w:gz") as bundle:
        for p in sources + [HERE / "source_inventory.json"]:
            bundle.add(p, arcname=p.relative_to(ROOT).as_posix())
    write_json(HERE / "source_packet.json", {"path": packet.relative_to(ROOT).as_posix(), "bytes": packet.stat().st_size, "sha256": file_hash(packet)})
    print(json.dumps({"runs": records, "packet_bytes": packet.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
