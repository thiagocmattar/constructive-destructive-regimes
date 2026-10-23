"""Allocate H022 calibration/production records and exact paired inputs once."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from src.fixed_ce_curvature.core import initialize, make_data, save_json, sha, tensor_hash


def main():
    here = Path(__file__).resolve().parent
    inputs = ROOT / "cache/fixed-ce-curvature"
    inputs.mkdir(parents=True, exist_ok=True)
    cohort = []
    calibration = ROOT / "runs/154-2026-10-23-fixed-ce-calibration"
    (calibration / "README.md").write_text(
        "# Run154: H022 mechanics and timing calibration\n\n"
        "User approved the Analysis026 protocol and execution on October23.\n"
        "Seed1000, three loss arms,200 updates each; full-shape model/data and\n"
        "ordinary/Adam-scaled spectra. Mechanics only, excluded from inference.\n"
        "The numbered worker and shared scientific core are frozen before execution.\n"
        "See Analysis026/PROTOCOL.md for measurement, price and retention contracts.\n",
        encoding="utf-8", newline="\n")
    for seed in range(10):
        model = initialize(seed)
        initial = inputs / f"initial-seed{seed}.pt"
        if initial.exists():
            raise RuntimeError("Refusing to regenerate frozen initial tensors")
        torch.save(model.state_dict(), initial)
        for index, arm in enumerate(("native32", "fixed32", "ce64")):
            identifier = 155 + seed * 3 + index
            run = f"{identifier:03d}-2026-10-23-fixed-ce-{arm}-seed{seed}"
            directory = ROOT / "runs" / run
            directory.mkdir()
            config = {"run": run, "hypothesis": "H022", "seed": seed, "arm": arm,
                      "updates": 10000, "initial_path": str(initial.relative_to(ROOT)).replace("\\", "/"),
                      "initial_file_sha256": sha(initial),
                      "initial_tensor_sha256": tensor_hash(model.parameters()),
                      "data_sha256": tensor_hash(make_data()),
                      "model": {"width": 64, "heads": 4, "ff": 256, "classes": 31},
                      "optimizer": {"type": "AdamW", "lr": 1e-3, "betas": [0.9, 0.98],
                                    "epsilon": 1e-8, "weight_decay": 1, "fused": False,
                                    "foreach": False},
                      "precision": "fp32-model-and-moments", "tf32": False,
                      "train_pairs": 480, "heldout_pairs": 481,
                      "curvature_objective": "whole-model-fp64-fixed-ce-all480"}
            save_json(directory / "config.json", config)
            (directory / "README.md").write_text(
                f"# Run{identifier}: H022 {arm}, seed{seed}\n\n"
                "Status: prepared, user-approved October23,2026; not yet executed.\n\n"
                "This is one condition in the ten-seed/three-arm comparison in\n"
                "[Analysis026](../../analyses/026-2026-10-23-fixed-ce-curvature/PROTOCOL.md).\n"
                "The immutable config names its exact initial tensor/data hashes,\n"
                "optimizer,10k full-batch updates, whole-model FP64 diagnostics,\n"
                "paired controls, metric/capture rules and interpretation limits.\n"
                "Training checkpoints, scalar histories, curvature and hash inventories\n"
                "belong under artifacts/attempt-01/. No manuscript claim yet.\n",
                encoding="utf-8", newline="\n")
            cohort.append({"id": identifier, "run": run, "seed": seed, "arm": arm})
    save_json(here / "cohort.json", cohort)
    sources = [ROOT / "src/fixed_ce_curvature/core.py", calibration / "02_worker.py"]
    sources += [ROOT / "runs" / r["run"] / "config.json" for r in cohort]
    sources += list(inputs.glob("*.pt"))
    save_json(here / "source_inventory.json", {"files": [
        {"path": str(p.relative_to(ROOT)).replace("\\", "/"), "bytes": p.stat().st_size,
         "sha256": sha(p)} for p in sources]})
    print(json.dumps({"runs": len(cohort), "first": cohort[0]["id"], "last": cohort[-1]["id"],
                      "data_sha256": tensor_hash(make_data()),
                      "parameters": sum(p.numel() for p in model.parameters())}))


if __name__ == "__main__":
    main()
