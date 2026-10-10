"""Extend Analysis003 using completed Analysis002 evidence; no training or plots."""
from pathlib import Path
import csv
import hashlib
import json
import sys

sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "analyses/002-2026-10-04-scale-seed-replication"
SIZES = ("14M", "31M", "70M")
SEEDS = (1234, 1235, 1236)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    generation = json.loads((SOURCE / "generation.json").read_text())
    for name, key in [("results.json", "results_sha256"),
                      ("figure_data.csv", "figure_data_sha256")]:
        assert sha(SOURCE / name) == generation[key], name
    evidence = json.loads((SOURCE / "results.json").read_text())
    source_hashes = {}
    checkpoint_inventory = []
    for item in evidence["source_lineage"]:
        for record in [*item["sources"].values(), item["acceptance"]]:
            path = ROOT / record["path"]
            assert sha(path) == record["sha256"], path
            source_hashes[record["path"]] = record["sha256"]
        if item["size"] in ("14M", "70M"):
            folder = ROOT / "runs" / item["folder"] / "artifacts" / item["attempt"]
            folder = folder / "full-pass/checkpoints/midpoint"
            files = {p.relative_to(folder).as_posix(): p.stat().st_size
                     for p in folder.rglob("*") if p.is_file()}
            assert files.get("recovery.pt", 0) > 0
            assert files.get("model/model.safetensors", 0) > 0
            checkpoint_inventory.append({"size": item["size"], "seed": item["seed"],
                "path": folder.relative_to(ROOT).as_posix(), "files_bytes": files,
                "scope": "Presence/size inventory only; resume mechanics not tested"})

    with (SOURCE / "figure_data.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 9 * 712
    statistics = []
    for seed in SEEDS:
        for size in SIZES:
            selected = [r for r in rows if r["model"] == size and int(r["seed"]) == seed]
            assert [int(r["update"]) for r in selected] == list(range(1, 713))
            assert selected[0]["raw_cosine"] == ""
            c = np.array([float(r["raw_cosine"]) for r in selected[1:]])
            norms = np.array([float(r["raw_norm"]) for r in selected])
            negative = c[c < 0]
            frequency = len(negative) / len(c)
            intensity = float(np.maximum(-c, 0).mean())
            magnitude = float(-negative.mean())
            assert np.isclose(intensity, frequency * magnitude, rtol=1e-14)
            original = next(r for r in evidence["runs"] if r["size"] == size and r["seed"] == seed)
            raw = original["all_updates"]["raw"]
            assert raw["negative_cosines"] == len(negative)
            assert np.isclose(raw["median_adjacent_cosine"], np.median(c), rtol=1e-14)
            assert np.isclose(raw["R"], float(selected[-1]["raw_R"]), rtol=1e-14)
            energy = float(norms @ norms)
            resultant_squared = float(selected[-1]["raw_B"]) ** 2
            lag_one = float(2 * np.sum(c * norms[:-1] * norms[1:]))
            floor = np.array([float(r["raw_cosine"]) for r in selected
                              if 576 <= int(r["update"]) <= 711])
            statistics.append({"size": size, "seed": seed,
                "negative_pairs": len(negative), "defined_pairs": len(c),
                "negative_fraction_F": frequency, "unconditional_intensity_D": intensity,
                "conditional_negative_magnitude_M": magnitude,
                "raw_median_cosine": float(np.median(c)), "raw_terminal_R": raw["R"],
                "final_validation_loss": original["final_validation_loss"],
                "raw_energy_E": energy, "raw_resultant_squared_B2": resultant_squared,
                "raw_all_pair_interaction_J": resultant_squared - energy,
                "raw_lag_one_interaction_L": lag_one,
                "lag_one_relative_error": (energy + lag_one - resultant_squared) / resultant_squared,
                "floor_preceded_full_batch_pairs": {"updates": [576, 711],
                    "defined_pairs": len(floor), "median_cosine": float(np.median(floor)),
                    "negative_fraction": float(np.mean(floor < 0))}})
    ordering = []
    for seed in SEEDS:
        block = [next(r for r in statistics if r["seed"] == seed and r["size"] == size)
                 for size in SIZES]
        result = {"seed": seed}
        for key, sign in [("negative_fraction_F", 1), ("unconditional_intensity_D", 1),
                          ("conditional_negative_magnitude_M", -1), ("raw_median_cosine", -1),
                          ("raw_terminal_R", -1), ("final_validation_loss", -1)]:
            result[key] = bool(np.all(sign * np.diff([r[key] for r in block]) > 0))
        ordering.append(result)
    protected = json.loads((HERE / "protected_inputs.json").read_text())
    for name, expected in protected.items():
        assert sha(ROOT / name) == expected, name
    output = {"hypotheses": ["H003", "H004"], "scope": "Exploratory replication audit of completed logs",
        "script_sha256": sha(Path(__file__)),
        "source_analysis": SOURCE.relative_to(ROOT).as_posix(),
        "source_analysis_hashes": {name: sha(SOURCE / name)
            for name in ("results.json", "figure_data.csv", "generation.json")},
        "run_source_hashes": source_hashes, "statistics": statistics,
        "strict_size_orderings": ordering,
        "all_nine_all_pair_interactions_positive": all(r["raw_all_pair_interaction_J"] > 0 for r in statistics),
        "checkpoint_inventory": checkpoint_inventory,
        "coverage": {"updates_per_run": 712, "defined_pairs_per_run": 711, "rows": len(rows),
            "floor_segment": "576-711; final partial update712 excluded only from phase summary",
            "replicate_unit": "Three seed blocks; initialization and data-order seed co-vary",
            "validation": "Original full 500-document validation, token weighted, including tail"},
        "verification": {"source_and_receipt_hashes_checked": len(source_hashes),
            "csv_matches_analysis002_summaries": True, "factorization_D_equals_F_times_M": True,
            "protected_files_unchanged": len(protected), "new_training_runs": 0, "cloud_calls": 0}}
    (HERE / "replication_check.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"orderings": ordering, "verification": output["verification"],
                      "all_nine_J_positive": output["all_nine_all_pair_interactions_positive"]}, indent=2))


if __name__ == "__main__":
    main()
