"""Reduce measured phase timings to explicitly qualified production scenarios."""
import argparse
import json
from pathlib import Path

import numpy as np

from cifar_eos import file_hash, stamp, write_json

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtx", type=Path, required=True)
    parser.add_argument("--spark", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((HERE / "config.json").read_text())
    budget = config["proposed_production"]["max_updates"]
    devices = {}
    for name, directory in (("rtx", args.rtx), ("spark", args.spark)):
        methods = {}
        for method in ("gd", "adam"):
            output = directory / method
            summary = json.loads((output / "summary.json").read_text())
            manifest = json.loads((output / "manifest.json").read_text())
            samples = summary["post_warmup_samples"]
            phases = {k: {"median": float(np.median([s[k] for s in samples])), "p90": float(np.percentile([s[k] for s in samples], 90)),
                          "minimum": min(s[k] for s in samples), "maximum": max(s[k] for s in samples)} for k in samples[0]}
            first, last = summary["curvature_samples"][0], summary["curvature_samples"][-1]
            plain_initial = first["hessian"]["seconds"]
            plain = last["hessian"]["seconds"]
            conditioned = last.get("paper_preconditioned_hessian", {}).get("seconds", 0.0)
            scenarios = {}
            for cadence in (100, 20, 1):
                # Includes initialization, all cadence states and final when off cadence.
                sampled_states = (budget + cadence - 1) // cadence
                curvature = plain_initial + sampled_states * (plain + conditioned)
                checkpoints = 2 + budget // config["proposed_production"]["checkpoint_every"]
                checkpoint_cost = max(summary["checkpoint_seconds"]) * checkpoints
                central = budget * phases["total_seconds"]["median"] + curvature + checkpoint_cost
                p90_phases = budget * phases["total_seconds"]["p90"] + curvature + checkpoint_cost
                # Qualified planning allowance: slower tail eigenproblems and host contention.
                allowance = 1.3 * budget * phases["total_seconds"]["p90"] + 3 * curvature + checkpoint_cost + 60
                scenarios[str(cadence)] = {"updates": budget, "post_initial_curvature_states": sampled_states,
                                           "measured_phase_projection_seconds": central,
                                           "p90_phase_projection_seconds": p90_phases,
                                           "planning_allowance_seconds": allowance,
                                           "projected_curvature_seconds": curvature,
                                           "projected_checkpoint_seconds": checkpoint_cost}
            methods[method] = {"source_summary_sha256": file_hash(output / "summary.json"), "environment": manifest["environment"],
                               "measured_updates": summary["completed_updates"], "post_warmup_samples": len(samples),
                               "phases_seconds": phases, "memory": summary["memory"],
                               "hessian_plain_seconds_initial": plain_initial, "hessian_plain_seconds_at24": plain,
                               "hessian_adam_preconditioned_seconds_at24": conditioned,
                               "hessian_hvp_calls": [e["hessian"]["hvp_calls"] for e in summary["curvature_samples"]],
                               "minimum_checkpoint_bytes": min(p.stat().st_size for p in output.glob("*.pt")),
                               "maximum_checkpoint_bytes": max(p.stat().st_size for p in output.glob("*.pt")),
                               "diagnostics_fraction_of_update_median": phases["diagnostic_seconds"]["median"] / phases["total_seconds"]["median"],
                               "scenarios_by_eigen_cadence": scenarios}
        devices[name] = methods
    result = {"utc": stamp(), "purpose": "Calibration; projections, not completed training or convergence estimates",
              "config_sha256": file_hash(HERE / "config.json"), "devices": devices,
              "default_eigen_cadence": 100, "production_launched": False,
              "limitations": ["24 early updates; 20 retained timing samples per method/device", "Only 2 plain Hessian solves and 1 Adam-preconditioned solve per method/device",
                             "Late eigenvalue clustering, ARPACK iteration growth and sustained thermals are unmeasured",
                             "Accuracy-goal stopping may occur earlier; the step at convergence is unknown",
                             "Planning allowance is a declared scenario, not a statistical confidence bound",
                             "PyTorch allocation/reservation excludes driver context, desktop applications and CPU ARPACK workspaces",
                             "No cloud rental; source/data/setup/download times are already spent and excluded from projection"]}
    write_json(HERE / "calibration_results.json", result)
    print(json.dumps({device: {method: {"update_ms": round(value["phases_seconds"]["total_seconds"]["median"] * 1000, 3),
                                                   "allocated_MiB": round(value["memory"]["peak_allocated_bytes"] / 2**20, 1),
                                                   "eigen100_minutes": round(value["scenarios_by_eigen_cadence"]["100"]["measured_phase_projection_seconds"] / 60, 2),
                                                   "planning_minutes": round(value["scenarios_by_eigen_cadence"]["100"]["planning_allowance_seconds"] / 60, 2)}
                                  for method, value in methods.items()} for device, methods in devices.items()}, indent=2))


if __name__ == "__main__":
    main()
