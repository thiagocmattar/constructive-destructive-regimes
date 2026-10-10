"""Estimate phase timing and cost; adapted from sparsity-spillover (see BOOTSTRAP)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any


PHASES = ("provision_seconds", "setup_seconds", "diagnostics_seconds",
          "checkpoint_seconds", "transfer_seconds")


def _nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite nonnegative number.")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return number


def _count(value: int, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _samples(calibration: dict[str, Any], key: str, required: bool) -> list[float]:
    values = calibration.get(key, [])
    if not isinstance(values, list) or (required and not values):
        raise ValueError(f"Calibration requires a non-empty {key} list for this phase.")
    samples = [_nonnegative(value, key) for value in values]
    if any(value == 0 for value in samples):
        raise ValueError(f"Every {key} sample must be positive.")
    return samples


def _timings(samples: list[float]) -> tuple[float, float]:
    if not samples:
        return 0.0, 0.0
    return statistics.median(samples), sorted(samples)[math.ceil(0.9 * len(samples)) - 1]


def estimate(calibration: dict[str, Any], *, steps: int, validation_passes: int,
             gpu_count: int = 1, hourly_gpu_price: float | None = None,
             maximum_hours: float | None = None, storage_cost: float = 0.0) -> dict[str, Any]:
    """One worker/group; p90 is an empirical planning scenario, not a guarantee."""
    _count(steps, "steps")
    _count(validation_passes, "validation_passes")
    _count(gpu_count, "gpu_count", 1)
    storage = _nonnegative(storage_cost, "storage_cost")
    price = None if hourly_gpu_price is None else _nonnegative(hourly_gpu_price, "hourly_gpu_price")
    hours = None if maximum_hours is None else _nonnegative(maximum_hours, "maximum_hours")
    if hours is not None and price is None:
        raise ValueError("Maximum-hours cost requires hourly_gpu_price.")
    step_samples = _samples(calibration, "optimizer_step_seconds", steps > 0)
    val_samples = _samples(calibration, "full_validation_seconds", validation_passes > 0)
    fixed = sum(_nonnegative(calibration.get(key, 0.0), key) for key in PHASES)
    median_step, upper_step = _timings(step_samples)
    median_val, upper_val = _timings(val_samples)
    median_seconds = fixed + steps * median_step + validation_passes * median_val
    upper_seconds = fixed + steps * upper_step + validation_passes * upper_val
    result: dict[str, Any] = {
        "steps": steps, "validation_passes": validation_passes,
        "step_samples": len(step_samples), "validation_samples": len(val_samples),
        "fixed_seconds": fixed,
        "median_step_seconds": median_step, "p90_step_seconds": upper_step,
        "median_validation_seconds": median_val, "p90_validation_seconds": upper_val,
        "median_etc_seconds": median_seconds, "p90_etc_seconds": upper_seconds,
        "median_etc_hours": median_seconds / 3600, "p90_etc_hours": upper_seconds / 3600,
    }
    if price is not None:
        result["projected_compute_cost_median"] = median_seconds / 3600 * gpu_count * price
        result["projected_compute_cost_p90"] = upper_seconds / 3600 * gpu_count * price
    if hours is not None:
        result["maximum_hours"] = hours
        result["maximum_total_cost"] = hours * gpu_count * price + storage
        result["p90_scenario_exceeds_maximum_hours"] = upper_seconds / 3600 > hours
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--validation-passes", type=int, required=True)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--hourly-gpu-price", type=float)
    parser.add_argument("--maximum-hours", type=float)
    parser.add_argument("--storage-cost", type=float, default=0.0)
    args = parser.parse_args()
    try:
        calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
        result = estimate(calibration, steps=args.steps, validation_passes=args.validation_passes,
                          gpu_count=args.gpu_count, hourly_gpu_price=args.hourly_gpu_price,
                          maximum_hours=args.maximum_hours, storage_cost=args.storage_cost)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
