"""Freeze new initial-point conditions without overwriting any prior run."""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main(point):
    if not all(math.isfinite(value) for value in point):
        raise ValueError("The starting point must be finite")
    cohort = []
    conditions = ((38, 35, "1e-2"), (39, 36, "midpoint"), (40, 37, "1e-3"))
    for number, control_number, slug in conditions:
        directory = f"runs/{number:03d}-2026-10-10-rosenbrock-adamw-new-start-lr-{slug}"
        if (ROOT / directory).exists():
            raise FileExistsError("Run inputs already exist: " + directory)
    for number, control_number, slug in conditions:
        control_directory = f"runs/{control_number:03d}-2026-10-10-rosenbrock-adamw-lr-{slug}"
        control = ROOT / control_directory
        template = json.loads((control / "config.json").read_text())
        directory = f"runs/{number:03d}-2026-10-10-rosenbrock-adamw-new-start-lr-{slug}"
        folder = ROOT / directory
        folder.mkdir(parents=True)
        label = f"{template['learning_rate']:.17g}"
        config = dict(template, run_id=f"Run{number:03d}", hypothesis="H012", initial_point=list(point),
                      label=f"Momentum, weight decay 0.1; start {tuple(point)}; LR {label}")
        (folder / "config.json").write_bytes((json.dumps(config, indent=2) + "\n").encode())
        (folder / "01_run.py").write_bytes((control / "01_run.py").read_bytes())
        cohort.append(dict(run_id=config["run_id"], directory=directory,
                           learning_rate=config["learning_rate"], lr_label=label,
                           control_directory=control_directory))
    (HERE / "cohort.json").write_bytes((json.dumps(cohort, indent=2) + "\n").encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("x", type=float)
    parser.add_argument("y", type=float)
    args = parser.parse_args()
    main((args.x, args.y))
