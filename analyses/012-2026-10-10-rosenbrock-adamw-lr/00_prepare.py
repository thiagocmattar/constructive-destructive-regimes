"""Freeze the authorized AdamW LR cohort; do not overwrite executed inputs."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    baseline = ROOT / "runs/032-2026-10-10-rosenbrock-lr-1e-2"
    template = json.loads((baseline / "config.json").read_text())
    cohort = []
    for number, lr, slug, label in ((35, 1e-2, "1e-2", "1e-2"),
                                   (36, math.sqrt(1e-3 * 1e-2), "midpoint", "3.1622776601683794e-3"),
                                   (37, 1e-3, "1e-3", "1e-3")):
        directory = f"runs/{number:03d}-2026-10-10-rosenbrock-adamw-lr-{slug}"
        folder = ROOT / directory
        if folder.exists():
            raise FileExistsError("Run inputs already exist: " + directory)
        folder.mkdir(parents=True)
        config = dict(template, run_id=f"Run{number:03d}", hypothesis="H011", weight_decay=0.1,
                      label="Momentum, weight decay 0.1; LR " + label, learning_rate=lr)
        (folder / "config.json").write_bytes((json.dumps(config, indent=2) + "\n").encode())
        (folder / "01_run.py").write_bytes((baseline / "01_run.py").read_bytes())
        cohort.append(dict(run_id=config["run_id"], directory=directory, learning_rate=lr, lr_label=label))
    (HERE / "cohort.json").write_bytes((json.dumps(cohort, indent=2) + "\n").encode())


if __name__ == "__main__":
    main()
