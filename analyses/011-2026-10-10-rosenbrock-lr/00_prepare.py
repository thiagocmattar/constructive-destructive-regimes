"""Freeze the user-requested 2,000-update, momentum/no-decay LR comparison."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    template = json.loads((ROOT / "runs/028-2026-10-10-rosenbrock-adam/config.json").read_text(encoding="utf-8"))
    cohort = []
    for number, lr, slug in ((32, 1e-2, "1e-2"), (33, 1e-3, "1e-3"), (34, 1e-4, "1e-4")):
        directory = f"runs/{number:03d}-2026-10-10-rosenbrock-lr-{slug}"
        folder = ROOT / directory
        if (folder / "artifacts").exists():
            raise FileExistsError("Executed inputs cannot be replaced: " + directory)
        folder.mkdir(parents=True, exist_ok=True)
        config = dict(template, run_id=f"Run{number:03d}", hypothesis="H010",
                      label="Momentum, no decay; LR " + slug, updates=2000,
                      learning_rate=lr, progress_every=200)
        (folder / "config.json").write_bytes((json.dumps(config, indent=2) + "\n").encode("utf-8"))
        wrapper = ('"""Execute this frozen Rosenbrock condition using the shared implementation."""\n'
                   'import sys\nfrom pathlib import Path\n'
                   'sys.path.insert(0, str(Path(__file__).resolve().parents[2]))\n'
                   'from src.controlled.rosenbrock import execute\n\n'
                   'if __name__ == "__main__":\n    execute(Path(__file__).resolve().parent)\n')
        (folder / "01_run.py").write_bytes(wrapper.encode("utf-8"))
        cohort.append({"run_id": config["run_id"], "directory": directory, "learning_rate": lr, "lr_label": slug})
    (HERE / "cohort.json").write_bytes((json.dumps(cohort, indent=2) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
