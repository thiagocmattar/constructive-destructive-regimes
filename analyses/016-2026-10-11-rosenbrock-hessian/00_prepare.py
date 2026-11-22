"""Freeze a new cohort matching Analysis013, adding only Hessian diagnostics."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.controlled.rosenbrock import write_json


def main():
    controls = json.loads((ROOT / "analyses/013-2026-10-10-rosenbrock-adamw-new-start/cohort.json").read_text())
    cohort = []
    for number, control, slug in zip(range(55, 58), controls, ("1e-2", "midpoint", "1e-3")):
        directory = f"runs/{number:03d}-2026-10-11-rosenbrock-hessian-lr-{slug}"
        if (ROOT / directory).exists():
            raise FileExistsError(directory)
        cohort.append(dict(run_id=f"Run{number:03d}", directory=directory, learning_rate=control["learning_rate"],
                           lr_label=control["lr_label"], control_directory=control["directory"]))
    for entry in cohort:
        config = json.loads((ROOT / entry["control_directory"] / "config.json").read_text())
        config.update(run_id=entry["run_id"], hypothesis="H015", label=f"Analysis013 replica with Hessian; LR {entry['lr_label']}",
                      curvature=dict(scope="full 2D unpenalized objective", points="all theta[t], t=0..2000",
                                     method="analytic Hessian, symmetric eigvalsh, float64",
                                     maximum="largest algebraic eigenvalue, not spectral radius",
                                     optimizer_feedback=False))
        folder = ROOT / entry["directory"]
        folder.mkdir(parents=True)
        write_json(folder / "config.json", config)
        (folder / "01_run.py").write_text('''"""Execute this frozen Rosenbrock condition with complete Hessian coverage."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.controlled.rosenbrock_curvature import execute

if __name__ == "__main__":
    execute(Path(__file__).resolve().parent)
''', encoding="utf-8")
    write_json(HERE / "cohort.json", cohort)


if __name__ == "__main__":
    main()
