"""Create four explicit scientific configurations; refuse to replace executed inputs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = [
    (28, "rosenbrock-adam", 0.9, 0.0, "Momentum, no decay"),
    (29, "rosenbrock-no-momentum", 0.0, 0.0, "No momentum, no decay"),
    (30, "rosenbrock-adamw", 0.9, 0.1, "Momentum, decay 0.1"),
    (31, "rosenbrock-no-momentum-decay", 0.0, 0.1, "No momentum, decay 0.1"),
]


def main():
    cohort = []
    for number, slug, beta1, decay, label in CONDITIONS:
        folder = f"{number:03d}-2026-10-10-{slug}"
        run_dir = ROOT / "runs" / folder
        if (run_dir / "artifacts").exists():
            raise FileExistsError(f"Executed inputs cannot be changed: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        config = {"schema_version": 1, "run_id": f"Run{number:03d}", "hypothesis": "H009", "label": label,
                  "objective": {"name": "Rosenbrock", "dimension": 2, "a": 1.0, "b": 100.0},
                  "initial_point": [-1.2, 1.0], "updates": 10000, "learning_rate": 0.01,
                  "betas": [beta1, 0.95], "epsilon": 1e-8, "weight_decay": decay,
                  "schedule": "constant from update 1", "precision": "float64", "device": "CPU",
                  "gradient": "exact analytic, full 2D objective, no clipping", "seed": None,
                  "progress_every": 1000, "history_every": 1,
                  "convergence": {"objective_at_most": 1e-8, "gradient_norm_at_most": 1e-4, "sustained_updates": 100},
                  "stopping": "fixed update budget, no threshold early stopping"}
        (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        cohort.append({"run_id": config["run_id"], "directory": "runs/" + folder, "label": label,
                       "beta1": beta1, "weight_decay": decay})
    (Path(__file__).parent / "cohort.json").write_text(json.dumps(cohort, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
