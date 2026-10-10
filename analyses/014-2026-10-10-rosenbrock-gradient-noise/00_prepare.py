"""Freeze the authorized one-seed noise sweep; reuse the deterministic controls."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    controls = [f"runs/{number:03d}-2026-10-10-rosenbrock-adamw-new-start-lr-{slug}"
                for number, slug in ((38, "1e-2"), (39, "midpoint"), (40, "1e-3"))]
    cohort = []
    wrapper = b'''"""Execute this frozen Gaussian-noise Rosenbrock condition."""\nimport sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parents[2]))\nfrom src.controlled.stochastic_rosenbrock import execute\n\nif __name__ == "__main__":\n    execute(Path(__file__).resolve().parent)\n'''
    for sigma_index, (sigma, sigma_label) in enumerate(((0., "0"), (.01, "1e-2"), (.1, "1e-1"), (1., "1"), (10., "10"))):
        for lr_index, (control_directory, lr_slug) in enumerate(zip(controls, ("1e-2", "midpoint", "1e-3"))):
            control = ROOT / control_directory
            template = json.loads((control / "config.json").read_text())
            if sigma == 0:
                cohort.append(dict(run_id=template["run_id"], directory=control_directory, sigma=sigma,
                                   sigma_label=sigma_label, learning_rate=template["learning_rate"],
                                   lr_label=lr_slug, control_directory=control_directory, reused=True))
                continue
            number = 41 + (sigma_index - 1) * 3 + lr_index
            directory = f"runs/{number:03d}-2026-10-10-rosenbrock-noise-sigma-{sigma_label}-lr-{lr_slug}"
            folder = ROOT / directory
            if folder.exists():
                raise FileExistsError("Inputs already exist: " + directory)
            folder.mkdir(parents=True)
            config = dict(template, run_id=f"Run{number:03d}", hypothesis="H013", sigma=sigma, noise_seed=1234,
                          label=f"Gaussian gradient noise sigma={sigma:g}; LR {lr_slug}",
                          gradient="exact Rosenbrock gradient plus independent N(0,sigma^2 I) noise; no clipping")
            (folder / "config.json").write_bytes((json.dumps(config, indent=2) + "\n").encode())
            (folder / "01_run.py").write_bytes(wrapper)
            cohort.append(dict(run_id=config["run_id"], directory=directory, sigma=sigma, sigma_label=sigma_label,
                               learning_rate=config["learning_rate"], lr_label=lr_slug,
                               control_directory=control_directory, reused=False))
    (HERE / "cohort.json").write_bytes((json.dumps(cohort, indent=2) + "\n").encode())


if __name__ == "__main__":
    main()
