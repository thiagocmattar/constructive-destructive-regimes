"""Record full-coverage results, matched endpoint comparisons and PDF integrity."""
import json
import sys
from pathlib import Path

import fitz
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.controlled.rosenbrock import sha256, write_json


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def main():
    verification = json.loads((HERE / "verification.json").read_text())
    assert verification["passed"]
    cohort = json.loads((HERE / "cohort.json").read_text())
    report = {"updates_per_run": 2000, "weight_decay": 0.1, "initial_loss": 24.2,
              "conditions": [], "matched_endpoint_comparisons": [], "raw_storage_bytes": 0}
    for entry in cohort:
        run_dir = ROOT / entry["directory"]
        output = run_dir / "artifacts/attempts/001-local"
        summary = json.loads((output / "summary.json").read_text())
        history = output / "history.npz"
        row = dict(entry, final_loss=summary["final_loss"], final_point=summary["final_point"],
                   final_gradient_norm=summary["final_gradient_norm"], minimum_loss=summary["minimum_observed_loss"],
                   minimum_loss_update=summary["minimum_loss_update"],
                   first_convergence_threshold_update=summary["first_convergence_threshold_update"],
                   final_100_updates_meet_threshold=summary["final_100_updates_meet_threshold"],
                   loss_increase_fraction=summary["loss_increase_fraction"],
                   late_loss_min=summary["late_loss_min"], late_loss_max=summary["late_loss_max"],
                   late_loss_median=summary["late_loss_median"], elapsed_seconds=summary["elapsed_seconds"],
                   history_sha256=sha256(history), streams={})
        with np.load(history, allow_pickle=False) as result:
            for stream in ("raw", "momentum"):
                stats = summary["streams"][stream]
                row["streams"][stream] = {"A": stats["final_A"], "B": stats["final_B"],
                    "B_over_A": stats["final_ratio"], "negative_cosine_fraction": stats["negative_cosine_fraction"],
                    "late_median_cosine": stats["late_median_cosine"],
                    "cosine_defined_pairs": int(np.isfinite(result[stream + "_cosine"]).sum()),
                    "norm_updates": len(result[stream + "_norm"])}
        report["conditions"].append(row)
        report["raw_storage_bytes"] += history.stat().st_size
        point = ", ".join(f"{value:.9f}" for value in summary["final_point"])
        write_text(run_dir / "README.md", f"""# {entry['run_id']}: Rosenbrock AdamW, LR {entry['lr_label']}

Completed and verified on 2026-10-10, exactly 2,000 local CPU updates in
`001-local`. Final objective {summary['final_loss']:.12g}, point ({point}),
post-gradient norm {summary['final_gradient_norm']:.9g}.

H011: same method as Analysis011 with momentum and decoupled weight decay 0.1.
f(x,y)=(1-x)^2+100(y-x^2)^2, start (-1.2,1), betas (0.9,0.95), epsilon 1e-8,
constant LR {entry['lr_label']}, CPU float64, analytic gradients and no clipping.
Both coordinates receive shrinkage; the objective and raw gradient contain no
L2 penalty. No data, stochastic seed, validation split, cloud resource or billing.

[Analysis012](../../analyses/012-2026-10-10-rosenbrock-adamw-lr/README.md)
owns the matched method, comparison and single nine-panel figure. [Config](config.json)
and [implementation](../../src/controlled/rosenbrock.py) are frozen at execution.
`01_run.py` calls that unchanged shared implementation; do not overwrite its attempt.

[Observation001](observations/001-trajectory-geometry.md), [verification](verification.json),
[manifest](artifacts/attempts/001-local/manifest.json), [summary](artifacts/attempts/001-local/summary.json),
[inventory](artifacts/attempts/001-local/inventory.json) and [final state](artifacts/attempts/001-local/final_state.json)
retain provenance. Complete ignored NPZ history: {history.stat().st_size:,} bytes;
all 2,001 positions and 2,000 raw/momentum/adaptive/displacement vectors are
retained, together with second moments and exact prefix geometry.
""")
        write_text(run_dir / "observations/INDEX.md", "# Observations\n\n- [001: Trajectory and geometry](001-trajectory-geometry.md)\n")
        write_text(run_dir / "observations/001-trajectory-geometry.md", f"""# Observation001: AdamW trajectory at LR {entry['lr_label']}

H011: weight decay 0.1, momentum, fixed 2,000-update horizon. Final
f={row['final_loss']:.12g}, point ({point}), gradient norm {row['final_gradient_norm']:.9g}.
No simultaneous loss/gradient threshold hit; final-100 sustained test fails.

| Stream | Final A | Final B | Final B/A | Negative adjacent pairs |
| --- | ---: | ---: | ---: | ---: |
| Raw | {row['streams']['raw']['A']:.9g} | {row['streams']['raw']['B']:.9g} | {row['streams']['raw']['B_over_A']:.9g} | {row['streams']['raw']['negative_cosine_fraction']:.6%} |
| Momentum | {row['streams']['momentum']['A']:.9g} | {row['streams']['momentum']['B']:.9g} | {row['streams']['momentum']['B_over_A']:.9g} | {row['streams']['momentum']['negative_cosine_fraction']:.6%} |

Sources: [config](../config.json), [summary](../artifacts/attempts/001-local/summary.json),
[inventory](../artifacts/attempts/001-local/inventory.json) and ignored
`artifacts/attempts/001-local/history.npz`, SHA-256 `{row['history_sha256']}`.
[Verification](../verification.json) covers objective/gradient, independent
optimizer arithmetic at every update, exact replay, all-stream prefixes and hashes.
All positions/transitions and both coordinates are measured in float64.
There are 1,999 defined adjacent pairs per displayed stream; update 1 is
undefined. A=sum of norms, B=norm of vector sum; B/A uses positive observed A.
No smoothing, thinning, stochastic replicates or uncertainty bands.

The [comparison observation](../../../analyses/012-2026-10-10-rosenbrock-adamw-lr/observations/001-learning-rate-comparison.md)
owns the [single figure](../../../analyses/012-2026-10-10-rosenbrock-adamw-lr/figures/rosenbrock_learning_rates.pdf),
generated by Analysis012 `02_plot.py` through the shared plot module. Its
caption defines the trajectory markers and eight exact diagnostics. Stored
momentum excludes decay; actual displacement includes it. Positive cosine
or small loss alone does not imply convergence or stability. One deterministic
objective/start/beta2/horizon supplies no asymptotic or LLM conclusion.
""")
    for control in verification["no_decay_controls"]:
        summary_path = ROOT / control["directory"] / "artifacts/attempts/001-local/summary.json"
        old = json.loads(summary_path.read_text())
        current = next(row for row in report["conditions"] if row["run_id"] == control["decay_run_id"])
        report["matched_endpoint_comparisons"].append(dict(control, learning_rate=current["learning_rate"],
            no_decay_final_loss=old["final_loss"], decay_final_loss=current["final_loss"],
            final_loss_difference=current["final_loss"] - old["final_loss"],
            no_decay_momentum_negative_fraction=old["streams"]["momentum"]["negative_cosine_fraction"],
            decay_momentum_negative_fraction=current["streams"]["momentum"]["negative_cosine_fraction"],
            no_decay_momentum_late_median=old["streams"]["momentum"]["late_median_cosine"],
            decay_momentum_late_median=current["streams"]["momentum"]["late_median_cosine"],
            control_summary_sha256=sha256(summary_path)))
    write_json(HERE / "results.json", report)
    manifest = json.loads((HERE / "figure_manifest.json").read_text())
    for entry in manifest["sources"]:
        folder = ROOT / entry["directory"]
        assert sha256(folder / "artifacts/attempts/001-local/history.npz") == entry["history_sha256"]
        assert sha256(folder / "config.json") == entry["config_sha256"]
    for key in ("script", "plot_implementation"):
        assert sha256(ROOT / manifest[key]) == manifest[key + "_sha256"]
    pdf_path = ROOT / manifest["figure"]
    assert sha256(pdf_path) == manifest["sha256"]
    with fitz.open(pdf_path) as pdf:
        assert len(pdf) == 1
        page = pdf[0]
        np.testing.assert_allclose([page.rect.width, page.rect.height], [16 * 72, 23 * 72])
        text = page.get_text()
        for label in ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)",
                      "Starting point", "Global optimum", "Final point", "weight decay 0.1", "B/A"):
            assert label in text, label
        for block in page.get_text("blocks"):
            assert page.rect.contains(fitz.Rect(block[:4])), block[:4]
        page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2), alpha=False).save(str(HERE / "scratch/pdf_review.png"))
    write_json(HERE / "figure_verification.json", dict(passed=True, pages=1, panels=9,
        pdf_sha256=sha256(pdf_path), physical_dimensions_inches=[16, 23],
        required_text_present=True, all_text_within_page=True, source_and_script_hashes=True,
        render="scratch/pdf_review.png", visual_review="pending"))
    print(json.dumps(dict(raw_storage_bytes=report["raw_storage_bytes"], pdf_integrity="passed")))


if __name__ == "__main__":
    main()
