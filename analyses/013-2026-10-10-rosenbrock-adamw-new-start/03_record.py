"""Record new-start results, three matched controls and figure integrity."""
import json
import sys
from pathlib import Path

import fitz
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.controlled.rosenbrock import sha256, write_json
from src.controlled.rosenbrock_plot import rolling_cosine


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def verify_figure(report):
    manifest = json.loads((HERE / "figure_manifest.json").read_text())
    for entry in manifest["sources"]:
        folder = ROOT / entry["directory"]
        assert sha256(folder / "artifacts/attempts/001-local/history.npz") == entry["history_sha256"]
        assert sha256(folder / "config.json") == entry["config_sha256"]
    for key in ("script", "plot_implementation"):
        assert sha256(ROOT / manifest[key]) == manifest[key + "_sha256"]
    assert manifest["initial_point"] == report["initial_point"] == [0., 1.]
    assert manifest["smoothing"] == dict(panels=["c", "d"], method="trailing arithmetic mean",
                                        window_updates=100, minimum_valid_values=100)
    assert manifest["y_scales"] == {panel: ("log" if panel == "b" else "linear") for panel in "bcdefghi"}
    coverage = []
    turns = manifest["trajectory_turns"]["points"]
    assert len(turns) == 3
    turn_checks = []
    bounds = manifest["trajectory_domain"]
    assert bounds == [-1.4, 1.22, -0.15, 1.48]  # Same full landscape as Analysis012.
    for entry in report["conditions"]:
        with np.load(ROOT / entry["directory"] / "artifacts/attempts/001-local/history.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["theta"][0], report["initial_point"])
            assert (saved["theta"] >= [bounds[0], bounds[2]]).all()
            assert (saved["theta"] <= [bounds[1], bounds[3]]).all()
            event = next(event for event in turns if event["run_id"] == entry["run_id"])
            theta = saved["theta"]
            # Direct position inequalities independently check the diff-based selector.
            minima = np.flatnonzero((theta[1:-1, 1] < theta[:-2, 1]) &
                                    (theta[1:-1, 1] < theta[2:, 1])) + 1
            assert event["update"] == int(minima[0])
            np.testing.assert_array_equal(event["point"], theta[event["update"]])
            assert event["incoming_delta_y"] == theta[event["update"], 1] - theta[event["update"] - 1, 1]
            assert event["outgoing_delta_y"] == theta[event["update"] + 1, 1] - theta[event["update"], 1]
            assert event["incoming_delta_y"] < 0 < event["outgoing_delta_y"]
            assert event["learning_rate"] == entry["learning_rate"]
            turn_checks.append(dict(run_id=entry["run_id"], update=event["update"],
                                    point=event["point"], first_y_reversal=True))
            for stream in ("raw", "momentum"):
                values = saved[stream + "_cosine"]
                expected = np.full_like(values, np.nan)
                for i in range(99, len(values)):
                    window = values[i - 99:i + 1]
                    if np.isfinite(window).all():
                        expected[i] = window.mean()
                actual = rolling_cosine(values, 100)
                np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14, equal_nan=True)
                valid = np.flatnonzero(np.isfinite(actual))
                assert len(valid) == 1900 and valid[0] + 1 == 101
                coverage.append(dict(run_id=entry["run_id"], stream=stream,
                                     first_update=101, displayed_means=1900))
    assert len(manifest["vertical_reference_lines"]) == 24
    for event in turns:
        lines = [line for line in manifest["vertical_reference_lines"] if line["run_id"] == event["run_id"]]
        assert {line["panel"] for line in lines} == set("bcdefghi")
        assert all(line["update"] == event["update"] and line["linestyle"] == "--" for line in lines)
        expected_color = {"Run038": "#0072B2", "Run039": "#8C4AA8", "Run040": "#D55E00"}[event["run_id"]]
        assert all(line["color"] == expected_color for line in lines)
    pdf_path = ROOT / manifest["figure"]
    assert sha256(pdf_path) == manifest["sha256"]
    with fitz.open(pdf_path) as pdf:
        assert len(pdf) == 1
        page = pdf[0]
        np.testing.assert_allclose([page.rect.width, page.rect.height], [16 * 72, 23 * 72])
        text = page.get_text()
        for label in ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)",
                      "Starting point", "Global optimum", "Final point", "Start (0, 1)",
                      "weight decay 0.1", "B/A", "100-step trailing mean", "First y-turn",
                      "turn t = 81", "turn t = 228", "turn t = 655"):
            assert label in text, label
        assert "log scale" not in text and "Start (−1.2, 1)" not in text
        for block in page.get_text("blocks"):
            assert page.rect.contains(fitz.Rect(block[:4])), block[:4]
        page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2), alpha=False).save(str(HERE / "scratch/pdf_review.png"))
    write_json(HERE / "figure_verification.json", dict(
        passed=True, pages=1, panels=9, pdf_sha256=sha256(pdf_path),
        physical_dimensions_inches=[16, 23], required_text_present=True,
        all_text_within_page=True, source_and_script_hashes=True, actual_initial_point_and_full_paths=True,
        rolling_means_against_direct_windows=True, rolling_coverage=coverage, y_scales=manifest["y_scales"],
        first_y_turns_against_direct_positions=turn_checks, vertical_reference_lines=24,
        render="scratch/pdf_review.png", visual_review="pending"))


def main():
    verification = json.loads((HERE / "verification.json").read_text())
    assert verification["passed"] and verification["initial_point"] == [0., 1.]
    cohort = json.loads((HERE / "cohort.json").read_text())
    report = dict(initial_point=verification["initial_point"], updates_per_run=2000, weight_decay=0.1,
                  initial_loss=101., conditions=[], initial_point_comparisons=[], raw_storage_bytes=0,
                  reduction_script=str(Path(__file__).relative_to(ROOT)), reduction_script_sha256=sha256(__file__))
    for entry in cohort:
        folder = ROOT / entry["directory"]
        output = folder / "artifacts/attempts/001-local"
        summary = json.loads((output / "summary.json").read_text())
        history = output / "history.npz"
        keys = ("initial_loss", "final_loss", "final_point", "final_gradient_norm", "minimum_observed_loss",
                "minimum_loss_update", "first_convergence_threshold_update", "final_100_updates_meet_threshold",
                "loss_increase_fraction", "late_loss_min", "late_loss_max", "late_loss_median", "elapsed_seconds")
        row = dict(entry, **{key: summary[key] for key in keys}, history_sha256=sha256(history), streams={})
        assert row["initial_loss"] == report["initial_loss"]
        with np.load(history, allow_pickle=False) as result:
            for stream in ("raw", "momentum"):
                stats = summary["streams"][stream]
                row["streams"][stream] = dict(A=stats["final_A"], B=stats["final_B"], B_over_A=stats["final_ratio"],
                    negative_cosine_fraction=stats["negative_cosine_fraction"], late_median_cosine=stats["late_median_cosine"],
                    cosine_defined_pairs=int(np.isfinite(result[stream + "_cosine"]).sum()),
                    norm_updates=len(result[stream + "_norm"]))
        report["conditions"].append(row)
        report["raw_storage_bytes"] += history.stat().st_size
        point = ", ".join(f"{value:.9f}" for value in row["final_point"])
        write_text(folder / "README.md", f"""# {entry['run_id']}: Rosenbrock AdamW from (0, 1), LR {entry['lr_label']}

Completed and verified on 2026-10-10, exactly 2,000 local CPU updates in
`001-local`. Final objective {row['final_loss']:.12g}, point ({point}),
post-gradient norm {row['final_gradient_norm']:.9g}.

H012 changes only the initial point from Analysis012 to the user's (0,1).
f(x,y)=(1-x)^2+100(y-x^2)^2; betas (0.9,0.95), epsilon 1e-8, decoupled
weight decay 0.1, constant LR {entry['lr_label']}, zero initial moments,
CPU float64, analytic gradients, no clipping. No data/seed or cloud billing.

[Analysis013](../../analyses/013-2026-10-10-rosenbrock-adamw-new-start/README.md)
owns the matched comparison with [{entry['control_directory'].split('/')[1][:3]}](../../{entry['control_directory']}/README.md)
and the single nine-panel figure. [Config](config.json) and
[numerical implementation](../../src/controlled/rosenbrock.py) are frozen;
`01_run.py` calls the unchanged shared implementation. Do not overwrite its attempt.

[Observation001](observations/001-trajectory-geometry.md), [verification](verification.json),
[manifest](artifacts/attempts/001-local/manifest.json), [summary](artifacts/attempts/001-local/summary.json),
[inventory](artifacts/attempts/001-local/inventory.json) and [final state](artifacts/attempts/001-local/final_state.json)
retain provenance. Complete ignored NPZ history: {history.stat().st_size:,} bytes;
all 2,001 positions/losses and 2,000 raw/momentum/adaptive/displacement vectors,
second moments and exact prefix geometry are retained.
""")
        write_text(folder / "observations/INDEX.md", "# Observations\n\n- [001: Trajectory and geometry](001-trajectory-geometry.md)\n")
        table = "| Stream | Final A | Final B | Final B/A | Negative adjacent pairs |\n| --- | ---: | ---: | ---: | ---: |\n"
        for stream in ("raw", "momentum"):
            stats = row["streams"][stream]
            table += f"| {stream.title()} | {stats['A']:.9g} | {stats['B']:.9g} | {stats['B_over_A']:.9g} | {stats['negative_cosine_fraction']:.6%} |\n"
        write_text(folder / "observations/001-trajectory-geometry.md", f"""# Observation001: AdamW trajectory from (0, 1), LR {entry['lr_label']}

H012: change only the start from the matched Analysis012 condition. Momentum,
weight decay 0.1, fixed 2,000 updates. Final f={row['final_loss']:.12g},
point ({point}), post-gradient norm {row['final_gradient_norm']:.9g}.
First joint loss/gradient threshold hit: {row['first_convergence_threshold_update']};
final-100 sustained criterion: {row['final_100_updates_meet_threshold']}.

{table}
Sources: [config](../config.json), [summary](../artifacts/attempts/001-local/summary.json),
[inventory](../artifacts/attempts/001-local/inventory.json) and ignored
`artifacts/attempts/001-local/history.npz`, SHA-256 `{row['history_sha256']}`.
[Verification](../verification.json) checks analytic gradients, every independent
optimizer update, exact replay, all-stream prefixes, decay displacement and hashes.
All positions/transitions and both coordinates are measured in float64;
1,999 defined adjacent pairs per displayed stream, first cosine undefined.
A=sum of vector norms; B=norm of vector sum; B/A undefined if A=0. No diagnostic
epsilon, clipping, weighting or thinning. Raw table statistics are unsmoothed.

The [comparison observation](../../../analyses/013-2026-10-10-rosenbrock-adamw-new-start/observations/001-initial-point-comparison.md)
owns the [single figure](../../../analyses/013-2026-10-10-rosenbrock-adamw-new-start/figures/rosenbrock_learning_rates.pdf),
generated by Analysis013 `02_plot.py` and the shared plot module. Cosine panels
use full finite 100-update trailing means, first at update 101; norms/A/B have
linear Y. Stored momentum excludes decay; actual displacement includes decay.
One deterministic point/function/recipe/horizon has no sampling uncertainty or
asymptotic/LLM implication; alignment sign and B/A alone do not establish stability.
""")
    for control in verification["initial_point_controls"]:
        old = json.loads((ROOT / control["directory"] / "artifacts/attempts/001-local/summary.json").read_text())
        assert sha256(ROOT / control["directory"] / "artifacts/attempts/001-local/summary.json") == control["summary_sha256"]
        new = next(row for row in report["conditions"] if row["run_id"] == control["new_run_id"])
        report["initial_point_comparisons"].append(dict(
            control, learning_rate=new["learning_rate"], old_initial_loss=old["initial_loss"],
            old_final_loss=old["final_loss"], new_final_loss=new["final_loss"],
            final_loss_difference=new["final_loss"] - old["final_loss"],
            old_streams={stream: old["streams"][stream] for stream in ("raw", "momentum")},
            new_streams=new["streams"]))
    write_json(HERE / "results.json", report)
    verify_figure(report)
    print(json.dumps(dict(raw_storage_bytes=report["raw_storage_bytes"], pdf_integrity="passed")))


if __name__ == "__main__":
    main()
