"""Reduce complete curvature histories, write observations and check both PDFs."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import fitz
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.controlled.rosenbrock import sha256, stamp, write_json
from src.controlled.rosenbrock_plot import rolling_cosine


def write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode("utf-8"))


def digest(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def verify_figures(histories, reviewed):
    reports = []
    for stem, full in (("rosenbrock_hessian", False), ("rosenbrock_hessian_diagnostics", True)):
        manifest = json.loads((HERE / f"figure_manifest_{stem}.json").read_text())
        assert manifest["initial_point"] == [0., 1.] and manifest["thinning"] is None
        for key in ("script", "plot_implementation", "rolling_implementation"):
            if key in manifest:
                assert sha256(ROOT / manifest[key]) == manifest[key + "_sha256"]
        for source in manifest["sources"]:
            folder = ROOT / source["directory"]
            assert sha256(folder / "config.json") == source["config_sha256"]
            assert sha256(folder / "artifacts/attempts/001-local/history.npz") == source["history_sha256"]
        expected = {p: [] for p in ("abcdefghijk" if full else "abcd")}

        def add(panel, x, y):
            expected[panel].append(dict(panel=panel, points=len(x), x_sha256=digest(x), y_sha256=digest(y)))

        for result, event in zip(histories, manifest["trajectory_turns"]["points"]):
            theta = result["theta"]
            minima = np.flatnonzero((theta[1:-1, 1] < theta[:-2, 1]) & (theta[1:-1, 1] < theta[2:, 1])) + 1
            assert event["update"] == int(minima[0])
            np.testing.assert_array_equal(event["point"], theta[event["update"]])
            bounds = manifest["trajectory_domain"]
            assert ((theta >= [bounds[0], bounds[2]]) & (theta <= [bounds[1], bounds[3]])).all()
            add("a", theta[:, 0], theta[:, 1])
            positions, updates = np.arange(2001), np.arange(1, 2001)
            means = {}
            for stream in ("raw", "momentum"):
                values = result[stream + "_cosine"]
                direct = np.full_like(values, np.nan)
                for i in range(99, 2000):
                    window = values[i - 99:i + 1]
                    if np.isfinite(window).all():
                        direct[i] = window.mean()
                means[stream] = rolling_cosine(values, 100)
                np.testing.assert_allclose(means[stream], direct, rtol=1e-13, atol=1e-14, equal_nan=True)
                assert np.flatnonzero(np.isfinite(means[stream]))[0] + 1 == 101
                assert np.isfinite(means[stream]).sum() == 1900
            if full:
                add("b", positions, result["loss"])
                add("c", updates, means["raw"])
                add("d", updates, means["momentum"])
                for stream in ("raw", "momentum"):
                    add("e", updates, result[stream + "_norm"])
                for panel, stream in (("f", "raw"), ("g", "momentum")):
                    for field in ("A", "B"):
                        add(panel, updates, result[stream + "_" + field])
                for panel, stream in (("h", "raw"), ("i", "momentum")):
                    a, b = result[stream + "_A"], result[stream + "_B"]
                    ratio = np.divide(b, a, out=np.full_like(a, np.nan), where=a > 0)
                    np.testing.assert_array_equal(ratio, result[stream + "_ratio"])
                    add(panel, updates, ratio)
                add("j", positions, result["hessian_lambda_max"])
                add("k", positions, result["hessian_lambda_min"])
            else:
                add("b", positions, result["hessian_lambda_max"])
                add("c", positions, result["loss"])
                add("d", updates, means["raw"])
                add("d", updates, means["momentum"])
        expected = [item for panel in expected.values() for item in panel]
        assert manifest["rendered_series"] == expected
        if full:
            assert manifest["y_scales"] == {p: ("log" if p == "b" else "linear") for p in "bcdefghijk"}
        else:
            assert manifest["y_scales"] == dict(b="linear", c="log", d="linear")
        reference_panels = "bcdefghijk" if full else "bcd"
        assert len(manifest["vertical_reference_lines"]) == 3 * len(reference_panels)
        for event, color in zip(manifest["trajectory_turns"]["points"], ("#0072B2", "#8C4AA8", "#D55E00")):
            lines = [line for line in manifest["vertical_reference_lines"] if line["run_id"] == event["run_id"]]
            assert {line["panel"] for line in lines} == set(reference_panels)
            assert all(line["update"] == event["update"] and line["color"] == color and line["linestyle"] == "--" for line in lines)
        pdf_path = ROOT / manifest["figure"]
        assert sha256(pdf_path) == manifest["sha256"]
        with fitz.open(pdf_path) as pdf:
            assert len(pdf) == 1
            page = pdf[0]
            np.testing.assert_allclose([page.rect.width, page.rect.height], np.array(manifest["dimensions_inches"]) * 72)
            text = page.get_text()
            for label in ("Largest Hessian eigenvalue", "Start (0, 1)", "First y-turn", "stability"):
                assert label in text, label
            for p in ("abcdefghijk" if full else "abcd"):
                assert f"({p})" in text
            for block in page.get_text("blocks"):
                assert page.rect.contains(fitz.Rect(block[:4])), block[:4]
        render = HERE / ("scratch/pdf_full_review.png" if full else "scratch/pdf_focus_review.png")
        if reviewed:
            assert render.is_file()
        reports.append(dict(figure=manifest["figure"], pdf_sha256=manifest["sha256"], pages=1,
                            all_text_within_page=True, source_script_hashes=True,
                            plotted_arrays_hashes_match=True, plotted_series=len(expected),
                            curvature_positions_per_run=2001, direct_rolling_windows=True,
                            first_y_turns=[e["update"] for e in manifest["trajectory_turns"]["points"]],
                            y_scales=manifest["y_scales"], vertical_references=3 * len(reference_panels),
                            visual_review="passed" if reviewed else "pending",
                            reviewed_utc=stamp() if reviewed else None,
                            poppler_render=render.relative_to(HERE).as_posix(),
                            render_sha256=sha256(render) if reviewed else None))
    write_json(HERE / "figure_verification.json", dict(passed=True, figures=reports))


def main(reviewed=False):
    assert json.loads((HERE / "verification.json").read_text())["passed"]
    cohort = json.loads((HERE / "cohort.json").read_text())
    histories, rows = [], []
    raw_bytes = 0
    for entry in cohort:
        folder = ROOT / entry["directory"]
        output = folder / "artifacts/attempts/001-local"
        summary = json.loads((output / "summary.json").read_text())
        history = output / "history.npz"
        with np.load(history, allow_pickle=False) as saved:
            result = {key: saved[key] for key in saved.files}
        histories.append(result)
        row = dict(entry, **summary, history_sha256=sha256(history), history_bytes=history.stat().st_size)
        dy = np.diff(result["theta"][:, 1])
        turn = int(np.flatnonzero((dy[:-1] < 0) & (dy[1:] > 0))[0] + 1)
        row["first_y_turn"] = dict(update=turn, point=result["theta"][turn].tolist(),
                                   lambda_max=float(result["hessian_lambda_max"][turn]),
                                   lambda_min=float(result["hessian_lambda_min"][turn]))
        nonnegative = np.flatnonzero(result["hessian_lambda_min"] >= 0)
        row["first_nonnegative_lambda_min_update"] = int(nonnegative[0]) if len(nonnegative) else None
        rows.append(row)
        raw_bytes += history.stat().st_size
        write_text(folder / "README.md", f"""# {entry['run_id']}: Rosenbrock with Hessian, LR {entry['lr_label']}

Completed and verified, 2026-10-11: 2,000 CPU float64 AdamW updates from (0,1).
Betas(0.9,0.95), epsilon 1e-8, decoupled decay 0.1, exact gradients, no clipping,
constant LR; no data/seed/cloud cost. Final f={summary['final_loss']:.12g};
lambda_max={summary['curvature']['final_lambda_max']:.9g}.

[Analysis016](../../analyses/016-2026-10-11-rosenbrock-hessian/README.md) owns
the method and both comparison PDFs. All original arrays are bitwise identical
to [{entry['control_directory'].split('/')[1][:3]}](../../{entry['control_directory']}/README.md).
[Config](config.json), [verification](verification.json),
[summary](artifacts/attempts/001-local/summary.json),
[manifest](artifacts/attempts/001-local/manifest.json),
[inventory](artifacts/attempts/001-local/inventory.json),
[final state](artifacts/attempts/001-local/final_state.json).

The complete ignored history retains 2,001 positions, full Hessians and both
eigenvalues, plus every original vector/geometry diagnostic: {history.stat().st_size:,} bytes.
Frozen attempt 001-local refuses overwrite. [Observation001](observations/001-curvature.md).
""")
        write_text(folder / "observations/INDEX.md", "# Observations\n\n- [001: Exact curvature](001-curvature.md)\n")
        write_text(folder / "observations/001-curvature.md", f"""# Observation001: Exact curvature along {entry['run_id']}

Question: what is lambda_max of the objective Hessian along the Analysis013
recipe at LR {entry['lr_label']}? Sources: [config](../config.json),
[summary](../artifacts/attempts/001-local/summary.json),
[inventory](../artifacts/attempts/001-local/inventory.json),
[verification](../verification.json); ignored history SHA-256 `{sha256(history)}`.
Complete 2,001 positions t=0..2000, float64, both coordinates; no filtering,
sampling, weighting or curvature smoothing. Raw/momentum cosines describe
different vector streams; the gradient used by update t is at theta[t-1].

Lambda_max starts 200, ends {summary['curvature']['final_lambda_max']:.12g}, and
peaks {summary['curvature']['maximum_lambda_max']:.12g} at update {summary['curvature']['maximum_lambda_max_update']}.
Final loss {summary['final_loss']:.12g}; final lambda_min{summary['curvature']['final_lambda_min']:.12g}.
The first y-turn is t={turn}, lambda_max={row['first_y_turn']['lambda_max']:.12g}.
Negative lambda_min occurs at {summary['curvature']['negative_lambda_min_positions']} saved positions.
Every old array matches its control bitwise; SciPy Hessians and closed-form
eigenvalues check every point, and independent AdamW checks every update.

Figure/caption/reduction: [Analysis016 observation](../../../analyses/016-2026-10-11-rosenbrock-hessian/observations/001-curvature.md),
generated by its 02_plot.py and 03_record.py. One deterministic setup supplies
no sampling uncertainty. The unpenalized objective Hessian excludes decoupled
decay; lambda_max alone is not an AdamW stability criterion or causal result.
""")
    report = dict(conditions=rows, updates_per_run=2000, curvature_positions_per_run=2001,
                  raw_storage_bytes=raw_bytes, simulation_and_save_seconds=sum(r["elapsed_seconds"] for r in rows),
                  reduction_script=Path(__file__).relative_to(ROOT).as_posix(), reduction_script_sha256=sha256(__file__))
    write_json(HERE / "results.json", report)
    verify_figures(histories, reviewed)
    table = "| LR | Final loss | Final lambda_max | Peak lambda_max (update) | lambda_max at y-turn | Final lambda_min |\n| --- | ---: | ---: | --- | ---: | ---: |\n"
    for row in rows:
        c = row["curvature"]
        table += f"| {row['learning_rate']:.7g} | {row['final_loss']:.9g} | {c['final_lambda_max']:.9f} | {c['maximum_lambda_max']:.9f} ({c['maximum_lambda_max_update']}) | {row['first_y_turn']['lambda_max']:.9f} | {c['final_lambda_min']:.9f} |\n"
    write_text(HERE / "observations/INDEX.md", "# Observations\n\n- [001: Curvature along matched trajectories](001-curvature.md)\n")
    write_text(HERE / "observations/001-curvature.md", f"""# Observation001: Curvature along matched Rosenbrock trajectories

H015 asks for largest Hessian eigenvalue along the Analysis013 recipe. Runs055-057
match038-040 at LR0.01/midpoint/0.001; all original arrays are bitwise identical.
[Method](../METHOD.md), [cohort](../cohort.json), [results](../results.json),
[scientific checks](../verification.json), [figure checks](../figure_verification.json).
Each results row identifies its source directory, history SHA-256 and size;
the owning inventories identify all raw/compact artifacts. Sources and script
hashes also appear in each figure manifest. Reduction: ../03_record.py;
plotting: ../02_plot.py, with the locally frozen extended layout ../plot_full.py.

Coverage: all 2,001 saved positions per rate, including t=0 and t=2000, and all
2,000 optimizer vectors. H uses both coordinates of the unpenalized objective;
eigenvalues are float64 symmetric eigvalsh, checked independently at every point.
No curvature/loss filtering, smoothing, weighting, sampling or thinning.
Only raw/momentum adjacent cosines use 100-update trailing arithmetic means of
complete finite windows, first at 101 (1,900 displayed means/stream/rate).
The gradient used by update t is at theta[t-1]; H and loss at t use theta[t].

{table}
Lambda_max starts at 200 for all rates, despite lambda_min=-398: the starting
Hessian is indefinite. Curvature rises along valley traversal and approaches
the optimum's lambda_max=1001.600639361. Large LR arrives sooner and has an
early overshoot/dip in curvature; its late plateau is lower. The smallest LR
has the largest final lambda_max and the lowest final loss in this horizon.
This is a positional association in matched trajectories, not a curvature
intervention. Max curvature does not by itself identify loss quality.

The first y-turn indices remain 81/228/655. Their lambda_max values are in the
table; these geometric events are not curvature extrema or stability thresholds.
The first nonnegative lambda_min positions are
{rows[0]['first_nonnegative_lambda_min_update']}/{rows[1]['first_nonnegative_lambda_min_update']}/{rows[2]['first_nonnegative_lambda_min_update']}.
Negative lambda_min covers 82/211/634 saved positions respectively; an initial
sign crossing does not imply curvature stays nonnegative thereafter.
All three fail the original joint loss/gradient stationarity criterion.

Caption - [focused figure](../figures/rosenbrock_hessian.pdf): **Exact Hessian
curvature along Rosenbrock AdamW trajectories.** Start (0,1); 2,000 updates;
betas (0.9,0.95), epsilon 1e-8, decoupled decay 0.1, CPU float64. (a) Complete
trajectories over objective contours, square start, star optimum, open final
points and diamond first y-turns; domain includes every trajectory point.
(b) Unsmoothed largest algebraic Hessian eigenvalue at each saved point; dotted
gray reference is its value at (1,1), not an optimizer threshold. (c) Objective
loss on log Y. (d) 100-update adjacent-cosine means, raw solid and stored momentum
dashed. LR colors match Analysis013; temporal first-y-turn references are dashed.

Caption - [complete diagnostics](../figures/rosenbrock_hessian_diagnostics.pdf):
**Analysis013 geometry with exact curvature.** (a) All trajectories on the
original full-domain contours; (b) loss; (c,d) 100-update raw/momentum adjacent
cosine means; (e) vector norms; (f,g) raw/momentum A=sum of norms and B=norm of
sum; (h,i) B/A; (j,k) largest/smallest algebraic Hessian eigenvalues at all
positions. Loss uses log Y; all other temporal axes are linear. All panels
retain LR-colored first-y-turn references at 81/228/655. Decay adds no L2 term
to the objective or Hessian. The first cosine and B/A at A=0 remain undefined.

Uncertainty/limits: one deterministic function/start/optimizer/horizon; no
sampling uncertainty or confidence bands. LR scales both adaptive descent and
decay. A quadratic plain-gradient-descent curvature boundary cannot be directly
applied to AdamW's evolving moments/preconditioner. No causal alignment benefit,
universal stability threshold, asymptotic convergence, generalization or LLM
transfer follows. Complete histories are locally retained and ignored.
New raw storage {raw_bytes:,} bytes; simulation/saving {report['simulation_and_save_seconds']:.3f}s.
No cloud resources, rental or storage billing.
""")
    write_text(HERE / "README.md", f"""# Analysis016: Rosenbrock with exact Hessian curvature

Completed and verified, 2026-10-11. **[Focused curvature figure](figures/rosenbrock_hessian.pdf)**;
[complete eleven-panel diagnostics](figures/rosenbrock_hessian_diagnostics.pdf).
Repeat Analysis013 from (0,1), at the same three LRs and 2,000 updates, measuring
the largest objective-Hessian eigenvalue at every saved position. New Runs055-057
retain every original diagnostic and both eigenvalues/full 2x2 Hessians.
All original arrays are bitwise identical to matched Runs038-040.

{table}
Curvature starts 200 and rises toward the optimum's 1001.60. Smallest LR has the
largest final curvature and lowest final loss here. First y-turns 81/228/655
are retained in both figures. Lambda_min starts -398, so maximum curvature alone
does not describe the initial indefinite geometry or an AdamW stability boundary.

Matched recipe: exact Rosenbrock f=(1-x)^2+100(y-x^2)^2, AdamW betas (0.9,0.95),
epsilon 1e-8, decoupled decay 0.1, zero initial moments, no clipping, constant LR,
CPU float64. Hessian/loss cover 2,001 positions t=0..2000; gradient for update t
uses theta[t-1]. Curvature is unsmoothed; only adjacent cosines use complete
finite 100-update trailing means, first at 101. All fail joint stationarity.

[Method](METHOD.md), [observation/captions](observations/001-curvature.md),
[results](results.json), [scientific verification](verification.json),
[figure verification](figure_verification.json). Full replay, independent AdamW
for 6,000 updates, SciPy Hessians/closed-form eigenvalues at 6,003 positions,
additional autograd/finite-difference checks, source/artifact sizes/hashes and
actual rendered-array identities pass. PDF layout review is recorded in the
figure verification receipt. Histories {raw_bytes:,} bytes, locally retained;
simulation/saving {report['simulation_and_save_seconds']:.3f}s; no cloud/billing.

Reproduce reductions with `python analyses/016-2026-10-11-rosenbrock-hessian/01_verify.py`,
then 02_plot.py and 03_record.py at this path. 00_prepare.py and owning 01_run.py
refuse existing inputs/attempts; do not execute into completed runs.
One deterministic setup establishes descriptive geometry, without sampling
uncertainty, an AdamW threshold, causal curvature benefit or LLM transfer.
""")
    print(json.dumps(dict(raw_storage_bytes=raw_bytes, figure_checks="passed")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-visual-review", action="store_true",
                        help="Record visual acceptance only after inspecting both current PDF renders")
    main(parser.parse_args().record_visual_review)
