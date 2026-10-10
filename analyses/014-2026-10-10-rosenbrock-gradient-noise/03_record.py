"""Reduce the verified sweep, document all figures and check PDF integrity."""
import json
import sys
from pathlib import Path

import fitz
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.controlled import rosenbrock as exact
from src.controlled.rosenbrock_plot import rolling_cosine


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def main():
    verification = json.loads((HERE / "verification.json").read_text())
    assert verification["passed"]
    cohort = json.loads((HERE / "cohort.json").read_text())
    manifest = json.loads((HERE / "figures_manifest.json").read_text())
    assert manifest["script_sha256"] == exact.sha256(ROOT / manifest["script"])
    for dependency in manifest["dependencies"]:
        assert dependency["sha256"] == exact.sha256(ROOT / dependency["path"])
    report = dict(conditions=[], seed=1234, sigmas=[0., .01, .1, 1., 10.],
                  updates_per_run=2000, new_updates=24000, raw_storage_bytes=0,
                  new_execution_seconds=0., reduction_script_sha256=exact.sha256(__file__),
                  uncertainty="One matched realization; no confidence intervals")
    rolling_checks = []
    max_prefix_energy_error = 0.
    for entry in cohort:
        folder = ROOT / entry["directory"]
        output = folder / "artifacts/attempts/001-local"
        history = output / "history.npz"
        summary = json.loads((output / "summary.json").read_text())
        row = dict(entry, history_sha256=exact.sha256(history), final_loss=summary["final_loss"],
                   final_point=summary["final_point"], final_gradient_norm=summary["final_gradient_norm"],
                   first_convergence_threshold_update=summary["first_convergence_threshold_update"],
                   final_100_updates_meet_threshold=summary["final_100_updates_meet_threshold"],
                   late_loss_median=summary["late_loss_median"], late_loss_min=summary["late_loss_min"],
                   late_loss_max=summary["late_loss_max"], loss_increase_fraction=summary["loss_increase_fraction"],
                   streams={}, noise={})
        with np.load(history, allow_pickle=False) as saved:
            for stream in ("raw", "momentum"):
                cosine = saved[stream + "_cosine"]
                row["streams"][stream] = dict(final_A=float(saved[stream + "_A"][-1]), final_B=float(saved[stream + "_B"][-1]),
                    final_B_over_A=float(saved[stream + "_ratio"][-1]), negative_cosine_fraction=float(np.nanmean(cosine[1:] < 0)),
                    late_mean_cosine=float(np.nanmean(cosine[-1000:])), late_median_cosine=float(np.nanmedian(cosine[-1000:])),
                    late_mean_norm=float(np.mean(saved[stream + "_norm"][-1000:])),
                    final_energy=float(saved[stream + "_energy"][-1]), final_all_pair=float(saved[stream + "_all_pair"][-1]),
                    final_lag_one=float(saved[stream + "_lag_one"][-1]), defined_pairs=int(np.isfinite(cosine).sum()))
                rolling = rolling_cosine(cosine, 100)
                direct = np.full_like(cosine, np.nan)
                for t in range(99, len(cosine)):
                    window = cosine[t-99:t+1]
                    if np.isfinite(window).all():
                        direct[t] = window.mean()
                np.testing.assert_allclose(rolling, direct, rtol=1e-13, atol=1e-14, equal_nan=True)
                valid = np.flatnonzero(np.isfinite(rolling))
                assert len(valid) == 1900 and valid[0] + 1 == 101
                rolling_checks.append(dict(run_id=entry["run_id"], stream=stream, displayed_means=1900, first_update=101))
            for stream in ("raw", "momentum", "adaptive", "displacement") + (() if entry["reused"] else ("signal", "noise")):
                vectors = saved[stream + "_vectors"]
                direct_energy = np.add.accumulate((vectors ** 2).sum(axis=1))
                error = float(np.max(np.abs(saved[stream + "_energy"] - direct_energy) / (1 + direct_energy)))
                assert error < 1e-12
                max_prefix_energy_error = max(max_prefix_energy_error, error)
                np.testing.assert_allclose(saved[stream + "_A_squared"], saved[stream + "_A"] ** 2, rtol=1e-13)
                np.testing.assert_allclose(saved[stream + "_triangle_gap"], saved[stream + "_A_squared"] - saved[stream + "_B_squared"], rtol=1e-13)
                np.testing.assert_allclose(saved[stream + "_beyond_lag_one"], saved[stream + "_all_pair"] - saved[stream + "_lag_one"], rtol=1e-13)
            if not entry["reused"]:
                report["raw_storage_bytes"] += history.stat().st_size
                report["new_execution_seconds"] += summary["elapsed_seconds"]
                row["noise"] = dict(empirical_mean=summary["noise_mean"], empirical_covariance=summary["noise_covariance"],
                    final_B_squared=float(saved["noise_B_squared"][-1]), final_energy=float(saved["noise_energy"][-1]),
                    ensemble_expectation_final_B_squared_and_energy=4000 * entry["sigma"] ** 2,
                    final_prefix_cross=float(saved["decomposition_prefix_cross"][-1]),
                    final_temporal_cross=float(saved["decomposition_temporal_cross"][-1]))
        report["conditions"].append(row)
        if entry["reused"]:
            continue
        point = ", ".join(f"{v:.9f}" for v in row["final_point"])
        write_text(folder / "README.md", f"""# {entry['run_id']}: Rosenbrock noisy gradients, sigma {entry['sigma']:g}, LR {entry['lr_label']}

Completed and verified on 2026-10-10: 2,000 local CPU float64 updates in
`001-local`. Final f={row['final_loss']:.12g}, point ({point}), true gradient
norm {row['final_gradient_norm']:.9g}. Noise seed1234, sigma={entry['sigma']:g},
g=grad f(theta_previous)+sigma*z; z has independent standard-normal coordinates.
Same z sequence across the sweep, zero initial moments and start(0,1).
AdamW: betas(0.9,0.95), epsilon1e-8, decay0.1, constant LR, no clipping.

[Analysis014](../../analyses/014-2026-10-10-rosenbrock-gradient-noise/README.md)
owns the matched comparison, figures and [method](../../analyses/014-2026-10-10-rosenbrock-gradient-noise/METHOD.md).
[Config](config.json), [numerical code](../../src/controlled/stochastic_rosenbrock.py)
and its dependencies are frozen and hash-pinned. Do not overwrite the attempt.
[Verification](verification.json), [summary](artifacts/attempts/001-local/summary.json),
[manifest](artifacts/attempts/001-local/manifest.json), [inventory](artifacts/attempts/001-local/inventory.json)
and [final state](artifacts/attempts/001-local/final_state.json) retain provenance.
Complete ignored history is {history.stat().st_size:,} bytes. No cloud billing.
""")
        write_text(folder / "observations/INDEX.md", "# Observations\n\n- [001: Noise and geometry](001-noise-geometry.md)\n")
        table = "| Stream | Final A | Final B | Final B/A | Late mean cosine |\n| --- | ---: | ---: | ---: | ---: |\n"
        for stream, stats in row["streams"].items():
            table += f"| {stream.title()} | {stats['final_A']:.9g} | {stats['final_B']:.9g} | {stats['final_B_over_A']:.9g} | {stats['late_mean_cosine']:.9g} |\n"
        index = [0., .01, .1, 1., 10.].index(entry["sigma"]) + 1
        observation = f"{index:03d}-sigma-{entry['sigma_label']}.md"
        write_text(folder / "observations/001-noise-geometry.md", f"""# Observation001: noisy raw/momentum geometry

H013, sigma={entry['sigma']:g}, one noise seed1234, LR {entry['lr_label']}.
Raw is the stochastic estimator used by AdamW; signal is the exact analytic
gradient on the same adapted path. Final f={row['final_loss']:.12g}; first joint
true loss/gradient threshold hit={row['first_convergence_threshold_update']},
sustained final100={row['final_100_updates_meet_threshold']}.

{table}
Sources: [summary](../artifacts/attempts/001-local/summary.json),
[inventory](../artifacts/attempts/001-local/inventory.json), full ignored NPZ
history SHA-256 `{row['history_sha256']}`, [verification](../verification.json).
Coverage: both coordinates; all2,001 positions and2,000 vectors per stream,
with1,999 adjacent pairs. A=sum norms; B=norm vector sum; ratio B/A; undefined
first cosine/zero vectors and A=0 ratios receive no epsilon. Table statistics
are unsmoothed; late means use the final1,000 updates. No clipping or thinning.

The [owning comparison observation](../../../analyses/014-2026-10-10-rosenbrock-gradient-noise/observations/{observation})
links the navigation/metric and identity PDFs and their generating scripts,
sources, reductions and captions. Cosine curves use100-update trailing means;
norm/A/B Y is linear. Every noise draw, true gradient, optimizer direction,
moment and displacement is retained. Identities are checked pathwise including
signal/noise cross terms. One realization supplies no sampling-error estimate,
universal noise/LR ordering, convergence guarantee or LLM conclusion.
""")
    exact.write_json(HERE / "results.json", report)
    observations = []
    for index, sigma in enumerate(report["sigmas"], 1):
        rows = [row for row in report["conditions"] if row["sigma"] == sigma]
        label = rows[0]["sigma_label"]
        name = f"{index:03d}-sigma-{label}.md"
        observations.append(f"- [{index:03d}: sigma {sigma:g}]({name})")
        table = "| LR | Final f | Late median f | Raw B/A | Momentum B/A | Late raw mean cosine | Late momentum mean cosine |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        for row in rows:
            g, m = row["streams"]["raw"], row["streams"]["momentum"]
            table += f"| {row['lr_label']} | {row['final_loss']:.9g} | {row['late_loss_median']:.9g} | {g['final_B_over_A']:.6f} | {m['final_B_over_A']:.6f} | {g['late_mean_cosine']:.6f} | {m['late_mean_cosine']:.6f} |\n"
        write_text(HERE / "observations" / name, f"""# Observation{index:03d}: gradient noise sigma {sigma:g}

H013: compare the same three LRs at sigma={sigma:g}, using the
[pre-execution method](../METHOD.md). Raw g=grad f(theta_previous)+xi;
xi is N(0,sigma^2 I_2), noise seed1234. Sigma0 reuses deterministic controls.
Sources: [cohort](../cohort.json), full run histories and receipts,
[results](../results.json), [verification](../verification.json), and
[figure manifest](../figures_manifest.json), pinning sources/scripts/PDF hashes.
Reduction: all2,001 exact objective values and2,000 raw/momentum vectors for
each LR. A=sum vector norms, B=norm vector sum, ratio B/A. First cosine and
zero-norm pairs are undefined; A=0 ratios undefined; no epsilon. All observed
raw/momentum vectors are nonzero. Table summaries are unsmoothed, equal-pair
weighted, with late statistics using the last1,000 updates. One realization,
no uncertainty bands or sampling-error estimate. LR also multiplies decay.

{table}
Caption 1: **Rosenbrock navigation with sigma={sigma:g} additive Gaussian
gradient noise.** [Navigation/metrics PDF](../figures/rosenbrock_sigma_{label}.pdf)
shows(a) full contours and trajectories, common start(0,1), optimum(1,1), final
points; (b) exact unpenalized objective; (c,d)100-update trailing raw noisy-gradient
and momentum cosine means, complete finite windows only, first at update101;
(e) raw/momentum norms; (f,g) cumulative A and B; (h,i) B/A. Norm/A/B axes use
linear Y, loss log Y; common full landscape domain, no zoom/thinning. The
analytic signal and added noise are distinct retained streams. First y-turn
markers are not interpreted under the stochastic paths.

Caption 2: **Pathwise squared-sum identities and noise realization, sigma={sigma:g}.**
[Identity PDF](../figures/identities_sigma_{label}.pdf) uses one LR per column.
Top row: B_g^2, energy E_g=sum||g||^2, exact all-pair I_g, with B_g^2=E_g+I_g.
Middle row: signal squared sum||H||^2, noise squared sum||X||^2, cross2<H,X>,
whose sum equals B_g^2. Bottom row: realized noise B_xi^2 and E_xi alongside
the dotted ensemble expectation2*t*sigma^2. Show every prefix with symmetric
log Y(linear within[-1,1]), common limits within each row. The expectation is
not a pathwise equality; cross terms are retained along the adapted path.
Sigma0's noise/cross terms are exactly zero. No sampled histories are averaged
across seeds, since there is one seed. Each identity holds up to reported FP64
roundoff; stochastic estimation changes its components, not its algebra.

Generating script: [02_plot.py](../02_plot.py) with shared plot/noise geometry;
[03_record.py](../03_record.py) writes reductions and validates every rolling
mean, energy prefix, source identity, PDF page/text and hash. See
[figure verification](../figure_verification.json) for rendered review.
Interpretation limits: descriptive finite-budget outcomes, no causal alignment
mechanism, formal stability/generalization, ensemble monotonicity or LLM transfer.
""")
    write_text(HERE / "observations/INDEX.md", "# Observations\n\n" + "\n".join(observations) +
               "\n- [006: Sigma comparison](006-sigma-comparison.md)\n"
               "- [007: Raw/momentum norm grid](007-norm-grid.md)\n")
    # PDF checks and renders for every figure, including all source identities.
    verified_figures = []
    assert len(manifest["figures"]) == 12
    for item in manifest["figures"]:
        path = ROOT / item["path"]
        assert exact.sha256(path) == item["sha256"]
        for source in item["sources"]:
            assert source["history_sha256"] == exact.sha256(ROOT / source["directory"] / "artifacts/attempts/001-local/history.npz")
        if item["kind"] == "navigation":
            frame = json.loads((ROOT / item["frame_manifest"]).read_text())
            assert frame["noise_sigma"] == item["sigma"]
            assert frame["trajectory_domain"] == manifest["common_trajectory_domain"]
            assert frame["smoothing"]["window_updates"] == 100
            assert frame["y_scales"] == {panel: ("log" if panel == "b" else "linear") for panel in "bcdefghi"}
            dimensions, required = [16, 23], ["Noisy-gradient", "100-step trailing mean", "Starting point", "Global optimum", "B/A"]
        elif item["kind"] == "identities":
            dimensions, required = [17, 14], ["exact identities", "Squared gradient units", "expectation", "Cross terms"]
        elif item["kind"] == "sigma_comparison":
            dimensions, required = [17, 10], ["increasing gradient noise", "Final objective", "Mean momentum cosine", "One seed"]
        else:
            assert item["kind"] == "norm_grid"
            dimensions, required = [17, 20], ["raw-gradient and momentum norms", "Stored momentum", "shared log Y", "Every update"]
        with fitz.open(path) as pdf:
            assert len(pdf) == 1
            page = pdf[0]
            np.testing.assert_allclose([page.rect.width, page.rect.height], np.array(dimensions) * 72)
            text = page.get_text()
            for word in required:
                assert word in text, (path, word)
            for block in page.get_text("blocks"):
                assert page.rect.contains(fitz.Rect(block[:4])), (path, block[:4])
            render = HERE / "scratch" / (path.stem + "_review.png")
            page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False).save(str(render))
        verified_figures.append(dict(path=item["path"], sha256=item["sha256"], pages=1, dimensions_inches=dimensions,
                                     text_and_page_bounds=True, render=str(render.relative_to(HERE))))
    exact.write_json(HERE / "figure_verification.json", dict(passed=True, figures=verified_figures,
        source_hashes=True, all_rolling_means_against_direct_windows=rolling_checks,
        max_energy_prefix_scaled_error=max_prefix_energy_error, visual_review="pending"))
    print(json.dumps(dict(new_raw_storage_bytes=report["raw_storage_bytes"], new_execution_seconds=report["new_execution_seconds"],
                          figures=len(verified_figures), figure_checks="passed")))


if __name__ == "__main__":
    main()
