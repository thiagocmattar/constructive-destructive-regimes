"""Overlay model sizes using complete 20-update means after excluding update712."""
from pathlib import Path
import argparse
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WINDOW = 20
FIELDS = ("loss", "eta_kappa_proxy", "rho", "raw_gradient_norm")
SIZES = ("14M", "31M", "70M")
COLORS = ("#2166ac", "#d95f02", "#1b9e77")
STYLES = ("-", "--", "-.")
PDF = HERE / "figures/training_size_overlay.pdf"
MANIFEST = HERE / "training_size_overlay_manifest.json"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def trailing_mean(values):
    output = np.full(len(values), np.nan)
    output[WINDOW - 1:] = np.convolve(values, np.ones(WINDOW) / WINDOW, mode="valid")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-reviewed", action="store_true")
    args = parser.parse_args()
    if args.accept_reviewed:
        record = json.loads(MANIFEST.read_text())
        assert sha(__file__) == record["script_sha256"]
        for name, spec in record["outputs"].items():
            assert sha(HERE / name) == spec["sha256"]
        render = HERE / "scratch/training_size_overlay-1.png"
        assert render.is_file()
        record["visual_review"] = dict(passed=True, renderer="Poppler pdftoppm",
            render_sha256=sha(render), inspection="All four panels, model overlays, legend, log-norm ticks and notes are legible; no clipping or overlap.")
        MANIFEST.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print("Recorded visual acceptance of the smoothed size overlay.")
        return

    source_manifest = HERE / "training_overview_manifest.json"
    previous = json.loads(source_manifest.read_text())
    source_csv = HERE / "training_overview_data.csv"
    assert sha(source_csv) == previous["outputs"][source_csv.name]["sha256"]
    for source in previous["sources"]:
        for name, spec in source["files"].items():
            path = ROOT / source["base"] / name
            assert path.stat().st_size == spec["bytes"] and sha(path) == spec["sha256"]
    preserved = {p.relative_to(HERE).as_posix(): sha(p) for p in [
        HERE / "training_overview_data.csv", HERE / "training_overview_results.json",
        source_manifest, HERE / "results.json", HERE / "artifact_validation.json",
        *sorted((HERE / "figures").glob("*.pdf"))] if p != PDF}
    with source_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    # Exclude the partial final batch BEFORE every reduction and moving mean.
    selected = [r for r in rows if int(r["update"]) != 712]
    assert len(rows) == 2136 and len(selected) == 2133
    assert all(r["partial_batch"] == "False" and int(r["sequences"]) == 1024 for r in selected)
    np.testing.assert_allclose(trailing_mean(np.arange(25.))[19:], np.arange(9.5, 15.5))
    zero_pair = np.arange(25., dtype=float)
    zero_pair[0] = np.nan
    assert np.isnan(trailing_mean(zero_pair)[:20]).all()

    curves, summaries, plot_rows = [], [], []
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.4), sharex=True)
    labels = ("Training loss", r"$\eta\widehat\kappa_t$ (proxy)",
              r"Raw adjacent cosine $\rho_t$", r"Raw gradient $\|g_t\|_2$")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "path.simplify": False})
    for size, color, style in zip(SIZES, COLORS, STYLES):
        source_rows = [r for r in selected if r["size"] == size]
        x = np.array([int(r["update"]) for r in source_rows])
        np.testing.assert_array_equal(x, np.arange(1, 712))
        raw = {field: np.array([float(r[field]) if r[field] else np.nan for r in source_rows]) for field in FIELDS}
        expected_q = 1 - raw["rho"][1:] * raw["raw_gradient_norm"][1:] / raw["raw_gradient_norm"][:-1]
        np.testing.assert_allclose(raw["eta_kappa_proxy"][1:], expected_q, rtol=1e-12, atol=1e-12)
        smooth = {field: trailing_mean(values) for field, values in raw.items()}
        for ax, field in zip(axes.flat, FIELDS):
            y = smooth[field]
            first = 21 if field in ("rho", "eta_kappa_proxy") else 20
            assert np.isfinite(y).sum() == 712 - first
            assert np.isnan(y[:first - 1]).all() and np.isfinite(y[first - 1:]).all()
            # Independent slicing checks every complete mean and placement.
            reference = [np.mean(raw[field][i - WINDOW + 1:i + 1]) for i in range(first - 1, 711)]
            np.testing.assert_allclose(y[first - 1:], reference, rtol=2e-13, atol=2e-13)
            line, = ax.plot(x, y, label="Pythia-" + size, color=color, ls=style, lw=1.9)
            np.testing.assert_array_equal(line.get_xdata(), x)
            np.testing.assert_array_equal(line.get_ydata(), y)
            curves.append(dict(size=size, metric=field, first_update=first, last_update=711,
                points=int(np.isfinite(y).sum()), arrays_sha256=hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()))
        for i, update in enumerate(x):
            plot_rows.append(dict(size=size, update=int(update), **{
                field + "_mean": float(smooth[field][i]) if np.isfinite(smooth[field][i]) else "" for field in FIELDS}))
        early = {}
        for stop in (10, 50, 100):
            loss = raw["loss"][:stop]
            norm = raw["raw_gradient_norm"][:stop]
            early[str(stop)] = dict(first_update=1, last_update=stop,
                loss_start=float(loss[0]), loss_end=float(loss[-1]),
                loss_drop=float(loss[0] - loss[-1]),
                average_loss_drop_per_update=float((loss[0] - loss[-1]) / (stop - 1)),
                ols_loss_slope=float(np.polyfit(x[:stop], loss, 1)[0]),
                mean_raw_gradient_norm=float(norm.mean()))
        summaries.append(dict(size=size, run=source_rows[0]["run"], windows=early,
            loss_at_update711=float(raw["loss"][-1]),
            full_batch_negative_pair_fraction=float(np.mean(raw["rho"][1:] < 0))))
    for ax, label in zip(axes.flat, labels):
        ax.set_ylabel(label)
        ax.set_xlabel("Optimizer update")
        ax.set_xlim(0, 720)
        ax.set_xticks([1, 100, 200, 300, 400, 500, 600, 711])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    axes[0, 1].axhline(1, color=".4", ls=":", lw=.9)
    axes[1, 0].axhline(0, color=".4", ls=":", lw=.9)
    axes[1, 1].set_yscale("log")
    axes[1, 1].yaxis.set_major_locator(FixedLocator([.2, .3, .5, 1., 2.]))
    axes[1, 1].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axes[1, 1].yaxis.set_minor_formatter(NullFormatter())
    fig.suptitle("Model size and MiniPile training dynamics", y=.985, fontsize=16)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.5, .947),
               ncol=3, frameon=False, handlelength=3)
    fig.text(.5, .043, "Complete 20-update trailing means; seed 1235. Update 712 excluded before smoothing.", ha="center", fontsize=9)
    fig.text(.5, .018, "First mean: update 20 (loss/norm), 21 (pairs). Norm axis: log Y. AdamW q is a gradient-change proxy.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.085, right=.98, top=.865, bottom=.13, hspace=.28, wspace=.23)
    fig.savefig(PDF)
    plt.close(fig)
    assert sum(c["points"] for c in curves) == 8298
    data_path = HERE / "training_size_overlay_data.csv"
    with data_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(plot_rows[0]))
        writer.writeheader()
        writer.writerows(plot_rows)
    with data_path.open(newline="", encoding="utf-8") as stream:
        exported = list(csv.DictReader(stream))
    for size in SIZES:
        wanted = [r for r in plot_rows if r["size"] == size]
        actual = [r for r in exported if r["size"] == size]
        for field in FIELDS:
            key = field + "_mean"
            np.testing.assert_array_equal([r[key] if r[key] != "" else np.nan for r in wanted],
                [float(r[key]) if r[key] else np.nan for r in actual])
    results = dict(smoothing="complete 20-update trailing arithmetic mean of each raw scalar",
        excluded_update=712, included_updates=[1, 711], finite_plotted_points=8298,
        comparison="Same tokens/order/AdamW recipe; architecture-specific tensors; seed1235",
        cohort=summaries, training_launched=False)
    result_path = HERE / "training_size_overlay_results.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    assert all(sha(HERE / name) == digest for name, digest in preserved.items())
    record = dict(script_sha256=sha(__file__), source_manifest_sha256=sha(source_manifest),
        source_csv_sha256=sha(source_csv), preserved=preserved,
        outputs={p.relative_to(HERE).as_posix(): dict(bytes=p.stat().st_size, sha256=sha(p))
                 for p in (PDF, data_path, result_path)}, curves=curves,
        checks=dict(original_sources_verified=True, update712_excluded_before_smoothing=True,
                    independent_means_checked=8298, plotted_arrays=True, csv_roundtrip=True,
                    raw_identity_pairs=2130, original_outputs_unchanged=True), visual_review=None)
    MANIFEST.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
