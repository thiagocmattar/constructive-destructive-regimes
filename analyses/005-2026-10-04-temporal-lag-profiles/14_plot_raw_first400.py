"""Model-size overlays over updates1..400, with optional short loss/log-loss differences."""
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
STEM = "training_size_overlay_raw_1_400"
FIELDS = ("loss", "eta_kappa_proxy", "rho", "raw_gradient_norm")
SIZES = ("14M", "31M", "70M")
COLORS = ("#2166ac", "#d95f02", "#1b9e77")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def loss_difference_mean(loss, window=5):
    """Signed adjacent differences; a complete trailing mean starts at update window+1."""
    result = np.full(len(loss), np.nan)
    result[window:] = np.convolve(np.diff(loss), np.ones(window) / window, mode="valid")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-reviewed", action="store_true")
    differences = parser.add_mutually_exclusive_group()
    differences.add_argument("--loss-diff", action="store_true",
                        help="Use a 5-update mean of signed adjacent loss changes in a separate figure.")
    differences.add_argument("--log-loss-diff", action="store_true",
                            help="Use a 5-update mean of adjacent natural-log loss differences.")
    args = parser.parse_args()
    use_diff = args.loss_diff or args.log_loss_diff
    diff_metric = "log_loss_diff_mean" if args.log_loss_diff else "loss_diff_mean"
    diff_column = "log_loss_diff" if args.log_loss_diff else "loss_diff"
    if args.log_loss_diff:
        stem = "training_size_overlay_log_loss_diff_n5_1_400"
    else:
        stem = "training_size_overlay_loss_diff_n5_1_400" if args.loss_diff else STEM
    pdf = HERE / "figures" / (stem + ".pdf")
    manifest = HERE / (stem + "_manifest.json")
    if args.accept_reviewed:
        record = json.loads(manifest.read_text())
        assert sha(__file__) == record["script_sha256"]
        for name, spec in record["outputs"].items():
            assert sha(HERE / name) == spec["sha256"]
        render = HERE / "scratch" / (stem + "-1.png")
        assert render.is_file()
        record["visual_review"] = dict(passed=True, renderer="Poppler pdftoppm",
            render_sha256=sha(render), inspection="All four panels, solid colour overlays, 1-400 axes, legend and reduction notes are legible; no clipping or overlap.")
        manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print("Recorded visual acceptance of the separate figure.")
        return

    source_csv = HERE / "training_overview_data.csv"
    source_manifest = HERE / "training_overview_manifest.json"
    previous = json.loads(source_manifest.read_text())
    assert sha(source_csv) == previous["outputs"][source_csv.name]["sha256"]
    for source in previous["sources"]:
        for name, spec in source["files"].items():
            path = ROOT / source["base"] / name
            assert path.stat().st_size == spec["bytes"] and sha(path) == spec["sha256"]
    preserved = {p.relative_to(HERE).as_posix(): sha(p) for p in [
        *sorted((HERE / "figures").glob("*.pdf")), source_csv, source_manifest,
        HERE / "training_size_overlay_data.csv", HERE / "training_size_overlay_results.json",
        HERE / "training_size_overlay_manifest.json", HERE / "13_plot_size_overlay.py"
    ] if p != pdf}
    if use_diff:
        for suffix in ("_data.csv", "_manifest.json"):
            path = HERE / (STEM + suffix)
            preserved[path.name] = sha(path)
        np.testing.assert_allclose(loss_difference_mean(np.arange(12.) * -2)[5:], -2.)
    if args.log_loss_diff:
        for suffix in ("_data.csv", "_manifest.json"):
            path = HERE / ("training_size_overlay_loss_diff_n5_1_400" + suffix)
            preserved[path.name] = sha(path)
        np.testing.assert_allclose(loss_difference_mean(np.log(np.exp(-.1 * np.arange(12.))))[5:], -.1)
    with source_csv.open(newline="", encoding="utf-8") as stream:
        rows = [r for r in csv.DictReader(stream) if 1 <= int(r["update"]) <= 400]
    assert len(rows) == 1200
    assert all(r["partial_batch"] == "False" and int(r["sequences"]) == 1024 for r in rows)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "path.simplify": False})
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.4), sharex=True)
    curves, plot_rows = [], []
    for size, color in zip(SIZES, COLORS):
        selected = [r for r in rows if r["size"] == size]
        x = np.array([int(r["update"]) for r in selected])
        np.testing.assert_array_equal(x, np.arange(1, 401))
        raw = {field: np.array([float(r[field]) if r[field] else np.nan for r in selected]) for field in FIELDS}
        plotted = dict(raw)
        if use_diff:
            assert np.isfinite(raw["loss"]).all() and (raw["loss"] > 0).all()
            loss_values = np.log(raw["loss"]) if args.log_loss_diff else raw["loss"]
            plotted["loss"] = loss_difference_mean(loss_values)
            # Independent telescoping form checks sign, scaling and endpoint placement.
            np.testing.assert_allclose(plotted["loss"][5:],
                (loss_values[5:] - loss_values[:-5]) / 5, rtol=1e-12, atol=1e-12)
            if args.log_loss_diff:
                np.testing.assert_allclose(plotted["loss"][5:],
                    np.log(raw["loss"][5:] / raw["loss"][:-5]) / 5, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(np.diff(loss_values),
                    np.log1p(np.diff(raw["loss"]) / raw["loss"][:-1]), rtol=1e-12, atol=1e-12)
        norm, rho = raw["raw_gradient_norm"], raw["rho"]
        assert np.isfinite(norm).all() and (norm > 0).all()
        np.testing.assert_allclose(raw["eta_kappa_proxy"][1:],
            1 - rho[1:] * norm[1:] / norm[:-1], rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(raw["eta_kappa_proxy"][1:] > 1, rho[1:] < 0)
        for ax, field in zip(axes.flat, FIELDS):
            y = plotted[field]
            first = 6 if field == "loss" and use_diff else (2 if field in ("rho", "eta_kappa_proxy") else 1)
            assert np.isnan(y[:first - 1]).all() and np.isfinite(y[first - 1:]).all()
            line, = ax.plot(x, y, color=color, ls="-", lw=1.1, label="Pythia-" + size)
            np.testing.assert_array_equal(line.get_xdata(), x)
            np.testing.assert_array_equal(line.get_ydata(), y)
            assert line.get_linestyle() == "-" and line.get_color() == color
            metric = diff_metric if field == "loss" and use_diff else field
            curves.append(dict(size=size, metric=metric, first_update=first, last_update=400,
                points=int(np.isfinite(y).sum()), arrays_sha256=hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()))
        for i, update in enumerate(x):
            plot_rows.append(dict(size=size, update=int(update), **{
                field: float(raw[field][i]) if np.isfinite(raw[field][i]) else "" for field in FIELDS}))
            if use_diff:
                plot_rows[-1][diff_column] = float(loss_values[i] - loss_values[i - 1]) if i else ""
                plot_rows[-1][diff_metric] = float(plotted["loss"][i]) if np.isfinite(plotted["loss"][i]) else ""
    if args.log_loss_diff:
        loss_label = "Change in ln(training loss)\n(5-update trailing mean)"
    else:
        loss_label = "Loss change / update\n(5-update trailing mean)" if args.loss_diff else "Training loss"
    labels = (loss_label, r"$\eta\widehat\kappa_t$ (proxy)",
              r"Raw adjacent cosine $\rho_t$", r"Raw gradient $\|g_t\|_2$")
    for ax, label in zip(axes.flat, labels):
        ax.set_ylabel(label)
        ax.set_xlabel("Optimizer update")
        ax.set_xlim(1, 400)
        ax.set_xticks([1, 100, 200, 300, 400])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    axes[0, 1].axhline(1, color=".4", ls=":", lw=.9)
    if use_diff:
        axes[0, 0].axhline(0, color=".4", ls=":", lw=.9)
    axes[1, 0].axhline(0, color=".4", ls=":", lw=.9)
    axes[1, 0].set_ylim(-1.05, 1.05)
    axes[1, 1].set_yscale("log")
    axes[1, 1].yaxis.set_major_locator(FixedLocator([.2, .3, .5, 1., 2., 4.]))
    axes[1, 1].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axes[1, 1].yaxis.set_minor_formatter(NullFormatter())
    fig.suptitle("Model size and MiniPile dynamics: updates 1-400", y=.985, fontsize=16)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.5, .947),
               ncol=3, frameon=False, handlelength=3)
    note = ("Loss panel: mean of L(t)-L(t-1), N=5; negative means falling loss. Other panels: unsmoothed."
            if args.loss_diff else "All variables unsmoothed; every recorded update 1-400. Seed 1235; solid colours identify model size.")
    if args.log_loss_diff:
        note = r"Loss panel: mean of $\ln L_t-\ln L_{t-1}$, N=5; negative means falling loss. Other panels: unsmoothed."
    detail = ("First loss-change mean at update 6. Seed 1235; first pair undefined. Norm: log Y. AdamW eta*kappa is a proxy."
              if use_diff else "First pair undefined. Norm axis: log Y. AdamW eta*kappa is a gradient-change proxy.")
    fig.text(.5, .043, note, ha="center", fontsize=9)
    fig.text(.5, .018, detail, ha="center", fontsize=9)
    fig.subplots_adjust(left=.085, right=.98, top=.865, bottom=.13, hspace=.28, wspace=.23)
    fig.savefig(pdf)
    plt.close(fig)
    point_count = 4779 if use_diff else 4794
    assert sum(c["points"] for c in curves) == point_count
    if use_diff:
        baseline = json.loads((HERE / (STEM + "_manifest.json")).read_text())
        original = {(c["size"], c["metric"]): c["arrays_sha256"] for c in baseline["curves"]}
        for curve in curves:
            if curve["metric"] != diff_metric:
                assert curve["arrays_sha256"] == original[curve["size"], curve["metric"]]
    data_path = HERE / (stem + "_data.csv")
    with data_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(plot_rows[0]))
        writer.writeheader()
        writer.writerows(plot_rows)
    with data_path.open(newline="", encoding="utf-8") as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == 1200
    columns = (*FIELDS, diff_column, diff_metric) if use_diff else FIELDS
    for field in columns:
        np.testing.assert_array_equal([float(r[field]) if r[field] != "" else np.nan for r in plot_rows],
                                      [float(r[field]) if r[field] else np.nan for r in exported])
    assert all(sha(HERE / name) == digest for name, digest in preserved.items())
    record = dict(script_sha256=sha(__file__), source_csv_sha256=sha(source_csv),
        source_manifest_sha256=sha(source_manifest), preserved=preserved, curves=curves,
        outputs={p.relative_to(HERE).as_posix(): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in (pdf, data_path)},
        included_updates=[1, 400], smoothing="loss/log-loss differences: complete 5-update trailing mean; other panels: none" if use_diff else "none for all variables",
        loss_difference_formula="ln(L[t]/L[t-5])/5; first defined at t=6" if args.log_loss_diff else ("(L[t]-L[t-5])/5; first defined at t=6" if args.loss_diff else None),
        log_base="natural" if args.log_loss_diff else None,
        checks=dict(original_sources_verified=True, raw_identity_pairs=1197, finite_plotted_points=point_count,
                    loss_difference_means_checked=1185 if use_diff else 0,
                    positive_loss_and_log_ratio_identities=bool(args.log_loss_diff),
                    csv_roundtrip=True, plotted_arrays=True, raw_diagnostic_arrays=True, solid_model_curves=True,
                    previous_overlay_unchanged=True), visual_review=None)
    manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record["checks"], indent=2))


if __name__ == "__main__":
    main()
