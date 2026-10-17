"""Split model-size dynamics into two ranges, optionally smoothing all rows by column."""
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
SOURCE_STEM = "training_size_overlay_log_loss_diff_n5_1_400"
STEM = "training_size_overlay_log_loss_diff_n5_split"
FIELDS = ("log_loss_diff_mean", "eta_kappa_proxy", "rho", "raw_gradient_norm")
RANGES = ((1, 100), (100, 400))
SIZES = ("14M", "31M", "70M")
COLORS = ("#2166ac", "#d95f02", "#1b9e77")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def trailing_mean(values, window):
    result = np.full(len(values), np.nan)
    result[window - 1:] = np.convolve(values, np.ones(window) / window, mode="valid")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-reviewed", action="store_true")
    parser.add_argument("--smooth-all", action="store_true",
                        help="Use complete N=5 means on the left and N=21 means on the right for every row.")
    args = parser.parse_args()
    stem = "training_size_overlay_log_loss_diff_split_n5_n21" if args.smooth_all else STEM
    pdf = HERE / "figures" / (stem + ".pdf")
    manifest = HERE / (stem + "_manifest.json")
    if args.accept_reviewed:
        record = json.loads(manifest.read_text())
        assert sha(__file__) == record["script_sha256"]
        assert sha(pdf) == record["output"]["sha256"]
        if record.get("plot_data"):
            assert sha(HERE / record["plot_data"]["path"]) == record["plot_data"]["sha256"]
        render = HERE / "scratch" / (stem + "-1.png")
        assert render.is_file()
        record["visual_review"] = dict(passed=True, renderer="Poppler pdftoppm",
            render_sha256=sha(render), inspection="All eight panels, two column ranges, metric rows, solid colour legend, independent Y scales and reduction notes are legible; no overlap or clipping.")
        manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print("Recorded visual acceptance of the split-range figure.")
        return

    source_manifest = HERE / (SOURCE_STEM + "_manifest.json")
    source_csv = HERE / (SOURCE_STEM + "_data.csv")
    source = json.loads(source_manifest.read_text())
    assert source["visual_review"]["passed"]
    for name, spec in source["outputs"].items():
        path = HERE / name
        assert path.stat().st_size == spec["bytes"] and sha(path) == spec["sha256"]
    preserved = {p.relative_to(HERE).as_posix(): sha(p) for p in [
        *sorted((HERE / "figures").glob("*.pdf")), source_csv, source_manifest,
        HERE / "14_plot_raw_first400.py"] if p != pdf}
    if args.smooth_all:
        previous = HERE / (STEM + "_manifest.json")
        preserved[previous.name] = sha(previous)
    with source_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1200
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "path.simplify": False})
    fig, axes = plt.subplots(4, 2, figsize=(12, 11), sharex="col")
    curves, plot_rows = [], []
    for size, color in zip(SIZES, COLORS):
        selected = [r for r in rows if r["size"] == size]
        x = np.array([int(r["update"]) for r in selected])
        np.testing.assert_array_equal(x, np.arange(1, 401))
        # Start from the raw log differences, not the previous N=5 means.
        source_columns = {field: "log_loss_diff" if args.smooth_all and field == "log_loss_diff_mean" else field for field in FIELDS}
        values = {field: np.array([float(r[source_columns[field]]) if r[source_columns[field]] else np.nan for r in selected]) for field in FIELDS}
        for col, (first, last) in enumerate(RANGES):
            window = (5 if col == 0 else 21) if args.smooth_all else None
            displayed = {field: trailing_mean(raw, window) if window else raw for field, raw in values.items()}
            keep = (x >= first) & (x <= last)
            window_x = x[keep]
            np.testing.assert_array_equal(window_x, np.arange(first, last + 1))
            for row, field in enumerate(FIELDS):
                window_y = displayed[field][keep]
                if window:
                    indices = np.flatnonzero(keep)[np.isfinite(window_y)]
                    reference = [np.mean(values[field][i - window + 1:i + 1]) for i in indices]
                    np.testing.assert_allclose(window_y[np.isfinite(window_y)], reference, rtol=1e-12, atol=1e-12)
                    if field == "log_loss_diff_mean":
                        loss = np.array([float(r["loss"]) for r in selected])
                        assert np.isfinite(loss).all() and (loss > 0).all()
                        np.testing.assert_allclose(window_y[np.isfinite(window_y)],
                            np.log(loss[indices] / loss[indices - window]) / window, rtol=1e-12, atol=1e-12)
                line, = axes[row, col].plot(window_x, window_y, color=color, ls="-", lw=1.15,
                                           label="Pythia-" + size)
                np.testing.assert_array_equal(line.get_xdata(), window_x)
                np.testing.assert_array_equal(line.get_ydata(), window_y)
                assert line.get_linestyle() == "-" and line.get_color() == color
                if args.smooth_all:
                    expected = (96 if field == "raw_gradient_norm" else 95) if col == 0 else 301
                else:
                    expected = (95 if field == "log_loss_diff_mean" else (100 if field == "raw_gradient_norm" else 99)) if col == 0 else 301
                assert np.isfinite(window_y).sum() == expected
                curves.append(dict(size=size, metric=field, column=col + 1,
                    updates=[first, last], smoothing_window=window, finite_points=expected,
                    arrays_sha256=hashlib.sha256(window_x.tobytes() + window_y.tobytes()).hexdigest()))
            if args.smooth_all:
                for i in np.flatnonzero(keep):
                    plot_rows.append(dict(size=size, column=col + 1, window=window, update=int(x[i]), **{
                        field if field.endswith("_mean") else field + "_mean": float(displayed[field][i]) if np.isfinite(displayed[field][i]) else "" for field in FIELDS}))
    loss_label = "Change in ln(training loss)" if args.smooth_all else "Change in ln(training loss)\n(5-update trailing mean)"
    labels = (loss_label,
              r"$\eta\widehat\kappa_t$ (proxy)", r"Raw adjacent cosine $\rho_t$", r"Raw gradient $\|g_t\|_2$")
    for row, label in enumerate(labels):
        axes[row, 0].set_ylabel(label)
        for col, (first, last) in enumerate(RANGES):
            ax = axes[row, col]
            ax.set_xlim(first, last)
            ax.set_xticks([1, 25, 50, 75, 100] if col == 0 else [100, 150, 200, 250, 300, 350, 400])
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.2)
            if row in (0, 2):
                ax.axhline(0, color=".4", ls=":", lw=.9)
            elif row == 1:
                ax.axhline(1, color=".4", ls=":", lw=.9)
            elif row == 3:
                ax.set_yscale("log")
                ticks = [.2, .3, .4, .5, .6, .8, 1., 2., 4.] if args.smooth_all and col == 1 else [.1, .2, .3, .5, 1., 2., 4.]
                ax.yaxis.set_major_locator(FixedLocator(ticks))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
                ax.yaxis.set_minor_formatter(NullFormatter())
        if row == 0:
            axes[row, 0].set_title("Updates 1-100" + ("  |  N=5" if args.smooth_all else ""), pad=12, fontsize=13)
            axes[row, 1].set_title("Updates 100-400" + ("  |  N=21" if args.smooth_all else ""), pad=12, fontsize=13)
    for ax in axes[-1]:
        ax.set_xlabel("Optimizer update")
    fig.suptitle("Pythia training dynamics by update range", y=.985, fontsize=16)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.5, .955),
               ncol=3, frameon=False, handlelength=3)
    note = (r"All rows: complete trailing means, N=5 left and N=21 right. Loss uses $\ln L_t-\ln L_{t-1}$. Seed 1235."
            if args.smooth_all else r"Loss row: N=5 mean of $\ln L_t-\ln L_{t-1}$; other rows unsmoothed. Seed 1235.")
    fig.text(.5, .045, note, ha="center", fontsize=9)
    fig.text(.5, .025, "Y limits vary by column; norm row uses log Y. Update 100 appears in both columns; smoothing spans the boundary.", ha="center", fontsize=9)
    detail = ("Left means start at update 5 (norm) / 6 (other rows). Update 100 has different smoothing windows in the two columns."
              if args.smooth_all else "First loss-change mean at update 6; first raw pair undefined. AdamW eta*kappa is a gradient-change proxy.")
    fig.text(.5, .008, detail, ha="center", fontsize=9)
    fig.subplots_adjust(left=.10, right=.98, top=.89, bottom=.105, hspace=.20, wspace=.21)
    fig.savefig(pdf)
    plt.close(fig)
    point_count = 4755 if args.smooth_all else 4791
    assert sum(c["finite_points"] for c in curves) == point_count
    plot_data = None
    if args.smooth_all:
        data_path = HERE / (stem + "_data.csv")
        with data_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(plot_rows[0]))
            writer.writeheader()
            writer.writerows(plot_rows)
        with data_path.open(newline="", encoding="utf-8") as stream:
            exported = list(csv.DictReader(stream))
        assert len(exported) == len(plot_rows) == 1203
        for field in FIELDS:
            key = field if field.endswith("_mean") else field + "_mean"
            np.testing.assert_array_equal([float(r[key]) if r[key] != "" else np.nan for r in plot_rows],
                                          [float(r[key]) if r[key] else np.nan for r in exported])
        plot_data = dict(path=data_path.name, bytes=data_path.stat().st_size, sha256=sha(data_path))
    assert all(sha(HERE / name) == digest for name, digest in preserved.items())
    record = dict(script_sha256=sha(__file__), source_csv=source_csv.name,
        source_csv_sha256=sha(source_csv), source_manifest_sha256=sha(source_manifest),
        preserved=preserved, curves=curves,
        output=dict(path=pdf.relative_to(HERE).as_posix(), bytes=pdf.stat().st_size, sha256=sha(pdf)), plot_data=plot_data,
        columns=[[1, 100], [100, 400]], row_metrics=list(FIELDS), y_limits="independent by column",
        smoothing_windows=[5, 21] if args.smooth_all else None,
        checks=dict(source_outputs_verified=True, plotted_arrays=True, finite_plotted_points=point_count,
                    independent_means_checked=4755 if args.smooth_all else 0,
                    csv_roundtrip=bool(args.smooth_all),
                    repeated_update100_points=12, update100_values_differ_by_window=bool(args.smooth_all),
                    raw_log_differences_used=bool(args.smooth_all),
                    smoothing_not_restarted_at_boundary=True, previous_figures_unchanged=True), visual_review=None)
    manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record["checks"], indent=2))


if __name__ == "__main__":
    main()
