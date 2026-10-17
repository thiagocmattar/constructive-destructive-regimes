"""Split the verified N=5 log-loss overlay into metric rows and two update ranges."""
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
PDF = HERE / "figures" / (STEM + ".pdf")
MANIFEST = HERE / (STEM + "_manifest.json")
FIELDS = ("log_loss_diff_mean", "eta_kappa_proxy", "rho", "raw_gradient_norm")
RANGES = ((1, 100), (100, 400))
SIZES = ("14M", "31M", "70M")
COLORS = ("#2166ac", "#d95f02", "#1b9e77")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-reviewed", action="store_true")
    args = parser.parse_args()
    if args.accept_reviewed:
        record = json.loads(MANIFEST.read_text())
        assert sha(__file__) == record["script_sha256"]
        assert sha(PDF) == record["output"]["sha256"]
        render = HERE / "scratch" / (STEM + "-1.png")
        assert render.is_file()
        record["visual_review"] = dict(passed=True, renderer="Poppler pdftoppm",
            render_sha256=sha(render), inspection="All eight panels, two column ranges, metric rows, solid colour legend, independent Y scales and reduction notes are legible; no overlap or clipping.")
        MANIFEST.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
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
        HERE / "14_plot_raw_first400.py"] if p != PDF}
    with source_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1200
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "path.simplify": False})
    fig, axes = plt.subplots(4, 2, figsize=(12, 11), sharex="col")
    curves = []
    for size, color in zip(SIZES, COLORS):
        selected = [r for r in rows if r["size"] == size]
        x = np.array([int(r["update"]) for r in selected])
        np.testing.assert_array_equal(x, np.arange(1, 401))
        values = {field: np.array([float(r[field]) if r[field] else np.nan for r in selected]) for field in FIELDS}
        for col, (first, last) in enumerate(RANGES):
            keep = (x >= first) & (x <= last)
            window_x = x[keep]
            np.testing.assert_array_equal(window_x, np.arange(first, last + 1))
            for row, field in enumerate(FIELDS):
                window_y = values[field][keep]
                line, = axes[row, col].plot(window_x, window_y, color=color, ls="-", lw=1.15,
                                           label="Pythia-" + size)
                np.testing.assert_array_equal(line.get_xdata(), window_x)
                np.testing.assert_array_equal(line.get_ydata(), window_y)
                assert line.get_linestyle() == "-" and line.get_color() == color
                expected = (95 if field == "log_loss_diff_mean" else (100 if field == "raw_gradient_norm" else 99)) if col == 0 else 301
                assert np.isfinite(window_y).sum() == expected
                curves.append(dict(size=size, metric=field, column=col + 1,
                    updates=[first, last], finite_points=expected,
                    arrays_sha256=hashlib.sha256(window_x.tobytes() + window_y.tobytes()).hexdigest()))
    labels = ("Change in ln(training loss)\n(5-update trailing mean)",
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
                ax.yaxis.set_major_locator(FixedLocator([.1, .2, .3, .5, 1., 2., 4.]))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
                ax.yaxis.set_minor_formatter(NullFormatter())
        if row == 0:
            axes[row, 0].set_title("Updates 1-100", pad=12, fontsize=13)
            axes[row, 1].set_title("Updates 100-400", pad=12, fontsize=13)
    for ax in axes[-1]:
        ax.set_xlabel("Optimizer update")
    fig.suptitle("Pythia training dynamics by update range", y=.985, fontsize=16)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.5, .955),
               ncol=3, frameon=False, handlelength=3)
    fig.text(.5, .045, r"Loss row: N=5 mean of $\ln L_t-\ln L_{t-1}$; other rows unsmoothed. Seed 1235.", ha="center", fontsize=9)
    fig.text(.5, .025, "Y limits vary by column; norm row uses log Y. Update 100 appears in both columns; smoothing spans the boundary.", ha="center", fontsize=9)
    fig.text(.5, .008, "First loss-change mean at update 6; first raw pair undefined. AdamW eta*kappa is a gradient-change proxy.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.10, right=.98, top=.89, bottom=.105, hspace=.20, wspace=.21)
    fig.savefig(PDF)
    plt.close(fig)
    assert sum(c["finite_points"] for c in curves) == 4791
    assert all(sha(HERE / name) == digest for name, digest in preserved.items())
    record = dict(script_sha256=sha(__file__), source_csv=source_csv.name,
        source_csv_sha256=sha(source_csv), source_manifest_sha256=sha(source_manifest),
        preserved=preserved, curves=curves,
        output=dict(path=PDF.relative_to(HERE).as_posix(), bytes=PDF.stat().st_size, sha256=sha(PDF)),
        columns=[[1, 100], [100, 400]], row_metrics=list(FIELDS), y_limits="independent by column",
        checks=dict(source_outputs_verified=True, plotted_arrays=True, finite_plotted_points=4791,
                    unique_finite_points=4779, repeated_boundary_points=12,
                    smoothing_not_restarted_at_boundary=True, previous_figures_unchanged=True), visual_review=None)
    MANIFEST.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record["checks"], indent=2))


if __name__ == "__main__":
    main()
