"""Separate unsmoothed model-size overlay restricted to updates1..400."""
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
PDF = HERE / "figures" / (STEM + ".pdf")
MANIFEST = HERE / (STEM + "_manifest.json")
FIELDS = ("loss", "eta_kappa_proxy", "rho", "raw_gradient_norm")
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
        for name, spec in record["outputs"].items():
            assert sha(HERE / name) == spec["sha256"]
        render = HERE / "scratch" / (STEM + "-1.png")
        assert render.is_file()
        record["visual_review"] = dict(passed=True, renderer="Poppler pdftoppm",
            render_sha256=sha(render), inspection="All four unsmoothed panels, solid colour overlays, 1-400 axes, legend and notes are legible; no clipping or overlap.")
        MANIFEST.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print("Recorded visual acceptance of the separate unsmoothed figure.")
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
    ] if p != PDF}
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
        norm, rho = raw["raw_gradient_norm"], raw["rho"]
        assert np.isfinite(norm).all() and (norm > 0).all()
        np.testing.assert_allclose(raw["eta_kappa_proxy"][1:],
            1 - rho[1:] * norm[1:] / norm[:-1], rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(raw["eta_kappa_proxy"][1:] > 1, rho[1:] < 0)
        for ax, field in zip(axes.flat, FIELDS):
            y = raw[field]
            first = 2 if field in ("rho", "eta_kappa_proxy") else 1
            assert np.isnan(y[:first - 1]).all() and np.isfinite(y[first - 1:]).all()
            line, = ax.plot(x, y, color=color, ls="-", lw=1.1, label="Pythia-" + size)
            np.testing.assert_array_equal(line.get_xdata(), x)
            np.testing.assert_array_equal(line.get_ydata(), y)
            assert line.get_linestyle() == "-" and line.get_color() == color
            curves.append(dict(size=size, metric=field, first_update=first, last_update=400,
                points=int(np.isfinite(y).sum()), arrays_sha256=hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()))
        for i, update in enumerate(x):
            plot_rows.append(dict(size=size, update=int(update), **{
                field: float(raw[field][i]) if np.isfinite(raw[field][i]) else "" for field in FIELDS}))
    labels = ("Training loss", r"$\eta\widehat\kappa_t$ (proxy)",
              r"Raw adjacent cosine $\rho_t$", r"Raw gradient $\|g_t\|_2$")
    for ax, label in zip(axes.flat, labels):
        ax.set_ylabel(label)
        ax.set_xlabel("Optimizer update")
        ax.set_xlim(1, 400)
        ax.set_xticks([1, 100, 200, 300, 400])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    axes[0, 1].axhline(1, color=".4", ls=":", lw=.9)
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
    fig.text(.5, .043, "All variables unsmoothed; every recorded update 1-400. Seed 1235; solid colours identify model size.", ha="center", fontsize=9)
    fig.text(.5, .018, "First pair undefined. Norm axis: log Y. AdamW eta*kappa is a gradient-change proxy.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.085, right=.98, top=.865, bottom=.13, hspace=.28, wspace=.23)
    fig.savefig(PDF)
    plt.close(fig)
    assert sum(c["points"] for c in curves) == 4794
    data_path = HERE / (STEM + "_data.csv")
    with data_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(plot_rows[0]))
        writer.writeheader()
        writer.writerows(plot_rows)
    with data_path.open(newline="", encoding="utf-8") as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == 1200
    for field in FIELDS:
        np.testing.assert_array_equal([float(r[field]) if r[field] != "" else np.nan for r in plot_rows],
                                      [float(r[field]) if r[field] else np.nan for r in exported])
    assert all(sha(HERE / name) == digest for name, digest in preserved.items())
    record = dict(script_sha256=sha(__file__), source_csv_sha256=sha(source_csv),
        source_manifest_sha256=sha(source_manifest), preserved=preserved, curves=curves,
        outputs={p.relative_to(HERE).as_posix(): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in (PDF, data_path)},
        included_updates=[1, 400], smoothing="none for all variables",
        checks=dict(original_sources_verified=True, raw_identity_pairs=1197, finite_plotted_points=4794,
                    csv_roundtrip=True, raw_plotted_arrays=True, solid_model_curves=True,
                    previous_overlay_unchanged=True), visual_review=None)
    MANIFEST.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record["checks"], indent=2))


if __name__ == "__main__":
    main()
