"""Export only panels (a) loss and (b) eta*lambda from the verified FP32 data."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA = HERE / "fp32_figure_data.json"
OUTPUT = HERE / "figures/fixed_ce_curvature_fp32_loss_lambda.png"
MANIFEST = HERE / "fp32_two_panel_png_manifest.json"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(reviewed=False):
    if reviewed:
        record = json.loads(MANIFEST.read_text())
        for key, path in (("script", Path(__file__)), ("data", DATA), ("png", OUTPUT)):
            assert record[key + "_sha256"] == sha(path)
        record.update(status="passed", visual_review="Actual PNG inspected: only panels(a,b), two FP32 arms, readable labels/legend/caption, no overlaps or clipping.")
        MANIFEST.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
        return
    previous = json.loads((HERE / "fp32_figure_manifest.json").read_text())
    assert previous["status"] == "passed" and previous["data_sha256"] == sha(DATA)
    data = json.loads(DATA.read_text())
    assert data["arms"] == ["native32", "fixed32"] and len(data["records"]) == 20
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.0))
    checked = 0
    for arm, color, style, label in (("native32", "#D55E00", "--", "Native CE (FP32)"),
                                     ("fixed32", "#0072B2", "-", "Fixed CE (FP32)")):
        selected = [r for r in data["records"] if r["arm"] == arm]
        assert [r["seed"] for r in selected] == list(range(10))
        for panel, (x, rows) in enumerate((
            (selected[0]["loss"]["x"], [r["loss"]["y"] for r in selected]),
            (data["states"], [r["eta_lambda"] for r in selected]))):
            values = np.asarray(rows)
            assert values.shape == (10, len(x)) and np.isfinite(values).all() and (values > 0).all()
            for row in [*values, np.median(values, axis=0)]:
                median = checked % 11 == 10
                artist, = axes[panel].plot(x, row, color=color, ls=style,
                    lw=2 if median else .6, alpha=1 if median else .23,
                    label=label if median else None, zorder=4 if median else 2,
                    marker="o" if median and panel == 1 else None, ms=2.6)
                np.testing.assert_array_equal(artist.get_xdata(), x)
                np.testing.assert_array_equal(artist.get_ydata(), row)
                checked += 1
        axes[0].scatter(np.full(10, 10000), [r["terminal_loss"] for r in selected],
                        s=11, color=color, alpha=.55, zorder=5, clip_on=False)
    for ax, title, ylabel in zip(axes,
            ["(a) Training loss", "(b) Maximum Hessian curvature"],
            ["Mean cross-entropy", r"$\eta\lambda_{\max}$"]):
        ax.set_title(title, loc="left", fontweight="bold", pad=9)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Completed updates (parameter state)")
        ax.set_yscale("log")
        ax.set_xlim(0, 10100)
        ax.set_xticks([0, 2500, 5000, 7500, 10000])
        ax.grid(alpha=.17, lw=.55)
    fig.suptitle("Fixed cross-entropy: FP32 training comparison", y=.98,
                 fontsize=15, fontweight="bold")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(.5, .922), ncol=2, frameon=False)
    fig.text(.5, .023,
        r"Mod-31 Transformer | 10 paired seeds | AdamW, $\eta=10^{-3}$ | thin: seeds; thick: median." "\n"
        "Loss: 100-state means; dots: final state. Curvature: 23 measured states, all 480 training pairs.\n"
        "Both training arms use FP32; loss / Hessian diagnostics retain the common FP64 evaluator.",
        ha="center", va="bottom", fontsize=8.5, linespacing=1.5)
    fig.subplots_adjust(left=.085, right=.98, top=.79, bottom=.25, wspace=.27)
    fig.canvas.draw()
    bounds = fig.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans)
    assert bounds.x0 >= 0 and bounds.y0 >= 0 and bounds.x1 <= fig.bbox.width+1 and bounds.y1 <= fig.bbox.height+1
    OUTPUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, facecolor="white")
    plt.close(fig)
    with Image.open(OUTPUT) as image:
        assert image.format == "PNG" and image.size == (3420, 1500)
        image.verify()
    assert checked == 44
    record = {"status": "awaiting_visual_review", "script_sha256": sha(__file__),
        "data_sha256": sha(DATA), "png_sha256": sha(OUTPUT), "png_bytes": OUTPUT.stat().st_size,
        "dimensions": [3420, 1500], "dpi": 300, "panels": ["a: loss", "b: eta*lambda_max"],
        "arms": data["arms"], "seeds_per_arm": 10, "source_curves_unchanged": True,
        "plotted_curves_checked": checked, "new_training": False}
    MANIFEST.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(OUTPUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--accept-reviewed", action="store_true")
    main(parser.parse_args().accept_reviewed)
