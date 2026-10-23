"""Publication PDF from verified complete H022 reductions; no training changes."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.fixed_ce_curvature.core import save_json, sha, stamp

COLORS = {"native32": "#D55E00", "fixed32": "#0072B2", "ce64": "#009E73"}
LABELS = {"native32": "Native CE / FP32", "fixed32": "Fixed CE / FP32", "ce64": "Native CE / FP64 loss only"}


def traces(ax, x, curves, color, label, logarithmic=True):
    matrix = np.array(curves, dtype=float)
    assert matrix.shape == (10, len(x))
    for row in matrix:
        ax.plot(x, row, color=color, lw=.45, alpha=.18)
    with np.errstate(all="ignore"):
        median = np.nanmedian(matrix, axis=0)
    ax.plot(x, median, color=color, lw=1.65, label=label)
    if logarithmic:
        positive = matrix[np.isfinite(matrix)]
        if positive.size and positive.min() <= 0:
            ax.set_yscale("symlog", linthresh=1e-6)
        elif positive.size and ax.get_yscale() != "symlog":
            ax.set_yscale("log")
    return matrix


def main():
    data = json.loads((HERE / "figure_data.json").read_text())
    result = json.loads((HERE / "results.json").read_text())
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axes = plt.subplots(4, 2, figsize=(11.8, 13.0))
    for ax in axes.ravel():
        ax.grid(alpha=.2, linewidth=.5)
        ax.spines[["right", "top"]].set_visible(False)
    titles = ["(a) Accurate training loss", "(b) Native-arm target-gradient collapse",
              "(c) Maximum Hessian curvature", "(d) Gradient-direction curvature",
              "(e) Actual-update curvature", "(f) Adam-scaled maximum curvature",
              "(g) Paired late-window curvature effects", "(h) Curvature at matched training loss"]
    for ax, title in zip(axes.ravel(), titles):
        ax.set_title(title, loc="left", fontweight="bold")
    for arm in COLORS:
        curves = data["curves"][arm]
        color, label = COLORS[arm], LABELS[arm]
        traces(axes[0, 0], curves["loss_steps"], curves["loss_seed_curves"], color, label)
        terminal = [r["final_common_loss"] for r in result["runs"] if r["arm"] == arm]
        axes[0, 0].scatter(np.full(10, 10000), terminal, color=color, s=10, alpha=.6,
                           clip_on=False, zorder=4)
        for ax, key in ((axes[1, 0], "ordinary_seed_curves"),
                        (axes[1, 1], "gradient_seed_curves"),
                        (axes[2, 0], "update_seed_curves"),
                        (axes[2, 1], "adam_seed_curves")):
            # Terminal update and initial Adam state are explicitly undefined.
            x = np.array(data["states"])
            values = np.array(curves[key], dtype=float)
            valid = ~np.all(np.isnan(values), axis=0)
            traces(ax, x[valid], values[:, valid].tolist(), color, label)
    baseline = data["curves"]["native32"]
    traces(axes[0, 1], baseline["loss_steps"], baseline["collapse_seed_curves"],
           COLORS["native32"], "Native CE", logarithmic=False)
    axes[0, 1].axhline(.1, color="0.4", ls="--", lw=.8, label="10% exposure threshold")
    axes[0, 1].set_ylim(-.005, max(.12, np.max(baseline["collapse_seed_curves"]) * 1.1))
    axes[0, 1].legend(loc="center right", frameon=False)
    axes[0, 0].set_ylabel("Mean cross-entropy")
    axes[0, 1].set_ylabel("Fraction of 480 training pairs")
    axes[1, 0].set_ylabel(r"$\lambda_{\max}(H)$")
    axes[1, 1].set_ylabel(r"$g^T H g/\|g\|^2$")
    axes[2, 0].set_ylabel(r"$\Delta^T H\Delta/\|\Delta\|^2$")
    axes[2, 1].set_ylabel(r"$\lambda_{\max}(D^{1/2} H D^{1/2})$")
    for ax in axes[:3].ravel():
        ax.set_xlabel("Completed updates (parameter state)")
        ax.set_xlim(0, 10000)
    ax = axes[3, 0]
    for index, (arm, key) in enumerate((("fixed32", "fixed32_over_native32"), ("ce64", "ce64_over_native32"))):
        summary = data["primary"][key]
        ratios = np.array(summary["seed_ratios"])
        ax.scatter(index + np.linspace(-.14, .14, 10), ratios, s=22, color=COLORS[arm], alpha=.85)
        value = summary["geometric_ratio"]
        lower, upper = summary["ci95"]
        ax.errorbar(index + .28, value, yerr=[[value - lower], [upper - value]],
                    color="black", marker="D", markersize=4, capsize=4, lw=1.2)
    ax.axhline(1, color="0.25", lw=.9)
    ax.axhline(.8, color="0.4", lw=.9, ls="--")
    ax.set_yscale("log")
    ax.set_xticks([0, 1], ["Fixed FP32 / native FP32", "FP64 loss / native FP32"])
    ax.tick_params(axis="x", labelsize=8)
    ax.set_ylabel("Ratio of late-window geometric means")
    ax = axes[3, 1]
    for arm in ("fixed32", "ce64"):
        selected = [r for r in data["matched_loss"] if r["arm"] == arm and r["qualified"]]
        for level in sorted({r["level"] for r in selected}):
            values = [r["curvature_ratio"] for r in selected if r["level"] == level]
            offset = .95 if arm == "fixed32" else 1.05
            ax.scatter(np.full(len(values), level * offset), values, s=15,
                       color=COLORS[arm], alpha=.35)
            ax.scatter([level * offset], [np.median(values)], marker="D", s=30, color=COLORS[arm])
    ax.axhline(1, color="0.25", lw=.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("First-crossing training-loss level")
    ax.set_ylabel("Curvature ratio versus native FP32")
    qualified = sum(r["qualified"] for r in data["matched_loss"])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .965), ncol=3, frameon=False)
    figure.suptitle("Fixed cross-entropy and curvature under paired AdamW training", fontsize=14, fontweight="bold", y=.99)
    qualification = data["qualification"]["qualified_native_seeds"]
    figure.text(.5, .025,
        "Modular addition mod 31; width-64 Transformer; 10 paired seeds × 3 arms × 10,000 updates. Final dots in (a): full-FP64 objective.\n"
        "Thin lines: individual seeds; thick lines: medians. Every Hessian uses the same full FP64 objective on all 480 training pairs.\n"
        f"Largest algebraic eigenpairs have relative residual ≤0.005. Collapse exposure qualified in {qualification}/10 native seeds (8 required).\n"
        f"(g) Dots: seeds; diamonds: geometric means + 95% CI; dashed: 20% threshold. (h) Dots: seeds; diamonds: medians; {qualified}/100 loss matches.",
        ha="center", va="bottom", fontsize=8)
    figure.subplots_adjust(left=.09, right=.98, bottom=.13, top=.935, hspace=.52, wspace=.28)
    directory = HERE / "figures"
    directory.mkdir(exist_ok=True)
    pdf = directory / "fixed_ce_curvature.pdf"
    figure.savefig(pdf, metadata={"Title": "Fixed CE curvature: ten paired seeds", "Author": "Constructive-destructive research workspace"})
    figure.savefig(ROOT / "tmp/analysis026/fixed_ce_curvature.png", dpi=130)
    plt.close(figure)
    save_json(HERE / "figure_manifest.json", {"utc": stamp(), "source_data_sha256": sha(HERE / "figure_data.json"),
        "source_results_sha256": sha(HERE / "results.json"), "script_sha256": sha(Path(__file__)),
        "pdf_sha256": sha(pdf), "pdf_bytes": pdf.stat().st_size,
        "coverage": result["coverage"], "loss_display_sampling": "pre-update states0,20,...,9980; all raw states retained",
        "terminal_loss_points": "state10000; full-FP64 objective from results final_common_loss; all10 seeds/arm",
        "curvature_sampling": "all23 prespecified regular states; undefined endpoints omitted",
        "uncertainty": "seed traces/medians; paired95% t CI in panelg; descriptive matched-loss control"})
    print(pdf)


if __name__ == "__main__":
    main()
