"""Reduce verified Run001 events into figures and a descriptive observation."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def identity(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def analyze(root, destination):
    root, destination = Path(root), Path(destination)
    verification = json.loads((root / "verification.json").read_text())
    if not verification["verified"]:
        raise ValueError("Require accepted full-pass verification before analysis.")
    summary = json.loads((root / "summary.json").read_text())
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    rows = [e for e in events if e["event"] == "update"]
    if len(rows) != summary["full_updates"]:
        raise ValueError("Incomplete update records.")
    figures = destination / "figures"
    observations = destination / "observations"
    scratch = destination / "scratch"
    for path in (figures, observations, scratch):
        path.mkdir(parents=True, exist_ok=True)
    x = np.array([e["update"] for e in rows])
    def values(key):
        return np.array([np.nan if e[key] is None else e[key] for e in rows], dtype=float)
    cosine = values("adjacent_cosine")
    norm_sum = values("cumulative_norm_sum")
    sum_norm = values("cumulative_gradient_norm")
    ratio = values("cumulative_ratio")
    gradient = values("gradient_norm")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(x, cosine, color="#2462a3", alpha=.5, lw=.7, label="Every update")
    rolling = np.array([np.nanmedian(cosine[max(1, i-24):i+1]) if i else np.nan
                        for i in range(len(rows))])
    ax.plot(x, rolling, color="#133c68", lw=1.6, label="Trailing 25-update median")
    ax.axhline(0, color="black", lw=.8, ls="--")
    ax.set(ylabel="Adjacent raw-gradient cosine", ylim=(-1.05, 1.05))
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.plot(x, norm_sum, label=r"$A_t=\sum_{u\leq t}\|g_u\|_2$", color="#d2771c", lw=1.5)
    ax.plot(x, sum_norm, label=r"$B_t=\|\sum_{u\leq t}g_u\|_2$", color="#2462a3", lw=1.5, ls="--")
    ax.set(ylabel="Cumulative gradient norms", ylim=(0, None))
    ax.legend(fontsize=9)
    axes[1, 0].plot(x, ratio, color="#28794d", lw=1.5)
    axes[1, 0].set(ylabel=r"Cumulative ratio $R_t=B_t/A_t$", ylim=(0, 1.05))
    axes[1, 1].plot(x, gradient, color="#8052a0", lw=.9)
    axes[1, 1].scatter([x[-1]], [gradient[-1]], color="#b72d2d", s=24,
                       label=f"Final smaller update ({rows[-1]['sequences']} sequences)", zorder=3)
    axes[1, 1].set(ylabel=r"Current raw-gradient norm $\|g_t\|_2$", ylim=(0, None))
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("Optimizer update")
        ax.grid(alpha=.18)
    fig.suptitle("Pythia-14M / MiniPile: raw-gradient geometry under AdamW\n"
                 "Full pass; seed 1234; capture before clipping and optimizer transformation", fontsize=12)
    fig.savefig(figures / "gradient_evolution.pdf")
    fig.savefig(scratch / "gradient_evolution.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, constrained_layout=True)
    axes[0].plot(x, values("loss"), color="#2462a3", lw=1, label="Training update loss (changing data)")
    axes[0].scatter([0, x[-1]], [summary['initial_validation']['loss'], summary['final_validation']['loss']],
                    color="#d2771c", marker="D", label="Full validation (fixed split)", zorder=3)
    axes[0].set_ylabel("Mean causal next-token loss")
    axes[0].legend(fontsize=8)
    axes[1].plot(x, values("learning_rate"), color="#28794d", lw=1)
    axes[1].set(ylabel="AdamW learning rate", xlabel="Optimizer update")
    for ax in axes:
        ax.grid(alpha=.18)
    fig.savefig(figures / "optimization_context.pdf")
    fig.savefig(scratch / "optimization_context.png", dpi=180)
    plt.close(fig)
    finite = cosine[np.isfinite(cosine)]
    phases = []
    for indices in np.array_split(np.arange(len(rows)), 4):
        c = cosine[indices]; c = c[np.isfinite(c)]
        phases.append({"first_update": int(x[indices[0]]), "last_update": int(x[indices[-1]]),
                       "defined_cosines": len(c), "median_cosine": float(np.median(c)),
                       "mean_cosine": float(np.mean(c)), "negative_cosines": int(np.sum(c < 0)),
                       "median_gradient_norm": float(np.median(gradient[indices])),
                       "end_cumulative_ratio": float(ratio[indices[-1]])})
    reduction = {"source_events_sha256": identity(root / "events.jsonl"),
                 "source_summary_sha256": identity(root / "summary.json"),
                 "script_sha256": identity(Path(__file__)), "seed": 1234, "updates": len(rows),
                 "defined_adjacent_cosines": len(finite), "mean_cosine": float(np.mean(finite)),
                 "median_cosine": float(np.median(finite)), "min_cosine": float(np.min(finite)),
                 "max_cosine": float(np.max(finite)), "negative_cosines": int(np.sum(finite < 0)),
                 "final_cumulative_norm_sum": float(norm_sum[-1]),
                 "final_cumulative_gradient_norm": float(sum_norm[-1]),
                 "final_cumulative_ratio": float(ratio[-1]), "phases": phases,
                 "figure_rows": [{k:e[k] for k in ("update", "adjacent_cosine", "gradient_norm",
                                    "cumulative_norm_sum", "cumulative_gradient_norm", "cumulative_ratio",
                                    "loss", "learning_rate", "sequences")} for e in rows]}
    (figures / "figure_data.json").write_text(json.dumps(reduction, indent=2, allow_nan=False) + "\n")
    table = "\n".join(f"| {p['first_update']}–{p['last_update']} | {p['median_cosine']:.5f} | "
                        f"{p['negative_cosines']}/{p['defined_cosines']} | {p['median_gradient_norm']:.5f} | "
                        f"{p['end_cumulative_ratio']:.5f} |" for p in phases)
    source_link = Path(os.path.relpath(root, observations)).as_posix()
    text = f"""# Observation 001: full-pass raw-gradient evolution

Question: how do adjacent-gradient alignment, accumulated gradient norms, and their ratio evolve during one dense AdamW pass?

Source: `{source_link}/events.jsonl` and `summary.json`, accepted by `03_verify.py` and the transfer inventory. SHA-256 identities and all plotted scalar rows are in `../figures/figure_data.json`. Reduction/plots: `../04_analyze.py`. Whole model, FP64 cumulative sums/reductions, raw unscaled gradients captured before clipping; one update includes accumulated microbatches. No filtering of updates. First adjacent cosine is undefined and plotted as a gap. The trailing median uses up to 25 defined update records and is descriptive smoothing, not uncertainty.

Coverage: {len(rows)} updates, {summary['completed_sequences']:,} packed sequences including the short tail, {summary['completed_input_tokens']:,} input tokens and {summary['completed_supervised_tokens']:,} supervised targets. Seed 1234; one trajectory. The last update contains {rows[-1]['sequences']} sequences rather than 1,024 and is flagged on the gradient-norm panel.

Adjacent cosines: median {np.median(finite):.6f}, mean {np.mean(finite):.6f}, range [{np.min(finite):.6f}, {np.max(finite):.6f}]; {np.sum(finite < 0)}/{len(finite)} defined pairs are negative.

At the final update, the cumulative sum of norms is {norm_sum[-1]:.6f}; the norm of the cumulative vector sum is {sum_norm[-1]:.6f}; their ratio is {ratio[-1]:.6f}. This ratio describes how much of total raw-gradient vector length remains in the cumulative resultant. It includes all temporal interactions among the recorded vectors and does not truncate at lag one. A small ratio can arise from orthogonality as well as opposing directions.

| Updates | Median adjacent cosine | Negative pairs | Median gradient norm | Ratio at phase end |
| --- | ---: | ---: | ---: | ---: |
{table}

Optimization context: full-validation loss moves from {summary['initial_validation']['loss']:.6f} to {summary['final_validation']['loss']:.6f}, using all 500 validation documents, including the short packed tail. Training losses use changing batches and are not fixed-probe losses. The first AdamW update has zero learning rate under the reference pre-step schedule; its gradient is still measured and included.

Figure caption (`gradient_evolution.pdf`): Raw-gradient geometry during a full MiniPile pass with randomly initialized Pythia-14M and AdamW. Top left: cosine between consecutive accumulated, unscaled gradients (all updates and trailing 25-update median). Top right: cumulative sum of gradient norms and norm of the cumulative gradient vector. Bottom left: their ratio. Bottom right: individual gradient norms, with the smaller final update marked. One seed, 712 updates; gradients captured before clipping and AdamW transformation. Undefined first cosine is omitted. No uncertainty band: this is one descriptive trajectory.

Figure caption (`optimization_context.pdf`): Token-weighted training update loss, initial/final full-validation loss, and reference learning-rate schedule for the same trajectory. Validation points use a fixed split; training batches change across updates.

Interpretation limits: raw-gradient alignment under AdamW does not reconstruct optimizer directions or parameter displacement. Neither alignment sign nor norm trend establishes stability, generalization, or causality. This run does not isolate per-sample energy from intra-batch alignment, validate the manuscript's lag-one approximation, or provide uncertainty across seeds. No manuscript text is changed by this observation.
"""
    (observations / "001-gradient-evolution.md").write_text(text, encoding="utf-8")
    (observations / "INDEX.md").write_text("# Run001 observations\n\n- [001: Full-pass raw-gradient evolution](001-gradient-evolution.md)\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in reduction.items() if k != "figure_rows"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--destination", default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()
    analyze(args.root, args.destination)
