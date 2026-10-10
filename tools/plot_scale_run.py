"""Plot and describe paired raw-gradient and adaptive-direction geometry."""

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


def stats(rows, adaptive=False):
    vectors = [e["adaptive"] if adaptive else e for e in rows]
    cosine = np.array([e["adjacent_cosine"] for e in vectors if e["adjacent_cosine"] is not None])
    norm_key = "direction_norm" if adaptive else "gradient_norm"
    sum_key = "cumulative_direction_norm" if adaptive else "cumulative_gradient_norm"
    minimum = min(zip(rows, vectors), key=lambda pair: pair[1]["cumulative_ratio"])
    return {"updates": len(rows), "defined_cosines": len(cosine),
            "median_cosine": float(np.median(cosine)), "mean_cosine": float(np.mean(cosine)),
            "min_cosine": float(np.min(cosine)), "max_cosine": float(np.max(cosine)),
            "negative_pairs": int(np.sum(cosine < 0)),
            "median_vector_norm": float(np.median([e[norm_key] for e in vectors])),
            "first_vector_norm": vectors[0][norm_key], "last_vector_norm": vectors[-1][norm_key],
            "minimum_cumulative_ratio": minimum[1]["cumulative_ratio"],
            "minimum_ratio_update": minimum[0]["update"],
            "end_norm_sum": vectors[-1]["cumulative_norm_sum"],
            "end_vector_sum_norm": vectors[-1][sum_key],
            "end_ratio": vectors[-1]["cumulative_ratio"]}


def paired_stats(rows):
    differences = [e["adaptive"]["adjacent_cosine"] - e["adjacent_cosine"]
                   for e in rows if e["adjacent_cosine"] is not None]
    return {"median_cosine_difference": float(np.median(differences)),
            "mean_cosine_difference": float(np.mean(differences)),
            "median_adaptive_to_raw_norm_ratio": float(np.median(
                [e["adaptive_to_raw_norm_ratio"] for e in rows])),
            "end_ratio_difference": rows[-1]["adaptive"]["cumulative_ratio"] - rows[-1]["cumulative_ratio"]}


def rolling(cosine):
    return np.array([np.nanmedian(cosine[max(1, i - 24):i + 1]) if i else np.nan
                     for i in range(len(cosine))])


def analyze(root, destination):
    root, destination = Path(root), Path(destination)
    receipt = destination / "prelaunch/attempt1-retrieval.json"
    if not receipt.exists() or not json.loads(receipt.read_text())["verified"]:
        raise ValueError("Require completed strict local artifact acceptance before plotting.")
    if not json.loads((root / "verification.json").read_text())["verified"]:
        raise ValueError("Require accepted scientific verification.")
    summary = json.loads((root / "summary.json").read_text())
    config = json.loads((root / "resolved_config.json").read_text())
    model_label = config["model"]["architecture"].split("/")[-1].replace("-deduped", "")
    rows = [e for line in (root / "events.jsonl").read_text().splitlines()
            if (e := json.loads(line))["event"] == "update"]
    if len(rows) != 712 or rows[-1]["sequences"] != 311:
        raise ValueError("Unexpected full-pass coverage.")
    displayed = rows[:-1]
    if any(e["sequences"] != 1024 for e in displayed):
        raise ValueError("Unexpected displayed batch size.")
    figures, scratch, observations = [destination / n for n in ("figures", "scratch", "observations")]
    for p in (figures, scratch, observations):
        p.mkdir(parents=True, exist_ok=True)
    x = np.array([e["update"] for e in displayed])
    colors = {"raw": "#2462a3", "adaptive": "#d2771c"}

    def values(key, adaptive=False):
        vectors = [e["adaptive"] if adaptive else e for e in displayed]
        return np.array([np.nan if e[key] is None else e[key] for e in vectors], dtype=float)

    def save(fig, name):
        fig.savefig(figures / (name + ".pdf"))
        fig.savefig(scratch / (name + ".png"), dpi=180, bbox_inches="tight", pad_inches=.05)
        plt.close(fig)

    def decorate(axes):
        for ax in axes.flat:
            ax.set_xlabel("Optimizer update")
            ax.grid(alpha=.18)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    for adaptive, name, label in [(False, "raw_gradient_evolution", "Raw gradient"),
                                  (True, "adaptive_evolution", "Adam adaptive direction")]:
        cosine = values("adjacent_cosine", adaptive)
        norm = "direction_norm" if adaptive else "gradient_norm"
        cum = "cumulative_direction_norm" if adaptive else "cumulative_gradient_norm"
        symbol = "d" if adaptive else "g"
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        axes[0, 0].plot(x, cosine, color="#2462a3", alpha=.5, lw=.7, label="Every displayed update")
        axes[0, 0].plot(x, rolling(cosine), color="#133c68", lw=1.6, label="Trailing 25-update median")
        axes[0, 0].axhline(0, color="black", lw=.8, ls="--")
        axes[0, 0].set(ylabel="Adjacent cosine", ylim=(-1.05, 1.05))
        axes[0, 0].legend(fontsize=8)
        axes[0, 1].plot(x, values("cumulative_norm_sum", adaptive), color="#d2771c", lw=1.5,
                        label=rf"$A_t=\sum_{{u\leq t}}\|{symbol}_u\|_2$")
        axes[0, 1].plot(x, values(cum, adaptive), color="#2462a3", lw=1.5, ls="--",
                        label=rf"$B_t=\|\sum_{{u\leq t}}{symbol}_u\|_2$")
        axes[0, 1].set(ylabel="Cumulative vector norms", ylim=(0, None))
        axes[0, 1].legend(fontsize=9)
        axes[1, 0].plot(x, values("cumulative_ratio", adaptive), color="#28794d", lw=1.5)
        axes[1, 0].set(ylabel=r"Cumulative ratio $R_t=B_t/A_t$", ylim=(0, 1.05))
        axes[1, 1].plot(x, values(norm, adaptive), color="#8052a0", lw=.9)
        axes[1, 1].set(ylabel=rf"Current vector norm $\|{symbol}_t\|_2$", ylim=(0, None))
        decorate(axes)
        fig.suptitle(f"{model_label} / MiniPile / AdamW: {label}\n"
                     "Seed 1234; updates 1–711; final partial batch (step 712) omitted", fontsize=12)
        save(fig, name)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for adaptive, label, color in [(False, "Raw gradient", colors["raw"]),
                                    (True, "Adam adaptive direction", colors["adaptive"])]:
        cosine = values("adjacent_cosine", adaptive)
        style = "--" if adaptive else "-"
        axes[0, 0].plot(x, cosine, color=color, ls=style, alpha=.2, lw=.6)
        axes[0, 0].plot(x, rolling(cosine), color=color, ls=style, lw=1.6, label=label + " (25-update median)")
        axes[0, 1].plot(x, values("cumulative_ratio", adaptive), color=color, ls=style, lw=1.5, label=label)
        axes[2, 0].plot(x, values("direction_norm" if adaptive else "gradient_norm", adaptive),
                        color=color, ls=style, lw=1, label=label)
        ax = axes[1, int(adaptive)]
        ax.plot(x, values("cumulative_norm_sum", adaptive), color=color, label="A: sum of norms", lw=1.5)
        ax.plot(x, values("cumulative_direction_norm" if adaptive else "cumulative_gradient_norm", adaptive),
                color=color, label="B: norm of vector sum", ls="--", lw=1.5)
        ax.set(ylabel=label + " cumulative norms", ylim=(0, None))
        ax.legend(fontsize=8)
    axes[0, 0].set(ylabel="Adjacent cosine", ylim=(-1.05, 1.05))
    axes[0, 0].axhline(0, color="black", ls="--", lw=.8)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(ylabel="Cumulative ratio B/A", ylim=(0, 1.05))
    axes[0, 1].legend(fontsize=8)
    axes[2, 0].set(ylabel="Current vector norm (log scale)", yscale="log")
    axes[2, 0].legend(fontsize=8)
    axes[2, 1].plot(x, [e["adaptive_to_raw_norm_ratio"] for e in displayed], color="#8052a0", lw=1)
    axes[2, 1].set(ylabel="Adaptive / raw norm (log scale)", yscale="log")
    decorate(axes)
    fig.suptitle(f"{model_label}: raw gradient versus bias-corrected Adam adaptive direction\n"
                 "No LR or weight decay in direction; seed 1234; steps 1–711", fontsize=12)
    save(fig, "raw_adaptive_comparison")


    reduction = {"source_events_sha256": identity(root / "events.jsonl"),
                 "source_summary_sha256": identity(root / "summary.json"),
                 "script_sha256": identity(Path(__file__)),
                 "entrypoint_sha256": identity(destination / "04_analyze.py"),
                 "full_pass": {"raw": stats(rows), "adaptive": stats(rows, True)},
                 "displayed_full_batches": {"raw": stats(displayed), "adaptive": stats(displayed, True)},
                 "paired_full_pass": paired_stats(rows),
                 "quarters": [{"first_update": int(a[0]+1), "last_update": int(a[-1]+1),
                      "raw": stats([rows[i] for i in a]), "adaptive": stats([rows[i] for i in a], True)}
                      for a in np.array_split(np.arange(len(rows)), 4)],
                 "display_filter": "Final partial update omitted from main plots; retained in full data/statistics",
                 "figure_rows": rows}
    (figures / "figure_data.json").write_text(json.dumps(reduction, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    raw, adaptive = reduction["full_pass"]["raw"], reduction["full_pass"]["adaptive"]
    phase_lines = "\n".join(f"| {q['first_update']}–{q['last_update']} | {q['raw']['median_cosine']:.6f} | {q['adaptive']['median_cosine']:.6f} | {q['raw']['end_ratio']:.6f} | {q['adaptive']['end_ratio']:.6f} |" for q in reduction["quarters"])
    observation = f"""# Observation 001: {model_label} raw and adaptive geometry

Question: measure the same raw-gradient and adaptive Adam-direction diagnostics as
Run002, to support the user-authorized scale comparison (H002).

Sources: attempt 01/full-pass events.jsonl, summary.json, input_identity.json,
environment.json, resolved_config.json, verification.json, and recovery checkpoints.
Strict byte/SHA acceptance and independent local scientific verification are in
../prelaunch/attempt1-retrieval.json. Source/script hashes, all 712 measurement rows,
displayed-prefix statistics, and fixed quarters are in ../figures/figure_data.json.

Generating command: python runs/{destination.name}/04_analyze.py
Entry point calls ../../../tools/plot_scale_run.py. Publication PDFs: raw_gradient_evolution.pdf,
adaptive_evolution.pdf, raw_adaptive_comparison.pdf. Scratch PNGs are ignored.

Coverage: all 712 accepted updates, {summary['completed_sequences']:,} sequences,
{summary['completed_input_tokens']:,} input tokens and {summary['completed_supervised_tokens']:,}
supervised targets, each once. Main plots show updates1–711, all full batches1024;
the final 311-sequence update is omitted from display only. All table/statistics
below retain it. Cosine begins at 2; undefined first value is a gap.

Raw g is unscaled after token-weighted accumulation, before global clipping.
Adaptive d is evaluated in FP64 from actual updated FP32 moments of clipped g:
bias-corrected m / (sqrt(bias-corrected v) + epsilon). It excludes LR and decay.
All unique trainable parameters, identical ordering across streams, FP64 cumulative
vectors/reductions; previous g FP32, previous d FP64. A=sum norms, B=norm sum,
R=B/A. Zero-denominator values are null with reason. The zero-LR first step still
advances moments, and its direction is included.

| Full pass | Raw g | Adaptive d |
| --- | ---: | ---: |
| Median adjacent cosine | {raw['median_cosine']:.6f} | {adaptive['median_cosine']:.6f} |
| Negative / defined cosine pairs | {raw['negative_pairs']}/{raw['defined_cosines']} | {adaptive['negative_pairs']}/{adaptive['defined_cosines']} |
| Final A | {raw['end_norm_sum']:.6f} | {adaptive['end_norm_sum']:.6f} |
| Final B | {raw['end_vector_sum_norm']:.6f} | {adaptive['end_vector_sum_norm']:.6f} |
| Final B/A | {raw['end_ratio']:.6f} | {adaptive['end_ratio']:.6f} |
| Median vector norm | {raw['median_vector_norm']:.6f} | {adaptive['median_vector_norm']:.6f} |

Median paired adaptive/raw norm ratio:
{reduction['paired_full_pass']['median_adaptive_to_raw_norm_ratio']:.6f}.

| Updates | Raw median cosine | Adaptive median cosine | Raw end R | Adaptive end R |
| --- | ---: | ---: | ---: | ---: |
{phase_lines}

Validation loss: {summary['initial_validation']['loss']:.6f} →
{summary['final_validation']['loss']:.6f}, all 500 fixed validation documents, including
the short tail, weighted by supervised targets. One random initialization seed 1234.

Captions: each single-vector 2×2 figure shows adjacent cosine (faint unsmoothed
values plus trailing 25-update median), A/B, R, and individual L2 norm. The paired
3×2 figure shows cosines and R on common axes, raw/adaptive A/B separately, norms
and paired norm ratio on log scales. Updates1–711 only, first cosine undefined.
No uncertainty band; trajectory quarters are fixed, not inferred regimes.

Limits: adaptive directions are not actual parameter displacements. No stability,
causal benefit, per-sample decomposition, or generalization across seeds follows
from these norms/alignments. B/A<1 can result from non-collinearity even when all
adjacent cosines are positive. Absolute scales differ after preconditioning.
Cross-size conclusions are owned by Analysis001; no scaling law is established
by this run alone. Frozen training inputs and original manuscript assets are preserved.
"""
    (observations / "001-gradient-geometry.md").write_text(observation, encoding="utf-8")
    (observations / "INDEX.md").write_text("# Observations\n\n- [001: Raw and adaptive geometry](001-gradient-geometry.md)\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in reduction.items() if k not in ["figure_rows","quarters"]}, indent=2))


def run_analysis(run):
    run = Path(run)
    analyze(run / "artifacts/attempt-01/full-pass", run)
