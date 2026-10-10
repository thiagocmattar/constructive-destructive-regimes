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
    return {"updates": len(rows), "defined_cosines": len(cosine),
            "median_cosine": float(np.median(cosine)), "mean_cosine": float(np.mean(cosine)),
            "min_cosine": float(np.min(cosine)), "max_cosine": float(np.max(cosine)),
            "negative_pairs": int(np.sum(cosine < 0)),
            "median_vector_norm": float(np.median([e[norm_key] for e in vectors])),
            "end_norm_sum": vectors[-1]["cumulative_norm_sum"],
            "end_vector_sum_norm": vectors[-1][sum_key],
            "end_ratio": vectors[-1]["cumulative_ratio"]}


def rolling(cosine):
    return np.array([np.nanmedian(cosine[max(1, i - 24):i + 1]) if i else np.nan
                     for i in range(len(cosine))])


def analyze(root, destination):
    root, destination = Path(root), Path(destination)
    if not json.loads((root / "verification.json").read_text())["verified"]:
        raise ValueError("Require accepted scientific verification.")
    summary = json.loads((root / "summary.json").read_text())
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
        fig.savefig(scratch / (name + ".png"), dpi=180)
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
        fig.suptitle(f"Pythia-14M / MiniPile / AdamW: {label}\n"
                     "Seed 1234; updates 1–711; final partial batch (step 712) omitted", fontsize=12)
        save(fig, name)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for adaptive, label, color in [(False, "Raw gradient", colors["raw"]),
                                    (True, "Adam adaptive direction", colors["adaptive"])]:
        cosine = values("adjacent_cosine", adaptive)
        axes[0, 0].plot(x, cosine, color=color, alpha=.2, lw=.6)
        axes[0, 0].plot(x, rolling(cosine), color=color, lw=1.6, label=label + " (25-update median)")
        axes[0, 1].plot(x, values("cumulative_ratio", adaptive), color=color, lw=1.5, label=label)
        axes[2, 0].plot(x, values("direction_norm" if adaptive else "gradient_norm", adaptive),
                        color=color, lw=1, label=label)
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
    fig.suptitle("Raw gradient versus bias-corrected Adam adaptive direction\n"
                 "No LR or weight decay in direction; seed 1234; steps 1–711", fontsize=12)
    save(fig, "raw_adaptive_comparison")

    reduction = {"source_events_sha256": identity(root / "events.jsonl"),
                 "source_summary_sha256": identity(root / "summary.json"),
                 "script_sha256": identity(Path(__file__)), "seed": 1234,
                 "full_pass": {"raw": stats(rows), "adaptive": stats(rows, True)},
                 "displayed_full_batches": {"raw": stats(displayed), "adaptive": stats(displayed, True)},
                 "display_filter": "Exclude final partial update 712 from all main plots only",
                 "displayed_updates": [e["update"] for e in displayed],
                 "median_adaptive_to_raw_norm_ratio": float(np.median([e["adaptive_to_raw_norm_ratio"] for e in rows])),
                 "phases": [{"first_update": int(indices[0]+1), "last_update": int(indices[-1]+1),
                             "raw": stats([rows[i] for i in indices]),
                             "adaptive": stats([rows[i] for i in indices], True)}
                            for indices in np.array_split(np.arange(len(rows)), 4)],
                 "figure_rows": rows}
    (figures / "figure_data.json").write_text(json.dumps(reduction, indent=2, allow_nan=False) + "\n")
    raw, adaptive = reduction["full_pass"]["raw"], reduction["full_pass"]["adaptive"]
    hypothesis_consistent = adaptive["median_cosine"] > raw["median_cosine"] and adaptive["end_ratio"] > raw["end_ratio"]
    lines = "\n".join(f"| {p['first_update']}–{p['last_update']} | {p['raw']['median_cosine']:.5f} | "
                      f"{p['adaptive']['median_cosine']:.5f} | {p['raw']['end_ratio']:.5f} | "
                      f"{p['adaptive']['end_ratio']:.5f} |" for p in reduction["phases"])
    source_link = Path(os.path.relpath(root, observations)).as_posix()
    observation = f"""# Observation 001: raw gradients versus Adam adaptive directions

Question: does adaptive Adam preconditioning produce higher adjacent alignment and
a cumulative resultant closer to the cumulative sum of vector norms? How do raw
and preconditioned norms compare? H001 is the user's descriptive expectation.

Sources: `{source_link}/events.jsonl`, `summary.json`, and
`verification.json`. Generated by `../04_analyze.py`. Exact source/script hashes,
all 712 records, full-pass statistics, displayed-prefix statistics, and phase
reductions are in `../figures/figure_data.json`.

Coverage: 712 accepted updates, {summary['completed_sequences']:,} packed sequences,
{summary['completed_input_tokens']:,} input tokens, {summary['completed_supervised_tokens']:,}
supervised targets. Every token is consumed once, including the short tail.
Seed 1234, one paired trajectory. All main figures display steps 1–711 with 1,024
sequences per step; the final 311-sequence update is omitted from display only.
The numerical table below includes all 712 updates. Cumulative values in figures
are the original prefixes, with no reweighting or filtering inside each prefix.

Raw g is measured after accumulation/unscaling, before clipping. Adaptive d is
the FP64 evaluation of bias-corrected m / (sqrt(bias-corrected v) + epsilon),
from actual updated FP32 AdamW moments. Those moments use clipped gradients.
No learning rate or weight decay enters d. Whole model, unique trainable parameters.
Both use FP64 cumulative vectors/reductions; previous g is FP32 and previous d FP64.
First cosine is undefined, including at the zero-LR first optimizer step: moments
still advance and d is included. Ratios/cosines with zero denominators are null.

| Full-pass measure | Raw gradient | Adam adaptive direction |
| --- | ---: | ---: |
| Median adjacent cosine | {raw['median_cosine']:.6f} | {adaptive['median_cosine']:.6f} |
| Mean adjacent cosine | {raw['mean_cosine']:.6f} | {adaptive['mean_cosine']:.6f} |
| Negative pairs / defined pairs | {raw['negative_pairs']}/{raw['defined_cosines']} | {adaptive['negative_pairs']}/{adaptive['defined_cosines']} |
| Median individual vector norm | {raw['median_vector_norm']:.6f} | {adaptive['median_vector_norm']:.6f} |
| Final A: cumulative sum of norms | {raw['end_norm_sum']:.6f} | {adaptive['end_norm_sum']:.6f} |
| Final B: norm of cumulative vector sum | {raw['end_vector_sum_norm']:.6f} | {adaptive['end_vector_sum_norm']:.6f} |
| Final B/A | {raw['end_ratio']:.6f} | {adaptive['end_ratio']:.6f} |

Median adaptive/raw individual-norm ratio: {reduction['median_adaptive_to_raw_norm_ratio']:.6f}.
H001's two directional comparisons (higher median cosine and higher final B/A)
are {'both observed' if hypothesis_consistent else 'not both observed'} in this one trajectory.
The A≈B expectation must be judged from the measured B/A, not adjacent signs alone.

| Updates | Raw median cosine | Adaptive median cosine | Raw end B/A | Adaptive end B/A |
| --- | ---: | ---: | ---: | ---: |
{lines}

Full fixed-split validation loss: {summary['initial_validation']['loss']:.6f} →
{summary['final_validation']['loss']:.6f}; all 500 documents including the short
tail, weighted by supervised tokens. Training batches change at each update.

Captions: `adaptive_evolution.pdf` and `raw_gradient_evolution.pdf` show adjacent
cosine (each point and trailing 25-update median), A and B, B/A, and individual
vector norm for the indicated vector. `raw_adaptive_comparison.pdf` compares
adjacent cosines and B/A on common axes; raw/adaptive A and B on separate norm
scales; individual norms and their paired ratio on log scales. All plots show
updates 1–711, seed 1234, with final partial update 712 omitted. Undefined first
cosine is a gap. Smoothing is descriptive; no cross-seed uncertainty band.

Limits: this is adaptive-direction geometry, not actual parameter-displacement
geometry. LR scaling, decay and subtraction rounding are excluded. Norms have
different scaling after preconditioning; a large ratio does not itself mean
greater physical step size or optimization benefit. Adjacent alignment does not
determine the all-lag cumulative ratio. No stability, causal improvement,
per-sample decomposition, generalization across seeds, or full SGD parameter-norm
identity is established. Original manuscript sources remain unchanged.
"""
    (observations / "001-adaptive-comparison.md").write_text(observation, encoding="utf-8")
    (observations / "INDEX.md").write_text(
        "# Run002 observations\n\n- [001: Raw gradients versus Adam adaptive directions](001-adaptive-comparison.md)\n")
    print(json.dumps({k:v for k,v in reduction.items() if k not in ("figure_rows", "displayed_updates", "phases")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--destination", default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()
    analyze(args.root, args.destination)
