"""Original raw/adaptive layout, added momentum panels, and matched LR comparison."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
STREAMS = {"raw": (None, "gradient_norm", "cumulative_gradient_norm", "Raw gradient", "#2462a3"),
           "momentum": ("momentum", "momentum_norm", "cumulative_momentum_norm", "Adam stored momentum m", "#28794d"),
           "adaptive": ("adaptive", "direction_norm", "cumulative_direction_norm", "Adam adaptive direction", "#d2771c")}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(folder):
    return [e for line in (folder / "events.jsonl").read_text().splitlines()
            if (e := json.loads(line)).get("event") == "update"]


def values(records, stream, key):
    subkey = STREAMS[stream][0]
    return np.array([np.nan if (v := (e[subkey] if subkey else e)[key]) is None else v for e in records])


def rolling(data):
    return np.array([np.nanmedian(data[max(1, i-24):i+1]) if i else np.nan for i in range(len(data))])


def decorate(axes):
    for ax in np.asarray(axes).flat:
        ax.set_xlabel("Optimizer update")
        ax.grid(alpha=.18)


def save(fig, destination, name):
    (destination / "figures").mkdir(parents=True, exist_ok=True)
    (destination / "scratch").mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / "figures" / (name + ".pdf"))
    fig.savefig(destination / "scratch" / (name + ".png"), dpi=150)
    plt.close(fig)


def plot_one(run, folder=None, destination=None):
    run = Path(run)
    folder = Path(folder) if folder else run / "artifacts/attempt-01/full-pass"
    destination = Path(destination) if destination else run
    config = json.loads((run / "config.json").read_text())
    all_rows = rows(folder)
    if not all_rows:
        return
    displayed = [e for e in all_rows if e["sequences"] == 1024]
    if not displayed:
        return
    x = [e["update"] for e in displayed]
    manifest = json.loads((folder / "manifest.json").read_text())
    scope = manifest["scope"]
    label = f"LR target {config['training']['peak_learning_rate']:g}, warmup only; seed 1234; {scope}; updates {x[0]}-{x[-1]}"
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for i, stream in enumerate(("raw", "adaptive")):
        _, norm, b, name, color = STREAMS[stream]
        style = "-" if i == 0 else "--"
        cosine = values(displayed, stream, "adjacent_cosine")
        axes[0, 0].plot(x, cosine, color=color, ls=style, alpha=.2, lw=.6)
        axes[0, 0].plot(x, rolling(cosine), color=color, ls=style, lw=1.6, label=name + " (25-update median)")
        axes[0, 1].plot(x, values(displayed, stream, "cumulative_ratio"), color=color, ls=style, label=name)
        axes[1, i].plot(x, values(displayed, stream, "cumulative_norm_sum"), color=color, label="A: sum of norms")
        axes[1, i].plot(x, values(displayed, stream, b), color=color, ls="--", label="B: norm of vector sum")
        axes[1, i].set(ylabel=name + " cumulative norms", ylim=(0, None))
        axes[1, i].legend(fontsize=8)
        axes[2, 0].plot(x, values(displayed, stream, norm), color=color, ls=style, label=name)
    axes[0, 0].set(ylabel="Adjacent cosine", ylim=(-1.05, 1.05))
    axes[0, 0].axhline(0, color="black", ls="--", lw=.8)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(ylabel="Cumulative ratio B/A", ylim=(0, 1.05))
    axes[0, 1].legend(fontsize=8)
    axes[2, 0].set(ylabel="Current vector norm (log scale)", yscale="log")
    axes[2, 0].legend(fontsize=8)
    axes[2, 1].plot(x, [e["adaptive_to_raw_norm_ratio"] for e in displayed], color="#8052a0")
    axes[2, 1].set(ylabel="Adaptive / raw norm (log scale)", yscale="log")
    decorate(axes)
    fig.suptitle("Raw gradient versus bias-corrected Adam adaptive direction\n" + label, fontsize=12)
    save(fig, destination, "raw_adaptive_comparison")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    cosine = values(displayed, "momentum", "adjacent_cosine")
    axes[0, 0].plot(x, cosine, color="#28794d", alpha=.3, lw=.7)
    axes[0, 0].plot(x, rolling(cosine), color="#14512e", label="Trailing 25-update median")
    axes[0, 0].axhline(0, color="black", ls="--", lw=.8)
    axes[0, 0].set(ylabel="Momentum adjacent cosine", ylim=(-1.05, 1.05))
    axes[0, 0].legend(fontsize=8)
    for key, text, style in (("cumulative_norm_sum", "A: sum of norms", "-"),
                             ("cumulative_momentum_norm", "B: norm of vector sum", "--")):
        axes[0, 1].plot(x, values(displayed, "momentum", key), color="#28794d", ls=style, label=text)
    axes[0, 1].set(ylabel="Momentum cumulative norms", ylim=(0, None))
    axes[0, 1].legend(fontsize=8)
    axes[0, 2].plot(x, values(displayed, "momentum", "cumulative_ratio"), color="#28794d")
    axes[0, 2].set(ylabel="Momentum cumulative B/A", ylim=(0, 1.05))
    axes[1, 0].plot(x, values(displayed, "momentum", "momentum_norm"), color="#28794d")
    axes[1, 0].set(ylabel="Stored momentum norm", ylim=(0, None))
    for ax, key, text in ((axes[1, 1], "momentum_to_raw_norm_ratio", "Momentum / raw norm"),
                          (axes[1, 2], "adaptive_to_momentum_norm_ratio", "Adaptive / momentum norm")):
        ax.plot(x, [np.nan if e[key] is None else e[key] for e in displayed], color="#8052a0")
        ax.set(ylabel=text + " (log scale)", yscale="log")
    decorate(axes)
    fig.suptitle("Actual Adam exp_avg m: no bias correction or preconditioning\n" + label, fontsize=12)
    save(fig, destination, "momentum_evolution")
    stats = {}
    for stream, (_, norm, b, _, _) in STREAMS.items():
        cos = values(all_rows, stream, "adjacent_cosine")
        vectors = [e[STREAMS[stream][0]] if STREAMS[stream][0] else e for e in all_rows]
        stats[stream] = {"median_cosine": float(np.nanmedian(cos)) if len(cos) > 1 else None,
                         "negative_pairs": int(np.sum(cos < 0)),
                         "median_norm": float(np.nanmedian(values(all_rows, stream, norm))),
                         "final_A": vectors[-1]["cumulative_norm_sum"], "final_B": vectors[-1][b],
                         "final_B_over_A": vectors[-1]["cumulative_ratio"]}
    data = {"source_events_sha256": sha(folder / "events.jsonl"), "script_sha256": sha(__file__),
            "scope": scope, "status": manifest["status"], "full_rows": all_rows, "statistics": stats,
            "display_filter": "Display full-sized updates only; retain partial update in statistics"}
    (destination / "figures/figure_data.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    observations = destination / "observations"
    observations.mkdir(exist_ok=True)
    (observations / "001-three-stream-geometry.md").write_text(
        f"# Observation 001: three-stream geometry\n\nSource: `{folder.as_posix()}`; script `04_plot.py`.\n\n"
        f"Coverage: {len(all_rows)} accepted updates, {sum(e['input_tokens'] for e in all_rows):,} input tokens. "
        f"Scope: {scope}; terminal status: {manifest['status']}. Main display has {len(displayed)} full batches.\n\n"
        "Caption: raw/adaptive figure retains Run002's 3x2 panels. Momentum figure shows adjacent cosine, "
        "A/B, B/A, individual norm, m/g and d/m norm ratios for actual stored FP32 exp_avg, with FP64 "
        "prefix accumulation. Faint cosine traces are unsmoothed; bold traces are trailing25-update medians. "
        "All numerical summaries include the final partial batch when present. Undefined values are gaps.\n\n"
        "Reduction and hashes: `../figures/figure_data.json`. One seed; no cross-seed uncertainty estimate. "
        "Neither alignment sign nor norm is a stability or loss-benefit claim. Each run retains all raw records.\n",
        encoding="utf-8")
    (observations / "INDEX.md").write_text("- [001: Three-stream geometry](001-three-stream-geometry.md)\n")


def plot_joint():
    cohort = json.loads((HERE / "cohort.json").read_text())
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
    colors = {"base": "#2462a3", "small": "#28794d", "large": "#ba422e"}
    reductions = {}
    for record in cohort["runs"]:
        run = ROOT / "runs" / record["run"]
        folder = run / "artifacts/attempt-01/full-pass"
        if not (folder / "events.jsonl").exists():
            continue
        selected = [e for e in rows(folder) if e["sequences"] == 1024]
        if not selected:
            continue
        reductions[record["run_id"]] = {"events_sha256": sha(folder / "events.jsonl"), "displayed_updates": len(selected)}
        x = [e["update"] for e in selected]
        for col, stream in enumerate(("raw", "momentum", "adaptive")):
            label = f"No decay, LR {record['peak_lr']:g}"
            for row, key in ((0, "adjacent_cosine"), (1, "cumulative_ratio"), (2, STREAMS[stream][1])):
                data = values(selected, stream, key)
                axes[row, col].plot(x, rolling(data) if row == 0 else data,
                                    color=colors[record["condition"]], label=label)
    for col, stream in enumerate(("raw", "momentum", "adaptive")):
        axes[0, col].set(title=STREAMS[stream][3], ylabel="Adjacent cosine (25-update median)", ylim=(-1.05, 1.05))
        axes[0, col].axhline(0, color="black", ls="--", lw=.8)
        axes[1, col].set(ylabel="Cumulative B/A", ylim=(0, 1.05))
        axes[2, col].set(ylabel="Current norm (log scale)", yscale="log")
        axes[0, col].legend(fontsize=8)
    decorate(axes)
    fig.suptitle("Pythia-14M: matched warmup-only learning rates\nIdentical initialization and data order; seed 1234; final partial batches omitted", fontsize=13)
    save(fig, HERE, "learning_rate_three_streams")
    (HERE / "figures/figure_data.json").write_text(json.dumps({"sources": reductions, "script_sha256": sha(__file__)}, indent=2) + "\n")
    (HERE / "observations").mkdir(exist_ok=True)
    (HERE / "observations/001-learning-rate-comparison.md").write_text(
        "# Observation 001: matched LR comparison\n\nSources: runs021–023 accepted full-pass events; "
        "per-run reductions and source/script hashes are in figure_data.json. Caption: columns are raw, "
        "stored momentum and adaptive direction; rows are trailing25-update median adjacent cosine, "
        "cumulative B/A and individual norm. All main curves omit the final partial batch. Each condition "
        "uses the same initial tensors and order, one seed, and all available accepted full-sized updates. "
        "Failed/partial conditions remain visibly shorter; no missing data is extended.\n\n"
        "Within-seed exploratory intervention only; LR also scales weight decay. No cross-seed error bars, "
        "stability proxy, or generalization claim. The run-specific figures retain A/B and norm ratios.\n",
        encoding="utf-8")
    (HERE / "observations/INDEX.md").write_text("- [001: LR comparison](001-learning-rate-comparison.md)\n")


if __name__ == "__main__":
    for record in json.loads((HERE / "cohort.json").read_text())["runs"]:
        plot_one(ROOT / "runs" / record["run"])
    plot_joint()
