"""Separate comparisons across learning rates from geometry within each rate."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import string
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LR_COLORS = {"021": "#2462a3", "022": "#d16b16", "023": "#ba422e"}
LR_STYLES = {"021": "-", "022": "--", "023": "-."}
STREAMS = {
    "raw": (None, "gradient_norm", "cumulative_gradient_norm", "Raw gradient g", "#2462a3", "-"),
    "momentum": ("momentum", "momentum_norm", "cumulative_momentum_norm", "Momentum m", "#28794d", "--"),
    "adaptive": ("adaptive", "direction_norm", "cumulative_direction_norm", "Adam direction d", "#d16b16", ":"),
}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def values(events, stream, key):
    sub = STREAMS[stream][0]
    result = [(e[sub] if sub else e)[key] for e in events]
    return np.array([np.nan if v is None else v for v in result])


def smooth(data, weights=None):
    result = []
    for i in range(len(data)):
        start = max(0, i - 24)
        part = data[start:i + 1]
        valid = np.isfinite(part)
        result.append(np.nan if not valid.any() else
                      np.median(part[valid]) if weights is None else
                      np.average(part[valid], weights=weights[start:i + 1][valid]))
    return np.asarray(result)


def load_run(record):
    run = ROOT / "runs" / record["run"]
    folder = run / "artifacts/attempt-01/full-pass"
    config = json.loads((run / "config.json").read_text())
    proof = json.loads((folder / "verification.json").read_text())
    receipt = json.loads((run / "prelaunch/retrieval-receipt.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    assert proof["verified"] and proof["scope"] == "full_pass" and receipt["accepted"]
    assert manifest["status"] == "completed"
    assert sha(folder / "transfer_inventory.json") == receipt["inventory_sha256"]
    inventory = json.loads((folder / "transfer_inventory.json").read_text())
    item = next(i for i in inventory["files"] if i["path"] == "events.jsonl")
    assert (folder / "events.jsonl").stat().st_size == item["bytes"]
    assert sha(folder / "events.jsonl") == item["sha256"]
    events = [e for line in (folder / "events.jsonl").read_text().splitlines()
              if (e := json.loads(line))["event"] == "update"]
    assert [e["update"] for e in events] == list(range(1, 713))
    assert sum(e["input_tokens"] for e in events) == 1491711416
    assert sum(e["supervised_tokens"] for e in events) == 1490983041
    displayed = [e for e in events if e["sequences"] == 1024]
    assert len(displayed) == 711 and events[-1]["sequences"] == 311
    lr = config["training"]["peak_learning_rate"]
    assert lr == record["peak_lr"] and all(e["learning_rate"] == lr for e in events[8:])
    old_data = HERE / "interim-021-022/comparison_data.json"
    if old_data.exists():
        old = json.loads(old_data.read_text())["runs"]
        if record["run_id"] in old:
            assert old[record["run_id"]]["sources"]["events.jsonl"] == item["sha256"]
    return {"id": record["run_id"], "lr": lr, "config": config,
            "identity": json.loads((folder / "input_identity.json").read_text()),
            "events": events, "displayed": displayed, "events_sha256": item["sha256"]}


def format_axes(axes):
    for ax in np.asarray(axes).flat:
        ax.set(xlabel="Optimizer update", xlim=(1, 711))
        ax.grid(alpha=.17)


def save(fig, out, name):
    fig.savefig(out / "figures" / f"{name}.pdf")
    fig.savefig(out / "scratch" / f"{name}.png", dpi=140)
    plt.close(fig)


def across_lrs(records, out):
    fig, axes = plt.subplots(3, 2, figsize=(13, 14))
    for record in records:
        e = record["displayed"]
        x = np.array([v["update"] for v in e])
        color, style = LR_COLORS[record["id"]], LR_STYLES[record["id"]]
        label = f"LR {record['lr']:g}"
        loss = np.array([v["loss"] for v in e])
        weights = np.array([v["supervised_tokens"] for v in e])
        axes[0, 0].plot(x, loss, color=color, alpha=.18, lw=.6)
        axes[0, 0].plot(x, smooth(loss, weights), color=color, ls=style, lw=1.8, label=label)
        for ax, stream in ((axes[0, 1], "raw"), (axes[1, 0], "momentum"), (axes[1, 1], "adaptive")):
            cosine = values(e, stream, "adjacent_cosine")
            ax.plot(x, cosine, color=color, alpha=.16, lw=.6)
            ax.plot(x, smooth(cosine), color=color, ls=style, lw=1.8, label=label)
        for ax, stream in ((axes[2, 0], "raw"), (axes[2, 1], "momentum")):
            ax.plot(x, values(e, stream, STREAMS[stream][1]), color=color, ls=style, lw=1.2, label=label)
    titles = ["(a) Training loss", "(b) Raw-gradient adjacent cosine", "(c) Momentum adjacent cosine",
              "(d) Adam-direction adjacent cosine", "(e) Raw-gradient vector norms", "(f) Momentum vector norms"]
    for i, ax in enumerate(axes.flat):
        ax.set_title(titles[i])
        ax.legend(fontsize=10, loc="best")
        ax.set_ylabel("Token-mean cross-entropy (nats)" if i == 0 else "Cosine" if i in (1, 2, 3) else "Vector norm")
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, color="black", lw=.8, ls=":")
    for ax in axes[2]:
        ax.set_ylim(0, None)
    format_axes(axes)
    fig.suptitle("Pythia-14M: comparison across learning rates\nWarmup only; matched initial weights and data order; seed 1234", fontsize=15)
    fig.text(.5, .022, "Light loss/cosine traces: individual updates. Bold loss: trailing 25-update token-weighted mean; bold cosines: trailing 25-update median.\n"
             "Norms are unsmoothed and use linear axes. All cosine panels use the same range. Display: full-sized updates 1-711.\n"
             "m is stored uncorrected Adam exp_avg; d is the bias-corrected adaptive direction. Partial update 712 remains in numerical results.",
             ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, .078, 1, .93), h_pad=2.3, w_pad=2.0)
    save(fig, out, "metrics_by_learning_rate")


def within_lrs(records, out):
    columns = len(records)
    fig, axes = plt.subplots(2, columns, figsize=(6.5 * columns, 9), squeeze=False)
    maximum = max(float(np.max(values(r["displayed"], s, "cumulative_norm_sum"))) for r in records for s in ("raw", "momentum"))
    for col, record in enumerate(records):
        e = record["displayed"]
        x = [v["update"] for v in e]
        for stream in ("raw", "momentum"):
            _, _, bkey, name, color, _ = STREAMS[stream]
            axes[0, col].plot(x, values(e, stream, "cumulative_norm_sum"), color=color, lw=1.8, label=f"{name}: A")
            axes[0, col].plot(x, values(e, stream, bkey), color=color, lw=1.8, ls="--", label=f"{name}: B")
        for stream, (_, _, _, name, color, style) in STREAMS.items():
            axes[1, col].plot(x, values(e, stream, "cumulative_ratio"), color=color, ls=style, lw=1.8, label=name)
        axes[0, col].set(title=f"({string.ascii_lowercase[col]}) LR {record['lr']:g}: cumulative norms",
                         ylabel="Cumulative norm", ylim=(0, maximum * 1.06))
        axes[1, col].set(title=f"({string.ascii_lowercase[columns + col]}) LR {record['lr']:g}: cumulative B/A",
                         ylabel="B/A", ylim=(0, 1.05))
        axes[0, col].legend(fontsize=9, loc="upper left", ncol=2)
        axes[1, col].legend(fontsize=10, loc="lower left")
    format_axes(axes)
    fig.suptitle("Pythia-14M: geometry within each learning rate\nWarmup only; matched initial weights and data order; seed 1234", fontsize=15)
    fig.text(.5, .028, "A = sum of vector norms; B = norm of vector sum. Colors identify raw gradient g, stored momentum m and adaptive direction d.\n"
             "Cumulative norms: solid A, dashed B. Both rows share scales across learning rates. Display: full-sized updates 1-711.\n"
             "All curves are unsmoothed. Partial update 712 remains in numerical results. One matched seed; no cross-seed uncertainty estimated.",
             ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, .115, 1, .91), h_pad=2.3, w_pad=2.0)
    save(fig, out, "geometry_within_learning_rate")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=["021", "022"], choices=list(LR_COLORS))
    parser.add_argument("--output", default="interim-021-022", help="Output directory relative to Analysis008")
    args = parser.parse_args()
    assert len(args.runs) == len(set(args.runs)) and 2 <= len(args.runs) <= 3
    out = (HERE / args.output).resolve()
    assert out.is_relative_to(HERE)
    for sub in ("figures", "scratch", "observations"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    cohort = {r["run_id"]: r for r in json.loads((HERE / "cohort.json").read_text())["runs"]}
    records = [load_run(cohort[rid]) for rid in args.runs]
    first = records[0]
    for r in records[1:]:
        assert r["identity"] == first["identity"]
        for key in ("model", "data", "seeds", "measurements", "validation", "checkpoints"):
            assert r["config"][key] == first["config"][key], key
        training = dict(r["config"]["training"])
        matched_training = dict(first["config"]["training"])
        training.pop("peak_learning_rate")
        matched_training.pop("peak_learning_rate")
        assert training == matched_training
    across_lrs(records, out)
    within_lrs(records, out)
    data = {"created_at": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(__file__),
            "run_order": args.runs, "main_layout": [3, 2], "within_lr_layout": [2, len(records)],
            "displayed_updates": [1, 711], "source_updates": 712, "partial_update_retained": True,
            "smoothing": "Loss: trailing 25-update token-weighted mean; cosine: trailing 25-update median; norms/cumulative metrics: none",
            "runs": {r["id"]: {"lr": r["lr"], "source_events_sha256": r["events_sha256"], "full_events": r["events"]} for r in records}}
    (out / "simplified_figure_data.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (out / "observations/002-simplified-layouts.md").write_text(
        "# Observation 002: simplified comparison layouts\n\n"
        "Question: distinguish comparison across learning rates from comparison between vector streams within one learning rate. "
        "This layout revision preserves the results and coverage in [the initial two-run observation](../../interim-021-022/observations/001-two-run-comparison.md).\n\n"
        "`metrics_by_learning_rate.pdf` is 3x2 in row-major order: training loss, raw cosine, momentum cosine, adaptive Adam cosine, "
        "raw individual norms and momentum individual norms. Every panel overlays all selected LRs; color and line style identify LR. "
        "Light loss/cosine traces are unsmoothed; bold loss is a trailing 25-update token-weighted mean; bold cosines are trailing 25-update medians. "
        "Norms are unsmoothed on linear axes; all three cosine panels share [-1.05, 1.05].\n\n"
        "`geometry_within_learning_rate.pdf` has one column per LR. Top panels compare raw/momentum A and B (solid A, dashed B); "
        "bottom panels compare raw/momentum/adaptive B/A. Color identifies stream. Both rows share scales across LR columns; no smoothing is applied.\n\n"
        "Sources: completed verified Run021/022 event logs, plus Run023 when explicitly selected after verified retrieval. "
        "Display: full-sized updates 1-711. All 712 updates including the partial final batch remain in `../simplified_figure_data.json`, "
        "with source/generator hashes. Undefined first cosines are gaps. One matched seed, identical initial tensors/order, LR-scaled AdamW decay; "
        "no new causal, stability or convergence claim. Previous numerical findings are unchanged.\n\n"
        "Generating script: `../../08_plot_simplified.py`. Future extension adds the third LR to every metric panel and adds a third geometry column, "
        "using `--runs 021 022 023 --output complete-three-lr`.\n",
        encoding="utf-8")
    index = out / "observations/INDEX.md"
    line = "- [002: Simplified comparison layouts](002-simplified-layouts.md)\n"
    previous = index.read_text(encoding="utf-8") if index.exists() else ""
    if line not in previous:
        index.write_text(previous + line, encoding="utf-8")
    print(json.dumps({"runs": args.runs, "output": str(out), "main_panels": 6, "geometry_panels": 2 * len(records)}))


if __name__ == "__main__":
    main()
