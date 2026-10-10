"""Frozen two-run comparison and exact sparse checkpoint temporal profiles."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "interim-021-022"
RUNS = [("021-2026-10-08-pythia14m-no-decay-base", "#2462a3", 1e-3),
        ("022-2026-10-08-pythia14m-no-decay-small", "#d16b16", 1e-4)]
STREAMS = {
    "raw": (None, "gradient_norm", "cumulative_gradient_norm", "Raw gradient", "geometry"),
    "momentum": ("momentum", "momentum_norm", "cumulative_momentum_norm", "Adam momentum m", "momentum_geometry"),
    "adaptive": ("adaptive", "direction_norm", "cumulative_direction_norm", "Adam direction d", "adaptive_geometry"),
}
STYLES = {"raw": "-", "momentum": "--", "adaptive": ":"}
SAMPLES = list(range(100, 701, 100))
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
torch.set_num_threads(4)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def arr(events, stream, key):
    sub = STREAMS[stream][0]
    return np.array([np.nan if (v := (e[sub] if sub else e)[key]) is None else v for e in events])


def rolling(values, weights=None):
    result = []
    for i in range(len(values)):
        start = max(0, i - 24)
        window = values[start:i + 1]
        valid = np.isfinite(window)
        result.append(np.nan if not valid.any() else
                      np.average(window[valid], weights=weights[start:i + 1][valid])
                      if weights is not None else np.median(window[valid]))
    return np.asarray(result)


def reduce(run, color, lr):
    folder = run / "artifacts/attempt-01/full-pass"
    config = json.loads((run / "config.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    identity = json.loads((folder / "input_identity.json").read_text())
    proof = json.loads((folder / "verification.json").read_text())
    receipt = json.loads((run / "prelaunch/retrieval-receipt.json").read_text())
    inventory = json.loads((folder / "transfer_inventory.json").read_text())
    assert proof["verified"] and proof["scope"] == "full_pass" and receipt["accepted"]
    assert digest(folder / "transfer_inventory.json") == receipt["inventory_sha256"]
    assert config["training"]["peak_learning_rate"] == lr
    source = {}
    by_path = {item["path"]: item for item in inventory["files"]}
    for name in ("events.jsonl", "summary.json", "input_identity.json", "verification.json", "resolved_config.json"):
        item = by_path[name]
        assert (folder / name).stat().st_size == item["bytes"]
        assert digest(folder / name) == item["sha256"]
        source[name] = item["sha256"]
    source["config.json"] = digest(run / "config.json")
    events = [e for line in (folder / "events.jsonl").read_text().splitlines()
              if (e := json.loads(line))["event"] == "update"]
    assert [e["update"] for e in events] == list(range(1, 713))
    assert sum(e["input_tokens"] for e in events) == 1491711416
    assert sum(e["supervised_tokens"] for e in events) == 1490983041
    assert summary["all_training_tokens_covered"] and events[-1]["sequences"] == 311
    assert summary["initial_validation"]["supervised_tokens"] == 693329
    assert summary["final_validation"]["supervised_tokens"] == 693329
    assert summary["final_validation"]["short_tail_included"]
    assert all(e["learning_rate"] == lr for e in events if e["update"] >= 9)
    displayed = [e for e in events if e["sequences"] == 1024]
    assert len(displayed) == 711
    stats = {}
    legacy_stats = json.loads((run / "figures/figure_data.json").read_text())["statistics"]
    for stream, (sub, norm, bkey, _, _) in STREAMS.items():
        cos = arr(events, stream, "adjacent_cosine")
        valid = cos[np.isfinite(cos)]
        final = events[-1][sub] if sub else events[-1]
        tail = displayed[-100:]
        stats[stream] = {"median_cosine": float(np.median(valid)), "defined_pairs": len(valid),
                         "negative_pairs": int(np.sum(valid < 0)), "negative_fraction": float(np.mean(valid < 0)),
                         "median_norm": float(np.median(arr(events, stream, norm))),
                         "final_A": final["cumulative_norm_sum"], "final_B": final[bkey],
                         "final_B_over_A": final["cumulative_ratio"], "final_norm": final[norm],
                         "last100_full_median_cosine": float(np.nanmedian(arr(tail, stream, "adjacent_cosine")))}
        assert len(valid) == 711 and np.isnan(cos[0])
        assert math.isclose(stats[stream]["final_B"] / stats[stream]["final_A"], stats[stream]["final_B_over_A"], rel_tol=1e-12)
        for key in ("median_cosine", "median_norm", "final_A", "final_B", "final_B_over_A"):
            assert math.isclose(stats[stream][key], legacy_stats[stream][key], rel_tol=1e-12, abs_tol=1e-12)
    tail = displayed[-100:]
    performance = {"initial_validation_loss": summary["initial_validation"]["loss"],
                   "final_validation_loss": summary["final_validation"]["loss"],
                   "final_validation_perplexity": math.exp(summary["final_validation"]["loss"]),
                   "last100_full_training_loss": float(np.average([e["loss"] for e in tail], weights=[e["supervised_tokens"] for e in tail])),
                   "all_update_token_weighted_training_loss": float(np.average([e["loss"] for e in events], weights=[e["supervised_tokens"] for e in events])),
                   "overflow_retries": sum(e["overflow_retries"] for e in events)}
    return {"id": config["run_id"], "run": run, "folder": folder, "lr": lr, "color": color,
            "config": config, "identity": identity, "events": events, "displayed": displayed,
            "stats": stats, "performance": performance, "source": source, "inventory": by_path}


def temporal(record):
    print(f"Computing exact checkpoint cosine matrices: Run{record['id']}", flush=True)
    states = []
    for step in SAMPLES:
        relative = f"checkpoints/recovery-{step:04d}/recovery.pt"
        path = record["folder"] / relative
        item = record["inventory"][relative]
        assert path.stat().st_size == item["bytes"]  # hashes verified by the completed transfer receipt
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        assert state["order_sha256"] == record["identity"]["order_sha256"]
        assert state["config_sha256"] == record["source"]["config.json"]
        record["source"][relative] = item["sha256"]
        states.append(state)
    result = {}
    for stream, (_, _, _, _, state_key) in STREAMS.items():
        saved = [s[state_key] for s in states]
        names = saved[0]["names"]
        assert all(s["names"] == names and s["updates"] == t for s, t in zip(saved, SAMPLES))
        assert sum(t.numel() for t in saved[0]["previous"]) == 14067712
        gram = torch.zeros((7, 7), dtype=torch.float64)
        for index in range(len(names)):
            vectors = [s["previous"][index].reshape(-1) for s in saved]
            assert all(t.shape == vectors[0].shape for t in vectors)
            for start in range(0, vectors[0].numel(), 262144):
                block = torch.stack([v[start:start + 262144].double() for v in vectors])
                gram += block @ block.T
        norms = gram.diag().sqrt()
        expected = torch.tensor([s["previous_norm"] for s in saved], dtype=torch.float64)
        torch.testing.assert_close(norms, expected, rtol=1e-10, atol=1e-12)
        assert torch.isfinite(gram).all() and (norms > 0).all()
        cos = (gram / (norms[:, None] * norms[None, :])).numpy()
        assert np.max(np.abs(cos - cos.T)) < 1e-12
        assert np.allclose(np.diag(cos), 1, rtol=0, atol=1e-12)
        pairs = [{"start_update": SAMPLES[i], "end_update": SAMPLES[j],
                  "lag": SAMPLES[j] - SAMPLES[i], "cosine": float(cos[i, j])}
                 for i in range(7) for j in range(i + 1, 7)]
        profile = [{"lag": lag, "pairs": len(values := [p["cosine"] for p in pairs if p["lag"] == lag]),
                    "mean_pair_cosine": float(np.mean(values))} for lag in range(100, 601, 100)]
        assert [p["pairs"] for p in profile] == [6, 5, 4, 3, 2, 1]
        result[stream] = {"sample_updates": SAMPLES, "cosine_matrix": cos.tolist(), "pairs": pairs, "profile": profile}
    return result


def save_figure(fig, name):
    fig.savefig(OUT / "figures" / f"{name}.pdf")
    fig.savefig(OUT / "scratch" / f"{name}.png", dpi=140)
    plt.close(fig)


def summary_figure(records):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    a = axes.flat
    for record in records:
        e = record["displayed"]
        x = [v["update"] for v in e]
        color, label = record["color"], f"LR {record['lr']:g}"
        loss = np.array([v["loss"] for v in e])
        weights = np.array([v["supervised_tokens"] for v in e])
        a[0].plot(x, loss, color=color, alpha=.18, lw=.6)
        a[0].plot(x, rolling(loss, weights), color=color, lw=1.8, label=label)
        raw_cos = arr(e, "raw", "adjacent_cosine")
        a[1].plot(x, raw_cos, color=color, alpha=.16, lw=.5)
        a[1].plot(x, rolling(raw_cos), color=color, lw=1.8, label=label)
        for stream in ("momentum", "adaptive"):
            cos = arr(e, stream, "adjacent_cosine")
            a[2].plot(x, cos, color=color, ls=STYLES[stream], alpha=.16, lw=.55)
            a[2].plot(x, rolling(cos), color=color, ls=STYLES[stream], lw=1.8,
                      label=f"{'m' if stream == 'momentum' else 'd'}, {label}")
        for stream, symbol, astyle, bstyle in (("raw", "g", "-", "--"), ("momentum", "m", "-.", ":")):
            a[3].plot(x, arr(e, stream, "cumulative_norm_sum"), color=color, ls=astyle, lw=1.7, label=f"{symbol}: A, {label}")
            a[3].plot(x, arr(e, stream, STREAMS[stream][2]), color=color, ls=bstyle, lw=1.7, label=f"{symbol}: B, {label}")
            a[5].plot(x, arr(e, stream, STREAMS[stream][1]), color=color, ls=STYLES[stream], lw=1.0,
                      label=f"{symbol}, {label}")
        for stream, symbol in (("raw", "g"), ("momentum", "m"), ("adaptive", "d")):
            a[4].plot(x, arr(e, stream, "cumulative_ratio"), color=color, ls=STYLES[stream], lw=1.8,
                      label=f"{symbol}, {label}")
    titles = ["(a) Training loss", "(b) Raw-gradient adjacent cosine", "(c) Adam direction and momentum cosine",
              "(d) Raw and momentum cumulative norms", "(e) Cumulative ratio B/A", "(f) Raw and momentum vector norms"]
    labels = ["Token-mean cross-entropy (nats)", "Cosine", "Cosine", "Cumulative norm", "B/A", "Vector norm (log scale)"]
    for i, ax in enumerate(axes.flat):
        ax.set(title=titles[i], ylabel=labels[i], xlabel="Optimizer update", xlim=(1, 711))
        ax.grid(alpha=.18)
        ax.legend(fontsize=8.2, ncol=2 if i >= 2 else 1,
                  loc="upper left" if i == 3 else "lower left" if i in (1, 2, 4) else "best")
    a[1].set_ylim(-1.05, 1.05)
    a[1].axhline(0, color="black", lw=.8, ls="--")
    minimum = min(np.nanmin(arr(r["displayed"], s, "adjacent_cosine")) for r in records for s in ("momentum", "adaptive"))
    a[2].set_ylim(max(-1.05, minimum - .05), 1.015)
    a[3].set_ylim(0, None)
    a[4].set_ylim(0, 1.05)
    a[5].set_yscale("log")
    fig.suptitle("Pythia-14M: Run021 (LR 0.001) versus Run022 (LR 0.0001)\n"
                 "Warmup only; identical initial weights and data order; seed 1234", fontsize=15)
    fig.text(.5, .024, "Light traces: individual updates. Bold loss: trailing 25-update token-weighted mean; bold cosines: trailing 25-update median.\n"
             "A = sum of vector norms; B = norm of vector sum. g: raw pre-clip gradient; m: uncorrected Adam first moment; d: adaptive direction.\n"
             "Display: full-sized updates 1-711. Numerical results retain partial update 712. No cross-seed uncertainty is estimated.",
             ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, .095, 1, .925), h_pad=2.0, w_pad=1.8)
    save_figure(fig, "six_panel_comparison")


def temporal_figures(records):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for ax, stream in zip(axes, STREAMS):
        for record in records:
            profile = record["temporal"][stream]["profile"]
            ax.plot([p["lag"] for p in profile], [p["mean_pair_cosine"] for p in profile],
                    color=record["color"], marker="o", ls="-" if record["id"] == "021" else "--",
                    label=f"LR {record['lr']:g}")
        ax.set(title=STREAMS[stream][3], xlabel="Lag in optimizer updates", ylabel="Mean sampled-pair cosine", ylim=(-1.05, 1.05))
        ax.set_xticks(range(100, 601, 100), [f"{lag}\nn={7-lag//100}" for lag in range(100, 601, 100)])
        ax.axhline(0, color="black", ls="--", lw=.8)
        ax.grid(alpha=.18)
        ax.legend(fontsize=9)
    fig.suptitle("Coarse temporal profiles from saved checkpoint vectors\n"
                 "Sampled updates 100, 200, ..., 700; n = sampled pairs at each lag; dense short-lag histories unavailable", fontsize=12)
    save_figure(fig, "checkpoint_lag_profiles")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
    for row, record in enumerate(records):
        for col, stream in enumerate(STREAMS):
            ax = axes[row, col]
            matrix = record["temporal"][stream]["cosine_matrix"]
            plot = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
            ax.set_xticks(range(7), SAMPLES)
            ax.set_yticks(range(7), SAMPLES)
            ax.set(title=f"Run{record['id']}, LR {record['lr']:g}: {STREAMS[stream][3]}",
                   xlabel="Sampled optimizer update", ylabel="Sampled optimizer update")
    fig.colorbar(plot, ax=axes.ravel().tolist(), label="Cosine between saved vectors", shrink=.9)
    fig.suptitle("Exact pairwise cosines at sparse checkpoints\n"
                 "Matched updates and parameter scope; each off-diagonal cell is one pair; diagonal = self-cosine", fontsize=13)
    save_figure(fig, "checkpoint_cosine_matrices")


def report(records):
    base, small = records
    diff = small["performance"]["final_validation_loss"] - base["performance"]["final_validation_loss"]
    lines = ["# Runs021/022: completed warmup-only learning-rate comparison", "",
             f"At the same full-pass budget, LR 0.001 reaches validation loss {base['performance']['final_validation_loss']:.6f}, "
             f"versus {small['performance']['final_validation_loss']:.6f} at LR 0.0001: a {diff:.6f}-nat difference favoring Run021.", "",
             "Raw adjacent gradients are much more often negative at the larger LR, while stored momentum and adaptive directions have positive adjacent cosines in both runs. "
             "The ordering of cumulative cancellation differs by stream: the smaller LR has higher raw and momentum B/A, but lower adaptive B/A. "
             "Thus greater geometric alignment does not consistently rank these two runs by validation performance.", "",
             "[Six-panel figure](figures/six_panel_comparison.pdf) | [Coarse lag profiles](figures/checkpoint_lag_profiles.pdf) | "
             "[Checkpoint cosine matrices](figures/checkpoint_cosine_matrices.pdf)", "", "## Loss and coverage", "",
             "| Metric | Run021: LR 0.001 | Run022: LR 0.0001 |", "| --- | ---: | ---: |"]
    for label, key in [("Initial validation loss", "initial_validation_loss"), ("Final validation loss", "final_validation_loss"),
                       ("Final validation perplexity = exp(loss)", "final_validation_perplexity"),
                       ("Training loss, updates 612-711 (token-weighted)", "last100_full_training_loss")]:
        lines.append(f"| {label} | {base['performance'][key]:.6f} | {small['performance'][key]:.6f} |")
    lines += ["", "Each run covers 712 accepted updates, 728,375 sequences, 1,491,711,416 input tokens and 1,490,983,041 supervised targets without repetition. "
              "Each initial/final validation covers 339 sequences, 693,668 input tokens and 693,329 targets, including the short tail. "
              "There are only initial and final fixed-validation evaluations; intermediate training curves use changing batches.", "",
              "## Geometry over the complete pass", "", "All scalar summaries below retain update 712; adjacent statistics use 711 defined pairs. "
              "A and B have stream-specific vector units and do not represent parameter displacement.", "",
              "| Stream / metric | Run021 | Run022 |", "| --- | ---: | ---: |"]
    for stream in STREAMS:
        for label, key in [("median adjacent cosine", "median_cosine"), ("negative adjacent pairs (%)", "negative_fraction"),
                           ("A at update 712", "final_A"), ("B at update 712", "final_B"),
                           ("B/A at update 712", "final_B_over_A"), ("median individual norm", "median_norm"),
                           ("median cosine, updates 612-711", "last100_full_median_cosine")]:
            values = [r["stats"][stream][key] * (100 if key == "negative_fraction" else 1) for r in records]
            lines.append(f"| {STREAMS[stream][3]}: {label} | {values[0]:.6f} | {values[1]:.6f} |")
    lines += ["", "Run022's positive full-pass raw median is driven partly by its long early positive phase; the terminal 100-update raw median is negative in both runs. "
              "Raw A is similar across the two passes, whereas the resultant B is much smaller in Run021. Momentum smooths clipped gradients, "
              "and both momentum streams remain positively aligned even when adjacent raw gradients are negative. "
              "Adaptive adjacent alignment is higher than momentum in Run021 but lower in Run022; preconditioning does not uniformly increase adjacent cosine. "
              "Adaptive B/A is below momentum B/A in both passes. These are within-trajectory transformations, not an ablation of momentum or preconditioning.", "",
              "## Temporal coverage and available profiles", "",
              "The saved recovery states contain the last raw, momentum and adaptive vector at each checkpoint. "
              "The supplement uses exactly updates 100, 200, 300, 400, 500, 600 and 700, common to both runs, with all 14,067,712 unique trainable parameters. "
              "Cosines are computed directly in FP64 from saved FP32 raw/momentum and FP64 adaptive vectors. "
              "Lag 100/200/300/400/500/600 means average 6/5/4/3/2/1 sampled-pair cosines, respectively. "
              "The matrices expose every sampled pair rather than hiding its training phase in the mean.", "",
              "At lag 100, mean sampled raw cosine is 0.107714 at LR 0.001 and 0.580164 at LR 0.0001. "
              "Momentum is positive in every sampled pair in both runs. Yet its longest sampled pair (updates 100 and 700) has cosine "
              "0.708184 at LR 0.001 versus 0.486947 at LR 0.0001, so the adjacent-alignment ordering does not extend uniformly to large lags. "
              "For adaptive directions, that same pair has cosine 0.408899 versus -0.252293. "
              "The smaller-LR trajectory can have highly positive adjacent adaptive alignment while early and late sampled directions oppose. "
              "These statements describe the saved pairs; they do not infer unobserved dense lag behavior.", "",
              "| Stream | Lag (updates) | Sampled pairs | Mean cosine, Run021 | Mean cosine, Run022 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for stream in STREAMS:
        for b, s in zip(base["temporal"][stream]["profile"], small["temporal"][stream]["profile"]):
            lines.append(f"| {STREAMS[stream][3]} | {b['lag']} | {b['pairs']} | {b['mean_pair_cosine']:.6f} | {s['mean_pair_cosine']:.6f} |")
    lines += ["",
              "These are exact sparse checkpoint comparisons, not dense phase-specific lag-1-to-64 profiles. "
              "Complete vector histories and temporal sketches were not recorded for these runs. Adjacent scalar cosines and cumulative vectors cannot reconstruct the missing lag profiles. "
              "Different lags average different endpoints/training phases, and a lag with one pair has no replication. No standard errors or cross-seed confidence intervals are shown. "
              "No replay or additional remote computation was launched.", "",
              "## Reduction, provenance and limits", "",
              "The requested main figure is 2x3, in row-major order: (a) training loss for both LRs; (b) raw adjacent cosine; "
              "(c) adaptive/momentum adjacent cosine; (d) overlay of raw/momentum A and B; (e) raw/momentum/adaptive B/A; "
              "(f) raw/momentum individual norms. Colors encode LR; line styles encode stream or A/B. "
              "Faint loss/cosine traces show unsmoothed updates. Bold loss is a trailing 25-update token-weighted mean; bold cosines are trailing 25-update medians. "
              "The main display uses full-sized updates 1-711; the partial update 712 is retained in data/statistics. Panel (c) uses a narrower y-range covering all its observed values; "
              "the norm panel uses a log scale. Undefined first cosines are gaps.", "",
              "Both runs share exact initial tensors, seed 1234 data order, pinned dataset/tokenizer, model, batch, FP16 compute/FP32 Adam moments, clipping and optimizer settings. "
              "Each LR has the same 1% warmup clock followed by a constant plateau. The intended LR difference also changes AdamW's LR-scaled weight decay. "
              "m is actual uncorrected exp_avg, accumulated from clipped gradients; d is the bias-corrected preconditioned direction excluding LR/decay. "
              "No causal mediation, stability, convergence time, information storage or cross-seed generalization claim follows from these measurements. "
              "Run023 is outside this two-run analysis and remains part of the pending full cohort comparison.", "",
              "Sources: each run's immutable `artifacts/attempt-01/full-pass/events.jsonl`, `summary.json`, `input_identity.json`, "
              "`verification.json`, transfer inventory and seven recovery states. Completed retrieval receipts verify all output sizes/SHA-256. "
              "This reduction rechecks the compact source hashes and recorded norms, exact clocks and scope of all sampled vectors. "
              "[Data and hashes](comparison_data.json), [sampled-pair CSV](checkpoint_pairs.csv), [figure observation](observations/001-two-run-comparison.md). "
              "Generating script: `../07_analyze_021_022.py`. Source-run artifacts and remote training are unchanged.", ""]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    for sub in ("figures", "scratch", "observations"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    records = [reduce(ROOT / "runs" / name, color, lr) for name, color, lr in RUNS]
    for key in ("model", "data", "seeds", "measurements", "validation", "checkpoints"):
        assert records[0]["config"][key] == records[1]["config"][key], key
    assert records[0]["identity"] == records[1]["identity"]
    settings = [dict(r["config"]["training"]) for r in records]
    for setting in settings:
        setting.pop("peak_learning_rate")
    assert settings[0] == settings[1]
    for record in records:
        record["temporal"] = temporal(record)
    data = {"created_at": datetime.now(timezone.utc).isoformat(), "script_sha256": digest(__file__),
            "coverage": {"accepted_updates_per_run": 712, "displayed_updates": [1, 711], "partial_update_in_statistics": True,
                         "checkpoint_updates": SAMPLES, "pair_counts_by_lag": [6, 5, 4, 3, 2, 1]},
            "runs": {r["id"]: {"lr": r["lr"], "sources": r["source"], "performance": r["performance"],
                                 "geometry": r["stats"], "temporal": r["temporal"], "events": r["events"]} for r in records}}
    save_json(OUT / "comparison_data.json", data)
    with (OUT / "checkpoint_pairs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run", "lr", "stream", "start_update", "end_update", "lag", "cosine"])
        writer.writeheader()
        for record in records:
            for name in STREAMS:
                for pair in record["temporal"][name]["pairs"]:
                    writer.writerow({"run": record["id"], "lr": record["lr"], "stream": name, **pair})
    summary_figure(records)
    temporal_figures(records)
    report(records)
    (OUT / "observations/001-two-run-comparison.md").write_text(
        "# Observation 001: completed two-run comparison\n\n"
        "Question, findings, sources, reductions, figure captions, full/filtered coverage and uncertainty limits are recorded in [the report](../README.md).\n\n"
        "Outputs: `six_panel_comparison.pdf`, `checkpoint_lag_profiles.pdf`, `checkpoint_cosine_matrices.pdf` under `../figures/`. "
        "Sources and generating-script hash are in `../comparison_data.json`; exact sampled pairs are in `../checkpoint_pairs.csv`. "
        "The main figure contains the six user-requested panels. Supplemental lags are sparse checkpoint observations, not dense reconstructed histories. "
        "One paired seed; no cross-seed error bars. Complete results retain the partial update omitted from the main display.\n",
        encoding="utf-8")
    (OUT / "observations/INDEX.md").write_text("- [001: Completed two-run comparison](001-two-run-comparison.md)\n", encoding="utf-8")
    print(json.dumps({r["id"]: {"performance": r["performance"], "geometry": r["stats"],
                    "coarse_raw_lag_profile": r["temporal"]["raw"]["profile"]} for r in records}, indent=2), flush=True)


if __name__ == "__main__":
    main()
