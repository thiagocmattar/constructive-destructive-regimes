"""Verified three-LR results; preserve previous two-run figures and generators."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import importlib.util
import json
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "complete-three-lr"


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


analysis = module("interim_analysis", "07_analyze_021_022.py")
plots = module("simplified_plots", "08_plot_simplified.py")


def focused(records):
    fig = plt.figure(figsize=(19.5, 10.5))
    grid = fig.add_gridspec(2, 6)
    loss_ax = fig.add_subplot(grid[0, :3])
    cosine_ax = fig.add_subplot(grid[0, 3:])
    norm_axes = [fig.add_subplot(grid[1, 2*i:2*i+2]) for i in range(3)]
    for record, ax in zip(records, norm_axes):
        events = record["displayed"]
        x = np.array([e["update"] for e in events])
        color, style = record["color"], plots.LR_STYLES[record["id"]]
        label = f"LR {record['lr']:g}"
        loss = np.array([e["loss"] for e in events])
        weights = np.array([e["supervised_tokens"] for e in events])
        cosine = plots.values(events, "raw", "adjacent_cosine")
        loss_ax.plot(x, loss, color=color, alpha=.18, lw=.6)
        loss_ax.plot(x, plots.smooth(loss, weights), color=color, ls=style, lw=1.8, label=label)
        cosine_ax.plot(x, cosine, color=color, alpha=.15, lw=.6)
        cosine_ax.plot(x, plots.smooth(cosine), color=color, ls=style, lw=1.8, label=label)
        first = record["extra"]["first_negative_raw_pair"]
        cosine_ax.scatter([first], [cosine[first-1]], color=color, s=35, edgecolor="white", lw=.6, zorder=5)
        offset = (13, 12) if record["id"] == "023" else (13, -18)
        cosine_ax.annotate(f"First negative: {first}", (first, cosine[first-1]),
                           xytext=offset, textcoords="offset points", color=color, fontsize=10,
                           bbox={"boxstyle": "round,pad=.2", "facecolor": "white", "edgecolor": "none", "alpha": .9})
        ax.axvspan(first, 711, color="#777777", alpha=.12, lw=0, zorder=0)
        ax.axvline(first, color="#555555", ls=":", lw=1.1)
        raw = plots.values(events, "raw", "gradient_norm")
        momentum = plots.values(events, "momentum", "momentum_norm")
        ax.plot(x, raw, color="#2462a3", lw=1.0, label="Raw gradient g")
        ax.plot(x, momentum, color="#28794d", ls="--", lw=1.8, label="Momentum m")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + [Patch(facecolor="#777777", alpha=.12)],
                  labels + [f"After first negative pair (t={first})"], fontsize=9, loc="upper left")
        ax.set(title=f"({'cde'[norm_axes.index(ax)]}) LR {record['lr']:g}: raw and momentum norms",
               ylabel="Vector norm", ylim=(0, 1.07*max(np.max(raw), np.max(momentum))))
    loss_ax.set(title="(a) Training loss", ylabel="Token-mean cross-entropy (nats)")
    cosine_ax.set(title="(b) Raw-gradient adjacent cosine", ylabel="Cosine", ylim=(-1.05, 1.05))
    cosine_ax.axhline(0, color="black", ls=":", lw=.8)
    for ax in (loss_ax, cosine_ax):
        ax.legend(fontsize=10, loc="best")
    plots.format_axes([loss_ax, cosine_ax, *norm_axes])
    fig.suptitle("Pythia-14M: loss, raw-gradient alignment and vector norms\n"
                 "Three completed warmup-only LR runs; matched initial weights and data order; seed 1234", fontsize=15)
    fig.text(.5, .022, "Light traces: individual updates. Bold loss: trailing 25-update token-weighted mean; bold cosine: trailing 25-update median.\n"
             "Dots/vertical markers: first negative unsmoothed raw pair. Shading marks subsequent time, including later positive pairs.\n"
             "Norm curves are unsmoothed; each lower panel has its own linear y-scale. g is pre-clipping; m is uncorrected exp_avg from clipped gradients.\n"
             "Display: updates 1-711; partial update 712 retained in results. The sign crossing does not define the onset of norm separation.",
             ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, .11, 1, .92), h_pad=2.5, w_pad=2.1)
    plots.save(fig, OUT, "loss_alignment_norms")


def temporal_figures(records):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    for ax, stream in zip(axes, analysis.STREAMS):
        for record in records:
            profile = record["temporal"][stream]["profile"]
            ax.plot([p["lag"] for p in profile], [p["mean_pair_cosine"] for p in profile],
                    color=record["color"], ls=plots.LR_STYLES[record["id"]], marker="o", label=f"LR {record['lr']:g}")
        ax.set(title=analysis.STREAMS[stream][3], xlabel="Lag in optimizer updates",
               ylabel="Mean sampled-pair cosine", ylim=(-1.05, 1.05))
        ax.set_xticks(range(100, 601, 100), [f"{lag}\nn={7-lag//100}" for lag in range(100, 601, 100)])
        ax.axhline(0, color="black", ls=":", lw=.8)
        ax.grid(alpha=.18)
        ax.legend(fontsize=9)
    fig.suptitle("Coarse temporal profiles: three completed learning rates\n"
                 "Exact saved vectors at updates 100, 200, ..., 700; n = sampled pairs; dense short-lag histories unavailable", fontsize=12)
    plots.save(fig, OUT, "checkpoint_lag_profiles")
    fig, axes = plt.subplots(3, 3, figsize=(16, 13), layout="constrained")
    for row, record in enumerate(records):
        for col, stream in enumerate(analysis.STREAMS):
            ax = axes[row, col]
            picture = ax.imshow(record["temporal"][stream]["cosine_matrix"], cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
            ax.set_xticks(range(7), analysis.SAMPLES)
            ax.set_yticks(range(7), analysis.SAMPLES)
            ax.set(title=f"Run{record['id']}, LR {record['lr']:g}: {analysis.STREAMS[stream][3]}",
                   xlabel="Sampled optimizer update", ylabel="Sampled optimizer update")
    fig.colorbar(picture, ax=axes.ravel().tolist(), label="Cosine between saved vectors", shrink=.85)
    fig.suptitle("Exact pairwise cosines at sparse checkpoints\n"
                 "Matched updates/scope; each off-diagonal cell is one pair; diagonal = self-cosine", fontsize=13)
    plots.save(fig, OUT, "checkpoint_cosine_matrices")


def extra(record):
    events = record["events"]
    first = next(e["update"] for e in events if e["adjacent_cosine"] is not None and e["adjacent_cosine"] < 0)
    window_loss = plots.smooth(np.array([e["loss"] for e in record["displayed"]]),
                               np.array([e["supervised_tokens"] for e in record["displayed"]]))
    best = int(np.argmin(window_loss[24:])) + 24
    relative = "checkpoints/final/recovery.pt"
    path = record["folder"] / relative
    item = record["inventory"][relative]
    assert path.stat().st_size == item["bytes"]
    record["source"][relative] = item["sha256"]
    state = torch.load(path, mmap=True, map_location="cpu", weights_only=False)
    g, m = state["geometry"], state["momentum_geometry"]
    assert g["names"] == m["names"] and g["updates"] == m["updates"] == 712
    gg = mm = gm = delta = clipped_estimate = 0.
    for G, M, last in zip(g["cumulative"], m["cumulative"], m["previous"]):
        gv, mv, lv = G.flatten(), M.flatten(), last.flatten()
        for start in range(0, gv.numel(), 262144):
            x, y, z = gv[start:start+262144], mv[start:start+262144], lv[start:start+262144].double()
            gg += float(x.square().sum()); mm += float(y.square().sum())
            gm += float((x*y).sum()); delta += float((x-y).square().sum())
            clipped_estimate += float((y+9*z).square().sum())
    assert math.isclose(math.sqrt(gg), record["stats"]["raw"]["final_B"], rel_tol=1e-10, abs_tol=1e-12)
    assert math.isclose(math.sqrt(mm), record["stats"]["momentum"]["final_B"], rel_tol=1e-10, abs_tol=1e-12)
    raw, momentum = record["stats"]["raw"], record["stats"]["momentum"]
    return {"first_negative_raw_pair": first,
            "first_negative_momentum_pair": next((e["update"] for e in events if e["momentum"]["adjacent_cosine"] is not None and e["momentum"]["adjacent_cosine"] < 0), None),
            "clipping_threshold_exceeded_updates": sum(e["gradient_norm"] > 1 for e in events),
            "best_trailing25_training_loss": float(window_loss[best]), "best_trailing25_end_update": best+1,
            "A_momentum_reduction_fraction": 1-momentum["final_A"]/raw["final_A"],
            "B_momentum_reduction_fraction": 1-momentum["final_B"]/raw["final_B"],
            "cumulative_raw_momentum_cosine": gm/math.sqrt(gg*mm),
            "cumulative_raw_momentum_difference_norm": math.sqrt(delta),
            "ideal_EMA_reconstructed_clipped_B": math.sqrt(clipped_estimate),
            "EMA_endpoint_correction_norm": 9*float(m["previous_norm"]),
            "clipped_reconstruction_caveat": "Uses sum(m)+9*m_T and m_0=0; ignores FP32 recurrence rounding, not an independently recorded clipped-gradient history"}


def report(records):
    titles = " | ".join(f"Run{r['id']}: LR {r['lr']:g}" for r in records)
    lines = ["# Completed three-LR warmup-only comparison", "",
             "LR 0.001 has the lowest final validation loss in this matched single-seed cohort. "
             "Both a tenfold reduction and a tenfold increase in LR give higher final loss. "
             "The large LR has much larger raw norms, more clipping and a smaller raw cumulative B/A. "
             "Positive momentum/adaptive adjacent alignment persists, despite worsening late training loss.", "",
             "[Combined six metrics](figures/metrics_by_learning_rate.pdf) | [Within-LR geometry](figures/geometry_within_learning_rate.pdf) | "
             "[Loss, alignment and norms](figures/loss_alignment_norms.pdf) | [Checkpoint lag profiles](figures/checkpoint_lag_profiles.pdf) | "
             "[Checkpoint cosine matrices](figures/checkpoint_cosine_matrices.pdf)", "",
             "## Performance and geometry", "", f"| Metric | {titles} |", "| --- | ---: | ---: | ---: |"]
    for label, key in (("Final validation loss", "final_validation_loss"),
                       ("Final validation perplexity", "final_validation_perplexity"),
                       ("Training loss, updates 612-711 (token weighted)", "last100_full_training_loss")):
        lines.append(f"| {label} | " + " | ".join(f"{r['performance'][key]:.6f}" for r in records) + " |")
    for stream in analysis.STREAMS:
        for label, key in (("median adjacent cosine", "median_cosine"), ("negative adjacent pairs (%)", "negative_fraction"),
                           ("final A", "final_A"), ("final B", "final_B"), ("final B/A", "final_B_over_A"),
                           ("median vector norm", "median_norm"), ("last100 median adjacent cosine", "last100_full_median_cosine")):
            factor = 100 if key == "negative_fraction" else 1
            lines.append(f"| {stream}: {label} | " + " | ".join(f"{r['stats'][stream][key]*factor:.6f}" for r in records) + " |")
    for label, key in (("First negative raw pair (update)", "first_negative_raw_pair"),
                       ("Updates exceeding raw clipping threshold", "clipping_threshold_exceeded_updates"),
                       ("Best trailing25 training loss", "best_trailing25_training_loss"),
                       ("Best trailing25 window end update", "best_trailing25_end_update"),
                       ("Ideal-EMA reconstructed clipped B", "ideal_EMA_reconstructed_clipped_B"),
                       ("EMA endpoint correction norm", "EMA_endpoint_correction_norm"),
                       ("Cosine between cumulative raw and momentum vectors", "cumulative_raw_momentum_cosine")):
        lines.append(f"| {label} | " + " | ".join(f"{r['extra'][key]:.6f}" for r in records) + " |")
    lines += ["", "## Reading the A/B comparison", "",
              "In all three runs, momentum lowers A substantially while B also decreases. "
              "The raw/momentum B curves almost overlap in Runs021/022; they separate more at the large LR. "
              "For normalized EMA m_t = 0.9*m_(t-1) + 0.1*h_t with m_0=0, the exact-arithmetic identity is "
              "sum(m_t) = sum(h_t) - 9*m_T, where h is the clipped gradient. "
              "Thus the cumulative clipped-gradient sum is preserved up to the terminal momentum correction, "
              "while averaging can reduce the sum of individual norms. "
              "The table's reconstructed clipped B uses this identity with actual stored FP32 moments; it ignores recurrence rounding "
              "and is not an independently captured clipped-gradient history. At LR 0.01, raw/momentum B separation is much larger "
              "than the endpoint correction. This is compatible with substantial clipping contributions; it does not isolate an optimization benefit of momentum.", "",
              "## Captions, reductions and coverage", "",
              "**Combined metrics:** 3x2 in row-major order: loss, raw cosine, momentum cosine, adaptive cosine, raw norms, momentum norms. "
              "Every plot overlays all three LRs. LR determines color/style; faint loss/cosine curves show unsmoothed records. "
              "Bold loss is a trailing 25-update token-weighted mean; bold cosine is a trailing 25-update median. "
              "Norms are unsmoothed and linear; all cosine panels share [-1.05, 1.05].", "",
              "**Within-LR geometry:** 2x3, columns Run021/022/023. Upper panels compare raw/momentum A (solid) and B (dashed); "
              "lower panels compare g/m/d B/A. Stream determines color, and each row shares axes across LRs. All curves are unsmoothed.", "",
              "**Focused view:** two shared loss/cosine panels above three per-LR raw/momentum norm panels. "
              "Dots and vertical lines mark the first negative unsmoothed raw pair; shading marks all subsequent time, including later positive pairs. "
              "Median sign is never used for shading. Each norm panel has its own declared linear scale to retain visibility across LR ranges. "
              "The first negative pair is an event reference, not an operational onset of norm separation. "
              "For Run023 it occurs at update 12, close to warmup and momentum initialization; a full 25-update pre-event window does not exist.", "",
              "All runs cover 712 accepted updates, 728,375 sequences, 1,491,711,416 input tokens and 1,490,983,041 supervised targets without repeats. "
              "The three main figures display full-sized updates 1-711; the partial final update 712 is retained in all source records and terminal statistics. "
              "Initial/final validation includes all 339 sequences, 693,668 input tokens and 693,329 targets, including the short tail. "
              "Training-loss smoothing weights supervised tokens; changing-batch training curves are not intermediate fixed-validation evaluations.", "",
              "**Temporal profiles and matrices:** exact FP64 whole-model cosines using the saved previous vectors at common updates 100-700 "
              "in increments of 100, all 14,067,712 unique parameters. Lag 100/200/300/400/500/600 averages 6/5/4/3/2/1 pairs. "
              "The 3x3 matrix layout has LR rows and raw/momentum/adaptive columns, with shared [-1,1] color limits. "
              "These sparse checkpoint samples mix training phases; they do not reconstruct dense short-lag profiles. "
              "No independent-seed confidence intervals or causal conclusions are available.", "",
              "At LR 0.01, sampled momentum checkpoint pairs can have negative cosine despite every adjacent momentum pair being positive. "
              "Its sampled adaptive lag-100 mean is also much smaller than its adjacent adaptive median. "
              "The following exact longest sampled pair (updates 100 and 700) illustrates the differing temporal orders:", "",
              f"| Stream: cosine between updates 100 and 700 | {titles} |", "| --- | ---: | ---: | ---: |"]
    for stream in analysis.STREAMS:
        lines.append(f"| {stream} | " + " | ".join(f"{r['temporal'][stream]['cosine_matrix'][0][-1]:.6f}" for r in records) + " |")
    lines += ["",
              "## Provenance and limits", "",
              "Sources are the completed immutable attempt-01 event logs, summaries, identities, configurations, verification files and recovery states. "
              "All retrieved files passed size/SHA-256 verification; Run023's 48 files were independently rechecked at closeout. "
              "The two previous checkpoint analyses are reused unchanged, after matching source hashes; Run023's exact matrices are newly computed. "
              "[All event records, reductions and source hashes](comparison_data.json); [189 sampled pairs](checkpoint_pairs.csv); "
              "[figure QA](verification.json). Generating script: `../10_complete_three_lr.py`, with unchanged verified helpers `../07_analyze_021_022.py` and `../08_plot_simplified.py`.", "",
              "All conditions share exact frozen initial tensors, data order/seed 1234, full token stream, optimizer, clipping, precision, batch and runtime. "
              "Only peak LR varies within this warmup-only cohort; LR also scales AdamW decay displacement. "
              "g is raw pre-clipping; m is actual uncorrected first moment of clipped gradients; d is bias-corrected/preconditioned, excluding LR/decay. "
              "A/B values have stream-specific units and are not actual parameter displacement. "
              "One matched seed does not support cross-seed generalization, a stability criterion, noise variance, information retention or a causal mediation claim. "
              "Previous interim PDFs and all executed scientific inputs are preserved; no new training, replay or cloud spending was launched.", ""]
    text = "\n".join(lines)
    (OUT / "README.md").write_text(text, encoding="utf-8")
    observation = text.replace("# Completed three-LR warmup-only comparison", "# Observation 001: completed three-LR results", 1)
    observation = observation.replace("](figures/", "](../figures/").replace("](comparison_data.json)", "](../comparison_data.json)")
    observation = observation.replace("](checkpoint_pairs.csv)", "](../checkpoint_pairs.csv)").replace("](verification.json)", "](../verification.json)")
    observation = observation.replace("`../10_complete_three_lr.py`", "`../../10_complete_three_lr.py`").replace("`../07_analyze_021_022.py`", "`../../07_analyze_021_022.py`").replace("`../08_plot_simplified.py`", "`../../08_plot_simplified.py`")
    (OUT / "observations/001-three-lr-results.md").write_text(observation, encoding="utf-8")
    (OUT / "observations/INDEX.md").write_text("- [001: Completed three-LR results](001-three-lr-results.md)\n", encoding="utf-8")


def main():
    for folder in ("figures", "scratch", "observations"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    cohort = json.loads((HERE / "cohort.json").read_text())["runs"]
    records = [analysis.reduce(ROOT / "runs" / r["run"], plots.LR_COLORS[r["run_id"]], r["peak_lr"]) for r in cohort]
    old = json.loads((HERE / "interim-021-022/comparison_data.json").read_text())["runs"]
    for record in records:
        assert record["identity"] == records[0]["identity"]
        for key in ("model", "data", "seeds", "measurements", "validation", "checkpoints"):
            assert record["config"][key] == records[0]["config"][key], key
        a, b = dict(record["config"]["training"]), dict(records[0]["config"]["training"])
        a.pop("peak_learning_rate"); b.pop("peak_learning_rate")
        assert a == b
        if record["id"] in old:
            cached = old[record["id"]]
            assert record["events"] == cached["events"] and record["source"]["events.jsonl"] == cached["sources"]["events.jsonl"]
            record["temporal"] = cached["temporal"]
            for step in analysis.SAMPLES:
                relative = f"checkpoints/recovery-{step:04d}/recovery.pt"
                assert cached["sources"][relative] == record["inventory"][relative]["sha256"]
                record["source"][relative] = cached["sources"][relative]
        else:
            record["temporal"] = analysis.temporal(record)
        record["extra"] = extra(record)
    plots.across_lrs(records, OUT)
    plots.within_lrs(records, OUT)
    focused(records)
    temporal_figures(records)
    data = {"created_at": datetime.now(timezone.utc).isoformat(), "script_sha256": analysis.digest(__file__),
            "helper_sha256": {name: analysis.digest(HERE / name) for name in ("07_analyze_021_022.py", "08_plot_simplified.py")},
            "coverage": {"source_updates": 712, "main_display_updates": [1, 711], "partial_update_retained": True,
                         "checkpoint_updates": analysis.SAMPLES, "sampled_pair_count": 189},
            "runs": {r["id"]: {"lr": r["lr"], "sources": r["source"], "performance": r["performance"],
                                  "geometry": r["stats"], "extra": r["extra"], "temporal": r["temporal"], "events": r["events"]} for r in records}}
    analysis.save_json(OUT / "comparison_data.json", data)
    with (OUT / "checkpoint_pairs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run_id", "lr", "stream", "start_update", "end_update", "lag", "cosine"])
        writer.writeheader()
        for record in records:
            for name, temporal in record["temporal"].items():
                for pair in temporal["pairs"]:
                    writer.writerow({"run_id": record["id"], "lr": record["lr"], "stream": name, **pair})
    report(records)
    print(json.dumps({r["id"]: {"performance": r["performance"], "geometry": r["stats"], "extra": r["extra"]} for r in records}, indent=2))


if __name__ == "__main__":
    main()
