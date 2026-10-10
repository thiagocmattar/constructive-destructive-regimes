"""Requested 2x2 view of loss, raw alignment and per-LR raw/momentum norms."""
from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
OUT = HERE / "interim-021-022"
HELPER = HERE / "08_plot_simplified.py"
spec = importlib.util.spec_from_file_location("simplified", HELPER)
plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plots)


def first_negative_band(ax, first):
    ax.axvspan(first, 711, color="#777777", alpha=.12, lw=0, zorder=0)
    ax.axvline(first, color="#555555", ls=":", lw=1.1)


def describe(record):
    events = record["displayed"]
    x = np.array([e["update"] for e in events])
    cosine = plots.values(events, "raw", "adjacent_cosine")
    median = plots.smooth(cosine)
    raw = plots.values(events, "raw", "gradient_norm")
    momentum = plots.values(events, "momentum", "momentum_norm")
    assert np.all(raw > 0) and np.all(momentum > 0)
    ratio = momentum / raw
    assert np.allclose(ratio, [e["momentum_to_raw_norm_ratio"] for e in events], rtol=1e-12, atol=1e-12)
    raw_delta = np.r_[np.nan, np.abs(np.diff(raw))]
    momentum_delta = np.r_[np.nan, np.abs(np.diff(momentum))]
    raw_log_delta = np.r_[np.nan, np.abs(np.diff(np.log(raw)))]
    momentum_log_delta = np.r_[np.nan, np.abs(np.diff(np.log(momentum)))]
    first = int(x[np.flatnonzero(cosine < 0)[0]])
    groups = {}
    for name, start, end in (("25_updates_before_first_negative", first - 25, first - 1),
                             ("25_updates_from_first_negative", first, first + 24),
                             ("following_25_updates", first + 25, first + 49)):
        assert start >= 26 and end <= 711
        mask = (x >= start) & (x <= end)
        assert mask.sum() == 25
        groups[name] = {"start_update": start, "end_update": end, "updates": int(mask.sum()),
                        "negative_raw_pairs": int(np.sum(cosine[mask] < 0)),
                        "median_momentum_over_raw_norm": float(np.median(ratio[mask])),
                        "median_absolute_raw_norm_change": float(np.median(raw_delta[mask])),
                        "median_absolute_momentum_norm_change": float(np.median(momentum_delta[mask])),
                        "median_absolute_raw_log_norm_change": float(np.median(raw_log_delta[mask])),
                        "median_absolute_momentum_log_norm_change": float(np.median(momentum_log_delta[mask]))}
    around = [{"update": int(x[i]), "raw_cosine": float(cosine[i]), "raw_norm": float(raw[i]),
               "momentum_norm": float(momentum[i]), "momentum_over_raw_norm": float(ratio[i])}
              for i in (first - 2, first - 1, first)]
    result = {"lr": record["lr"], "source_events_sha256": record["events_sha256"],
              "first_negative_individual_raw_cosine_update": first,
              "first_negative_trailing25_raw_cosine_median_update": int(x[np.flatnonzero(median < 0)[0]]),
              "norms_around_first_negative_pair": around, "first_negative_event_windows": groups,
              "displayed_updates": x.tolist(), "after_first_negative_mask": (x >= first).tolist(),
              "full_events": record["events"]}
    return result, (x, median, raw, momentum)


def main():
    cohort = {r["run_id"]: r for r in json.loads((HERE / "cohort.json").read_text())["runs"]}
    records = [plots.load_run(cohort[rid]) for rid in ("021", "022")]
    assert records[0]["identity"] == records[1]["identity"]
    for key in ("model", "data", "seeds", "measurements", "validation", "checkpoints"):
        assert records[0]["config"][key] == records[1]["config"][key]
    results, series = {}, []
    for record in records:
        result, arrays = describe(record)
        results[record["id"]] = result
        series.append(arrays)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    maximum = max(float(np.max(v)) for _, _, raw, momentum in series for v in (raw, momentum)) * 1.07
    for col, (record, arrays) in enumerate(zip(records, series)):
        x, median, raw, momentum = arrays
        events = record["displayed"]
        color, style = plots.LR_COLORS[record["id"]], plots.LR_STYLES[record["id"]]
        label = f"LR {record['lr']:g}"
        loss = np.array([e["loss"] for e in events])
        weights = np.array([e["supervised_tokens"] for e in events])
        axes[0, 0].plot(x, loss, color=color, alpha=.18, lw=.6)
        axes[0, 0].plot(x, plots.smooth(loss, weights), color=color, ls=style, lw=1.8, label=label)
        cosine = plots.values(events, "raw", "adjacent_cosine")
        axes[0, 1].plot(x, cosine, color=color, alpha=.16, lw=.6)
        axes[0, 1].plot(x, median, color=color, ls=style, lw=1.8, label=label)
        first = results[record["id"]]["first_negative_individual_raw_cosine_update"]
        axes[0, 1].scatter([first], [cosine[first - 1]], color=color, s=35, edgecolor="white", lw=.6, zorder=5)
        axes[0, 1].annotate(f"First negative: {first}", (first, cosine[first - 1]),
                          xytext=(9, -13), textcoords="offset points", color=color, fontsize=9,
                          bbox={"boxstyle": "round,pad=.2", "facecolor": "white", "edgecolor": "none", "alpha": .85})
        ax = axes[1, col]
        first_negative_band(ax, first)
        ax.plot(x, raw, color="#2462a3", lw=1.15, label="Raw gradient g")
        ax.plot(x, momentum, color="#28794d", ls="--", lw=1.8, label="Momentum m")
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Patch(facecolor="#777777", alpha=.12))
        labels.append(f"After first negative raw pair (t={first})")
        ax.legend(handles, labels, fontsize=9, loc="upper left")
        ax.set(title=f"({'c' if col == 0 else 'd'}) LR {record['lr']:g}: raw and momentum norms",
               ylabel="Vector norm", ylim=(0, maximum))
    axes[0, 0].set(title="(a) Training loss", ylabel="Token-mean cross-entropy (nats)")
    axes[0, 1].set(title="(b) Raw-gradient adjacent cosine", ylabel="Cosine", ylim=(-1.05, 1.05))
    axes[0, 1].axhline(0, color="black", ls=":", lw=.8)
    for ax in axes[0]:
        ax.legend(fontsize=10, loc="best")
    plots.format_axes(axes)
    fig.suptitle("Pythia-14M: loss, raw-gradient alignment and vector norms\n"
                 "Warmup-only LR 0.001 and 0.0001; matched initial weights and data order; seed 1234", fontsize=14)
    fig.text(.5, .022, "Light loss/cosine traces: individual updates. Bold loss: trailing 25-update token-weighted mean; bold cosine: trailing 25-update median.\n"
             "Dots and vertical markers: first negative unsmoothed raw pair. Gray regions: time after that event, including later positive pairs.\n"
             "Norm curves are unsmoothed on a shared linear scale. g is pre-clipping; m is uncorrected exp_avg from clipped gradients.\n"
             "Display: full-sized updates 1-711; partial update 712 retained in data. Event timing describes an association; it does not define norm separation.",
             ha="center", va="bottom", fontsize=8.7)
    fig.tight_layout(rect=(0, .11, 1, .91), h_pad=2.3, w_pad=2.)
    plots.save(fig, OUT, "loss_alignment_norms")
    data = {"created_at": datetime.now(timezone.utc).isoformat(), "script_sha256": plots.sha(__file__),
            "helper_sha256": plots.sha(HELPER), "panel_layout": [2, 2], "runs": results,
            "jaggedness_definition": "Median absolute first differences of vector norms and log vector norms; scalar trajectory fluctuations, not gradient-noise variance",
            "group_definition": "Three consecutive 25-update windows: before, from, and following the first unsmoothed negative raw pair; temporally dependent observational samples",
            "shading_definition": "From first unsmoothed negative raw pair through update 711; not an indicator of the sign of later raw pairs"}
    (OUT / "loss_alignment_norms_data.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Observation 003: raw alignment, momentum norm separation and norm fluctuations", "",
             "User intuition: when raw-gradient adjacent cosine becomes negative, raw and momentum norms separate and the raw norm becomes noisy. "
             "This observation examines that intuition descriptively along the two completed matched trajectories.", "",
             "## Figure and coverage", "",
             "[The 2x2 figure](../figures/loss_alignment_norms.pdf) overlays both LRs in (a) training loss and (b) raw adjacent cosine; "
             "(c) and (d) show raw and momentum individual norms at LR 0.001 and 0.0001. Dots in the cosine panel and vertical lines in the norm panels "
             "mark the first negative unsmoothed raw pair, at updates 54 and 385. Gray regions mark all subsequent time, including later positive pairs. "
             "The shading no longer uses a smoothed sign; the trailing median remains only a curve display aid. "
             "Bottom panels share a linear y-range; raw/momentum norm curves are unsmoothed. "
             "Faint loss/cosine traces show individual updates; bold loss is a trailing token-weighted mean and bold cosine a trailing median, both with window 25.", "",
             "The main display contains full-sized updates 1-711; all 712 accepted updates and the partial final batch are retained in "
             "[data and source/generator hashes](../loss_alignment_norms_data.json). Source event hashes match the previous verified analysis. "
             "Both runs cover the complete identical token stream/order and initial tensors; LR also scales AdamW decay. "
             "No new training or replay is performed.", "", "## Operational meaning and findings", "",
             "Here 'separation' means the ratio of stored-momentum norm to raw-gradient norm, ||m||/||g||. "
             "'Jaggedness' is measured as median absolute consecutive changes of norms and log norms. "
             "These are scalar trajectory fluctuations, not estimates of stochastic gradient-noise variance at a fixed parameter state. "
             "The first event is selected by rho_t < 0 on the original unsmoothed record, without a persistence requirement. "
             "The following table examines the preceding, event and next update, avoiding a smoothed crossing as a proxy for onset.", "",
             "| LR | Update | Position relative to first negative pair | Raw adjacent cosine | Raw norm | Momentum norm | m/g norm ratio |",
             "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for rid, result in results.items():
        for label, s in zip(("preceding update", "first negative pair", "next update"), result["norms_around_first_negative_pair"]):
            lines.append(f"| {result['lr']:g} | {s['update']} | {label} | {s['raw_cosine']:.6f} | {s['raw_norm']:.6f} | "
                         f"{s['momentum_norm']:.6f} | {s['momentum_over_raw_norm']:.6f} |")
    lines += ["", "At the preceding updates 53 and 384, raw adjacent cosine is still positive (0.084 and 0.049), "
              "yet momentum norm is already only 62.7% and 61.0% of raw norm. At the first negative pair itself, "
              "m/g increases to 73.6% and 62.8%, rather than suddenly dropping. Thus the first negative pair does not identify "
              "the exact start of norm separation. Loss of directional coherence is already visible before the sign crossing. "
              "A strictly defined onset of 'disconnection' would require an additional declared criterion, not the first cosine sign alone.", "",
              "First negative individual raw pairs occur at updates 54 and 385; trailing 25-update median sign crossings occur much later, "
              "at 112 and 496. The first pair is a direct event reference and the shaded region marks time after it, not a permanently negative regime.", "",
              "## Local windows around the first event", "",
              "These three consecutive 25-update windows describe timing without claiming a causal change point. "
              "The raw-norm fluctuation at the window's first update compares with the immediately preceding update. "
              "Windows are temporally dependent, differ in training age and gradient scale, and have no independent-seed uncertainty.", "",
              "| LR | Window | Updates | Negative raw pairs | Median m/g norm ratio | Median absolute raw norm change | Median absolute momentum norm change | Median absolute raw log-norm change | Median absolute momentum log-norm change |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for rid, result in results.items():
        for name, s in result["first_negative_event_windows"].items():
            label = name.replace("_", " ")
            lines.append(f"| {result['lr']:g} | {label} | {s['start_update']}-{s['end_update']} | {s['negative_raw_pairs']} | {s['median_momentum_over_raw_norm']:.6f} | "
                         f"{s['median_absolute_raw_norm_change']:.6f} | {s['median_absolute_momentum_norm_change']:.6f} | "
                         f"{s['median_absolute_raw_log_norm_change']:.6f} | {s['median_absolute_momentum_log_norm_change']:.6f} |")
    lines += ["", "Compared with the preceding 25-update window, the first 25 updates from the event have a lower median m/g ratio "
              "(1.165 to 0.709 at LR 0.001; 0.848 to 0.717 at LR 0.0001). "
              "Their raw absolute norm fluctuations increase about 5.8-fold and 2.5-fold, respectively, while momentum fluctuations "
              "fall or remain similar. This is consistent with raw norms becoming more jagged near the first observed negative pair. "
              "The per-update table nevertheless shows that the norm gap is already present before that pair.", "",
              "## Interpretation limits", "",
              "Raw g is pre-clipping; stored m is formed from globally clipped gradients and is uncorrected for initialization bias. "
              "Clipping, EMA history and changing gradient magnitudes all affect the norm gap. "
              "At initialization, m starts at zero, so early norm differences also include its buildup. "
              "In exact arithmetic, m_t = beta*m_(t-1) + (1-beta)*g_t_clipped, so its squared norm includes a cross term "
              "between previous momentum and the current clipped gradient. Adjacent raw cosine alone does not determine the momentum norm. "
              "The figure is consistent with cancellation/smoothing contributing to norm separation, but it does not isolate their causal role. "
              "One matched seed, temporally dependent observations and two LR-scaled trajectories provide no cross-seed uncertainty, stability or convergence guarantee.", "",
              "Generating script: `../../09_plot_loss_alignment_norms.py`; verified event-loading/smoothing helper: `../../08_plot_simplified.py`. "
              "Each source event SHA-256, all raw records, display mask and group reductions are in the figure data. "
              "Prior analyses and executed scientific inputs remain unchanged.", ""]
    (OUT / "observations/003-loss-alignment-norms.md").write_text("\n".join(lines), encoding="utf-8")
    index = OUT / "observations/INDEX.md"
    line = "- [003: Raw alignment and norm separation](003-loss-alignment-norms.md)\n"
    content = index.read_text(encoding="utf-8")
    if line not in content:
        index.write_text(content + line, encoding="utf-8")
    print(json.dumps({rid: {k: v for k, v in r.items() if k not in ("full_events", "displayed_updates", "after_first_negative_mask")} for rid, r in results.items()}, indent=2))


if __name__ == "__main__":
    main()
