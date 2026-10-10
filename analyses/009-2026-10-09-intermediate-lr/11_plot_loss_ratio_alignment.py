"""Seven-LR loss, raw B/A, adjacent cosines and vector norms with requested smoothing."""
from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lr_plot_helpers", HERE / "10_plot_lr_geometry.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
OUT = helpers.OUT
NAME = "loss_ratio_alignment_norms"


def trailing_mean(values, window):
    result = np.full(len(values), np.nan)
    for i in range(len(values)):
        part = values[max(0, i-window+1):i+1]
        valid = np.isfinite(part)
        if valid.any():
            result[i] = np.mean(part[valid])
    return result


def serial(values):
    return [float(v) if np.isfinite(v) else None for v in values]


def main():
    rows = helpers.records()
    prior = json.loads((OUT / "requested_figures_data.json").read_text())
    assert helpers.sha(helpers.SOURCE) == prior["source_comparison_sha256"]
    prior_colors = {r["id"]: r["color"] for r in prior["palette"]["conditions"]}
    fig, axes = plt.subplots(3, 2, figsize=(14, 13.5))
    x = np.arange(1, 712)
    plotted = {}
    for r in rows:
        assert to_hex(r["color"]) == prior_colors[r["id"]]
        e = r["displayed"]
        loss = np.array([v["loss"] for v in e])
        weights = np.array([v["supervised_tokens"] for v in e])
        raw_cos = helpers.plots.values(e, "raw", "adjacent_cosine")
        momentum_cos = helpers.plots.values(e, "momentum", "adjacent_cosine")
        series = {
            "training_loss": helpers.plots.smooth(loss, weights),
            "raw_B_over_A": helpers.plots.values(e, "raw", "cumulative_ratio"),
            "raw_adjacent_cosine": trailing_mean(raw_cos, 50),
            "momentum_adjacent_cosine": trailing_mean(momentum_cos, 25),
            "raw_vector_norm": trailing_mean(helpers.plots.values(e, "raw", "gradient_norm"), 10),
            "momentum_vector_norm": helpers.plots.values(e, "momentum", "momentum_norm"),
        }
        assert np.isnan(series["raw_adjacent_cosine"][0]) and np.isnan(series["momentum_adjacent_cosine"][0])
        assert np.isfinite(series["raw_adjacent_cosine"][1:]).all()
        assert np.isfinite(series["momentum_adjacent_cosine"][1:]).all()
        assert (series["raw_vector_norm"] > 0).all()
        axes[0,0].plot(x, loss, color=r["color"], alpha=.10, lw=.6)
        for ax, values in zip(axes.flat, series.values()):
            ax.plot(x, values, color=r["color"], ls=r["hardware_style"], lw=1.8)
        plotted[r["id"]] = {"lr": r["lr"], "hardware": r["hardware"], "color": to_hex(r["color"]),
                              "line_style": r["hardware_style"], "series": {key:serial(v) for key,v in series.items()}}
    titles = ["(a) Training loss", "(b) Raw-gradient g: cumulative B/A",
              "(c) Raw-gradient adjacent cosine", "(d) Momentum adjacent cosine",
              "(e) Raw-gradient vector norm", "(f) Momentum vector norm"]
    for i, ax in enumerate(axes.flat):
        ax.set_title(titles[i])
        ax.set_ylabel("Cross-entropy (nats)" if i==0 else "B/A" if i==1 else "Cosine" if i in (2,3) else "Vector norm")
        if i==1:
            ax.set_ylim(0,1.05)
        elif i in (2,3):
            ax.set_ylim(-1.05,1.05); ax.axhline(0,color="black",ls=":",lw=.7)
        elif i==4:
            ax.set_yscale("log"); ax.set_ylabel("Vector norm (log scale)"); ax.grid(which="minor",alpha=.08)
        elif i==5:
            ax.set_ylim(0,None)
    helpers.plots.format_axes(axes)
    fig.suptitle("Pythia-14M: loss, raw cumulative geometry, alignment and vector norms",fontsize=17,y=.988)
    helpers.legend(fig,rows,hardware_style=True,y=.965)
    fig.text(.5,.015,"(a) Trailing 25-update token-weighted loss mean; light traces: individual losses. (b) Raw B/A: unsmoothed.\n"
             "(c) Raw adjacent cosine: trailing 50-update arithmetic mean. (d) Momentum adjacent cosine: trailing 25-update arithmetic mean.\n"
             "(e) Raw norm: trailing 10-update arithmetic mean, logarithmic Y. (f) Momentum norm: unsmoothed, linear Y.\n"
             "Windows include the current update, use available prefix data, and exclude undefined first cosines; no future data.\n"
             "Updates 1-711 shown; partial update 712 retained in results. g is preclip; m is stored exp_avg before preconditioning.\n"
             "Seed 1234; frozen initialization/order; different hardware across S/R; LR scales AdamW decay; one seed.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.102,1,.90),h_pad=2.1,w_pad=2.1)
    path = OUT / "figures" / (NAME + ".pdf")
    fig.savefig(path); plt.close(fig)
    metadata = {"generated_at":datetime.now(timezone.utc).isoformat(), "source_comparison_sha256":helpers.sha(helpers.SOURCE),
                "generator_sha256":helpers.sha(__file__), "helper_sha256":helpers.sha(HERE/"10_plot_lr_geometry.py"),
                "palette":prior["palette"], "layout":[3,2], "displayed_updates":[1,711], "source_updates":712,
                "partial_update_retained":True, "figure_sha256":helpers.sha(path),
                "smoothing":{"training_loss":"trailing25 token-weighted arithmetic mean",
                             "raw_B_over_A":"none", "raw_adjacent_cosine":"trailing50 arithmetic mean",
                             "momentum_adjacent_cosine":"trailing25 arithmetic mean", "raw_vector_norm":"trailing10 arithmetic mean before log display",
                             "momentum_vector_norm":"none", "window_policy":"available prefix including current update; omit nonfinite first cosine"},
                "y_scales":["linear","linear","linear","linear","log","linear"], "updates":x.tolist(), "runs":plotted}
    (OUT / (NAME + "_data.json")).write_text(json.dumps(metadata,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"figure":str(path),"layout":[3,2],"raw_cosine_mean_window":50,"raw_norm_mean_window":10,"source_unchanged":True}))


if __name__ == "__main__": main()
