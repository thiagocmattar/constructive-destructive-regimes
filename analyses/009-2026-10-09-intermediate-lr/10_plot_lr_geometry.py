"""Requested log-norm revision and cumulative geometry across seven learning rates."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "complete-seven-lr"
SOURCE = OUT / "comparison_data.json"
HELPER = ROOT / "analyses/008-2026-10-08-no-decay-momentum/08_plot_simplified.py"
spec = importlib.util.spec_from_file_location("verified_plot_helpers", HELPER)
plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plots)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "pdf.fonttype": 42})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lightness(color):
    rgb = np.array(color[:3])
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    y = float(linear @ np.array([.2126729, .7151522, .0721750]))
    return 116 * np.cbrt(y) - 16 if y > (6 / 29) ** 3 else (29 / 3) ** 3 * y


def records():
    source = json.loads(SOURCE.read_text())
    assert sha(HELPER) == source["helper_sha256"]
    rows = sorted(({"id": rid, **r} for rid, r in source["runs"].items()), key=lambda r: r["lr"])
    assert len(rows) == 7 and source["coverage"]["source_updates"] == 712
    lo, hi = np.log10(rows[0]["lr"]), np.log10(rows[-1]["lr"])
    cmap = plt.get_cmap("plasma_r")
    for r in rows:
        events = r["events"]
        assert [e["update"] for e in events] == list(range(1, 713))
        assert events[-1]["sequences"] == 311
        r["displayed"] = events[:711]
        assert all(e["sequences"] == 1024 for e in r["displayed"])
        r["color"] = cmap(.10 + .85 * (np.log10(r["lr"]) - lo) / (hi - lo))
        r["lightness"] = lightness(r["color"])
        r["hardware_style"] = "--" if r["hardware"] == "Spark" else "-"
        for stream in ("raw", "momentum"):
            for key in ("cumulative_norm_sum", plots.STREAMS[stream][2], "cumulative_ratio", plots.STREAMS[stream][1]):
                assert np.isfinite(plots.values(r["displayed"], stream, key)).all()
    assert all(a["lightness"] > b["lightness"] for a, b in zip(rows, rows[1:]))
    return rows


def legend(fig, rows, hardware_style=False, y=.95):
    handles = [Line2D([], [], color=r["color"], ls=r["hardware_style"] if hardware_style else "-", lw=2.2,
                      label=f"{r['lr']:.6g} ({'S' if r['hardware']=='Spark' else 'R'})") for r in rows]
    title = "Peak LR: light yellow to dark purple as LR increases; S = Spark, R = RTX4090"
    if hardware_style:
        title += "\nDashed = Spark; solid = RTX4090"
    fig.legend(handles=handles, title=title, loc="upper center", bbox_to_anchor=(.5,y), ncol=7,
               frameon=False, fontsize=9, title_fontsize=10, columnspacing=1.2, handlelength=2.2)


def metrics(rows):
    fig, axes = plt.subplots(3, 2, figsize=(14, 13.5))
    x = np.arange(1, 712)
    for r in rows:
        e, color, style = r["displayed"], r["color"], r["hardware_style"]
        loss = np.array([v["loss"] for v in e]); weights = np.array([v["supervised_tokens"] for v in e])
        axes[0,0].plot(x, loss, color=color, alpha=.12, lw=.6)
        axes[0,0].plot(x, plots.smooth(loss, weights), color=color, ls=style, lw=1.8)
        for ax, stream in ((axes[0,1],"raw"),(axes[1,0],"momentum"),(axes[1,1],"adaptive")):
            cos = plots.values(e, stream, "adjacent_cosine")
            ax.plot(x, cos, color=color, alpha=.12, lw=.6)
            ax.plot(x, plots.smooth(cos), color=color, ls=style, lw=1.8)
        for ax, stream in ((axes[2,0],"raw"),(axes[2,1],"momentum")):
            norm = plots.values(e, stream, plots.STREAMS[stream][1])
            assert (norm > 0).all()
            ax.plot(x, norm, color=color, ls=style, lw=1.1)
    titles = ["(a) Training loss", "(b) Raw-gradient adjacent cosine", "(c) Momentum adjacent cosine",
              "(d) Adam-direction adjacent cosine", "(e) Raw-gradient vector norm", "(f) Momentum vector norm"]
    for i, ax in enumerate(axes.flat):
        ax.set_title(titles[i]); ax.set_ylabel("Cross-entropy (nats)" if i==0 else "Cosine" if i in (1,2,3) else "Vector norm")
        if i in (1,2,3):
            ax.set_ylim(-1.05,1.05); ax.axhline(0,color="black",ls=":",lw=.7)
        elif i==4:
            ax.set_yscale("log"); ax.set_ylabel("Vector norm (log scale)")
            ax.grid(which="minor",alpha=.08)
        elif i==5:
            ax.set_ylim(0,None)
    plots.format_axes(axes)
    fig.suptitle("Pythia-14M: seven warmup-only learning rates",fontsize=17,y=.988)
    legend(fig,rows,hardware_style=True,y=.965)
    fig.text(.5,.018,"Light traces: updates; bold loss: trailing 25-update token-weighted mean; bold cosine: trailing 25-update median.\n"
             "Norms are unsmoothed: panel (e) logarithmic Y, panel (f) linear Y. Display 1-711; partial update 712 retained in results.\n"
             "Same frozen initialization/order, seed 1234. Different hardware across S/R; one seed; LR also scales AdamW decay.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.075,1,.90),h_pad=2.1,w_pad=2.1)
    path=OUT/"figures/metrics_by_learning_rate.pdf"; fig.savefig(path); plt.close(fig)
    return path


def cumulative(rows):
    fig,axes=plt.subplots(2,2,figsize=(14,10))
    x=np.arange(1,712)
    for r in rows:
        e,color=r["displayed"],r["color"]
        for col,stream in enumerate(("raw","momentum")):
            axes[0,col].plot(x,plots.values(e,stream,"cumulative_norm_sum"),color=color,ls="-",lw=1.8)
            axes[0,col].plot(x,plots.values(e,stream,plots.STREAMS[stream][2]),color=color,ls="--",lw=1.8)
            axes[1,col].plot(x,plots.values(e,stream,"cumulative_ratio"),color=color,lw=1.8)
    for col,stream in enumerate(("Raw-gradient g","Momentum m")):
        axes[0,col].set(title=f"({'ab'[col]}) {stream}: A and B",ylabel="Cumulative norm",ylim=(0,None))
        axes[1,col].set(title=f"({'cd'[col]}) {stream}: B/A",ylabel="B/A")
        if col == 0:
            axes[1,col].set_ylim(0,1.05)
        else:
            axes[1,col].margins(y=.05)
        axes[0,col].legend(handles=[Line2D([],[],color="#333333",lw=1.8,label="A: sum of norms"),
                                   Line2D([],[],color="#333333",ls="--",lw=1.8,label="B: norm of sum")],loc="upper left",fontsize=10)
    plots.format_axes(axes)
    fig.suptitle("Pythia-14M: cumulative raw-gradient and momentum geometry across learning rates",fontsize=16,y=.985)
    legend(fig,rows,y=.95)
    fig.text(.5,.021,"Colors identify LR using reversed plasma, uniformly sampled in log10(LR); higher LR is darker.\n"
             "Top: solid A = sum of vector norms, dashed B = norm of vector sum. Bottom: B/A; all curves unsmoothed, linear Y.\n"
             "g is raw before clipping; m is stored uncorrected Adam exp_avg from clipped gradients, before preconditioning.\n"
             "Display: updates 1-711; partial update 712 retained in results. Seed 1234; hardware differs across S/R; LR scales decay.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.105,1,.885),h_pad=2.2,w_pad=2.1)
    path=OUT/"figures/geometry_across_learning_rates.pdf"; fig.savefig(path); plt.close(fig)
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure",choices=("metrics","cumulative"))
    args=parser.parse_args(); rows=records()
    path=metrics(rows) if args.figure=="metrics" else cumulative(rows)
    metadata_path=OUT/"requested_figures_data.json"
    metadata=json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata.update(source_comparison_sha256=sha(SOURCE),generator_sha256=sha(__file__),helper_sha256=sha(HELPER),
                    coverage={"source_updates":712,"displayed_updates":[1,711],"partial_update_retained":True},
                    palette={"name":"plasma_r","normalization":"log10(peak_lr)","colormap_range":[.10,.95],
                             "monotonically_decreasing_CIELAB_lightness_verified":True,
                             "conditions":[{"id":r["id"],"lr":r["lr"],"color":to_hex(r["color"]),"CIELAB_L_star":r["lightness"]} for r in rows]})
    metadata.setdefault("figures",{})[path.name]={"sha256":sha(path),"generated_at":datetime.now(timezone.utc).isoformat(),
                       "layout":[3,2] if args.figure=="metrics" else [2,2],
                       "y_scales":["linear","linear","linear","linear","log","linear"] if args.figure=="metrics" else ["linear"]*4,
                       "generator_sha256":sha(__file__)}
    if args.figure == "cumulative":
        metadata["figures"][path.name]["y_limits_policy"] = [
            "zero lower bound; automatic upper bound", "zero lower bound; automatic upper bound",
            "fixed 0-1.05", "automatic displayed-curve range with 5% padding"]
    metadata_path.write_text(json.dumps(metadata,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"figure":str(path),"sha256":sha(path),"palette":"plasma_r","monotonic_lightness":True}))


if __name__=="__main__": main()
