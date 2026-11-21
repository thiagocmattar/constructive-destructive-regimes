"""Redraw selected, already measured manuscript series at lecture size."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE.parent / "work"
ANALYSES = HERE.parents[2] / "analyses"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
INK, TEAL, ORANGE, BLUE = "#172f40", "#007568", "#b64a20", "#286ca1"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "axes.labelsize": 13,
    "axes.labelcolor": INK, "axes.edgecolor": "#99a7aa", "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.right": False,
    "axes.spines.top": False, "legend.frameon": False, "pdf.fonttype": 42,
    "axes.grid": True, "grid.alpha": .18, "axes.axisbelow": True,
    "axes.titlelocation": "left", "axes.titlesize": 14,
})


def save(fig, name):
    fig.savefig(OUT / (name + ".pdf"), bbox_inches="tight", pad_inches=.05,
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def plot_series(ax, arrays, key, **style):
    ax.plot(arrays[key + ":x"], arrays[key + ":y"], **style)


# An exact, one-parameter example. Both panels display the same GD iterations.
fig, (pos, loss) = plt.subplots(1, 2, figsize=(8.6, 2.6), layout="constrained")
t = np.arange(10)
for rate, color in [(.4, TEAL), (1.8, ORANGE)]:
    w = (1-rate)**t
    pos.plot(t, w, "o-", color=color, ms=4, label=fr"$\eta={rate}$")
    loss.semilogy(t, .5*w*w, "o-", color=color, ms=4)
pos.axhline(0, color=INK, lw=.8)
pos.legend(loc="lower right", fontsize=11)
pos.set(ylabel="Weight", xlabel="GD update")
loss.set(ylabel="Loss (log scale)", xlabel="GD update")
save(fig, "quadratic")

with np.load(WORK / "figure_revision_20261114/plotted_arrays.npz") as a:
    fig, (l, c) = plt.subplots(1, 2, figsize=(9.1, 2.8), layout="constrained")
    for rid, color, label in [("059", TEAL, "0.004"), ("122", ORANGE, "0.04")]:
        plot_series(l, a, "GD:loss:"+rid, color=color, lw=1.2, label=label)
    plot_series(c, a, "GD:Q:centered:122", color=ORANGE, lw=2, label="Along the step")
    plot_series(c, a, "GD:spectrum:centered:122", color=BLUE, ls="--", lw=1.8,
                label="Sharpest direction")
    c.axhline(2, color=INK, ls=":", lw=1.1)
    l.set(xlabel="GD update", ylabel="Training loss")
    c.set(xlabel="GD update", ylabel="Scaled curvature", ylim=(0,3.25))
    l.legend(title="Learning rate", fontsize=10)
    c.legend(fontsize=10, loc="lower right")
    save(fig, "gd-curvature")

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 2.7), layout="constrained")
    for ax, metric, label in zip(axes, ["rho_smoothed", "gradient_norms_smoothed"],
                                 ["Gradient cosine", "Gradient norm"]):
        plot_series(ax, a, "GD:"+metric+"_run122", color=TEAL, lw=1.7)
        ax.set(xlabel="GD update", ylabel=label)
    axes[0].axhline(0, color=INK, ls=":", lw=1)
    axes[0].set_ylim(-1.08, 1.08)
    save(fig, "gd-reversal")

    fig, ax = plt.subplots(figsize=(7.5, 2.45), layout="constrained")
    lag, cos = a["backward:mlp:32:x"], a["backward:mlp:32:y"]
    ax.axhline(0, color=INK, lw=.9)
    for odd, color, label in [(True, ORANGE, "Odd lag"), (False, TEAL, "Even lag")]:
        take = (lag % 2 == int(odd)) & (lag <= 12)
        ax.vlines(lag[take], 0, cos[take], color=color, lw=2)
        ax.scatter(lag[take],cos[take],color=color,s=28,label=label)
    ax.set(xlabel="Updates back from step 2,256", ylabel="Gradient cosine",
           ylim=(-1.08,1.08), xticks=[1,2,4,6,8,10,12])
    ax.legend(loc="upper right", ncol=2, fontsize=10)
    save(fig, "backward")

    fig, (l, c) = plt.subplots(1,2, figsize=(9.1,2.7), layout="constrained")
    plot_series(l,a,"Adam:resnet:0",color=INK,lw=1.1)
    l.set(xlabel="Adam update",ylabel="Training loss",yscale="log")
    l.axvspan(390,450,color=ORANGE,alpha=.18)
    x, y = a["Adam:resnet:2:x"], a["Adam:resnet:2:y"]
    take=(x>=390)&(x<=450)
    c.plot(x[take],y[take],lw=1.7,color=TEAL)
    c.axvline(422,color=ORANGE,lw=1.4,ls="--",label="Local loss peak")
    c.axhline(0,color=INK,lw=.7,ls=":")
    c.set(xlabel="Adam update",ylabel="Gradient cosine",title="The shaded interval",ylim=(-1.05,1.15))
    c.legend(loc="lower right",fontsize=9)
    save(fig,"adam")

# Use the seed-0 table underlying the CURRENT Figure 4, including its full-GD endpoint.
with (ANALYSES / "037-2026-11-02-sgd-batch-extension/seed0_cells.csv").open() as f:
    samples = [r for r in csv.DictReader(f) if r["model"] == "transformer"]
fig, ax = plt.subplots(figsize=(7.6,2.65),layout="constrained")
for state, color, label in [("0",TEAL,"From initialization"),("2000",ORANGE,"After 2,000 GD updates")]:
    cells=sorted((r for r in samples if r["state"]==state),key=lambda r:int(r["plotted_batch"]))
    x=np.array([int(r["plotted_batch"]) for r in cells])
    y=np.array([float(r["mean"]) for r in cells])
    ax.semilogx(x,y,"o-",color=color,lw=2,ms=5,label=label)
    ax.plot(x[-1],y[-1],"o",mfc="white",mec=color,ms=7)
ax.axhline(0,color=INK,lw=.8)
ax.set(xticks=x,xticklabels=["8","32","128","512","Full GD"],ylim=(-1.12,1.12),
       xlabel="Sequences per update",ylabel="Mean gradient cosine")
ax.legend(loc="upper left",fontsize=10)
save(fig,"sampling")

with np.load(WORK / "resnet_peak_revision_20261114/plotted_arrays.npz") as m:
    x=m["trace:resnet:full-loss:x"]
    fig, axes=plt.subplots(1,2,figsize=(9.2,2.75),layout="constrained",sharey=True)
    for ax,idx,title,cmap,low in zip(axes,[1,2],["Consecutive-change cosine","Size relative to each site's peak"],
                                      ["RdBu_r","viridis"],[-1,0]):
        data=m[f"heatmap:resnet:{idx}"]
        im=ax.imshow(data,extent=(x[0]-.5,x[-1]+.5,5.5,-.5),aspect="auto",interpolation="nearest",
                     cmap=cmap,vmin=low,vmax=1)
        ax.grid(False)
        ax.axvline(415,color="black",lw=1.2,ls="--")
        ax.set(title=title,xlabel="Adam update",yticks=range(6),
               yticklabels=["Conv 1","Group 1","Group 2","Group 3","Pooling","Output"])
        fig.colorbar(im,ax=ax,shrink=.8,ticks=[low,1])
    save(fig,"hidden")

with np.load(WORK / "resnet_temporal_revision_20261114/plotted_arrays.npz") as a:
    fig, axes = plt.subplots(1,3,figsize=(9.4,2.4),layout="constrained")
    for ax,case,title in zip(axes,["mlp","resnet","transformer"],["MLP","ResNet","Transformer"]):
        plot_series(ax,a,"output-coefficient:"+case,color=ORANGE,lw=1.5)
        ax.axhline(0,color=INK,lw=.8)
        ax.set(title=title,xlabel="Adam update")
    axes[0].set_ylabel("Signed change")
    save(fig,"output-direction")

# Matched ResNet continuations from Figure 8. Normalize the full-precision
# metrics first; labels derived from rounded table entries would shift the dots.
summary=json.loads((HERE.parent/"evidence/05-adam/c-intervention/data/matched-results.json").read_text())
rows={r["arm"]:r for r in summary["summary"] if r["case"]=="resnet"}
control=rows["control"]
fig, ax=plt.subplots(figsize=(7.6,2.8),layout="constrained")
ax.axvline(1,color="#939d9f",lw=1)
ax.axhline(1,color="#939d9f",lw=1)
for arm, label, marker, color, offset in [
        ("control","Unchanged Adam","x",INK,(7,6)),
        ("cap_q25","q25 cap","D",TEAL,(-4,-17)),
        ("lr_0p1","10-fold lower learning rate","o",ORANGE,(-150,5))]:
    x=rows[arm]["final_loss"]/control["final_loss"]
    y=rows[arm]["positive_variation"]/control["positive_variation"]
    ax.scatter(x,y,s=62,marker=marker,color=color,zorder=3)
    ax.annotate(label,(x,y),xytext=offset,textcoords="offset points",color=color,fontsize=12)
ax.set(xscale="log",yscale="log",xlim=(.53,51),ylim=(.006,2.7),
       xlabel="Final loss / Adam final loss",ylabel="Upward variation\n/ Adam variation")
ax.set_xticks([1,2,5,10,40],labels=["1","2","5","10","40"])
ax.set_yticks([.01,.1,1],labels=["0.01","0.1","1"])
save(fig,"tradeoff")
