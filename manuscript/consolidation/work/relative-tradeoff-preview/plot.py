"""Preview 11 matched interventions per model relative to unchanged Adam.

Inputs are the full-precision matched outcomes and training-loss trajectories.
Each panel uses one initialization and the same update budget for all arms.
The plot compares observed ratios within each model, without uncertainty bars.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "evidence/05-adam/c-intervention/data"
records = json.loads((DATA / "matched-results.json").read_text())["summary"]
controls = {r["case"]: r for r in records if r["arm"] == "control"}
cases = ["mlp", "resnet", "transformer"]
titles = ["(a) CIFAR-10 / MLP", "(b) CIFAR-10 / ResNet-32", "(c) WikiText-2 / Transformer"]
markers = {"lr": "o", "beta2": "s", "clipping": "^", "cap": "D", "control": "x"}
cap_colors = {"cap_q25": "#85c9ab", "cap_q50": "#23885b", "cap_q75": "#064c32"}
plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix",
                     "font.size": 8, "axes.linewidth": .6, "pdf.fonttype": 42})
x_ticks = {"mlp": [.3, .5, .75, 1], "resnet": [1, 4, 16, 32], "transformer": [.6, .7, .8, .9, 1]}
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.95), sharey=True)
with np.load(DATA / "matched-curves.npz") as curves:
    for ax, case, title in zip(axes, cases, titles):
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=.005, linscale=.65)
        ax.margins(x=.08)
        ax.set_ylim(-.0009, 8)
        ax.axvline(1, color=".65", lw=.65)
        ax.axhline(1, color=".65", lw=.65)
        control = controls[case]
        reference = np.array([control["final_loss"], control["positive_variation"]])
        assert np.all(reference > 0)
        group = [r for r in records if r["case"] == case]
        assert len(group) == 12
        for r in group:
            arm = r["arm"]
            loss = curves[f"{case}__{arm}__loss"]
            metrics = np.array([r["final_loss"], r["positive_variation"]])
            np.testing.assert_allclose(metrics, [loss[-1], np.maximum(np.diff(loss), 0).sum()],
                                       rtol=2e-12, atol=2e-12)
            x, y = metrics / reference
            family = "control" if arm == "control" else r["family"]
            color = ".1" if family == "control" else cap_colors.get(arm, ".55")
            ax.scatter(x, y, marker=markers[family], s=24, color=color,
                       linewidths=.8, zorder=4 if family in ("control", "cap") else 3)
            if arm == "control":
                assert (x, y) == (1, 1)
            if family == "cap":
                offset = (5, -7) if case == "resnet" and arm == "cap_q25" else (5, 2)
                if case == "resnet" and arm == "cap_q75":
                    offset = (5, -7)
                ax.annotate(arm[4:], (x, y), xytext=offset, textcoords="offset points",
                            fontsize=7, color=".18")
        ax.set_title(title, fontsize=9, pad=6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks(x_ticks[case], labels=[f"{tick:g}" for tick in x_ticks[case]])
        ax.set_yticks([0, .01, .1, 1, 5], labels=["0", "0.01", "0.1", "1", "5"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(length=3, pad=2)
axes[0].set_ylabel(r"Upward loss variation  $\mathcal{P}/\mathcal{P}^{\mathrm{Adam}}$")
fig.supxlabel(r"Final training loss  $L_{t_f}/L_{t_f}^{\mathrm{Adam}}$", fontsize=9, y=.11)
handles = [Line2D([], [], ls="", marker=markers[f], color=c, markersize=4.4, label=l)
           for f, c, l in [("control", ".1", "Unchanged Adam"), ("lr", ".55", "LR reduction"),
                           ("beta2", ".55", r"$\beta_2$ reduction"), ("clipping", ".55", "Gradient clipping"),
                           ("cap", cap_colors["cap_q50"], "Displacement cap")]]
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, handletextpad=.25,
           columnspacing=1.3, fontsize=7.5)
fig.subplots_adjust(left=.094, right=.987, bottom=.26, top=.91, wspace=.16)
fig.savefig(HERE / "relative-tradeoff-preview.pdf", metadata={"CreationDate": None, "ModDate": None})
