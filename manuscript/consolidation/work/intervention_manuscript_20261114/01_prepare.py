"""Journal layouts from frozen H051 reductions; all outputs stay in consolidation."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
EVIDENCE = BASE / "evidence/05-adam/c-intervention"
DATA = EVIDENCE / "data"
FIGURES = HERE / "figures"
TABLES = HERE / "tables"
CASES = ["mlp", "resnet", "transformer"]
ARMS = ["control", "lr_0p75", "lr_0p5", "lr_0p1", "beta2_0p95",
        "beta2_0p9", "beta2_0p5", "clip_0p5", "clip_1p0", "cap_q25", "cap_q50", "cap_q75"]
COLORS = {"control": "#81878d", "lr_0p75": "#80b8dd", "lr_0p5": "#2879b5",
          "lr_0p1": "#083e70", "beta2_0p95": "#f1bd70", "beta2_0p9": "#d78024",
          "beta2_0p5": "#8b4407", "clip_0p5": "#a174b1", "clip_1p0": "#593375",
          "cap_q25": "#85c9ab", "cap_q50": "#23885b", "cap_q75": "#064c32"}
LABEL_COLORS = {**COLORS, "lr_0p75": "#365f7b", "beta2_0p95": "#835210",
                "cap_q25": "#29654e", "clip_0p5": "#704180"}
MARKERS = {"lr": "o", "beta2": "s", "clipping": "^", "cap": "D"}
POINT_LABELS = {"lr_0p75": r"$.75\eta$", "lr_0p5": r"$.5\eta$", "lr_0p1": r"$.1\eta$",
                "beta2_0p95": r"$\beta_2{=}.95$", "beta2_0p9": r"$\beta_2{=}.9$",
                "beta2_0p5": r"$\beta_2{=}.5$", "clip_0p5": "Clip .5", "clip_1p0": "Clip 1",
                "cap_q25": "q25", "cap_q50": "q50", "cap_q75": "q75"}
TITLES = {"mlp": "(a) CIFAR-10 / MLP", "resnet": "(b) CIFAR-10 / ResNet-32",
          "transformer": "(c) WikiText-2 / Transformer"}
STAGE_LABELS = {"mlp": ["Linear 1", "Tanh 1", "Linear 2", "Tanh 2", "Output"],
                "resnet": ["Conv 1", "Stage 1", "Stage 2", "Stage 3", "Pool", "Output"],
                "transformer": ["Embed.", "Attn 1", "Block 1", "Attn 2", "Block 2", "Output"]}

plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.labelsize": 9,
                     "axes.titlesize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "legend.fontsize": 9, "axes.linewidth": .6, "lines.linewidth": 1,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "stixsans"})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=.6, length=2.6, pad=2)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def main():
    for folder in (FIGURES, TABLES, HERE / "data", HERE / "scratch"):
        folder.mkdir(parents=True, exist_ok=True)
    report = json.loads((DATA / "matched-results.json").read_text(encoding="utf-8"))
    assert report["complete_matrix"] and report["conditions"] == 36
    curves = np.load(DATA / "matched-curves.npz")
    rows = {(r["case"], r["arm"]): r for r in report["summary"]}
    assert set(rows) == {(c, a) for c in CASES for a in ARMS}
    events = list(csv.DictReader((DATA / "matched-events.csv").open(encoding="utf-8")))
    manifest = {"status": "generated", "scope": "Presentation only; frozen H051 data and windows",
                "sources": {p.relative_to(BASE).as_posix(): sha(p) for p in
                            [DATA / "matched-results.json", DATA / "matched-curves.npz", DATA / "matched-events.csv"]},
                "generator_sha256": sha(__file__), "figures": [], "table": {}}
    v = lambda c, a, m: curves[f"{c}__{a}__{m}"]

    # Callout coordinates move labels only. Point positions are never jittered.
    layout = {
        "mlp": {"lr_0p75": (.41, .75), "lr_0p5": (.33, .97), "lr_0p1": (.67, .90),
                "beta2_0p95": (.43, .48), "beta2_0p9": (.43, .62), "beta2_0p5": (.69, .33),
                "clip_0p5": (.02, .60), "clip_1p0": (.43, .06),
                "cap_q25": (.02, .76), "cap_q50": (.02, .915), "cap_q75": (.04, .99)},
        "resnet": {"lr_0p75": (.52, .58), "lr_0p5": (.73, .69), "lr_0p1": (.73, .95),
                   "beta2_0p95": (.30, .05), "beta2_0p9": (.02, .40), "beta2_0p5": (.45, .85),
                   "clip_0p5": (.54, .76), "clip_1p0": (.51, .31),
                   "cap_q25": (.20, .83), "cap_q50": (.02, .62), "cap_q75": (.02, .98)},
        "transformer": {"lr_0p75": (.76, .32), "lr_0p5": (.60, .55), "lr_0p1": (.70, .94),
                        "beta2_0p95": (.23, .07), "beta2_0p9": (.16, .27), "beta2_0p5": (.63, .70),
                        "clip_0p5": (.40, .84), "clip_1p0": (.04, .55),
                        "cap_q25": (.34, .96), "cap_q50": (.02, .87), "cap_q75": (.02, .98)}}
    limits = {"mlp": ((-53, 160), (-510, 195)), "resnet": ((-40, 110), (-310, 170)),
              "transformer": ((-24, 3), (-30, 128))}
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 3.65))
    points, annotations = [], []
    for ax, case in zip(axes, CASES):
        ax.set_xlim(*limits[case][0]); ax.set_ylim(*limits[case][1])
        ax.axhline(0, color="#8b9094", lw=.65); ax.axvline(0, color="#8b9094", lw=.65)
        for arm in ARMS[1:]:
            row = rows[case, arm]
            x, y = row["mean_loss_change_percent"], row["pv_reduction_percent"]
            artist = ax.scatter(x, y, s=32, color=COLORS[arm], marker=MARKERS[row["family"]],
                                edgecolors="white", linewidths=.45, zorder=5)
            np.testing.assert_array_equal(artist.get_offsets().data, [[x, y]])
            label = ax.annotate(POINT_LABELS[arm], (x, y), xytext=layout[case][arm],
                                textcoords="axes fraction", fontsize=9, ha="left", va="center",
                                color=LABEL_COLORS[arm], zorder=6,
                                bbox=dict(facecolor="white", edgecolor="none", alpha=.93, pad=.12),
                                arrowprops=dict(arrowstyle="-", color=LABEL_COLORS[arm], lw=.5,
                                                alpha=.7, shrinkA=2, shrinkB=4))
            annotations.append((case, arm, label))
            points.append(dict(case=case, arm=arm, x=x, y=y))
        ax.scatter([0], [0], c=COLORS["control"], marker="x", s=30, zorder=5)
        ax.set_title(TITLES[case].replace(" / ", "\n"), pad=9)
        tidy(ax)
        ax.set_xticks({"mlp": [-50, 0, 50, 100, 150], "resnet": [-25, 0, 50, 100],
                      "transformer": [-20, -10, 0]}[case])
        ax.set_yticks({"mlp": [-400, -200, 0, 100], "resnet": [-300, -200, -100, 0, 100],
                      "transformer": [0, 50, 100]}[case])
    axes[0].set_ylabel("Positive-variation reduction (%)", labelpad=5)
    fig.text(.54, .18, r"Mean training-loss change (%)  $\leftarrow$ lower", ha="center", fontsize=9)
    handles = [Line2D([], [], marker=MARKERS[f], color=COLORS[a], ls="", label=label, markersize=5.5)
               for f, a, label in [("lr", "lr_0p5", "LR reduction"), ("beta2", "beta2_0p9", r"Late $\beta_2$ reduction"),
                                   ("clipping", "clip_1p0", "Gradient clipping"), ("cap", "cap_q50", "Displacement cap")]]
    handles.append(Line2D([], [], marker="x", color=COLORS["control"], ls="", label="Unchanged Adam", markersize=5.5))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.54, .005), ncol=3,
               frameon=False, columnspacing=1.25, handletextpad=.45)
    fig.subplots_adjust(left=.105, right=.985, bottom=.27, top=.87, wspace=.37)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for point in points:
        point["display_xy_pixels"] = list(axes[CASES.index(point["case"])].transData.transform([point["x"], point["y"]]))
    boxes = [{"case": c, "arm": a, "bounds_pixels": list(t.get_bbox_patch().get_window_extent(renderer).bounds)}
             for c, a, t in annotations]
    name = "24-main-intervention-tradeoff.pdf"
    fig.savefig(FIGURES / name, metadata={"Title": "Matched intervention tradeoff", "CreationDate": None})
    fig.savefig(HERE / "scratch/tradeoff.png", dpi=170)
    plt.close(fig)
    manifest["figures"].append(dict(file=name, points=points, labels=boxes,
                                    point_count=33, baseline_count=3, width_inches=7.15,
                                    height_inches=3.65, ordinary_font_points=9))

    # Compact paired maps retain every stage, update, and original denominator.
    fig = plt.figure(figsize=(7.15, 5.7))
    axes = np.empty((5, 3), dtype=object)
    row_bottoms = [.775, .58, .415, .25, .085]
    rel, den, maximum = {}, {}, 1.
    for case in CASES:
        den[case] = v(case, "control", "stage_change_rms")[:, 1:].max(axis=1)
        assert np.all(den[case] > 0)
        for arm in ["control", "cap_q50"]:
            rel[case, arm] = v(case, arm, "stage_change_rms")[:, 1:] / den[case][:, None]
            maximum = max(maximum, float(rel[case, arm].max()))
    map_records, line_records = [], []
    for col, case in enumerate(CASES):
        state = v(case, "control", "state"); x = state[1:]
        stage_labels = STAGE_LABELS[case]
        selected = max([e for e in events if e["case"] == case and e["arm"] == "control" and e["window_height_percent"]],
                       key=lambda e: float(e["window_height_percent"]))
        peak = int(selected["window_peak_state"])
        for row, bottom in enumerate(row_bottoms):
            ax = axes[row, col] = fig.add_axes([.145 + col * .28, bottom, .20, .15 if row == 0 else .14])
            ax.set_xlim(state[0], state[-1]); tidy(ax)
            if row < 4: ax.tick_params(axis="x", labelbottom=False)
        for arm in ["control", "cap_q50"]:
            y = v(case, arm, "loss")
            line, = axes[0, col].plot(state, y, color=COLORS[arm], lw=1.0,
                                      ls="-" if arm == "control" else (0, (4, 1.4)))
            np.testing.assert_array_equal(line.get_ydata(), y)
            np.testing.assert_array_equal(line.get_xdata(), state)
            line_records.append(dict(case=case, arm=arm, n=y.size, y_sha256=array_sha(y), x_sha256=array_sha(state)))
        axes[0, col].set_title(TITLES[case].replace(" / ", "\n"), pad=8)
        axes[0, col].set_ylim(0, max(v(case, a, "loss").max() for a in ["control", "cap_q50"]) * 1.12)
        axes[0, col].yaxis.set_major_locator(MaxNLocator(3, steps=[1, 2, 2.5, 5, 10]))
        matrices = [(a, "stage_rho", v(case, a, "stage_rho")[:, 1:]) for a in ["control", "cap_q50"]]
        matrices += [(a, "relative_rms", rel[case, a]) for a in ["control", "cap_q50"]]
        for row, (arm, metric, values) in enumerate(matrices, 1):
            image = axes[row, col].imshow(values, aspect="auto", interpolation="none",
                       extent=[x[0]-.5, x[-1]+.5, len(stage_labels)-.5, -.5],
                       cmap="RdBu_r" if row < 3 else "viridis", vmin=-1 if row < 3 else 0,
                       vmax=1 if row < 3 else maximum)
            np.testing.assert_array_equal(image.get_array().data, values)
            axes[row, col].set_yticks(range(len(stage_labels)), labels=stage_labels)
            axes[row, col].tick_params(axis="y", length=0)
            map_records.append(dict(case=case, arm=arm, metric=metric, shape=list(values.shape),
                                    values_sha256=array_sha(values), maximum=float(values.max()),
                                    window=[int(state[0]), int(state[-1])], stage_labels=stage_labels,
                                    baseline_peak=peak, rms_denominator=den[case].tolist()))
            if row < 3: cosine_image = image
            else: rms_image = image
        for ax in axes[:, col]:
            reference = ax.axvline(peak, color="#52575b", ls="--", lw=.65)
            reference.set_path_effects([pe.Stroke(linewidth=1.35, foreground="white"), pe.Normal()])
            ax.set_xlim(state[0], state[-1])
            ax.set_xticks([int(state[0]), int((state[0]+state[-1])/2), int(state[-1])])
        axes[-1, col].set_xlabel("Training step $s$", labelpad=3)
    for row, label in enumerate(["Training loss", "Adam: cosine", "q50: cosine", "Adam: rel. RMS", "q50: rel. RMS"]):
        axes[row, 0].set_ylabel(label, labelpad=6)
    fig.legend(handles=[Line2D([], [], color=COLORS["control"], label="Unchanged Adam"),
                        Line2D([], [], color=COLORS["cap_q50"], ls=(0, (4, 1.4)), label="Cap q50"),
                        Line2D([], [], color="#52575b", ls="--", label="Baseline peak reference")],
               loc="center", bbox_to_anchor=(.55, .747), ncol=3, frameon=False,
               handlelength=1.6, columnspacing=1.4)
    cax = fig.add_axes([.934, .415, .012, .305]); fig.colorbar(cosine_image, cax=cax, ticks=[-1, 0, 1])
    cax.tick_params(width=.5, length=2, pad=2)
    cax = fig.add_axes([.934, .085, .012, .305]); fig.colorbar(rms_image, cax=cax, ticks=[0, .5, 1] if maximum == 1 else None)
    cax.tick_params(width=.5, length=2, pad=2)
    name = "25-main-q50-functional-motion.pdf"
    fig.savefig(FIGURES / name, metadata={"Title": "Matched q50 functional motion", "CreationDate": None})
    fig.savefig(HERE / "scratch/motion.png", dpi=170)
    plt.close(fig)
    manifest["figures"].append(dict(file=name, maps=map_records, lines=line_records,
                                    rms_color_maximum=maximum, width_inches=7.15,
                                    height_inches=5.7, ordinary_font_points=9))

    # All values come from the same full-precision summary; no outcome selection.
    labels = ["Adam", r"LR $\times .75$", r"LR $\times .5$", r"LR $\times .1$",
              r"$\beta_2=.95$", r"$\beta_2=.9$", r"$\beta_2=.5$", "Clip .5", "Clip 1",
              "Cap q25", r"\textbf{Cap q50}", "Cap q75"]
    metrics = ["pv_reduction_percent", "mean_loss_change_percent", "final_loss_change_percent", "terminal_test_loss_change_percent"]
    tex = [r"% Generated by work/intervention_manuscript_20261114/01_prepare.py.", r"\begin{table*}[!t]",
           r"\caption{Matched intervention outcomes relative to unchanged Adam (\%).}", r"\label{tab:vc-outcomes}",
           r"\centering\fontsize{9}{10.5}\selectfont", r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{1.06}",
           r"\begin{tabular}{lrrrrrrrrrrrr}", r"\toprule",
           r" & \multicolumn{4}{c}{CIFAR-10 / MLP} & \multicolumn{4}{c}{CIFAR-10 / ResNet-32} & \multicolumn{4}{c}{WikiText-2 / Transformer} \\",
           r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}",
           r"Setting & PV$\uparrow$ & Mean$\downarrow$ & Final$\downarrow$ & Test$\downarrow$ & PV$\uparrow$ & Mean$\downarrow$ & Final$\downarrow$ & Test$\downarrow$ & PV$\uparrow$ & Mean$\downarrow$ & Final$\downarrow$ & Test$\downarrow$ \\", r"\midrule"]
    entries = []
    for arm, label in zip(ARMS, labels):
        if arm in ["lr_0p75", "beta2_0p95", "clip_0p5", "cap_q25"]: tex.append(r"\addlinespace[2pt]")
        values = ["0" if arm == "control" else f"{rows[case, arm][metric]:+.2f}" for case in CASES for metric in metrics]
        tex.append(" & ".join([label] + values) + r" \\")
        entries.append(dict(arm=arm, display_values=values, values=[rows[c, arm][m] for c in CASES for m in metrics]))
    tex += [r"\bottomrule", r"\end{tabular}", r"\par\vspace{3pt}\begin{minipage}{\textwidth}\fontsize{9}{10.5}\selectfont",
            r"PV reports reduction in positive loss variation; other columns report loss changes. Arrows mark the favorable direction. Mean excludes the common fork; Final/Test use the fixed endpoint. Windows: 512--800 / 256--576 / 544--928. One initialization/model; bold marks preselected q50. No seed-level uncertainty is estimated.\par",
            r"Adam absolute references (PV, Mean, Final, Test):"]
    for case, title in zip(CASES, ["MLP", "ResNet-32", "Transformer"]):
        baseline = rows[case, "control"]
        numbers = [f"{baseline[m]:.5g}" for m in ["positive_variation", "mean_loss", "final_loss", "terminal_test_loss"]]
        tex.append(title + ": (" + ", ".join(numbers) + r").\par")
    tex += [r"\end{minipage}", r"\end{table*}", ""]
    table_path = TABLES / "matched-main-outcomes.tex"
    table_path.write_text("\n".join(tex), encoding="utf-8", newline="\n")
    manifest["table"] = dict(file=table_path.relative_to(HERE).as_posix(), rows=entries, metrics=metrics,
                              body_font_points=9, body_cells=144, sha256=sha(table_path))
    for entry in manifest["figures"]: entry["sha256"] = sha(FIGURES / entry["file"])
    dump(HERE / "data/presentation_manifest.json", manifest)
    print(json.dumps({"figures": len(manifest["figures"]), "table_cells": 144,
                      "plotted_values": sum(m["shape"][0]*m["shape"][1] for m in map_records) + sum(l["n"] for l in line_records) + 66,
                      "rms_color_maximum": maximum}))


if __name__ == "__main__":
    main()
