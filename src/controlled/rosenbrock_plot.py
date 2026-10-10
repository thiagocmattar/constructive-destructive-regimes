"""Shared 2,000-update Rosenbrock layout, promoted from frozen Analysis011."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.controlled.rosenbrock import sha256, write_json

COLORS = ("#0072B2", "#D55E00", "#8C4AA8")
STYLES = ("-", (0, (7, 3)), (0, (2, 2)))


def rolling_cosine(values, width):
    """Trailing mean requiring a complete window of finite adjacent cosines."""
    values = np.asarray(values, dtype=np.float64)
    result = np.full_like(values, np.nan)
    if len(values) < width:
        return result
    finite = np.isfinite(values)
    sums = np.convolve(np.where(finite, values, 0), np.ones(width), mode="valid")
    counts = np.convolve(finite.astype(float), np.ones(width), mode="valid")
    result[width - 1:] = np.where(counts == width, sums / width, np.nan)
    return result


def plot(here, *, title, labels, colors=COLORS, styles=STYLES,
         cosine_window=1, norm_scale="log", trajectory_turns=None,
         cohort_entries=None, figure_stem="rosenbrock_learning_rates", noise_sigma=None,
         trajectory_domain=None):
    if not isinstance(cosine_window, int) or cosine_window < 1:
        raise ValueError("cosine_window must be a positive integer")
    if norm_scale not in {"log", "linear"}:
        raise ValueError("norm_scale must be log or linear")
    HERE = Path(here)
    COLORS, STYLES, LABELS = colors, styles, labels
    figure_title = title
    cohort = cohort_entries if cohort_entries is not None else json.loads((HERE / "cohort.json").read_text())
    histories, sources = [], []
    for entry in cohort:
        run_dir = ROOT / entry["directory"]
        receipt = json.loads((run_dir / "verification.json").read_text())
        assert receipt["passed"]
        path = run_dir / "artifacts/attempts/001-local/history.npz"
        assert receipt["history_sha256"] == sha256(path)
        assert receipt["config_sha256"] == sha256(run_dir / "config.json")
        with np.load(path, allow_pickle=False) as saved:
            histories.append({key: saved[key] for key in saved.files})
        sources.append(dict(entry, history_sha256=sha256(path),
                            config_sha256=sha256(run_dir / "config.json")))
    turns = trajectory_turns["points"] if trajectory_turns else None
    if turns:
        assert len(turns) == len(histories)
        for event, source, result in zip(turns, sources, histories):
            assert event["run_id"] == source["run_id"] and event["learning_rate"] == source["learning_rate"]
            assert 0 < event["update"] < len(result["theta"]) - 1
            np.testing.assert_array_equal(event["point"], result["theta"][event["update"]])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 13, "axes.labelsize": 11, "legend.fontsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#66717D", "axes.labelcolor": "#26313D",
                         "text.color": "#26313D", "xtick.color": "#455363", "ytick.color": "#455363",
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(16, 23), facecolor="white")
    grid = fig.add_gridspec(5, 2, height_ratios=[2.55, 1, 1, 1, 1],
                           left=0.09, right=0.92, bottom=0.074, top=0.905, wspace=0.24, hspace=0.47)
    landscape = fig.add_subplot(grid[0, :])
    axes = [fig.add_subplot(grid[row, col]) for row in range(1, 5) for col in range(2)]
    fig.suptitle(figure_title, fontsize=20, y=0.979)
    fig.text(0.5, 0.959, r"2,000 updates per learning rate  ·  $\beta_1=0.9$, $\beta_2=0.95$  ·  CPU float64",
             ha="center", fontsize=12)
    legend_labels = ([label + f" · turn t = {event['update']}" for label, event in zip(LABELS, turns)]
                     if turns else LABELS)
    lr_handles = [Line2D([], [], color=color, ls=style, lw=2.5, label=label)
                  for color, style, label in zip(COLORS, STYLES, legend_labels)]
    fig.legend(handles=lr_handles, loc="upper center", bbox_to_anchor=(0.5, 0.945),
               ncol=3, frameon=False, handlelength=3.5, columnspacing=3.5,
               title=("Learning rate · vertical dashed line at first y-turn" if turns
                      else "Optimization trajectory · learning rate"),
               title_fontsize=10)

    bounds = tuple(trajectory_domain) if trajectory_domain is not None else (-1.4, 1.22, -0.15, 1.48)
    start = histories[0]["theta"][0]
    for result in histories:
        np.testing.assert_array_equal(result["theta"][0], start)
    # Retain the existing landscape domain, expanding only to include full paths.
    all_points = np.concatenate([result["theta"] for result in histories])
    low, high = all_points.min(axis=0), all_points.max(axis=0)
    padding = 0.1 * np.maximum(high - low, 1)
    bounds = (low[0] - padding[0] if low[0] < bounds[0] else bounds[0],
              high[0] + padding[0] if high[0] > bounds[1] else bounds[1],
              low[1] - padding[1] if low[1] < bounds[2] else bounds[2],
              high[1] + padding[1] if high[1] > bounds[3] else bounds[3])
    x = np.linspace(bounds[0], bounds[1], 700)
    y = np.linspace(bounds[2], bounds[3], 550)
    xx, yy = np.meshgrid(x, y)
    zz = (1 - xx) ** 2 + 100 * (yy - xx ** 2) ** 2
    levels = np.array([0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000])
    contour_map = LinearSegmentedColormap.from_list("contours", ["#6EB6AA", "#50839D", "#233D66"])
    landscape.contourf(xx, yy, zz, levels=levels, cmap=contour_map, norm=LogNorm(0.01, 1000), alpha=0.10, extend="both")
    contours = landscape.contour(xx, yy, zz, levels=levels, cmap=contour_map,
                                 norm=LogNorm(0.01, 1000), linewidths=0.85, alpha=0.85)
    landscape.clabel(contours, levels=[0.1, 1, 10, 100, 500], inline=True, fontsize=9, fmt="%g")
    # Each path uses every recorded coordinate, including the common starting point.
    for result, color, style in zip(histories, COLORS, STYLES):
        point = result["theta"]
        assert ((point[:, 0] >= bounds[0]) & (point[:, 0] <= bounds[1]) &
                (point[:, 1] >= bounds[2]) & (point[:, 1] <= bounds[3])).all()
        landscape.plot(point[:, 0], point[:, 1], color=color, ls=style, lw=2.1, zorder=4)
        landscape.scatter(*point[-1], s=150, facecolor="white", edgecolor=color, linewidth=2.3, zorder=6)
        # A finite path segment shows travel direction without a thinned trajectory.
        begin, end = 550, 700
        landscape.annotate("", xy=point[end], xytext=point[begin],
                           arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, mutation_scale=16), zorder=5)
    landscape.scatter(*start, marker="s", s=100, color="#202A35", edgecolor="white", linewidth=1, zorder=8)
    landscape.scatter(1, 1, marker="*", s=150, color="#E9B949", edgecolor="#202A35", linewidth=0.8, zorder=9)
    annotation = dict(fontsize=11, color="#202A35", bbox=dict(facecolor="white", alpha=0.92, edgecolor="none", pad=2))
    landscape.annotate(f"Start ({start[0]:g}, {start[1]:g})", xy=start, xytext=(0.06, 0.42),
                       textcoords="axes fraction",
                       arrowprops=dict(arrowstyle="-", color="#566474"), **annotation)
    landscape.annotate("Optimum (1, 1)", xy=(1, 1), xytext=(0.55, 1.32),
                       arrowprops=dict(arrowstyle="-", color="#566474"), **annotation)
    point_handles = [Line2D([], [], marker="s", color="none", markerfacecolor="#202A35", markersize=8, label="Starting point (t = 0)"),
                     Line2D([], [], marker="*", color="none", markerfacecolor="#E9B949", markeredgecolor="#202A35", markersize=13, label="Global optimum"),
                     Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor="#455363", markeredgewidth=1.8, markersize=8, label="Final point (t = 2,000)")]
    if turns:
        for event, color in zip(turns, COLORS):
            landscape.scatter(*event["point"], marker="D", s=115, facecolor="white",
                              edgecolor=color, linewidth=2, zorder=8)
        point_handles.append(Line2D([], [], marker="D", color="none", markerfacecolor="white",
                                   markeredgecolor="#455363", markeredgewidth=1.8,
                                   markersize=8, label="First y-turn"))
    landscape.legend(handles=point_handles, loc="lower left", frameon=True, framealpha=0.95,
                     edgecolor="#DCE2E7", fontsize=11, borderpad=0.8, labelspacing=0.6,
                     title="Trajectory markers", title_fontsize=11)
    landscape.set(xlim=bounds[:2], ylim=bounds[2:], xlabel="$x$", ylabel="$y$")
    landscape.set_aspect("equal", adjustable="box")
    landscape.set_title("(a) Full optimization trajectories and objective level curves", loc="left", pad=12)
    landscape.spines[["top", "right"]].set_visible(True)
    # Separate color scale for the objective, not the LR palette.
    cax = landscape.inset_axes([1.02, 0.06, 0.015, 0.88])
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=LogNorm(0.01, 1000), cmap=contour_map), cax=cax,
                           ticks=[0.01, 0.1, 1, 10, 100, 1000])
    colorbar.set_label("Level value: $f(x,y)$", labelpad=10)
    colorbar.ax.tick_params(labelsize=10)

    titles = ["(b) Loss", "(c) Raw-gradient adjacent cosine", "(d) Momentum adjacent cosine",
              "(e) Raw-gradient and momentum vector norms", "(f) A and B for raw gradients",
              "(g) A and B for momentum", "(h) B/A for raw gradients", "(i) B/A for momentum"]
    quantities = [r"$f(x_t,y_t)$", r"$\cos(g_{t-1},g_t)$", r"$\cos(m_{t-1},m_t)$",
                  "Euclidean norm (log scale)", "Cumulative norm (log scale)",
                  "Cumulative norm (log scale)", "$B_g/A_g$", "$B_m/A_m$"]
    if cosine_window > 1:
        titles[1] = f"(c) Raw-gradient cosine: {cosine_window}-step trailing mean"
        titles[2] = f"(d) Momentum cosine: {cosine_window}-step trailing mean"
    if noise_sigma is not None:
        titles[1] = titles[1].replace("Raw-gradient", "Noisy-gradient")
        titles[3] = "(e) Noisy-gradient and momentum vector norms"
        titles[4] = "(f) A and B for noisy raw gradients"
        titles[6] = "(h) B/A for noisy raw gradients"
    if norm_scale == "linear":
        quantities[3:6] = ["Euclidean norm", "Cumulative norm", "Cumulative norm"]
    for ax, title, label in zip(axes, titles, quantities):
        ax.set_title(title, loc="left", pad=9)
        ax.set(xlim=(0, 2000), xlabel="Optimizer update t", ylabel=label)
        ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax.grid(axis="y", color="#E2E7EC", linewidth=0.75)
        ax.set_axisbelow(True)
    axes[0].set_yscale("log")
    for ax in axes[3:6]:
        ax.set_yscale(norm_scale)
    for ax in axes[1:3]:
        ax.set_ylim(-1.08, 1.08)
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.axhline(0, color="#9CA7B2", lw=0.8)

    for result, color, style in zip(histories, COLORS, STYLES):
        updates = np.arange(1, 2001)
        axes[0].plot(np.arange(2001), result["loss"], color=color, ls=style, lw=1.5)
        for ax, stream in zip(axes[1:3], ("raw", "momentum")):
            if cosine_window > 1:
                ax.plot(updates, rolling_cosine(result[stream + "_cosine"], cosine_window),
                        color=color, ls=style, lw=1.5)
            else:
                # Joining fast +1/-1 alternation creates an opaque curtain; show every pair as a point.
                ax.scatter(updates, result[stream + "_cosine"], color=color, s=3, alpha=0.65, linewidths=0)
        for stream, quantity_style in (("raw", "-"), ("momentum", (0, (5, 3)))):
            axes[3].plot(updates, result[stream + "_norm"], color=color, ls=quantity_style, lw=1.2)
        for stream, norm_ax, ratio_ax in (("raw", axes[4], axes[6]), ("momentum", axes[5], axes[7])):
            for field, field_style in (("A", "-"), ("B", (0, (5, 3)))):
                norm_ax.plot(updates, result[stream + "_" + field], color=color, ls=field_style, lw=1.5)
            a, b = result[stream + "_A"], result[stream + "_B"]
            ratio = np.divide(b, a, out=np.full_like(a, np.nan), where=a > 0)
            ratio_ax.plot(updates, ratio, color=color, ls=style, lw=1.5)
    axes[3].legend(handles=[Line2D([], [], color="#26313D", lw=1.5, label=r"Raw $\|g_t\|$"),
                            Line2D([], [], color="#26313D", ls=(0, (5, 3)), lw=1.5, label=r"Momentum $\|m_t\|$")], frameon=False)
    for ax in axes[4:6]:
        ax.legend(handles=[Line2D([], [], color="#26313D", lw=1.5, label=r"$A=\sum_{s\leq t}\|z_s\|$"),
                           Line2D([], [], color="#26313D", ls=(0, (5, 3)), lw=1.5, label=r"$B=\|\sum_{s\leq t}z_s\|$")],
                  frameon=True, framealpha=0.95, edgecolor="white",
                  loc="center right" if norm_scale == "linear" else "lower right")
    for ax in axes[6:]:
        ax.set_ylim(0, 1.04)
        ax.axhline(1, color="#9CA7B2", lw=0.8)
    if norm_scale == "linear":
        for ax in axes[3:6]:
            ax.set_ylim(bottom=0)
    reference_lines = []
    if turns:
        for event, color in zip(turns, COLORS):
            for panel, ax in zip("bcdefghi", axes):
                line = ax.axvline(event["update"], color=color, ls="--", lw=1.15, alpha=0.8, zorder=1.8)
                reference_lines.append(dict(panel=panel, run_id=event["run_id"],
                                            update=int(line.get_xdata()[0]), color=line.get_color(),
                                            linestyle=line.get_linestyle()))
    cosine_note = (f"Panels (c,d): {cosine_window}-step trailing means; complete finite windows only. Other series unsmoothed."
                   if cosine_window > 1 else "All updates shown; no smoothing.")
    fig.text(0.09, 0.042, r"$f(x,y)=(1-x)^2+100(y-x^2)^2$.  " + cosine_note, fontsize=10)
    momentum_note = r"$m_t$ is the stored, uncorrected first moment."
    if turns:
        momentum_note += "  Diamonds and LR-colored dashed vertical lines mark the first y-turn."
    fig.text(0.09, 0.032, momentum_note, fontsize=10)
    if noise_sigma is not None:
        context = "Reused deterministic controls." if noise_sigma == 0 else "Noise seed: 1234."
        indexing_note = (r"$g_t=\nabla f(x_{t-1},y_{t-1})+\xi_t$, $\xi_t\sim\mathcal{N}(0,\sigma^2 I_2)$; loss is exact.  "
                         + context + "  First cosine and B/A at A = 0 are undefined.")
    else:
        indexing_note = r"Vectors at update $t$ use $g_t=\nabla f(x_{t-1},y_{t-1})$; loss uses $(x_t,y_t)$.  First adjacent cosines are undefined.  B/A is undefined if A = 0."
    fig.text(0.09, 0.023, indexing_note, fontsize=10)
    destination = HERE / f"figures/{figure_stem}.pdf"
    destination.parent.mkdir(exist_ok=True)
    scratch = HERE / "scratch"
    scratch.mkdir(exist_ok=True)
    fig.savefig(destination)
    fig.savefig(scratch / f"{figure_stem}.png", dpi=130)
    plt.close(fig)
    manifest_name = "figure_manifest.json" if figure_stem == "rosenbrock_learning_rates" else f"figure_manifest_{figure_stem}.json"
    write_json(HERE / manifest_name, {
        "figure": str(destination.relative_to(ROOT)), "sha256": sha256(destination),
        "script": str((HERE / "02_plot.py").relative_to(ROOT)), "script_sha256": sha256(HERE / "02_plot.py"),
        "plot_implementation": str(Path(__file__).relative_to(ROOT)), "plot_implementation_sha256": sha256(Path(__file__)),
        "title": figure_title,
        "sources": sources, "panels": 9, "dimensions_inches": [16, 23], "updates_per_run": 2000,
        "smoothing": (dict(panels=["c", "d"], method="trailing arithmetic mean",
                           window_updates=cosine_window, minimum_valid_values=cosine_window)
                      if cosine_window > 1 else None),
        "thinning": None, "trajectory_domain": list(bounds), "initial_point": start.tolist(),
        "trajectory_turns": trajectory_turns, "vertical_reference_lines": reference_lines,
        "noise_sigma": noise_sigma,
        "y_scales": {panel: ax.get_yscale() for panel, ax in zip("bcdefghi", axes)},
        "adjacent_cosine_rendering": ("connected trailing means; complete finite windows only"
                                      if cosine_window > 1 else "one point per defined pair; no connecting lines"),
        "ratio_orientation": "B/A", "zero_A": "undefined", "first_cosine": "undefined"})
    print(destination)
