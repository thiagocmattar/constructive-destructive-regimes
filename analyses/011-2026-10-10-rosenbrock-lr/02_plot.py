"""One full trajectory panel and the eight requested unsmoothed metric panels."""
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
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.controlled.rosenbrock import sha256, write_json

COLORS = ("#0072B2", "#D55E00", "#8C4AA8")
STYLES = ("-", (0, (7, 3)), (0, (2, 2)))
LABELS = (r"$\eta=10^{-2}$", r"$\eta=10^{-3}$", r"$\eta=10^{-4}$")


def main():
    cohort = json.loads((HERE / "cohort.json").read_text())
    histories, sources = [], []
    for entry in cohort:
        run_dir = ROOT / entry["directory"]
        assert json.loads((run_dir / "verification.json").read_text())["passed"]
        path = run_dir / "artifacts/attempts/001-local/history.npz"
        with np.load(path, allow_pickle=False) as saved:
            histories.append({key: saved[key] for key in saved.files})
        sources.append(dict(entry, history_sha256=sha256(path),
                            config_sha256=sha256(run_dir / "config.json")))
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
    fig.suptitle("Rosenbrock · Adam with momentum, no weight decay", fontsize=20, y=0.979)
    fig.text(0.5, 0.959, r"2,000 updates per learning rate  ·  $\beta_1=0.9$, $\beta_2=0.95$  ·  CPU float64",
             ha="center", fontsize=12)
    lr_handles = [Line2D([], [], color=color, ls=style, lw=2.5, label=label)
                  for color, style, label in zip(COLORS, STYLES, LABELS)]
    fig.legend(handles=lr_handles, loc="upper center", bbox_to_anchor=(0.5, 0.945),
               ncol=3, frameon=False, handlelength=3.5, columnspacing=3.5,
               title="Optimization trajectory · learning rate",
               title_fontsize=10)

    bounds = (-1.4, 1.22, -0.15, 1.48)
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
    landscape.scatter(-1.2, 1, marker="s", s=100, color="#202A35", edgecolor="white", linewidth=1, zorder=8)
    landscape.scatter(1, 1, marker="*", s=150, color="#E9B949", edgecolor="#202A35", linewidth=0.8, zorder=9)
    annotation = dict(fontsize=11, color="#202A35", bbox=dict(facecolor="white", alpha=0.92, edgecolor="none", pad=2))
    landscape.annotate("Start (−1.2, 1)", xy=(-1.2, 1), xytext=(-1.30, 0.72),
                       arrowprops=dict(arrowstyle="-", color="#566474"), **annotation)
    landscape.annotate("Optimum (1, 1)", xy=(1, 1), xytext=(0.55, 1.32),
                       arrowprops=dict(arrowstyle="-", color="#566474"), **annotation)
    point_handles = [Line2D([], [], marker="s", color="none", markerfacecolor="#202A35", markersize=8, label="Starting point (t = 0)"),
                     Line2D([], [], marker="*", color="none", markerfacecolor="#E9B949", markeredgecolor="#202A35", markersize=13, label="Global optimum"),
                     Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor="#455363", markeredgewidth=1.8, markersize=8, label="Final point (t = 2,000)")]
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
    for ax, title, label in zip(axes, titles, quantities):
        ax.set_title(title, loc="left", pad=9)
        ax.set(xlim=(0, 2000), xlabel="Optimizer update t", ylabel=label)
        ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax.grid(axis="y", color="#E2E7EC", linewidth=0.75)
        ax.set_axisbelow(True)
    for index in (0, 3, 4, 5):
        axes[index].set_yscale("log")
    for ax in axes[1:3]:
        ax.set_ylim(-1.08, 1.08)
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.axhline(0, color="#9CA7B2", lw=0.8)

    for result, color, style in zip(histories, COLORS, STYLES):
        updates = np.arange(1, 2001)
        axes[0].plot(np.arange(2001), result["loss"], color=color, ls=style, lw=1.5)
        for ax, stream in zip(axes[1:3], ("raw", "momentum")):
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
                  frameon=True, framealpha=0.95, edgecolor="white", loc="lower right")
    for ax in axes[6:]:
        ax.set_ylim(0, 1.04)
        ax.axhline(1, color="#9CA7B2", lw=0.8)
    fig.text(0.09, 0.037, r"$f(x,y)=(1-x)^2+100(y-x^2)^2$.  All updates shown; no smoothing.  $m_t$ is the stored, uncorrected first moment.", fontsize=11)
    fig.text(0.09, 0.023, r"Vectors at update $t$ use $g_t=\nabla f(x_{t-1},y_{t-1})$; loss uses $(x_t,y_t)$.  First adjacent cosines are undefined.  B/A is undefined if A = 0.", fontsize=10)
    destination = HERE / "figures/rosenbrock_learning_rates.pdf"
    destination.parent.mkdir(exist_ok=True)
    scratch = HERE / "scratch"
    scratch.mkdir(exist_ok=True)
    fig.savefig(destination)
    fig.savefig(scratch / "rosenbrock_learning_rates.png", dpi=130)
    plt.close(fig)
    write_json(HERE / "figure_manifest.json", {
        "figure": str(destination.relative_to(ROOT)), "sha256": sha256(destination),
        "script": str(Path(__file__).relative_to(ROOT)), "script_sha256": sha256(Path(__file__)),
        "sources": sources, "panels": 9, "dimensions_inches": [16, 23], "updates_per_run": 2000,
        "smoothing": None, "thinning": None, "trajectory_domain": list(bounds),
        "adjacent_cosine_rendering": "one point per defined pair; no connecting lines",
        "ratio_orientation": "B/A", "zero_A": "undefined", "first_cosine": "undefined"})
    print(destination)


if __name__ == "__main__":
    main()
