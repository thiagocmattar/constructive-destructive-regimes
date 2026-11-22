"""Render focused curvature and complete extended Analysis013 diagnostics."""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.controlled.rosenbrock import sha256, write_json
from src.controlled.rosenbrock_plot import rolling_cosine
from plot_full import plot

COLORS = ("#0072B2", "#8C4AA8", "#D55E00")
STYLES = ("-", (0, (2, 2)), (0, (7, 3)))
LABELS = (r"$\eta=10^{-2}$", r"$\eta=3.1623\times10^{-3}$", r"$\eta=10^{-3}$")


def digest(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def load():
    cohort = json.loads((HERE / "cohort.json").read_text())
    histories, sources, turns = [], [], []
    for entry in cohort:
        folder = ROOT / entry["directory"]
        receipt = json.loads((folder / "verification.json").read_text())
        history = folder / "artifacts/attempts/001-local/history.npz"
        assert receipt["passed"] and receipt["history_sha256"] == sha256(history)
        assert receipt["config_sha256"] == sha256(folder / "config.json")
        with np.load(history, allow_pickle=False) as saved:
            result = {k: saved[k] for k in saved.files}
        histories.append(result)
        sources.append(dict(entry, history_sha256=sha256(history), config_sha256=sha256(folder / "config.json")))
        dy = np.diff(result["theta"][:, 1])
        turn = int(np.flatnonzero((dy[:-1] < 0) & (dy[1:] > 0))[0] + 1)
        turns.append(dict(run_id=entry["run_id"], learning_rate=entry["learning_rate"], update=turn,
                          point=result["theta"][turn].tolist(), incoming_delta_y=float(dy[turn - 1]),
                          outgoing_delta_y=float(dy[turn])))
    return histories, sources, dict(
        definition="First strict local minimum of y: y[t] < y[t-1] and y[t] < y[t+1]",
        indexing="Position theta[t] after optimizer update t; no shift for vector or rolling panels", points=turns)


def focused(histories, sources, turns):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#26313D", "text.color": "#26313D"})
    fig, grid = plt.subplots(2, 2, figsize=(14, 11))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.81, bottom=0.13, wspace=0.25, hspace=0.35)
    axes = grid.ravel()
    fig.suptitle("Rosenbrock: curvature along the AdamW trajectory", fontsize=20, y=0.965)
    fig.text(0.5, 0.932, "Start (0, 1)  |  2,000 updates  |  betas (0.9, 0.95)  |  weight decay 0.1  |  CPU float64",
             ha="center", fontsize=11)
    handles = [Line2D([], [], color=c, ls=s, lw=2, label=label + f"  (y-turn: {e['update']})")
               for c, s, label, e in zip(COLORS, STYLES, LABELS, turns["points"])]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=3, frameon=False,
               title="Learning rate; dashed vertical references mark first y-turn", title_fontsize=10)
    x = np.linspace(-0.08, 1.12, 500)
    y = np.linspace(0.25, 1.12, 500)
    xx, yy = np.meshgrid(x, y)
    zz = (1 - xx) ** 2 + 100 * (yy - xx ** 2) ** 2
    contours = axes[0].contour(xx, yy, zz, levels=[0.01, 0.1, 1, 5, 10, 25, 50, 100],
                               colors="#BDC8CE", linewidths=0.8)
    axes[0].clabel(contours, levels=[0.1, 1, 10, 100], fontsize=9, fmt="%g")
    axes[0].set(xlim=(-0.08, 1.12), ylim=(0.25, 1.12), xlabel="x", ylabel="y")
    axes[0].set_title("(a) Complete trajectories on objective contours", loc="left", pad=12)
    titles = ("(b) Largest Hessian eigenvalue at every point", "(c) Objective loss",
              "(d) Adjacent cosine: 100-update trailing mean")
    ylabels = (r"$\lambda_{\max}(H(\theta_t))$", r"$f(\theta_t)$", "Cosine")
    for ax, title, label in zip(axes[1:], titles, ylabels):
        ax.set_title(title, loc="left", pad=12)
        ax.set(xlim=(0, 2000), xlabel="Optimizer update t", ylabel=label)
        ax.grid(axis="y", color="#E2E7EC", linewidth=0.75)
        ax.set_axisbelow(True)
    axes[2].set_yscale("log")
    axes[1].set_ylim(150, 1050)
    axes[3].set_ylim(-1.06, 1.06)
    axes[3].axhline(0, color="#9CA7B2", lw=0.8)
    references = []
    for result, color, style, event in zip(histories, COLORS, STYLES, turns["points"]):
        theta = result["theta"]
        assert ((theta >= [-0.08, 0.25]) & (theta <= [1.12, 1.12])).all()
        axes[0].plot(theta[:, 0], theta[:, 1], color=color, ls=style, lw=1.8)
        axes[0].scatter(*theta[-1], s=80, facecolor="white", edgecolor=color, linewidth=1.8, zorder=6)
        axes[0].scatter(*event["point"], marker="D", s=70, facecolor="white", edgecolor=color, linewidth=1.8, zorder=6)
        axes[1].plot(np.arange(2001), result["hessian_lambda_max"], color=color, ls=style, lw=1.8)
        axes[2].plot(np.arange(2001), result["loss"], color=color, ls=style, lw=1.1)
        for stream, stream_style in (("raw", "-"), ("momentum", (0, (5, 3)))):
            axes[3].plot(np.arange(1, 2001), rolling_cosine(result[stream + "_cosine"], 100),
                         color=color, ls=stream_style, lw=1.3)
        for panel, ax in zip("bcd", axes[1:]):
            line = ax.axvline(event["update"], color=color, ls="--", lw=1, alpha=0.7)
            references.append(dict(panel=panel, run_id=event["run_id"], update=int(line.get_xdata()[0]),
                                   color=line.get_color(), linestyle=line.get_linestyle()))
    optimum_curvature = float(np.linalg.eigvalsh([[802., -400.], [-400., 200.]])[-1])
    axes[1].axhline(optimum_curvature, color="#6B7682", ls=":", lw=1.1)
    axes[1].text(75, optimum_curvature + 13, f"At (1, 1): {optimum_curvature:.2f}", fontsize=10, color="#53616E")
    axes[0].scatter(0, 1, marker="s", s=65, color="#26313D", zorder=8)
    axes[0].scatter(1, 1, marker="*", s=130, color="#E9B949", edgecolor="#26313D", zorder=8)
    axes[0].annotate("Start (0, 1)", (0, 1), (0.03, 1.055), fontsize=10)
    axes[0].annotate("Optimum (1, 1)", (1, 1), (0.73, 1.07), fontsize=10)
    axes[0].legend(handles=[Line2D([], [], marker="D", ls="none", markerfacecolor="white", color="#53616E", label="First y-turn"),
                            Line2D([], [], marker="o", ls="none", markerfacecolor="white", color="#53616E", label="Final point")],
                   loc="lower left", frameon=True, fontsize=9)
    axes[3].legend(handles=[Line2D([], [], color="#26313D", lw=1.5, label="Raw gradient"),
                            Line2D([], [], color="#26313D", ls=(0, (5, 3)), lw=1.5, label="Stored momentum")],
                   loc="lower right", frameon=True, fontsize=9)
    fig.text(0.08, 0.062, r"$H_{xx}=2+1200x^2-400y,\quad H_{xy}=H_{yx}=-400x,\quad H_{yy}=200$",
             fontsize=12)
    fig.text(0.08, 0.038, "Curvature and loss use all 2,001 saved positions; no smoothing. Cosines use complete finite windows, first at t = 101.", fontsize=10)
    fig.text(0.08, 0.019, "Objective Hessian excludes decoupled decay. Lambda_max measures local curvature; no AdamW stability threshold is implied.", fontsize=10)
    destination = HERE / "figures/rosenbrock_hessian.pdf"
    destination.parent.mkdir(exist_ok=True)
    (HERE / "scratch").mkdir(exist_ok=True)
    # Hash the actual plotted arrays, including undefined rolling prefixes.
    rendered = [dict(panel=p, points=len(line.get_xdata()), x_sha256=digest(line.get_xdata()),
                     y_sha256=digest(line.get_ydata()))
                for p, ax in zip("abcd", axes) for line in ax.lines if len(line.get_xdata()) > 2]
    fig.savefig(destination)
    fig.savefig(HERE / "scratch/rosenbrock_hessian.png", dpi=130)
    plt.close(fig)
    write_json(HERE / "figure_manifest_rosenbrock_hessian.json", dict(
        figure=destination.relative_to(ROOT).as_posix(), sha256=sha256(destination),
        script=Path(__file__).relative_to(ROOT).as_posix(), script_sha256=sha256(__file__),
        rolling_implementation="src/controlled/rosenbrock_plot.py",
        rolling_implementation_sha256=sha256(ROOT / "src/controlled/rosenbrock_plot.py"),
        sources=sources, panels=4, dimensions_inches=[14, 11], updates_per_run=2000,
        curvature_positions_per_run=2001, curvature_smoothing=None, trajectory_domain=[-0.08, 1.12, 0.25, 1.12],
        initial_point=[0., 1.], trajectory_turns=turns, rendered_series=rendered, vertical_reference_lines=references,
        y_scales=dict(b="linear", c="log", d="linear"), cosine_window=100,
        optimum_lambda_max=optimum_curvature, thinning=None,
    ))
    print(destination)


def main():
    histories, sources, turns = load()
    plot(HERE, title="Rosenbrock: AdamW geometry and exact Hessian curvature", labels=LABELS,
         colors=COLORS, styles=STYLES, cosine_window=100, norm_scale="linear",
         trajectory_turns=turns, figure_stem="rosenbrock_hessian_diagnostics")
    focused(histories, sources, turns)


if __name__ == "__main__":
    main()
