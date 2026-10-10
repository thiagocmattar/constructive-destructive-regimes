"""Publication PDFs: trajectories, all collector streams, squared/temporal geometry."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
COLORS = ["#2563a6", "#c57618", "#20916f", "#a84988"]
STYLES = ["-", "--", "-.", ":"]
STREAMS = ["raw", "momentum", "adaptive", "displacement"]
SYMBOLS = ["g", "m", "d", "u"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "axes.titlesize": 11})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records():
    rows = []
    for entry in json.loads((HERE / "cohort.json").read_text()):
        folder = ROOT / entry["directory"]
        receipt = json.loads((folder / "verification.json").read_text())
        assert receipt["passed"]
        path = folder / "artifacts/attempts/001-local/history.npz"
        assert sha(path) == receipt["history_sha256"]
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        rows.append(dict(entry, folder=folder, data=data, path=path,
                         summary=json.loads((path.parent / "summary.json").read_text())))
    return rows


def contours(ax, xlim=(-1.5, 1.35), ylim=(-.35, 1.6), levels=None):
    x = np.linspace(*xlim, 500)
    y = np.linspace(*ylim, 500)
    xx, yy = np.meshgrid(x, y)
    z = (1 - xx) ** 2 + 100 * (yy - xx ** 2) ** 2
    if levels is None:
        levels = [.001, .01, .1, .5, 1, 2, 5, 10, 25, 50, 100, 250, 500]
    ax.contour(xx, yy, z, levels=levels, cmap="Greys", norm=LogNorm(vmin=min(levels), vmax=max(levels)), linewidths=.65, alpha=.65)
    ax.plot(1, 1, "*", color="black", ms=12, label="Global minimum (1,1)")
    ax.set(xlim=xlim, ylim=ylim, xlabel="x coordinate", ylabel="y coordinate")
    ax.set_aspect("equal", adjustable="box")


def smooth(values, width=100):
    """Trailing mean, undefined until the complete window is available."""
    result = np.full(len(values), np.nan)
    finite = np.isfinite(values)
    summed = np.convolve(np.where(finite, values, 0), np.ones(width), mode="valid")
    counts = np.convolve(finite.astype(float), np.ones(width), mode="valid")
    result[width - 1:] = np.where(counts == width, summed / width, np.nan)
    return result


def finish(fig, axes, title, note, path):
    for ax in np.asarray(axes).flat:
        if ax.get_xlabel() == "":
            ax.set_xlabel("Optimizer update")
        ax.grid(alpha=.18)
    fig.suptitle(title, y=.99, fontsize=15)
    fig.text(.5, .012, note, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, .065, 1, .89 if fig.legends else .955), h_pad=2, w_pad=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Title": title})
    scratch = HERE / "scratch"
    scratch.mkdir(exist_ok=True)
    fig.savefig(scratch / (path.parent.parent.name + "-" + path.stem + ".png"), dpi=110)
    plt.close(fig)
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(path)}


def main():
    rows = records()
    outputs = []
    for row in rows:
        d = row["data"]
        t = np.arange(1, len(d["loss"]))
        fig, axes = plt.subplots(3, 2, figsize=(13, 12))
        contours(axes[0, 0])
        axes[0, 0].plot(*d["theta"].T, color=COLORS[0], lw=1, label="Trajectory")
        axes[0, 0].plot(*d["theta"][0], "s", color=COLORS[0], ms=6, label="Start")
        axes[0, 0].plot(*d["theta"][-1], "o", mfc="none", mec=COLORS[0], ms=9, label="Final")
        axes[0, 0].set_title("(a) Rosenbrock contours and all 10,000 steps")
        axes[0, 0].legend(fontsize=8, loc="lower left")
        axes[0, 1].semilogy(np.arange(len(d["loss"])), d["loss"], lw=.8, color=COLORS[0])
        axes[0, 1].set(title="(b) Objective after update", ylabel="f(x,y), log scale")
        for name, symbol, color, style in zip(STREAMS, SYMBOLS, COLORS, STYLES):
            axes[1, 0].plot(t, smooth(d[name + "_cosine"]), color=color, ls=style, label=symbol, lw=1)
            axes[1, 1].semilogy(t, d[name + "_norm"], color=color, ls=style, label=symbol, lw=.7)
            axes[2, 0].semilogy(t, d[name + "_A"], color=color, ls="-", lw=1.4, label=symbol + ": A")
            axes[2, 0].semilogy(t, d[name + "_B"], color=color, ls="--", lw=1.2, label=symbol + ": B")
            axes[2, 1].plot(t, d[name + "_ratio"], color=color, ls=style, label=symbol, lw=1.2)
        axes[1, 0].axhline(0, color="gray", lw=.6)
        axes[1, 0].set(title="(c) Adjacent cosine: trailing 100-update mean", ylabel="Cosine (dimensionless)", ylim=(-1.05, 1.05))
        axes[1, 1].set(title="(d) Individual vector norms", ylabel="L2 norm (stream units), log")
        axes[2, 0].set(title="(e) A = sum of norms; B = norm of sum", ylabel="Cumulative norm (stream units), log")
        axes[2, 1].set(title="(f) Cumulative B/A", ylabel="B/A (dimensionless)", ylim=(-.02, 1.02))
        for ax in axes.flat[2:]:
            ax.legend(fontsize=8, ncol=2)
        outputs.append(finish(fig, axes, f"{row['run_id']} | {row['label']}",
            "g: exact raw gradient; m: uncorrected first moment; d: adaptive direction without LR/decay; u: actual displacement.\n"
            "Loss: positions 0-10,000; diagnostics: transitions 1-10,000. First cosine undefined; means require 100 valid pairs.\n"
            "LR 0.01; beta2 0.95; float64; exact objective; one deterministic start; no uncertainty bars.",
            row["folder"] / "figures/trajectory_geometry.pdf"))

    fig, axes = plt.subplots(3, 2, figsize=(13, 11.5))
    contours(axes[0, 0])
    contours(axes[0, 1], (.70, 1.035), (.50, 1.045), [1e-7, 1e-5, 1e-3, .01, .05, .1, .5, 1, 2, 5, 10])
    axes[0, 0].plot(-1.2, 1, "s", color="black", ms=6)
    for row, color, style in zip(rows, COLORS, STYLES):
        d = row["data"]
        t = np.arange(1, len(d["loss"]))
        label = row["run_id"] + ": " + row["label"]
        for ax in axes[0]:
            ax.plot(*d["theta"].T, color=color, ls=style, lw=.85)
            ax.plot(*d["theta"][-1], "o", color=color, ms=5)
        axes[1, 0].semilogy(np.arange(len(d["loss"])), d["loss"], color=color, ls=style, lw=.7, label=label)
        axes[1, 1].plot(t, smooth(d["raw_cosine"]), color=color, ls=style, lw=1)
        axes[2, 0].plot(t, smooth(d["momentum_cosine"]), color=color, ls=style, lw=1)
        axes[2, 1].semilogy(t, d["raw_norm"], color=color, ls=style, lw=.7)
    titles = ["(a) Full trajectories: shared start", "(b) Close-up near the minimum: final circles",
              "(c) Objective after update", "(d) Raw adjacent cosine, trailing mean (100)",
              "(e) Momentum adjacent cosine, trailing mean (100)", "(f) Raw gradient norm"]
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
    axes[1, 0].set_ylabel("f(x,y), log scale")
    for ax in (axes[1, 1], axes[2, 0]):
        ax.set(ylim=(-1.05, 1.05), ylabel="Cosine (dimensionless)")
        ax.axhline(0, color="gray", lw=.6)
    axes[2, 1].set_ylabel("L2 norm, log scale")
    fig.legend(*axes[1, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .953), ncol=2, fontsize=8)
    outputs.append(finish(fig, axes, "Controlled Rosenbrock: momentum × decoupled decay",
        "All 10,000 updates and identical (-1.2,1) starts; LR=0.01, beta2=0.95, epsilon=1e-8.\n"
        "Independent trajectories; one objective/start/LR. Final loss is not proof of sustained convergence. Raw cosine alone does not diagnose loss improvement.",
        HERE / "figures/controlled_comparison.pdf"))

    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    for row, color, style in zip(rows, COLORS, STYLES):
        d = row["data"]
        t = np.arange(1, len(d["loss"]))
        for col, name in enumerate(("raw", "momentum")):
            for r, quantity in enumerate(("A_squared", "B_squared", "triangle_gap")):
                values = d[name + "_" + quantity]
                axes[r, col].semilogy(t, np.ma.masked_less_equal(values, 0), color=color, ls=style, lw=1,
                                     label=row["run_id"] + ": " + row["label"])
                axes[r, col].set(title=f"{'Raw g' if col == 0 else 'Momentum m'}: {['A squared','B squared','A squared minus B squared'][r]}",
                                 ylabel="Squared norm, log scale")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .953), ncol=2, fontsize=8)
    outputs.append(finish(fig, axes, "Rosenbrock: Analysis009 squared cumulative geometry",
        "Exact full prefixes, updates 1-10,000. A²-B² is a nonnegative triangle gap, not the signed pair interaction.\n"
        "Independent log Y scales; zero/nonpositive values masked only for display, no epsilon replacement. One deterministic trajectory per condition.",
        HERE / "figures/squared_geometry.pdf"))

    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    for row, color, style in zip(rows, COLORS, STYLES):
        d = row["data"]
        t = np.arange(1, len(d["loss"]))
        for col, name in enumerate(("raw", "momentum")):
            for r, quantity in enumerate(("all_pair", "lag_one", "beyond_lag_one")):
                axes[r, col].plot(t, d[name + "_" + quantity], color=color, ls=style, lw=1,
                                  label=row["run_id"] + ": " + row["label"])
                axes[r, col].set(title=f"{'Raw g' if col == 0 else 'Momentum m'}: {['I = B² - sum squared norms','Lag-one interaction L1','Beyond-lag-one residual I - L1'][r]}",
                                 ylabel="Signed squared norm")
                axes[r, col].axhline(0, color="gray", lw=.6)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .953), ncol=2, fontsize=8)
    outputs.append(finish(fig, axes, "Rosenbrock: exact pair interaction versus lag-one truncation",
        "I=2 sum over all i<j of dot(z_i,z_j); L1=2 sum of adjacent dot products. Exact full histories, all 10,000 updates.\n"
        "A negative adjacent cosine does not imply a negative complete-history interaction. No long-lag approximation or stochastic uncertainty.",
        HERE / "figures/exact_temporal_interaction.pdf"))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for row, color, style in zip(rows, COLORS, STYLES):
        d = row["data"]
        t = np.arange(1, len(d["loss"]))
        for ax, quantity in zip(axes.flat, ("momentum_to_raw_norm_ratio", "adaptive_to_raw_norm_ratio", "adaptive_to_momentum_norm_ratio", "post_gradient_norm")):
            values = d[quantity][1:] if quantity == "post_gradient_norm" else d[quantity]
            ax.semilogy(t, values, color=color, ls=style, lw=.7, label=row["run_id"] + ": " + row["label"])
    for ax, title in zip(axes.flat, ("||m|| / ||g||", "||d|| / ||g||", "||d|| / ||m||", "Gradient norm after each update")):
        ax.set(title=title, ylabel="Ratio (dimensionless), log" if " / " in title else "L2 norm, log")
    axes[1, 1].axhline(1e-4, color="gray", lw=.8, label="Gradient threshold 1e-4")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .953), ncol=2, fontsize=8)
    outputs.append(finish(fig, axes, "Rosenbrock: paired collector norm ratios and stationarity",
        "Every update, no smoothing; log axes do not replace undefined denominators. Matched start/LR/beta2 and 10,000-update budget.\n"
        "The gradient threshold additionally requires objective <=1e-8; a one-step threshold hit is not sustained convergence.",
        HERE / "figures/norm_ratios.pdf"))
    manifest = {"generator_sha256": sha(Path(__file__)), "source_histories": [{"run": r["run_id"], "sha256": sha(r["path"])} for r in rows],
                "coverage": "every update 1..10000, objective positions 0..10000", "cosine_smoothing": "trailing 100 valid adjacent pairs",
                "contours": "analytic f on 500x500 meshes; exact trajectories, no decimation", "figures": outputs}
    (HERE / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pdf_count": len(outputs), "figures": [o["path"] for o in outputs]}))


if __name__ == "__main__":
    main()
