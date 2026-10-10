"""One navigation/metric PDF and one identity PDF per sigma, plus a sweep view."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.controlled.rosenbrock import sha256, write_json
from src.controlled.noise_geometry import noise_terms
from src.controlled.rosenbrock_plot import plot

COLORS = ("#0072B2", "#8C4AA8", "#D55E00")
STYLES = ("-", (0, (2, 2)), (0, (7, 3)))
LABELS = (r"$\eta=10^{-2}$", r"$\eta=3.1623\times10^{-3}$", r"$\eta=10^{-3}$")


def load(entry):
    folder = ROOT / entry["directory"]
    receipt = json.loads((folder / "verification.json").read_text())
    path = folder / "artifacts/attempts/001-local/history.npz"
    assert receipt["passed"] and receipt["history_sha256"] == sha256(path)
    assert receipt["config_sha256"] == sha256(folder / "config.json")
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def save(fig, stem, kind, sigma, entries, figures):
    path = HERE / "figures" / (stem + ".pdf")
    fig.savefig(path)
    fig.savefig(HERE / "scratch" / (stem + ".png"), dpi=110)
    plt.close(fig)
    figures.append(dict(path=str(path.relative_to(ROOT)), sha256=sha256(path), kind=kind, sigma=sigma,
                        sources=[dict(run_id=e["run_id"], directory=e["directory"],
                                      history_sha256=sha256(ROOT / e["directory"] / "artifacts/attempts/001-local/history.npz"))
                                 for e in entries]))


def identities(entries, sigma, figures):
    histories = [load(entry) for entry in entries]
    fig, axes = plt.subplots(3, 3, figsize=(17, 14), sharex=True)
    fig.subplots_adjust(left=.09, right=.97, top=.89, bottom=.14, hspace=.40, wspace=.38)
    fig.suptitle(f"Rosenbrock noisy gradients · exact identities · σ = {sigma:g}", fontsize=20, y=.98)
    seed_note = "Reused deterministic controls" if sigma == 0 else "One paired noise seed (1234)"
    fig.text(.5, .947, seed_note + "; every prefix shown; signed values use symmetric log Y (linear within ±1).",
             ha="center", fontsize=11)
    keys = ("#0072B2", "#D55E00", "#8C4AA8", "#68737D")
    t = np.arange(1, 2001)
    for col, (entry, h, label) in enumerate(zip(entries, histories, LABELS)):
        signal = h["signal_vectors"] if not entry["reused"] else h["raw_vectors"]
        noise = h["noise_vectors"] if not entry["reused"] else np.zeros_like(signal)
        terms = noise_terms(signal, noise)
        series = ((h["raw_B_squared"], h["raw_energy"], h["raw_all_pair"]),
                  (terms["signal_B_squared"], terms["noise_B_squared"], terms["prefix_cross"]),
                  (terms["noise_B_squared"], terms["noise_energy"], 2 * t * sigma ** 2))
        names = ((r"$B_g^2$", r"$E_g=\sum\|g_t\|^2$", r"$I_g=2\sum_{r<s}\langle g_r,g_s\rangle$"),
                 (r"$\|H\|^2$ (signal)", r"$\|X\|^2$ (noise)", r"$2\langle H,X\rangle$ (cross)"),
                 (r"$B_\xi^2=\|X\|^2$", r"$E_\xi=\sum\|\xi_t\|^2$", r"$2t\sigma^2$ (expectation)"))
        for row in range(3):
            ax = axes[row, col]
            for value, name, color, style in zip(series[row], names[row], keys, ("-", "--", ":")):
                ax.plot(t, value, label=name, color=color, ls=style, lw=1.45)
            ax.set_yscale("symlog", linthresh=1)
            ax.axhline(0, color="#9CA7B2", lw=.7)
            ax.set(xlim=(0, 2000), ylabel="Squared gradient units" if col == 0 else "")
            if row == 2:
                ax.set_xlabel("Optimizer update t")
            ax.set_title(f"({chr(97 + row * 3 + col)}) " + (label if row == 0 else ("Signal/noise/cross" if row == 1 else "Noise realization vs expectation")), loc="left")
            ax.grid(axis="y", color="#E2E7EC", lw=.7)
            ax.legend(loc="best", fontsize=9, framealpha=.9)
    # Equal limits within each row make the LR columns comparable.
    for row in axes:
        bottom = min(ax.get_ylim()[0] for ax in row)
        top = max(ax.get_ylim()[1] for ax in row)
        for ax in row:
            ax.set_ylim(bottom, top)
            if max(abs(bottom), abs(top)) < 1:
                ax.yaxis.set_major_locator(MaxNLocator(5))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:g}"))
    fig.text(.08, .065, r"Exact: $B_g^2=E_g+I_g=\|H\|^2+\|X\|^2+2\langle H,X\rangle$, $H=\sum h_t$, $X=\sum\xi_t$, $h_t=\nabla f(\theta_{t-1})$.", fontsize=12)
    fig.text(.08, .041, "The dotted ensemble expectation is not a pathwise identity. Cross terms remain on the adapted path; zero-mean noise does not set them to zero.", fontsize=10)
    save(fig, "identities_sigma_" + entries[0]["sigma_label"], "identities", sigma, entries, figures)


def sweep(cohort, figures):
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.subplots_adjust(left=.075, right=.97, top=.87, bottom=.10, hspace=.37, wspace=.31)
    fig.suptitle("Rosenbrock · increasing gradient noise", fontsize=20, y=.98)
    fig.legend(handles=[Line2D([], [], color=color, ls=style, marker="o", label=label)
                        for color, style, label in zip(COLORS, STYLES, LABELS)], loc="upper center",
               bbox_to_anchor=(.5, .947), ncol=3, frameon=False)
    titles = ("(a) Final objective", "(b) Median objective, last 1,000 updates", "(c) Raw noisy-gradient B/A, full prefix",
              "(d) Momentum B/A, full prefix", "(e) Mean raw cosine, last 1,000 updates", "(f) Mean momentum cosine, last 1,000 updates")
    sigmas = [0., .01, .1, 1., 10.]
    for lr_index, (color, style) in enumerate(zip(COLORS, STYLES)):
        rows = []
        for sigma in sigmas:
            entry = [e for e in cohort if e["sigma"] == sigma][lr_index]
            h = load(entry)
            rows.append([h["loss"][-1], np.median(h["loss"][-1000:]), h["raw_ratio"][-1], h["momentum_ratio"][-1],
                         np.nanmean(h["raw_cosine"][-1000:]), np.nanmean(h["momentum_cosine"][-1000:])])
        for col, ax in enumerate(axes.flat):
            ax.plot(sigmas, np.array(rows)[:, col], color=color, ls=style, marker="o", lw=1.6)
    for i, (ax, title) in enumerate(zip(axes.flat, titles)):
        ax.set_title(title, loc="left")
        ax.set_xscale("symlog", linthresh=.01)
        ax.set_xticks(sigmas, ["0", "0.01", "0.1", "1", "10"])
        ax.set_xlabel("σ: gradient-coordinate standard deviation")
        ax.grid(axis="y", color="#E2E7EC", lw=.7)
        if i < 2:
            ax.set_yscale("log")
            ax.set_ylabel("Exact objective f")
        elif i < 4:
            ax.set(ylim=(0, 1.04), ylabel="B/A")
        else:
            ax.set(ylim=(-1.05, 1.05), ylabel="Mean adjacent cosine")
            ax.axhline(0, color="#9CA7B2", lw=.7)
    fig.text(.075, .03, "One seed (1234), paired normal draws across σ/LR; σ = 0 reuses deterministic controls. No error bars or sampling uncertainty estimate.", fontsize=11)
    save(fig, "sigma_comparison", "sigma_comparison", None, cohort, figures)


def norm_grid(cohort, figures):
    """All steps, one noise level per row and one learning rate per column."""
    fig, axes = plt.subplots(5, 3, figsize=(17, 20), sharex=True, sharey=True)
    fig.subplots_adjust(left=.10, right=.97, top=.92, bottom=.075, hspace=.24, wspace=.15)
    fig.suptitle("Rosenbrock · raw-gradient and momentum norms", fontsize=20, y=.98)
    colors = ("#0072B2", "#D55E00")
    fig.legend(handles=[Line2D([], [], color=color, ls=style, label=label)
                        for color, style, label in zip(colors, ("-", "--"),
                        (r"Noisy raw gradient $\|g_t\|$", r"Stored momentum $\|m_t\|$"))],
               loc="upper center", bbox_to_anchor=(.5, .96), ncol=2, frameon=False)
    t = np.arange(1, 2001)
    lower, upper = np.inf, 0.
    for row, sigma in enumerate((0., .01, .1, 1., 10.)):
        entries = [entry for entry in cohort if entry["sigma"] == sigma]
        for col, entry in enumerate(entries):
            h = load(entry)
            ax = axes[row, col]
            for stream, color, style in zip(("raw", "momentum"), colors, ("-", "--")):
                values = h[stream + "_norm"]
                assert np.all(values > 0), "Log norm grid requires positive norms"
                lower, upper = min(lower, values.min()), max(upper, values.max())
                ax.plot(t, values, color=color, ls=style, lw=.9)
            ax.set_yscale("log")
            ax.set_xlim(1, 2000)
            ax.grid(axis="y", color="#E2E7EC", lw=.7)
            ax.set_title(f"({chr(97 + row * 3 + col)}) " + (LABELS[col] if row == 0 else ""), loc="left")
            if col == 0:
                ax.set_ylabel(rf"$\sigma={sigma:g}$" + "\nVector norm")
            if row == 4:
                ax.set_xlabel("Optimizer update t")
    axes[0, 0].set_ylim(lower / 1.3, upper * 1.3)
    fig.text(.10, .039, "Every update, no averaging or thinning; shared log Y limits. Columns: learning rate; rows: gradient-coordinate noise standard deviation.", fontsize=11)
    fig.text(.10, .023, "AdamW, start (0,1), decay 0.1; one paired noise seed1234. Sigma0 reuses deterministic controls; momentum is before bias correction/adaptive scaling.", fontsize=10)
    save(fig, "gradient_momentum_norms_grid", "norm_grid", None, cohort, figures)


def main():
    verification = json.loads((HERE / "verification.json").read_text())
    assert verification["passed"]
    cohort = json.loads((HERE / "cohort.json").read_text())
    points = np.concatenate([load(entry)["theta"] for entry in cohort])
    low, high = points.min(axis=0), points.max(axis=0)
    pad = .1 * np.maximum(high - low, 1)
    domain = [min(-1.4, low[0] - pad[0]), max(1.22, high[0] + pad[0]),
              min(-.15, low[1] - pad[1]), max(1.48, high[1] + pad[1])]
    figures = []
    for sigma in (0., .01, .1, 1., 10.):
        entries = [entry for entry in cohort if entry["sigma"] == sigma]
        stem = "rosenbrock_sigma_" + entries[0]["sigma_label"]
        frame_manifest = HERE / f"figure_manifest_{stem}.json"
        title = f"Rosenbrock · AdamW noisy gradients · σ = {sigma:g}, weight decay 0.1"
        plot(HERE, title=title, labels=LABELS, colors=COLORS, styles=STYLES,
             cosine_window=100, norm_scale="linear", cohort_entries=entries,
             figure_stem=stem, noise_sigma=sigma, trajectory_domain=domain)
        frame = json.loads(frame_manifest.read_text())
        figures.append(dict(path=frame["figure"], sha256=frame["sha256"], kind="navigation", sigma=sigma,
                            sources=frame["sources"], frame_manifest=str(frame_manifest.relative_to(ROOT))))
        identities(entries, sigma, figures)
    sweep(cohort, figures)
    norm_grid(cohort, figures)
    write_json(HERE / "figures_manifest.json", dict(script=str(Path(__file__).relative_to(ROOT)),
        script_sha256=sha256(__file__), dependencies=[dict(path=p, sha256=sha256(ROOT / p))
        for p in ("src/controlled/rosenbrock_plot.py", "src/controlled/noise_geometry.py")],
        common_trajectory_domain=domain, figures=figures, seed=1234, sigma_zero_reused=True,
        cosine_window=100, uncertainty="None: one matched noise realization", thinning=None))


if __name__ == "__main__":
    main()
