"""Matched LR overlays across sigma, with and without realized noise epsilon."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from noise_kappa import HERE, ROOT, FIELDS, digest, sha256, write_json

COLORS = ("#0072B2", "#8C4AA8", "#D55E00")
STYLES = ("-", (0, (2, 2)), (0, (7, 3)))
LABELS = (r"$\eta=10^{-2}$", r"$\eta=3.1623\times10^{-3}$", r"$\eta=10^{-3}$")
SIGMAS = (0, .01, .1, 1, 10)


def nullable(values):
    return [float(v) if np.isfinite(v) else None for v in values]


def main():
    assert json.loads((HERE / "verification.json").read_text())["passed"]
    report = json.loads((HERE / "results.json").read_text())
    series, data = [], []
    for source in report["sources"]:
        path = ROOT / source["derived_history"]
        assert sha256(path) == source["derived_history_sha256"]
        with np.load(path, allow_pickle=False) as saved:
            values = {k: saved[k] for k in saved.files}
        series.append((source, values))
        data.append(dict(run_id=source["run_id"], sigma=source["sigma"], learning_rate=source["learning_rate"],
                         arrays={k: nullable(values[k]) for k in ("gradient_only_mean100", "noise_corrected_mean100")}))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": "#26313D",
                         "text.color": "#26313D"})
    fig, axes = plt.subplots(5, 2, figsize=(16, 18.5), sharex=True)
    fig.subplots_adjust(left=.085, right=.97, top=.846, bottom=.107, hspace=.43, wspace=.24)
    fig.suptitle("Rosenbrock: gradient-identity kappa under increasing noise", fontsize=22, y=.983)
    fig.text(.5, .950, "Three matched AdamW LRs | Start (0, 1) | 2,000 updates | Weight decay 0.1 | Paired noise seed 1234", ha="center", fontsize=12)
    handles = [plt.Line2D([0], [0], color=c, ls=s, lw=2.2, label=l) for c,s,l in zip(COLORS, STYLES, LABELS)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5,.935), ncol=3, frameon=False, handlelength=3.2)
    fig.text(.296, .891, r"Gradient-only term: $q_t=1-\rho_t\|g_t\|/\|g_{t-1}\|$", ha="center", fontsize=15)
    fig.text(.781, .891, r"Including known noise: $\eta\widehat\kappa_t=q_t+\varepsilon_t$", ha="center", fontsize=15)
    fig.text(.296, .865, "Same quantity as the previous overlay; common linear Y scale", ha="center", fontsize=10.5)
    fig.text(.781, .865, "Realized noise correction; linear Y range varies by row", ha="center", fontsize=10.5)
    all_left = np.concatenate([v["gradient_only_mean100"][100:] for _,v in series])
    spread = np.ptp(all_left)
    left_limits = (min(0, all_left.min()) - .065 * spread, all_left.max() + .065 * spread)
    rendered = []
    for row, sigma in enumerate(SIGMAS):
        selected = [(s,v) for s,v in series if s["sigma"] == sigma]
        for col, metric in enumerate(("gradient_only", "noise_corrected")):
            ax = axes[row, col]
            for (source, values), color, style in zip(selected, COLORS, STYLES):
                y = values[metric + "_mean100"]
                line, = ax.plot(values["updates"], y, color=color, ls=style, lw=1.7)
                np.testing.assert_array_equal(line.get_ydata(), y)
                rendered.append(dict(panel=[row,col], run_id=source["run_id"], metric=metric,
                                    points=2000, finite_points=1900,
                                    x_sha256=digest(line.get_xdata()), y_sha256=digest(line.get_ydata()),
                                    color=color))
            if col == 0:
                limits = left_limits
            else:
                finite = np.concatenate([v[metric + "_mean100"][100:] for _,v in selected])
                span = max(np.ptp(finite), .001)
                limits = (min(0, finite.min()) - .08 * span, finite.max() + .08 * span)
            ax.set(xlim=(0,2000), ylim=limits, ylabel="100-pair mean", title=rf"({chr(97 + 2*row + col)}) $\sigma={sigma:g}$")
            ax.grid(axis="y", color="#E2E7EC", lw=.8)
            ax.axhline(0, color="#66727E", lw=.7, alpha=.5)
            ax.set_axisbelow(True)
            ax.set_xticks([0,500,1000,1500,2000])
            ax.tick_params(axis="x", labelbottom=True)
            for record in rendered[-3:]:
                record.update(xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()), y_scale="linear")
            if row == 4:
                ax.set_xlabel("Optimizer update t")
    fig.text(.085,.070, r"$g_t=\nabla f(\theta_{t-1})+\xi_t$,  $\xi_t\sim\mathcal{N}(0,\sigma^2 I)$;  $\varepsilon_t=g_{t-1}^{\mathsf{T}}(\xi_t-\xi_{t-1})/\|g_{t-1}\|^2$.", fontsize=13)
    fig.text(.085,.048, "Compute each adjacent-pair quantity first, then average 100 pairs. First complete mean: t = 101; all 1,900 means per curve shown.", fontsize=10.5)
    fig.text(.085,.028, "No clipping, denominator floor or seed averaging. Noise correction removes the explicit noise difference; AdamW step mismatch remains.", fontsize=10.5)
    destination = HERE / "figures/rosenbrock_kappa_eta_noise_overlay.pdf"
    fig.savefig(destination)
    plt.close(fig)
    for original in report["protected_reference_figures"]:
        assert sha256(ROOT / original["path"]) == original["sha256"]
    data_path = HERE / "plot_data.json"
    write_json(data_path, dict(updates=list(range(1,2001)), smoothing_window=100,
                              normalization="multiply kappa by constant eta; no maximum rescaling", sources=data))
    write_json(HERE / "figure_manifest.json", dict(figure=destination.relative_to(ROOT).as_posix(),
               sha256=sha256(destination), panels=10, dimensions_inches=[16,18.5],
               script_sha256=sha256(__file__), metric_sha256=sha256(HERE / "noise_kappa.py"),
               plot_data_sha256=sha256(data_path), window=100, thinning=None,
               left_common_y_scale=True, right_per_row_y_limits=True, rendered_series=rendered))
    print(destination)


if __name__ == "__main__":
    main()
