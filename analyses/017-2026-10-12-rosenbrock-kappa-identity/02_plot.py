"""Compare raw and averaged kappa with exact curvature; retain the full layout."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from kappa import HERE, ROOT, digest, load_sources, sha256, write_json
from plot_full import plot

COLORS = ("#0072B2", "#8C4AA8", "#D55E00")
STYLES = ("-", (0, (2, 2)), (0, (7, 3)))
LABELS = (r"$\eta=10^{-2}$", r"$\eta=3.1623\times10^{-3}$", r"$\eta=10^{-3}$")


def main():
    assert json.loads((HERE / "verification.json").read_text())["passed"]
    histories, sources, turns = load_sources()
    report = json.loads((HERE / "results.json").read_text())
    for h, row in zip(histories, report["sources"]):
        path = ROOT / row["derived_history"]
        assert sha256(path) == row["derived_history_sha256"]
        with np.load(path, allow_pickle=False) as extra:
            h.update({key: extra[key] for key in extra.files})
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#26313D", "text.color": "#26313D"})
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.subplots_adjust(left=.065, right=.98, top=.77, bottom=.14, wspace=.20, hspace=.34)
    fig.suptitle("Rosenbrock: gradient-identity kappa versus exact Hessian curvature", fontsize=21, y=.965)
    fig.text(.5, .925, "Same AdamW trajectories as Analysis016 | Start (0, 1) | 2,000 updates | Weight decay 0.1", ha="center")
    fig.text(.5, .878, r"$\widehat{\kappa}_t=\eta^{-1}(1-\rho_t\|g_t\|/\|g_{t-1}\|+\varepsilon_t),\quad\varepsilon_t=0$", ha="center", fontsize=16)
    handles = [Line2D([], [], color="#0072B2", lw=2, label="Identity kappa proxy (LR color)"),
               Line2D([], [], color="#27313D", lw=1.7, label="Largest Hessian eigenvalue"),
               Line2D([], [], color="#5D7D72", ls="--", lw=1.4, label="Exact gradient-direction curvature")]
    fig.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(.5, .838), frameon=False)
    series, references = [], []
    for row in range(2):
        suffix = "" if row == 0 else "_mean100"
        all_values = np.concatenate([h[field + suffix] for h in histories for field in ("kappa_proxy", "lambda_max", "kappa_direction")])
        finite = all_values[np.isfinite(all_values)]
        low, high = finite.min(), finite.max(); padding = .07 * (high - low)
        for col, (h, color, label, event) in enumerate(zip(histories, COLORS, LABELS, turns["points"])):
            ax = axes[row, col]; panel = "abcdef"[3 * row + col]
            ax.set_title(f"({panel}) " + label + (" | unsmoothed" if row == 0 else " | 100-pair trailing means"), loc="left", fontsize=12, pad=12)
            ax.set(xlim=(0, 2000), ylim=(low-padding, high+padding), xlabel="Optimizer update t")
            if col == 0:
                ax.set_ylabel("Curvature / identity proxy")
            ax.set_axisbelow(True); ax.grid(axis="y", color="#E2E7EC", lw=.8)
            ax.axhline(0, color="#A8B1BB", lw=.7)
            for field, line_color, style, width, alpha in (
                ("kappa_proxy", color, "-", .65 if row == 0 else 2., .35 if row == 0 else 1.),
                ("lambda_max", "#27313D", "-", 1.7, 1.),
                ("kappa_direction", "#5D7D72", "--", .8 if row == 0 else 1.5, .65 if row == 0 else 1.)):
                expected = h[field + suffix]
                line, = ax.plot(h["updates"], expected, color=line_color, ls=style, lw=width, alpha=alpha)
                np.testing.assert_array_equal(line.get_ydata(), expected)
                series.append(dict(panel=panel, run_id=event["run_id"], field=field+suffix,
                                   points=2000, x_sha256=digest(line.get_xdata()), y_sha256=digest(line.get_ydata())))
            line = ax.axvline(event["update"], color=color, ls="--", lw=1., alpha=.8)
            references.append(dict(panel=panel, run_id=event["run_id"], update=int(line.get_xdata()[0])))
    fig.text(.065, .087, "Top: every defined raw pair (faint proxy). Bottom: each complete 100-pair arithmetic mean, first at t = 101. Common Y limits within each row.", fontsize=10.5)
    fig.text(.065, .064, r"For pair $t$: $g_t=\nabla f(\theta_{t-1})$; Hessian references use $H(\theta_{t-2})$. First pair is undefined. Vertical dashed lines mark original y-turns.", fontsize=10.5)
    fig.text(.065, .040, "Epsilon is the normalized noise term (zero here), not AdamW's numerical epsilon. No clipping, normalization, thinning or confidence bands.", fontsize=10.5)
    fig.text(.065, .018, "The identity assumes plain SGD to first order. On AdamW, this is a diagnostic proxy; exact directional curvature is not generally lambda_max.", fontsize=10.5)
    (HERE / "figures").mkdir(exist_ok=True); (HERE / "scratch").mkdir(exist_ok=True)
    path = HERE / "figures/rosenbrock_kappa_comparison.pdf"
    fig.savefig(path); fig.savefig(HERE / "scratch/rosenbrock_kappa_comparison.png", dpi=130)
    plt.close(fig)
    write_json(HERE / "figure_manifest_rosenbrock_kappa_comparison.json", dict(
        figure=path.relative_to(ROOT).as_posix(), sha256=sha256(path), sources=report["sources"],
        script_sha256=sha256(__file__), metric_implementation_sha256=sha256(HERE / "kappa.py"),
        panels=6, dimensions_inches=[18, 11], smoothing_window=100, rendered_series=series,
        vertical_reference_lines=references, y_scales={p:"linear" for p in "abcdef"}, thinning=None,
        reference_indexing="H(theta[t-2]) at pair t", undefined_first_pair=True))
    plot(HERE, title="Rosenbrock: AdamW geometry, Hessian and gradient-identity kappa", labels=LABELS,
         colors=COLORS, styles=STYLES, cosine_window=100, norm_scale="linear",
         trajectory_turns=turns, figure_stem="rosenbrock_kappa_diagnostics")
    print(path)


if __name__ == "__main__":
    main()
