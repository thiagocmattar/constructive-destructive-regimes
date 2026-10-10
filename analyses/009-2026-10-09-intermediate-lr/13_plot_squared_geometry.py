"""Plot A squared, B squared and their difference for raw gradients and momentum."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lr_plot_helpers", HERE / "10_plot_lr_geometry.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
OUT = helpers.OUT
NAME = "squared_cumulative_geometry"
QUANTITIES = ("A_squared", "B_squared", "A_squared_minus_B_squared")
LABELS = (r"$A^2$", r"$B^2$", r"$A^2-B^2$")


def main():
    rows = helpers.records()
    prior = json.loads((OUT / "requested_figures_data.json").read_text(encoding="utf-8"))
    assert helpers.sha(helpers.SOURCE) == prior["source_comparison_sha256"]
    colors = {r["id"]: r["color"] for r in prior["palette"]["conditions"]}
    fig, axes = plt.subplots(3, 2, figsize=(14, 13.5))
    x = np.arange(1, 712)
    plotted = {}
    for run in rows:
        assert to_hex(run["color"]) == colors[run["id"]]
        data = {"lr": run["lr"], "hardware": run["hardware"],
                "color": to_hex(run["color"]), "line_style": run["hardware_style"],
                "series": {}, "terminal_update_712": {}}
        for col, stream in enumerate(("raw", "momentum")):
            A = helpers.plots.values(run["events"], stream, "cumulative_norm_sum")
            B = helpers.plots.values(run["events"], stream, helpers.plots.STREAMS[stream][2])
            series = dict(zip(QUANTITIES, (A*A, B*B, A*A-B*B)))
            assert all(np.isfinite(values).all() for values in series.values())
            assert (series["A_squared_minus_B_squared"] >= 0).all()
            for row, quantity in enumerate(QUANTITIES):
                axes[row, col].plot(x, series[quantity][:711], color=run["color"],
                                    ls=run["hardware_style"], lw=1.8)
            data["series"][stream] = {key: values[:711].tolist() for key, values in series.items()}
            data["terminal_update_712"][stream] = {key: float(values[-1]) for key, values in series.items()}
        plotted[run["id"]] = data
    for row, label in enumerate(LABELS):
        for col, stream in enumerate(("Raw-gradient g", "Momentum m")):
            ax = axes[row, col]
            ax.set_title(f"({'abcdef'[2*row+col]}) {stream}: {label}")
            ax.set_ylabel("Squared norm")
            ax.set_ylim(0, None)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    helpers.plots.format_axes(axes)
    fig.suptitle("Pythia-14M: squared cumulative geometry across learning rates", fontsize=17, y=.988)
    helpers.legend(fig, rows, hardware_style=True, y=.965)
    fig.text(.5, .026,
             r"$A_t=(\sum_{s=1}^{t}\|v_s\|)$; $B_t=\|\sum_{s=1}^{t}v_s\|$. Rows: $A_t^2$, $B_t^2$, $A_t^2-B_t^2$."
             "\nUnsmoothed curves; independent linear Y scales. Columns: raw g before clipping; stored Adam momentum m before preconditioning."
             "\nUpdates 1-711 shown; partial update 712 retained in results. Same frozen initialization/order; seed 1234."
             "\nS = Spark, R = RTX4090; hardware differs across groups. Momentum uses clipped gradients; LR also scales AdamW decay.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .095, 1, .90), h_pad=2.1, w_pad=2.1)
    path = OUT / "figures" / (NAME + ".pdf")
    limits = [[float(v) for v in ax.get_ylim()] for ax in axes.flat]
    fig.savefig(path)
    plt.close(fig)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_comparison_sha256": helpers.sha(helpers.SOURCE),
        "generator_sha256": helpers.sha(Path(__file__)),
        "helper_sha256": helpers.sha(HERE / "10_plot_lr_geometry.py"),
        "value_helper_sha256": helpers.sha(helpers.HELPER),
        "figure_sha256": helpers.sha(path), "palette": prior["palette"],
        "layout": [3, 2], "columns": ["raw", "momentum"], "rows": list(QUANTITIES),
        "definitions": {"A": "sum_{s<=t} ||v_s||", "B": "||sum_{s<=t} v_s||",
                        "A_squared": "(sum of vector norms)^2; not sum of squared norms",
                        "A_squared_minus_B_squared": "squared deficit relative to aligned accumulation; not signed total interaction J"},
        "source_updates": 712, "displayed_updates": [1, 711], "partial_update_retained": True,
        "smoothing": "none", "y_scales": ["linear"]*6,
        "y_limits_policy": "independent per panel, zero lower bound, automatic upper bound",
        "y_limits_row_major": limits, "updates": x.tolist(), "runs": plotted,
    }
    (OUT / (NAME + "_data.json")).write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"figure": str(path), "layout": [3, 2], "source_sha256": metadata["source_comparison_sha256"]}))


if __name__ == "__main__":
    main()
