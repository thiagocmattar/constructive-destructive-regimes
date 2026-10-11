"""Unsmoothed training loss, then one raw/momentum norm panel per learning rate."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lr_plot_helpers", HERE / "10_plot_lr_geometry.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
OUT = helpers.OUT
NAME = "loss_and_norms_by_learning_rate"


def main():
    rows = helpers.records()
    prior = json.loads((OUT / "requested_figures_data.json").read_text())
    assert helpers.sha(helpers.SOURCE) == prior["source_comparison_sha256"]
    source = json.loads(helpers.SOURCE.read_text())["runs"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 13, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    fig.subplots_adjust(left=.085, right=.97, top=.89, bottom=.095, hspace=.37, wspace=.24)
    fig.suptitle("Pythia-14M · unsmoothed loss and norms by learning rate", fontsize=20, y=.98)
    handles = [Line2D([], [], color=r["color"], ls=r["hardware_style"], lw=1.5,
                     label=f"{r['lr']:.6g} ({'S' if r['hardware']=='Spark' else 'R'})") for r in rows]
    fig.legend(handles=handles, title="Panel (a): peak learning rate · S = Spark, R = RTX4090",
               loc="upper center", bbox_to_anchor=(.5, .954), ncol=7,
               frameon=False, fontsize=9, title_fontsize=11, columnspacing=1.1)
    x = np.arange(1, 713)
    plotted = {}
    loss_ax = axes.flat[0]
    loss_ax.set(title="(a) Unsmoothed training loss · all learning rates", ylabel="Cross-entropy (nats)")
    norm_limits = [np.inf, 0.]
    for panel, run in enumerate(rows, 1):
        events = run["events"]
        loss = np.array([event["loss"] for event in events])
        raw = helpers.plots.values(events, "raw", "gradient_norm")
        momentum = helpers.plots.values(events, "momentum", "momentum_norm")
        assert np.isfinite(loss).all() and (raw > 0).all() and (momentum > 0).all()
        # Independent direct extraction verifies the helper's stream placement.
        np.testing.assert_array_equal(raw, [event["gradient_norm"] for event in source[run["id"]]["events"]])
        np.testing.assert_array_equal(momentum, [event["momentum"]["momentum_norm"] for event in source[run["id"]]["events"]])
        np.testing.assert_array_equal(loss, [event["loss"] for event in source[run["id"]]["events"]])
        loss_line, = loss_ax.plot(x, loss, color=run["color"], ls=run["hardware_style"], lw=1.1)
        loss_ax.plot(x[-1], loss[-1], "o", color=run["color"], ms=3)
        ax = axes.flat[panel]
        ax.set_title(f"({'abcdefgh'[panel]}) " + rf"$\eta={run['lr']:.6g}$" + f" · {run['hardware']}", loc="left")
        lines = [ax.plot(x, values, color=color, ls=style, lw=1.1, label=label)[0]
                 for values, color, style, label in zip((raw, momentum), ("#0072B2", "#D55E00"), ("-", "--"),
                    (r"Raw gradient $\|g_t\|$", r"Stored momentum $\|m_t\|$"))]
        for line, values in zip(lines, (raw, momentum)):
            np.testing.assert_array_equal(line.get_xdata(), x)
            np.testing.assert_array_equal(line.get_ydata(), values)
            ax.plot(x[-1], values[-1], "o", color=line.get_color(), ms=3)
        np.testing.assert_array_equal(loss_line.get_ydata(), loss)
        ax.set_yscale("log")
        ax.set_ylabel("Euclidean norm (log scale)")
        ax.legend(loc="best", fontsize=9, framealpha=.9)
        norm_limits[0] = min(norm_limits[0], raw.min(), momentum.min())
        norm_limits[1] = max(norm_limits[1], raw.max(), momentum.max())
        plotted[run["id"]] = dict(panel="abcdefgh"[panel], lr=run["lr"], hardware=run["hardware"],
            loss_color=to_hex(run["color"]), loss_line_style=run["hardware_style"],
            training_loss=loss.tolist(), raw_norm=raw.tolist(), momentum_norm=momentum.tolist(),
            supervised_tokens=[event["supervised_tokens"] for event in events],
            event_source_sha256=run["sources"]["events.jsonl"])
    norm_limits = [float(norm_limits[0] / 1.25), float(norm_limits[1] * 1.25)]
    for index, ax in enumerate(axes.flat):
        ax.set(xlim=(0, 725), xlabel="Optimizer update t")
        ax.grid(axis="y", color="#E2E7EC", lw=.7)
        if index:
            ax.set_ylim(norm_limits)
    fig.text(.085, .057, "All 712 accepted updates; all curves unsmoothed. Circles mark update 712: the partial final batch (311 sequences versus 1,024).", fontsize=10)
    fig.text(.085, .041, "Loss is the recorded token-mean training cross-entropy per update. Norm panels share logarithmic Y limits; no normalization or positive floor.", fontsize=10)
    fig.text(.085, .025, "Raw g is before clipping; stored Adam first moment m uses clipped gradients and precedes bias correction/preconditioning. Seed1234; hardware differs across groups.", fontsize=9)
    pdf = OUT / "figures" / (NAME + ".pdf")
    fig.savefig(pdf)
    metadata = dict(generated_at=datetime.now(timezone.utc).isoformat(),
        generator_sha256=helpers.sha(__file__), helper_sha256=helpers.sha(HERE / "10_plot_lr_geometry.py"),
        value_helper_sha256=helpers.sha(helpers.HELPER), source_comparison_sha256=helpers.sha(helpers.SOURCE),
        figure_sha256=helpers.sha(pdf), layout=[4, 2], panel_order=["training_loss"]+[r["id"] for r in rows],
        source_updates=712, displayed_updates=[1, 712], partial_final_update_displayed=True,
        smoothing="none for loss and both norm streams", thinning=None,
        y_scales=[ax.get_yscale() for ax in axes.flat], norm_y_limits=norm_limits,
        norm_colors=dict(raw="#0072B2", momentum="#D55E00"),
        definitions=dict(loss="per-update supervised-token-mean training cross-entropy, nats",
                         raw="whole-model pre-clipping accumulated gradient norm",
                         momentum="whole-model stored uncorrected exp_avg norm, based on clipped gradients"),
        updates=x.tolist(), runs=plotted)
    assert metadata["y_scales"] == ["linear"]+["log"]*7
    plt.close(fig)
    data_path = OUT / (NAME + "_data.json")
    data_path.write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    with fitz.open(pdf) as document:
        assert len(document) == 1
        page = document[0]
        np.testing.assert_array_equal([page.rect.width, page.rect.height], [16*72, 18*72])
        text = page.get_text()
        for label in ("Unsmoothed training loss", "Stored momentum", "Raw gradient", "All 712 accepted updates", "log scale"):
            assert label in text, label
        for block in page.get_text("blocks"):
            assert page.rect.contains(fitz.Rect(block[:4])), block[:4]
        scratch = OUT / "scratch"
        scratch.mkdir(exist_ok=True)
        render = scratch / (NAME + "_review.png")
        page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2), alpha=False).save(str(render))
    verification = dict(passed=True, figure_sha256=helpers.sha(pdf), data_sha256=helpers.sha(data_path),
        source_comparison_sha256=helpers.sha(helpers.SOURCE), all_21_series_equal_source=True,
        loss_artist_unsmoothed=True, all_norm_artists_equal_source=True, source_and_displayed_updates=712,
        panels=8, norm_panels=7, common_log_norm_limits=norm_limits, positive_norms=True,
        all_points_within_axes=True, page_text_and_bounds=True, pages=1,
        render_sha256=helpers.sha(render), visual_review="pending")
    (OUT / (NAME + "_verification.json")).write_text(json.dumps(verification, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(figure=str(pdf), panels=8, updates=712, smoothing="none")))


if __name__ == "__main__":
    main()
