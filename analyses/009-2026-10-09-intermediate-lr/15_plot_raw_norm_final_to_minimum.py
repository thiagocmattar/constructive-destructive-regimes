"""Final raw-gradient norm divided by its trajectory minimum versus peak LR."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "complete-seven-lr"
NAME = "raw_norm_final_to_minimum"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source_path = OUT / "loss_and_norms_by_learning_rate_data.json"
    receipt = json.loads((OUT / "loss_and_norms_by_learning_rate_verification.json").read_text())
    prior_pdf = OUT / "figures/loss_and_norms_by_learning_rate.pdf"
    comparison_path = OUT / "comparison_data.json"
    assert receipt["passed"] and receipt["data_sha256"] == sha(source_path)
    assert receipt["figure_sha256"] == sha(prior_pdf)
    assert receipt["source_comparison_sha256"] == sha(comparison_path)
    source = json.loads(source_path.read_text())
    comparison = json.loads(comparison_path.read_text())["runs"]
    assert source["updates"] == list(range(1, 713)) and len(source["runs"]) == 7
    rows = []
    for rid, run in sorted(source["runs"].items(), key=lambda item: item[1]["lr"]):
        norms = np.asarray(run["raw_norm"], dtype=np.float64)
        direct = [event["gradient_norm"] for event in comparison[rid]["events"]]
        np.testing.assert_array_equal(norms, direct)
        assert np.isfinite(norms).all() and (norms > 0).all()
        final, minimum = float(norms[-1]), float(norms.min())
        ratio = final / minimum
        assert ratio == direct[-1] / min(direct) and ratio >= 1
        rows.append(dict(run_id=rid, peak_lr=run["lr"], hardware=run["hardware"],
            final_update=712, final_raw_norm=final, minimum_raw_norm=minimum,
            minimum_update=int(norms.argmin())+1, final_to_minimum=ratio,
            color=run["loss_color"]))
    csv_path = OUT / (NAME + ".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.subplots_adjust(left=.115, right=.97, top=.86, bottom=.24)
    fig.suptitle("Pythia-14M · final raw-gradient norm relative to its minimum", fontsize=18, y=.965)
    x = [row["peak_lr"] for row in rows]
    y = [row["final_to_minimum"] for row in rows]
    line, = ax.plot(x, y, color="#6C7782", lw=1.4, zorder=1)
    np.testing.assert_array_equal(line.get_ydata(), y)
    for row in rows:
        marker = "o" if row["hardware"] == "Spark" else "s"
        ax.scatter(row["peak_lr"], row["final_to_minimum"], color=row["color"], marker=marker,
                   s=85, edgecolor="#3E4852", linewidth=.7, zorder=3)
        ax.annotate(f"{row['final_to_minimum']:.2f}", (row["peak_lr"], row["final_to_minimum"]),
                    xytext=(0, 10), textcoords="offset points", ha="center", fontsize=11)
    ax.set_xscale("log")
    ax.set(xlim=(min(x)/1.2, max(x)*1.2), ylim=(0, max(y)*1.16),
           xlabel="Peak learning rate (log scale)", ylabel="Final / minimum raw-gradient norm")
    ax.set_xticks(x, [r"$10^{-4}$", r"$2.154\times10^{-4}$", r"$4.642\times10^{-4}$", r"$10^{-3}$",
                     r"$2.154\times10^{-3}$", r"$4.642\times10^{-3}$", r"$10^{-2}$"], fontsize=10)
    ax.axhline(1, color="#9AA4AE", ls=":", lw=.9)
    ax.grid(axis="y", color="#E2E7EC", lw=.7)
    ax.legend(handles=[Line2D([], [], marker=marker, ls="none", markerfacecolor="#CAD2DA",
                             markeredgecolor="#3E4852", label=label)
                       for marker, label in (("o", "Spark"), ("s", "RTX4090"))],
              loc="upper left", title="Hardware", frameon=False, fontsize=11)
    fig.text(.115, .155, r"Metric: $\|g_{712}\|\,/\,\min_{1\leq t\leq712}\|g_t\|$; raw gradients before clipping, no smoothing.", fontsize=12)
    fig.text(.115, .109, "Final update712 is the partial batch: 311 sequences versus 1,024. The minimum uses all 712 updates.", fontsize=10)
    fig.text(.115, .069, "One seed1234; hardware differs across groups. Lines connect observed points; no fit or sampling uncertainty estimate.", fontsize=10)
    pdf = OUT / "figures" / (NAME + ".pdf")
    fig.savefig(pdf)
    plt.close(fig)
    data = dict(generated_at=datetime.now(timezone.utc).isoformat(), generator_sha256=sha(__file__),
        source_plotted_data_sha256=sha(source_path), source_comparison_sha256=sha(comparison_path),
        source_pdf_sha256=sha(prior_pdf), figure_sha256=sha(pdf), csv_sha256=sha(csv_path),
        metric="||g_712|| / min_{1<=t<=712} ||g_t||", units="dimensionless",
        raw_scope="whole-model unscaled accumulated gradient before clipping",
        final_update=712, minimum_window=[1,712], smoothing=None, minimum_tie_policy="first update",
        zero_minimum_policy="undefined; no epsilon; all observed minima are positive",
        partial_final_batch_included=True, x_scale="log", y_scale="linear", rows=rows)
    data_path = OUT / (NAME + "_data.json")
    data_path.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    with fitz.open(pdf) as document:
        assert len(document) == 1
        page = document[0]
        np.testing.assert_array_equal([page.rect.width, page.rect.height], [11*72, 7*72])
        for label in ("final raw-gradient norm", "Peak learning rate", "minimum", "partial batch", "Hardware"):
            assert label in page.get_text(), label
        for block in page.get_text("blocks"):
            assert page.rect.contains(fitz.Rect(block[:4])), block[:4]
        render = OUT / "scratch" / (NAME + "_review.png")
        page.get_pixmap(matrix=fitz.Matrix(1.5,1.5), alpha=False).save(str(render))
    verification = dict(passed=True, all_norms_equal_source=True, all_seven_ratios_independently_reduced=True,
        point_order="increasing peak LR", plotted_line_equal_metric=True, pages=1, page_text_and_bounds=True,
        source_comparison_sha256=sha(comparison_path), data_sha256=sha(data_path), csv_sha256=sha(csv_path),
        figure_sha256=sha(pdf), render_sha256=sha(render), visual_review="pending")
    (OUT / (NAME + "_verification.json")).write_text(json.dumps(verification, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(figure=str(pdf), rows=rows), indent=2))


if __name__ == "__main__":
    main()
