"""Final raw-gradient norm minus its trajectory minimum versus learning rate."""
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
NAME = "raw_norm_final_minus_minimum"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    prior_path = OUT / "raw_norm_final_to_minimum_data.json"
    prior = json.loads(prior_path.read_text())
    receipt = json.loads((OUT / "raw_norm_final_to_minimum_verification.json").read_text())
    source_path = OUT / "loss_and_norms_by_learning_rate_data.json"
    assert receipt["passed"] and receipt["data_sha256"] == sha(prior_path)
    assert prior["source_plotted_data_sha256"] == sha(source_path)
    assert prior["source_comparison_sha256"] == sha(OUT / "comparison_data.json")
    source = json.loads(source_path.read_text())["runs"]
    rows = []
    for old in prior["rows"]:
        norms = source[old["run_id"]]["raw_norm"]
        assert len(norms) == 712 and np.isfinite(norms).all()
        assert old["final_raw_norm"] == norms[-1] and old["minimum_raw_norm"] == min(norms)
        difference = old["final_raw_norm"] - old["minimum_raw_norm"]
        assert difference == norms[-1] - min(norms) and difference >= 0
        row = {key: value for key, value in old.items() if key != "final_to_minimum"}
        row["final_minus_minimum"] = difference
        rows.append(row)
    assert len(rows) == 7 and [r["peak_lr"] for r in rows] == sorted(r["peak_lr"] for r in rows)
    csv_path = OUT / (NAME + ".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.subplots_adjust(left=.115, right=.97, top=.86, bottom=.24)
    fig.suptitle("Pythia-14M · final raw-gradient norm minus its minimum", fontsize=18, y=.965)
    x = [row["peak_lr"] for row in rows]
    y = [row["final_minus_minimum"] for row in rows]
    line, = ax.plot(x, y, color="#6C7782", lw=1.4, zorder=1)
    np.testing.assert_array_equal(line.get_xdata(), x)
    np.testing.assert_array_equal(line.get_ydata(), y)
    for row in rows:
        point = (row["peak_lr"], row["final_minus_minimum"])
        ax.scatter(*point, color=row["color"], marker="o" if row["hardware"] == "Spark" else "s",
                   s=85, edgecolor="#3E4852", linewidth=.7, zorder=3)
        ax.annotate(f"{point[1]:.3f}", point, xytext=(0, 10), textcoords="offset points", ha="center", fontsize=11)
    ax.set_xscale("log")
    ax.set(xlim=(min(x)/1.2, max(x)*1.2), ylim=(0, max(y)*1.16),
           xlabel="Peak learning rate (log scale)", ylabel="Final raw-gradient norm minus minimum")
    ax.set_xticks(x, [r"$10^{-4}$", r"$2.154\times10^{-4}$", r"$4.642\times10^{-4}$", r"$10^{-3}$",
                     r"$2.154\times10^{-3}$", r"$4.642\times10^{-3}$", r"$10^{-2}$"], fontsize=10)
    ax.grid(axis="y", color="#E2E7EC", lw=.7)
    ax.legend(handles=[Line2D([], [], marker=marker, ls="none", markerfacecolor="#CAD2DA",
                             markeredgecolor="#3E4852", label=label)
                       for marker, label in (("o", "Spark"), ("s", "RTX4090"))],
              loc="upper left", title="Hardware", frameon=False, fontsize=11)
    fig.text(.115, .155, r"Metric: $\|g_{712}\|-\min_{1\leq t\leq712}\|g_t\|$; raw gradients before clipping, no smoothing.", fontsize=12)
    fig.text(.115, .109, "Final update712 is the partial batch: 311 sequences versus 1,024. The minimum uses all 712 updates.", fontsize=10)
    fig.text(.115, .069, "One seed1234; hardware differs across groups. Lines connect observed points; no fit or sampling uncertainty estimate.", fontsize=10)
    pdf = OUT / "figures" / (NAME + ".pdf")
    fig.savefig(pdf)
    plt.close(fig)
    data = dict(generated_at=datetime.now(timezone.utc).isoformat(), generator_sha256=sha(__file__),
        prior_reduction_sha256=sha(prior_path), source_plotted_data_sha256=sha(source_path),
        source_comparison_sha256=prior["source_comparison_sha256"], figure_sha256=sha(pdf), csv_sha256=sha(csv_path),
        metric="||g_712|| - min_{1<=t<=712} ||g_t||", units="raw-gradient norm units",
        final_update=712, minimum_window=[1,712], smoothing=None, minimum_tie_policy="first update",
        partial_final_batch_included=True, x_scale="log", y_scale="linear", rows=rows)
    data_path = OUT / (NAME + "_data.json")
    data_path.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    with fitz.open(pdf) as document:
        assert len(document) == 1
        page = document[0]
        np.testing.assert_array_equal([page.rect.width, page.rect.height], [11*72, 7*72])
        for label in ("minus its minimum", "Peak learning rate", "partial batch", "Hardware"):
            assert label in page.get_text(), label
        for block in page.get_text("blocks"):
            assert page.rect.contains(fitz.Rect(block[:4])), block[:4]
        render = OUT / "scratch" / (NAME + "_review.png")
        page.get_pixmap(matrix=fitz.Matrix(1.5,1.5), alpha=False).save(str(render))
    verification = dict(passed=True, seven_differences_equal_source=True, plotted_line_equal_metric=True,
        pages=1, page_text_and_bounds=True, data_sha256=sha(data_path), csv_sha256=sha(csv_path),
        figure_sha256=sha(pdf), render_sha256=sha(render), visual_review="pending")
    (OUT / (NAME + "_verification.json")).write_text(json.dumps(verification, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(figure=str(pdf), differences=y)))


if __name__ == "__main__":
    main()
