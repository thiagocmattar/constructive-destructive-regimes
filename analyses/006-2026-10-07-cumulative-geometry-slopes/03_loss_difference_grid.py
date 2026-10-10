"""Plot exact loss differences against A/B/ratio slopes in a 3x2 grid."""

import csv
import hashlib
import json
from pathlib import Path

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    design = json.loads((HERE / "panel_design.json").read_text())
    original = json.loads((HERE / "results.json").read_text())
    events_path = ROOT / "runs/002-2026-10-03-pythia14m-adamw-adaptive-geometry/artifacts/attempt-02/full-pass/events.jsonl"
    inputs = [HERE / "curve_data.csv", HERE / "results.json", events_path]
    before = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in inputs}
    assert sha(HERE / "curve_data.csv") == original["outputs_sha256"]["curve_data.csv"]
    for path, expected in original["source_hashes_before"].items():
        assert sha(ROOT / path) == expected
    with (HERE / "curve_data.csv").open(newline="", encoding="utf-8") as stream:
        base = list(csv.DictReader(stream))
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    updates = [e for e in events if e["event"] == "update"]
    assert len(base) == len(updates) == 712
    steps = np.array([int(float(row["update"])) for row in base])
    np.testing.assert_array_equal(steps, np.arange(1, 713))
    loss = np.array([float(row["training_loss"]) for row in base])
    np.testing.assert_array_equal(loss, [e["loss"] for e in updates])
    loss_diff = np.r_[np.nan, np.diff(loss)]
    np.testing.assert_allclose(np.sum(loss_diff[1:]), loss[-1] - loss[0], atol=1e-13)
    display = (steps >= 2) & (steps <= 711)
    assert display.sum() == design["display_points_per_panel"] == 710
    assert all(e["sequences"] == 1024 for e in updates[:-1])
    assert updates[-1]["sequences"] == 311
    arrays = {"update": steps, "displayed": display.astype(int),
              "sequences": np.array([e["sequences"] for e in updates]),
              "training_loss": loss, "training_loss_difference": loss_diff}
    checked = {}
    for name in ("raw", "adaptive"):
        vectors = updates if name == "raw" else [e["adaptive"] for e in updates]
        b_key = "cumulative_gradient_norm" if name == "raw" else "cumulative_direction_norm"
        a = np.array([e["cumulative_norm_sum"] for e in vectors])
        b = np.array([e[b_key] for e in vectors])
        expected = {"A": np.diff(np.r_[0, a]), "B": np.diff(np.r_[0, b]),
                    "ratio": np.r_[np.nan, np.diff(b / a)]}
        checked[name] = {}
        for key, values in expected.items():
            recorded = np.array([float(row[name + "_delta_" + key])
                                 if row[name + "_delta_" + key] else np.nan for row in base])
            np.testing.assert_allclose(recorded, values, rtol=1e-12, atol=1e-14, equal_nan=True)
            arrays[name + "_slope_" + key] = recorded
            assert np.all(np.isfinite(recorded[display]))
            checked[name][key] = {"points": int(display.sum()),
                                 "minimum": float(recorded[display].min()),
                                 "maximum": float(recorded[display].max())}
    output_csv = HERE / "loss_difference_grid_data.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(arrays))
        writer.writeheader()
        for i in range(712):
            writer.writerow({k: None if not np.isfinite(v[i]) else float(v[i]) for k, v in arrays.items()})
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharey=True, constrained_layout=True)
    labels = [
        ("A", r"Slope of A: $A_t-A_{t-1}$ (norm units / update)"),
        ("B", r"Slope of B: $B_t-B_{t-1}$ (norm units / update)"),
        ("ratio", r"Slope of cumulative B/A: $(B/A)_t-(B/A)_{t-1}$"),
    ]
    for col, name in enumerate(("raw", "adaptive")):
        axes[0, col].set_title("Raw gradients" if name == "raw" else "Adam adaptive directions")
        for row, (key, xlabel) in enumerate(labels):
            ax = axes[row, col]
            x = arrays[name + "_slope_" + key][display]
            y = loss_diff[display]
            points = ax.scatter(x, y, c=steps[display], cmap="viridis", vmin=2, vmax=711,
                                s=15, alpha=.72, linewidths=0)
            np.testing.assert_array_equal(np.asarray(points.get_offsets()), np.column_stack([x, y]))
            ax.axhline(0, color="#555555", ls=":", lw=.8)
            ax.grid(alpha=.15)
            ax.set_xlabel(xlabel)
            if col == 0:
                ax.set_ylabel(r"Training-loss difference $L_t-L_{t-1}$")
            if row == 2:
                ax.set_xscale("symlog", linthresh=1e-4)
                ax.set_xlim(-.09, .01)
                ax.axvline(0, color="#555555", ls=":", lw=.8)
    bar = fig.colorbar(points, ax=axes, shrink=.85, pad=.02)
    bar.set_label("Optimizer update")
    fig.suptitle("Per-step training-loss difference versus cumulative-geometry slopes\n"
                 "Pythia-14M, seed 1234; 710 paired points per panel, updates 2-711; no smoothing\n"
                 "Ratio-slope x-axes use symmetric log scaling; loss is measured on changing batches",
                 fontsize=11)
    pdf = HERE / "figures/loss_difference_vs_slopes.pdf"
    fig.savefig(pdf)
    plt.close(fig)
    preview = HERE / "scratch/loss_difference_vs_slopes_pdf.png"
    with fitz.open(pdf) as document:
        assert len(document) == 1
        assert "Raw gradients" in document[0].get_text()
        assert "Adam adaptive directions" in document[0].get_text()
        document[0].get_pixmap(matrix=fitz.Matrix(1.7, 1.7), alpha=False).save(preview)
    after = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in inputs}
    assert before == after
    results = {
        "design_sha256": sha(HERE / "panel_design.json"),
        "script_sha256": sha(Path(__file__)),
        "source_hashes_before": before, "source_hashes_after": after,
        "sources_unchanged": True, "retained_rows": 712,
        "finite_loss_differences": 711, "display_updates": [2, 711],
        "points_per_panel": 710, "panels": checked,
        "loss_difference_display_range": [float(loss_diff[display].min()), float(loss_diff[display].max())],
        "checks": {"source_slopes_match": True, "loss_matches_events": True,
                   "loss_difference_telescoping": True, "scatter_coordinates_match": True,
                   "one_pdf_page": True},
        "outputs_sha256": {str(p.relative_to(HERE)).replace("\\", "/"): sha(p) for p in (pdf, output_csv)},
        "pdf_preview": str(preview.relative_to(HERE)).replace("\\", "/"),
    }
    (HERE / "loss_difference_grid_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sources_unchanged": True, "retained_rows": 712, "points_per_panel": 710,
                      "pdf": str(pdf.relative_to(ROOT)).replace("\\", "/"),
                      "loss_difference_display_range": results["loss_difference_display_range"]}, indent=2))


if __name__ == "__main__":
    main()
