"""Generate matched-lag 3x2 loss/geometry difference grids for k=2,5,20."""

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


def lag_difference(values, lag):
    result = np.full(len(values), np.nan)
    result[lag:] = values[lag:] - values[:-lag]
    return result


def main():
    design = json.loads((HERE / "lagged_panel_design.json").read_text())
    original = json.loads((HERE / "results.json").read_text())
    events_path = ROOT / "runs/002-2026-10-03-pythia14m-adamw-adaptive-geometry/artifacts/attempt-02/full-pass/events.jsonl"
    generated_names = {f"loss_difference_vs_slopes_lag{lag:02d}.pdf" for lag in design["lags"]}
    protected = [HERE / "curve_data.csv", HERE / "results.json", events_path,
                 HERE / "loss_difference_grid_data.csv", HERE / "loss_difference_grid_results.json",
                 *[p for p in sorted((HERE / "figures").glob("*.pdf")) if p.name not in generated_names]]
    before = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in protected}
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
    levels = {key: np.array([float(row[key]) for row in base])
              for key in ("training_loss", "raw_A", "raw_B", "raw_ratio",
                          "adaptive_A", "adaptive_B", "adaptive_ratio")}
    np.testing.assert_array_equal(levels["training_loss"], [e["loss"] for e in updates])
    for name in ("raw", "adaptive"):
        vectors = updates if name == "raw" else [e["adaptive"] for e in updates]
        b_key = "cumulative_gradient_norm" if name == "raw" else "cumulative_direction_norm"
        for key, source_key in (("A", "cumulative_norm_sum"), ("B", b_key), ("ratio", "cumulative_ratio")):
            np.testing.assert_array_equal(levels[name + "_" + key], [e[source_key] for e in vectors])
    assert all(e["sequences"] == 1024 for e in updates[:-1]) and updates[-1]["sequences"] == 311
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    csv_rows, records, outputs = [], {}, []
    for lag in design["lags"]:
        changes = {key: lag_difference(values, lag) for key, values in levels.items()}
        for key, values in levels.items():
            # Independent check: each k-step change equals the sum of k one-step changes.
            expected = np.convolve(np.diff(values), np.ones(lag), mode="valid")
            np.testing.assert_allclose(changes[key][lag:], expected, rtol=1e-11, atol=1e-9)
            assert np.all(np.isnan(changes[key][:lag]))
        display = (steps > lag) & (steps <= 711)
        assert display.sum() == 711 - lag
        for i in range(712):
            row = {"lag": lag, "update": int(steps[i]), "start_update": int(steps[i] - lag) if i >= lag else None,
                   "displayed": int(display[i]), "sequences": updates[i]["sequences"]}
            row.update({key + "_difference": None if not np.isfinite(values[i]) else float(values[i])
                        for key, values in changes.items()})
            csv_rows.append(row)
        fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharey=True, constrained_layout=True)
        labels = [
            ("A", rf"A change over {lag} updates: $A_t-A_{{t-{lag}}}$ (norm units)"),
            ("B", rf"B change over {lag} updates: $B_t-B_{{t-{lag}}}$ (norm units)"),
            ("ratio", rf"Cumulative B/A change: $(B/A)_t-(B/A)_{{t-{lag}}}$"),
        ]
        ratio_values = np.concatenate([changes[name + "_ratio"][display] for name in ("raw", "adaptive")])
        low, high = min(0, float(ratio_values.min())), max(0, float(ratio_values.max()))
        padding = max((high - low) * .1, 1e-5)
        panels = {}
        for col, name in enumerate(("raw", "adaptive")):
            axes[0, col].set_title("Raw gradients" if name == "raw" else "Adam adaptive directions")
            for row, (key, xlabel) in enumerate(labels):
                ax = axes[row, col]
                x, y = changes[name + "_" + key][display], changes["training_loss"][display]
                assert len(x) == len(y) == 711 - lag and np.all(np.isfinite(x)) and np.all(np.isfinite(y))
                points = ax.scatter(x, y, c=steps[display], cmap="viridis", vmin=2, vmax=711,
                                    s=15, alpha=.72, linewidths=0)
                np.testing.assert_array_equal(np.asarray(points.get_offsets()), np.column_stack([x, y]))
                ax.axhline(0, color="#555555", ls=":", lw=.8)
                ax.grid(alpha=.15)
                ax.set_xlabel(xlabel)
                if col == 0:
                    ax.set_ylabel(rf"Training-loss change $L_t-L_{{t-{lag}}}$")
                if row == 2:
                    ax.set_xscale("symlog", linthresh=1e-4)
                    ax.set_xlim(low - padding, high + padding)
                    ax.axvline(0, color="#555555", ls=":", lw=.8)
                panels[name + "_" + key] = {"points": len(x), "x_range": [float(x.min()), float(x.max())]}
        bar = fig.colorbar(points, ax=axes, shrink=.85, pad=.02)
        bar.set_label("Optimizer update (pair endpoint)")
        fig.suptitle(f"Lag {lag}: training-loss change versus cumulative-geometry changes\n"
                     f"Pythia-14M, seed 1234; {711-lag} points per panel, updates {lag+1}-711; same lag on both axes\n"
                     "No smoothing or division by lag; ratio x-axes use symmetric log scaling", fontsize=11)
        pdf = HERE / "figures" / f"loss_difference_vs_slopes_lag{lag:02d}.pdf"
        fig.savefig(pdf)
        plt.close(fig)
        preview = HERE / "scratch" / (pdf.stem + "_pdf.png")
        with fitz.open(pdf) as document:
            assert len(document) == 1
            assert "Raw gradients" in document[0].get_text() and "Adam adaptive directions" in document[0].get_text()
            document[0].get_pixmap(matrix=fitz.Matrix(1.7, 1.7), alpha=False).save(preview)
        outputs.append(pdf)
        records[str(lag)] = {"display_updates": [lag + 1, 711], "points_per_panel": 711 - lag,
                             "loss_difference_range": [float(y.min()), float(y.max())],
                             "panels": panels, "pdf": str(pdf.relative_to(HERE)).replace("\\", "/"),
                             "preview": str(preview.relative_to(HERE)).replace("\\", "/")}
    output_csv = HERE / "lagged_loss_difference_grid_data.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    outputs.append(output_csv)
    after = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in protected}
    assert before == after
    results = {"design_sha256": sha(HERE / "lagged_panel_design.json"), "script_sha256": sha(Path(__file__)),
               "sources_before": before, "sources_after": after, "previous_evidence_unchanged": True,
               "data_rows": len(csv_rows), "lags": records,
               "checks": {"source_levels_match": True, "same_lag_for_all_variables": True,
                          "lag_changes_equal_sums_of_single_step_changes": True,
                          "undefined_initial_differences_blank": True, "scatter_coordinates_match": True,
                          "each_pdf_one_page": True},
               "outputs_sha256": {str(p.relative_to(HERE)).replace("\\", "/"): sha(p) for p in outputs}}
    (HERE / "lagged_loss_difference_grid_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"previous_evidence_unchanged": True, "data_rows": len(csv_rows),
                      "figures": {k: {"points": v["points_per_panel"], "pdf": v["pdf"]} for k, v in records.items()}}, indent=2))


if __name__ == "__main__":
    main()
