"""Regenerate within-LR geometry with independent linear cumulative-norm axes."""
from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import json
import string
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "complete-three-lr"
HELPER = HERE / "08_plot_simplified.py"
spec = importlib.util.spec_from_file_location("geometry_styles", HELPER)
plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plots)


def main():
    cohort = json.loads((HERE / "cohort.json").read_text())["runs"]
    records = [plots.load_run(record) for record in cohort]
    data_path = OUT / "comparison_data.json"
    data = json.loads(data_path.read_text())
    for record in records:
        saved = data["runs"][record["id"]]
        assert record["events"] == saved["events"]
        assert record["events_sha256"] == saved["sources"]["events.jsonl"]
    limits = {}
    for record in records:
        norms = [plots.values(record["displayed"], stream, key)
                 for stream in ("raw", "momentum")
                 for key in ("cumulative_norm_sum", plots.STREAMS[stream][2])]
        assert all(np.all(np.isfinite(values)) and np.all(values >= 0) for values in norms)
        limits[record["id"]] = (0, max(float(np.max(values)) for values in norms) * 1.06)
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 9), squeeze=False)
    for col, record in enumerate(records):
        events = record["displayed"]
        x = [event["update"] for event in events]
        for stream in ("raw", "momentum"):
            _, _, bkey, name, color, _ = plots.STREAMS[stream]
            axes[0, col].plot(x, plots.values(events, stream, "cumulative_norm_sum"), color=color, lw=1.8, label=f"{name}: A")
            axes[0, col].plot(x, plots.values(events, stream, bkey), color=color, lw=1.8, ls="--", label=f"{name}: B")
        for stream, (_, _, _, name, color, style) in plots.STREAMS.items():
            axes[1, col].plot(x, plots.values(events, stream, "cumulative_ratio"), color=color, ls=style, lw=1.8, label=name)
        axes[0, col].set(title=f"({string.ascii_lowercase[col]}) LR {record['lr']:g}: cumulative norms",
                         ylabel="Cumulative norm", yscale="linear", ylim=limits[record["id"]])
        axes[1, col].set(title=f"({string.ascii_lowercase[3+col]}) LR {record['lr']:g}: cumulative B/A",
                         ylabel="B/A", ylim=(0, 1.05))
        axes[0, col].legend(fontsize=9, loc="upper left", ncol=2)
        axes[1, col].legend(fontsize=10, loc="lower left")
    plots.format_axes(axes)
    fig.suptitle("Pythia-14M: geometry within each learning rate\nWarmup only; matched initial weights and data order; seed 1234", fontsize=15)
    fig.text(.5, .028, "A = sum of vector norms; B = norm of vector sum. Colors identify raw gradient g, stored momentum m and adaptive direction d.\n"
             "Cumulative norms: solid A, dashed B; independent linear y-ranges emphasize shape within each LR. B/A: shared linear y-scale.\n"
             "All curves are unsmoothed. Display: full-sized updates 1-711; partial update 712 retained in results. One matched seed; no cross-seed uncertainty estimated.",
             ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, .115, 1, .91), h_pad=2.3, w_pad=2.0)
    plots.save(fig, OUT, "geometry_within_learning_rate")
    metadata = {"created_at": datetime.now(timezone.utc).isoformat(), "script_sha256": plots.sha(__file__),
                "helper_sha256": plots.sha(HELPER), "source_data_sha256": plots.sha(data_path),
                "event_sources": {r["id"]: r["events_sha256"] for r in records},
                "first_row_scale": "linear", "first_row_y_limits_by_run": limits,
                "first_row_shared_scale": False, "second_row_scale": "linear",
                "main_display_updates": [1, 711], "partial_update_retained_in_results": True}
    (OUT / "geometry-figure.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
