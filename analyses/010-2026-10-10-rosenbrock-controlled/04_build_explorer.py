"""Embed every recorded update into a self-contained, offline scientific explorer."""
import base64
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pilot_plots", HERE / "02_plot.py")
plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plots)


def rounded(value):
    return float(f"{float(value):.8g}") if np.isfinite(value) else None


def contour_paths(xlim, ylim, levels, size):
    fig, ax = plots.plt.subplots()
    xx, yy = np.meshgrid(np.linspace(*xlim, size), np.linspace(*ylim, size))
    values = (1 - xx) ** 2 + 100 * (yy - xx ** 2) ** 2
    contour = ax.contour(xx, yy, values, levels=levels)
    paths = [[[rounded(x), rounded(y)] for x, y in segment]
             for level in contour.allsegs for segment in level if len(segment) > 1]
    plots.plt.close(fig)
    return {"x": xlim, "y": ylim, "levels": levels, "mesh": [size, size], "paths": paths}


def main():
    rows = plots.records()
    columns = ["step", "x", "y", "loss", "gradient_norm"]
    quantities = ("cosine", "norm", "A", "B", "ratio", "all_pair", "lag_one")
    columns += [name + "_" + quantity for name in plots.STREAMS for quantity in quantities]
    ratios = ["momentum_to_raw_norm_ratio", "adaptive_to_raw_norm_ratio", "adaptive_to_momentum_norm_ratio"]
    columns += ratios
    payload = {"columns": columns, "runs": [], "contours": {
        "full": contour_paths([-1.5, 1.35], [-.35, 1.6], [.001,.01,.1,.5,1,2,5,10,25,50,100,250,500], 500)}}
    for row in rows:
        d = row["data"]
        values = []
        for step in range(10001):
            record = [step, rounded(d["theta"][step, 0]), rounded(d["theta"][step, 1]),
                      rounded(d["loss"][step]), rounded(d["post_gradient_norm"][step])]
            record += [rounded(d[name + "_" + quantity][step-1]) if step else None
                       for name in plots.STREAMS for quantity in quantities]
            record += [rounded(d[key][step-1]) if step else None for key in ratios]
            values.append(record)
        payload["runs"].append({"id": row["run_id"], "label": row["label"], "beta1": row["beta1"],
                                "decay": row["weight_decay"], "rows": values})
    encoded = base64.b64encode(gzip.compress(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode(), mtime=0)).decode()
    template = HERE / "03_explorer_template.html"
    output = HERE / "rosenbrock-explorer.html"
    output.write_text(template.read_text(encoding="utf-8").replace("__PAYLOAD__", encoded), encoding="utf-8")
    metadata = {"source_histories": [{"run": row["run_id"], "sha256": plots.sha(row["path"])} for row in rows],
                "generator_sha256": plots.sha(Path(__file__)), "template_sha256": plots.sha(template),
                "html_sha256": plots.sha(output), "size_bytes": output.stat().st_size,
                "updates_per_condition": 10000, "positions_per_condition": 10001,
                "display_rounding": "8 significant digits; exact NPZ source retained", "step_sampling": "none",
                "contours": {k: {key: value for key, value in v.items() if key != "paths"} for k, v in payload["contours"].items()},
                "transition_zoom": "100x100 analytic mesh centered at selected post-point; halfwidth=max(0.002,4*displacement_norm); linearly interpolated contour crossings",
                "external_dependencies": [], "undefined_metric_representation": "null"}
    (HERE / "explorer_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    # Check compressed delivery integrity and exact step/column coverage.
    decoded = json.loads(gzip.decompress(base64.b64decode(encoded)))
    assert decoded == payload
    assert all(len(run["rows"]) == 10001 and all(len(r) == len(columns) for r in run["rows"]) for run in decoded["runs"])
    print(json.dumps({"explorer": str(output), "size_bytes": output.stat().st_size, "payload_verified": True}))


if __name__ == "__main__":
    main()
