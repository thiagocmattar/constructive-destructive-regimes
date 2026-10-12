"""Compute both eta-scaled identities, retaining realized epsilon separately."""
import json

import numpy as np

from noise_kappa import HERE, ROOT, REFERENCE, FIELDS, load_sources, reduce, sha256, summary, write_json


def main():
    (HERE / "artifacts").mkdir(exist_ok=True)
    (HERE / "figures").mkdir(exist_ok=True)
    (HERE / "scratch").mkdir(exist_ok=True)
    histories, sources = load_sources()
    rows = []
    for history, source in zip(histories, sources):
        values = reduce(history, source["learning_rate"])
        path = HERE / "artifacts" / (source["run_id"] + "_kappa_noise.npz")
        np.savez_compressed(path, **values)
        row = dict(source, **summary(values), derived_history=path.relative_to(ROOT).as_posix(),
                   derived_history_bytes=path.stat().st_size, derived_history_sha256=sha256(path))
        row["smallest_previous_raw_norm_squared"] = float(np.sum(history["raw_vectors"][:-1] ** 2, axis=1).min())
        rows.append(row)
    protected = list(REFERENCE.glob("figures/*.pdf"))
    write_json(HERE / "results.json", dict(hypothesis="H017", sources=rows, runs=15,
               defined_pairs=15 * 1999, means_per_metric=15 * 1900,
               scientific_inputs=dict(config_sha256=sha256(HERE / "config.json"),
                   cohort_sha256=sha256(HERE / "cohort.json"),
                   metric_sha256=sha256(HERE / "noise_kappa.py"),
                   reference_data_sha256=sha256(REFERENCE / "eta_overlay_data.json"),
                   formula_snapshot_sha256=sha256(REFERENCE / "formula_source_snapshot.md")),
               protected_reference_figures=[dict(path=p.relative_to(ROOT).as_posix(), sha256=sha256(p)) for p in protected]))
    print(json.dumps(dict(runs=len(rows), pairs=len(rows) * 1999, means_per_metric=len(rows) * 1900)))
    for sigma in (0, .01, .1, 1, 10):
        print(json.dumps(dict(sigma=sigma, last100=[{k:r["last100_pair_means"][k] for k in FIELDS[:3]}
                  for r in rows if r["sigma"] == sigma])))


if __name__ == "__main__":
    main()
