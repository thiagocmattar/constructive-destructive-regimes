"""Compute the new diagnostic without altering the source optimizer runs."""
import json
import time
from datetime import datetime, timezone

import numpy as np

from kappa import HERE, ROOT, SOURCE, digest, load_sources, reduce, sha256, summary, write_json


def main():
    started = time.perf_counter()
    config = json.loads((HERE / "config.json").read_text())
    assert config["epsilon"] == 0 and config["smoothing_window"] == 100
    histories, sources, turns = load_sources()
    (HERE / "artifacts").mkdir(exist_ok=True)
    rows = []
    for history, source in zip(histories, sources):
        result = reduce(history, source["learning_rate"])
        path = HERE / "artifacts" / (source["run_id"] + "_kappa.npz")
        np.savez_compressed(path, **result)
        rows.append(dict(source, **summary(result), derived_history=path.relative_to(ROOT).as_posix(),
                         derived_history_bytes=path.stat().st_size, derived_history_sha256=sha256(path),
                         arrays={key: dict(shape=list(value.shape), sha256=digest(value)) for key, value in result.items()}))
    write_json(HERE / "cohort.json", json.loads((SOURCE / "cohort.json").read_text()))
    write_json(HERE / "results.json", dict(hypothesis="H016", generated_at=datetime.now(timezone.utc).isoformat(),
        config_sha256=sha256(HERE / "config.json"), formula_source_sha256=sha256(HERE / "formula_source_snapshot.md"),
        formula_source_snapshot="formula_source_snapshot.md",
        source_analysis_verification_sha256=sha256(SOURCE / "verification.json"), sources=rows,
        trajectory_turns=turns, reduction_seconds=time.perf_counter() - started,
        derived_storage_bytes=sum(row["derived_history_bytes"] for row in rows),
        source_storage_reused_bytes=sum(row["history_bytes"] for row in rows),
        scripts={name: sha256(HERE / name) for name in ("00_compute.py", "kappa.py")},
        epsilon=0, pair_clock="t=2..2000; Hessian at theta[t-2]", smoothing_window=100,
        no_new_training=True, no_cloud=True))
    print(json.dumps([{k: row[k] for k in ("run_id", "pearson_proxy_lambda_raw", "pearson_proxy_lambda_mean100", "last100_pair_means")} for row in rows], indent=2))


if __name__ == "__main__":
    main()
