"""Exact Hessian diagnostics for the three Analysis016 Rosenbrock runs.

The unchanged Rosenbrock simulator supplies the optimizer trajectory. Curvature
is reconstructible from its complete positions and never enters the update.
"""
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from src.controlled import rosenbrock as base
from src.controlled.rosenbrock import (
    STREAMS, geometry, gradient, objective, sha256, stamp, write_json,
)


def hessian(theta, b=100.0):
    """Full objective Hessian for one point or a (..., 2) array, float64."""
    theta = np.asarray(theta, dtype=np.float64)
    x, y = theta[..., 0], theta[..., 1]
    result = np.empty(theta.shape[:-1] + (2, 2), dtype=np.float64)
    result[..., 0, 0] = 2 + 12 * b * x * x - 4 * b * y
    result[..., 0, 1] = result[..., 1, 0] = -4 * b * x
    result[..., 1, 1] = 2 * b
    return result


def simulate(config, progress=None):
    result = base.simulate(config, progress)
    result["hessian"] = hessian(result["theta"], config["objective"]["b"])
    result["hessian_eigenvalues"] = np.linalg.eigvalsh(result["hessian"])
    result["hessian_lambda_min"] = result["hessian_eigenvalues"][:, 0]
    result["hessian_lambda_max"] = result["hessian_eigenvalues"][:, 1]
    return result


def summarize(config, result):
    summary = base.summarize(config, result)
    maximum = result["hessian_lambda_max"]
    minimum = result["hessian_lambda_min"]
    summary["curvature"] = dict(
        positions=len(maximum), initial_lambda_max=float(maximum[0]),
        final_lambda_max=float(maximum[-1]), minimum_lambda_max=float(maximum.min()),
        minimum_lambda_max_update=int(maximum.argmin()), maximum_lambda_max=float(maximum.max()),
        maximum_lambda_max_update=int(maximum.argmax()), initial_lambda_min=float(minimum[0]),
        final_lambda_min=float(minimum[-1]), negative_lambda_min_positions=int((minimum < 0).sum()),
    )
    return summary


def execute(run_dir):
    run_dir = Path(run_dir).resolve()
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = run_dir / "artifacts/attempts/001-local"
    if output.exists():
        raise FileExistsError(f"Executed attempt is immutable: {output}")
    output.mkdir(parents=True)
    source = Path(__file__).resolve()
    root = run_dir.parents[1]
    manifest = dict(
        schema_version=1, run_id=config["run_id"], attempt="001-local", status="running",
        started_utc=stamp(), config_sha256=sha256(config_path), source_sha256=sha256(source),
        source=source.relative_to(root).as_posix(),
        base_source=Path(base.__file__).relative_to(root).as_posix(), base_source_sha256=sha256(base.__file__),
        launcher_sha256=sha256(run_dir / "01_run.py"),
        initial_point_sha256=hashlib.sha256(np.array(config["initial_point"], dtype="<f8").tobytes()).hexdigest(),
        execution="local CPU / NumPy float64", command=sys.argv,
        environment=dict(python=sys.version, numpy=np.__version__, platform=platform.platform()),
        required_outputs=["history.npz", "summary.json", "final_state.json"],
        curvature_indexing="H(theta[t]), t=0..2000; post-update point at t>0; no L2 penalty",
    )
    write_json(output / "manifest.json", manifest)
    write_json(output / "resolved_config.json", config)
    clock = time.perf_counter()

    def progress(step, loss):
        elapsed = time.perf_counter() - clock
        event = dict(utc=stamp(), update=step, loss=loss, elapsed_seconds=elapsed,
                     updates_per_second=step / elapsed, remaining_seconds=(config["updates"] - step) * elapsed / step)
        with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")

    try:
        result = simulate(config, progress)
        np.savez_compressed(output / "history.npz", **result)
        summary = summarize(config, result)
        summary["elapsed_seconds"] = time.perf_counter() - clock
        write_json(output / "summary.json", summary)
        write_json(output / "final_state.json", dict(
            step=config["updates"], theta=result["theta"][-1].tolist(),
            m=result["momentum_vectors"][-1].tolist(), v=result["second_moment"][-1].tolist(),
            rng=None, scheduler=None, data_order=None,
            note="Deterministic complete histories retained; no resume requested.",
        ))
        write_json(output / "inventory.json", dict(files=[
            dict(path=p.name, size_bytes=p.stat().st_size, sha256=sha256(p))
            for p in sorted(output.iterdir()) if p.name != "manifest.json" and p.is_file()
        ]))
        manifest.update(status="executed_pending_verification", completed_utc=stamp(), completed_updates=config["updates"])
        write_json(output / "manifest.json", manifest)
        print(json.dumps(dict(run=config["run_id"], loss=summary["final_loss"],
                              curvature=summary["curvature"], seconds=summary["elapsed_seconds"])))
    except Exception as exc:
        manifest.update(status="failed", failure=repr(exc), failed_utc=stamp())
        write_json(output / "manifest.json", manifest)
        raise
