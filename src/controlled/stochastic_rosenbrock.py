"""Rosenbrock gradient estimates with additive Gaussian noise; frozen on execution."""
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from src.controlled import rosenbrock as exact
from src.controlled.noise_geometry import noise_terms

ROOT = Path(__file__).resolve().parents[2]
STREAMS = exact.STREAMS + ("signal", "noise")


def simulate(config, progress=None):
    n = config["updates"]
    rng = np.random.Generator(np.random.PCG64(config["noise_seed"]))
    standard_noise = rng.standard_normal((n, 2))
    noise = config["sigma"] * standard_noise
    theta = np.empty((n + 1, 2), dtype=np.float64)
    loss = np.empty(n + 1, dtype=np.float64)
    post_gradient_norm = np.empty(n + 1, dtype=np.float64)
    vectors = {name: np.empty((n, 2), dtype=np.float64) for name in STREAMS}
    second_moment = np.empty((n, 2), dtype=np.float64)
    theta[0] = config["initial_point"]
    a, b = config["objective"]["a"], config["objective"]["b"]
    loss[0] = exact.objective(theta[0], a, b)
    post_gradient_norm[0] = np.linalg.norm(exact.gradient(theta[0], a, b))
    m, v = np.zeros(2), np.zeros(2)
    for i in range(n):
        signal = exact.gradient(theta[i], a, b)
        g = signal + noise[i]
        theta[i + 1], m, v, d = exact.adamw_step(theta[i], m, v, g, i + 1, config)
        loss[i + 1] = exact.objective(theta[i + 1], a, b)
        post_gradient_norm[i + 1] = np.linalg.norm(exact.gradient(theta[i + 1], a, b))
        for name, value in zip(STREAMS, (g, m, d, theta[i + 1] - theta[i], signal, noise[i])):
            vectors[name][i] = value
        second_moment[i] = v
        if not np.isfinite(loss[i + 1]) or not np.isfinite(theta[i + 1]).all():
            raise RuntimeError(f"Nonfinite trajectory at update {i + 1}")
        if progress and (i + 1) % config["progress_every"] == 0:
            progress(i + 1, float(loss[i + 1]))
    result = dict(theta=theta, loss=loss, post_gradient_norm=post_gradient_norm,
                  second_moment=second_moment, standard_noise_vectors=standard_noise,
                  distance_to_minimum=np.linalg.norm(theta - [a, a * a], axis=1),
                  loss_change=np.diff(loss),
                  linear_loss_prediction=np.einsum("ij,ij->i", vectors["raw"], vectors["displacement"]),
                  signal_linear_loss_prediction=np.einsum("ij,ij->i", vectors["signal"], vectors["displacement"]))
    for name in STREAMS:
        result[name + "_vectors"] = vectors[name]
        result.update({name + "_" + key: value for key, value in exact.geometry(vectors[name]).items()})
    for numerator, denominator in (("momentum", "raw"), ("adaptive", "raw"), ("adaptive", "momentum")):
        ratio = np.full(n, np.nan)
        bottom = result[denominator + "_norm"]
        np.divide(result[numerator + "_norm"], bottom, out=ratio, where=bottom > 0)
        result[numerator + "_to_" + denominator + "_norm_ratio"] = ratio
    result.update({"decomposition_" + key: value for key, value in noise_terms(vectors["signal"], vectors["noise"]).items()})
    return result


def execute(run_dir):
    run_dir = Path(run_dir).resolve()
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text())
    output = run_dir / "artifacts/attempts/001-local"
    if output.exists():
        raise FileExistsError("Executed attempt is immutable: " + str(output))
    output.mkdir(parents=True)
    source = Path(__file__)
    dependencies = [Path(exact.__file__), ROOT / "src/controlled/noise_geometry.py"]
    manifest = dict(run_id=config["run_id"], attempt="001-local", status="running", started_utc=exact.stamp(),
                    source=str(source.relative_to(ROOT)), source_sha256=exact.sha256(source),
                    dependencies=[dict(path=str(p.relative_to(ROOT)), sha256=exact.sha256(p)) for p in dependencies],
                    config_sha256=exact.sha256(config_path), execution="local CPU / NumPy float64", command=sys.argv,
                    environment=dict(python=sys.version, numpy=np.__version__, platform=platform.platform()),
                    noise=dict(sigma=config["sigma"], seed=config["noise_seed"], generator="NumPy PCG64",
                               distribution="independent standard normal, shape (2000,2), scaled by sigma"))
    exact.write_json(output / "manifest.json", manifest)
    exact.write_json(output / "resolved_config.json", config)
    started = time.perf_counter()
    def progress(step, value):
        elapsed = time.perf_counter() - started
        event = dict(utc=exact.stamp(), update=step, loss=value, elapsed_seconds=elapsed,
                     updates_per_second=step / elapsed, remaining_seconds=(config["updates"] - step) * elapsed / step)
        with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
    try:
        result = simulate(config, progress)
        np.savez_compressed(output / "history.npz", **result)
        summary = exact.summarize(config, result)
        summary.update(sigma=config["sigma"], noise_seed=config["noise_seed"],
                       noise_mean=result["noise_vectors"].mean(axis=0).tolist(),
                       noise_covariance=np.cov(result["noise_vectors"], rowvar=False).tolist(),
                       elapsed_seconds=time.perf_counter() - started)
        exact.write_json(output / "summary.json", summary)
        final_rng = np.random.Generator(np.random.PCG64(config["noise_seed"]))
        final_rng.standard_normal((config["updates"], 2))
        exact.write_json(output / "final_state.json", dict(step=config["updates"], theta=result["theta"][-1].tolist(),
                         m=result["momentum_vectors"][-1].tolist(), v=result["second_moment"][-1].tolist(),
                         rng=final_rng.bit_generator.state,
                         noise_draws_consumed=config["updates"] * 2,
                         note="Full pre-generated standard-normal sequence retained in history; no resume requested."))
        inventory = dict(files=[dict(path=p.name, size_bytes=p.stat().st_size, sha256=exact.sha256(p))
                        for p in sorted(output.iterdir()) if p.is_file() and p.name != "manifest.json"])
        exact.write_json(output / "inventory.json", inventory)
        manifest.update(status="executed_pending_verification", completed_utc=exact.stamp(), completed_updates=config["updates"])
        exact.write_json(output / "manifest.json", manifest)
        print(json.dumps(dict(run=config["run_id"], sigma=config["sigma"], lr=config["learning_rate"],
                             final_loss=summary["final_loss"], seconds=summary["elapsed_seconds"])))
    except Exception as exc:
        manifest.update(status="failed", failure=repr(exc), failed_utc=exact.stamp())
        exact.write_json(output / "manifest.json", manifest)
        raise
