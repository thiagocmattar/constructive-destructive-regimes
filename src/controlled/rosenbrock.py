"""Exact 2D Rosenbrock / AdamW experiment shared across controlled runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

STREAMS = ("raw", "momentum", "adaptive", "displacement")


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def objective(theta, a=1.0, b=100.0):
    x, y = np.asarray(theta, dtype=np.float64)
    return (a - x) ** 2 + b * (y - x * x) ** 2


def gradient(theta, a=1.0, b=100.0):
    x, y = np.asarray(theta, dtype=np.float64)
    return np.array([2 * (x - a) - 4 * b * x * (y - x * x),
                     2 * b * (y - x * x)], dtype=np.float64)


def adamw_step(theta, m, v, g, step, config):
    """Ordinary bias-corrected AdamW, epsilon outside sqrt, decay before descent."""
    beta1, beta2 = config["betas"]
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g * g
    d = (m / (1 - beta1 ** step)) / (np.sqrt(v / (1 - beta2 ** step)) + config["epsilon"])
    updated = theta * (1 - config["learning_rate"] * config["weight_decay"]) - config["learning_rate"] * d
    return updated, m, v, d


def geometry(vectors):
    """Every exact prefix; NaN storage means undefined, with reasons in contract."""
    norms = np.linalg.norm(vectors, axis=1)
    cumulative = np.cumsum(vectors, axis=0)
    a = np.cumsum(norms)
    b = np.linalg.norm(cumulative, axis=1)
    ratio = np.full(len(a), np.nan)
    np.divide(b, a, out=ratio, where=a > 0)
    cosine = np.full(len(a), np.nan)
    denominator = norms[:-1] * norms[1:]
    dot = np.einsum("ij,ij->i", vectors[:-1], vectors[1:])
    np.divide(dot, denominator, out=cosine[1:], where=denominator > 0)
    energy = np.cumsum(norms ** 2)
    lag_one = np.concatenate(([0.0], 2 * np.cumsum(dot)))
    all_pair = b ** 2 - energy
    return {"norm": norms, "A": a, "B": b, "ratio": ratio, "cosine": cosine,
            "A_squared": a ** 2, "B_squared": b ** 2, "triangle_gap": a ** 2 - b ** 2,
            "energy": energy, "all_pair": all_pair, "lag_one": lag_one,
            "beyond_lag_one": all_pair - lag_one}


def simulate(config, progress=None):
    n = config["updates"]
    theta = np.empty((n + 1, 2), dtype=np.float64)
    loss = np.empty(n + 1, dtype=np.float64)
    post_gradient_norm = np.empty(n + 1, dtype=np.float64)
    vectors = {name: np.empty((n, 2), dtype=np.float64) for name in STREAMS}
    second_moment = np.empty((n, 2), dtype=np.float64)
    theta[0] = config["initial_point"]
    a, b = config["objective"]["a"], config["objective"]["b"]
    loss[0] = objective(theta[0], a, b)
    post_gradient_norm[0] = np.linalg.norm(gradient(theta[0], a, b))
    m = np.zeros(2, dtype=np.float64)
    v = np.zeros(2, dtype=np.float64)
    for i in range(n):
        g = gradient(theta[i], a, b)
        theta[i + 1], m, v, d = adamw_step(theta[i], m, v, g, i + 1, config)
        loss[i + 1] = objective(theta[i + 1], a, b)
        post_gradient_norm[i + 1] = np.linalg.norm(gradient(theta[i + 1], a, b))
        for name, value in zip(STREAMS, (g, m, d, theta[i + 1] - theta[i])):
            vectors[name][i] = value
        second_moment[i] = v
        if not np.isfinite(loss[i + 1]) or not np.isfinite(theta[i + 1]).all():
            raise RuntimeError(f"Nonfinite trajectory at update {i + 1}")
        if progress and (i + 1) % config["progress_every"] == 0:
            progress(i + 1, float(loss[i + 1]))
    result = {"theta": theta, "loss": loss, "post_gradient_norm": post_gradient_norm,
              "second_moment": second_moment,
              "distance_to_minimum": np.linalg.norm(theta - np.array([a, a * a]), axis=1),
              "loss_change": np.diff(loss),
              "linear_loss_prediction": np.einsum("ij,ij->i", vectors["raw"], vectors["displacement"])}
    for name in STREAMS:
        result[name + "_vectors"] = vectors[name]
        result.update({name + "_" + key: value for key, value in geometry(vectors[name]).items()})
    for numerator, denominator in (("momentum", "raw"), ("adaptive", "raw"), ("adaptive", "momentum")):
        value = np.full(n, np.nan)
        bottom = result[denominator + "_norm"]
        np.divide(result[numerator + "_norm"], bottom, out=value, where=bottom > 0)
        result[numerator + "_to_" + denominator + "_norm_ratio"] = value
    return result


def summarize(config, result):
    criterion = config["convergence"]
    hit = (result["loss"] <= criterion["objective_at_most"]) & (result["post_gradient_norm"] <= criterion["gradient_norm_at_most"])
    indices = np.flatnonzero(hit)
    n = config["updates"]
    summary = {"run_id": config["run_id"], "updates": n, "initial_loss": float(result["loss"][0]),
               "final_loss": float(result["loss"][-1]), "minimum_observed_loss": float(result["loss"].min()),
               "minimum_loss_update": int(result["loss"].argmin()), "final_point": result["theta"][-1].tolist(),
               "final_gradient_norm": float(result["post_gradient_norm"][-1]),
               "final_distance_to_minimum": float(result["distance_to_minimum"][-1]),
               "first_convergence_threshold_update": int(indices[0]) if len(indices) else None,
               "final_100_updates_meet_threshold": bool(hit[-criterion["sustained_updates"]:].all()),
               "loss_increase_fraction": float(np.mean(result["loss_change"] > 0)),
               "late_loss_median": float(np.median(result["loss"][-1000:])),
               "late_loss_min": float(result["loss"][-1000:].min()),
               "late_loss_max": float(result["loss"][-1000:].max()), "streams": {}}
    for name in STREAMS:
        cosine = result[name + "_cosine"]
        finite = np.isfinite(cosine)
        decrease = result["loss_change"] < 0
        positive = finite & (cosine > 0)
        negative = finite & (cosine < 0)
        summary["streams"][name] = {
            "final_A": float(result[name + "_A"][-1]), "final_B": float(result[name + "_B"][-1]),
            "final_ratio": float(result[name + "_ratio"][-1]),
            "median_cosine": float(np.median(cosine[finite])),
            "negative_cosine_fraction": float(np.mean(cosine[finite] < 0)),
            "late_median_cosine": float(np.nanmedian(cosine[-1000:])),
            "negative_cosine_and_decreased_loss_count": int((negative & decrease).sum()),
            "positive_cosine_and_increased_loss_count": int((positive & (result["loss_change"] > 0)).sum()),
            "final_energy": float(result[name + "_energy"][-1]),
            "final_all_pair": float(result[name + "_all_pair"][-1]),
            "final_lag_one": float(result[name + "_lag_one"][-1]),
            "final_beyond_lag_one": float(result[name + "_beyond_lag_one"][-1])}
    return summary


def execute(run_dir):
    run_dir = Path(run_dir).resolve()
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = run_dir / "artifacts" / "attempts" / "001-local"
    if output.exists():
        raise FileExistsError(f"Executed attempt is immutable: {output}")
    output.mkdir(parents=True)
    source = Path(__file__).resolve()
    started = stamp()
    manifest = {"schema_version": 1, "run_id": config["run_id"], "attempt": "001-local",
                "status": "running", "started_utc": started,
                "config_sha256": sha256(config_path), "source_sha256": sha256(source),
                "source": str(source.relative_to(run_dir.parents[1])).replace("\\", "/"),
                "initial_point_sha256": hashlib.sha256(np.array(config["initial_point"], dtype="<f8").tobytes()).hexdigest(),
                "execution": "local CPU / NumPy float64", "command": sys.argv,
                "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
                "required_outputs": ["history.npz", "summary.json", "final_state.json"]}
    write_json(output / "manifest.json", manifest)
    write_json(output / "resolved_config.json", config)
    clock = time.perf_counter()
    def progress(step, loss):
        elapsed = time.perf_counter() - clock
        event = {"utc": stamp(), "update": step, "loss": loss, "elapsed_seconds": elapsed,
                 "updates_per_second": step / elapsed, "remaining_seconds": (config["updates"] - step) * elapsed / step}
        with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
    try:
        result = simulate(config, progress)
        np.savez_compressed(output / "history.npz", **result)
        summary = summarize(config, result)
        summary["elapsed_seconds"] = time.perf_counter() - clock
        write_json(output / "summary.json", summary)
        write_json(output / "final_state.json", {"step": config["updates"], "theta": result["theta"][-1].tolist(),
                   "m": result["momentum_vectors"][-1].tolist(), "v": result["second_moment"][-1].tolist(),
                   "rng": None, "scheduler": None, "data_order": None,
                   "note": "Deterministic exact gradients; complete histories retained; no resume requested."})
        inventory = {"files": [{"path": p.name, "size_bytes": p.stat().st_size, "sha256": sha256(p)}
                               for p in sorted(output.iterdir()) if p.name != "manifest.json" and p.is_file()]}
        write_json(output / "inventory.json", inventory)
        manifest.update(status="executed_pending_verification", completed_utc=stamp(), completed_updates=config["updates"])
        write_json(output / "manifest.json", manifest)
        print(json.dumps({"run": config["run_id"], "loss": summary["final_loss"],
                          "point": summary["final_point"], "seconds": summary["elapsed_seconds"]}))
    except Exception as exc:
        manifest.update(status="failed", failure=repr(exc), failed_utc=stamp())
        write_json(output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    execute(parser.parse_args().run_dir)
