"""Run-local reductions for the requested identity on immutable AdamW histories."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "analyses/016-2026-10-11-rosenbrock-hessian"
sys.path.insert(0, str(ROOT))
from src.controlled.rosenbrock import sha256, write_json


def digest(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def rolling(values, width=100):
    values = np.asarray(values, dtype=np.float64)
    result = np.full_like(values, np.nan)
    finite = np.isfinite(values)
    sums = np.convolve(np.where(finite, values, 0), np.ones(width), "valid")
    counts = np.convolve(finite.astype(float), np.ones(width), "valid")
    result[width - 1:] = np.where(counts == width, sums / width, np.nan)
    return result


def identity(vectors, eta, epsilon=0.0):
    if eta <= 0:
        raise ValueError("eta must be positive")
    g = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(g, axis=1)
    proxy = np.full(len(g), np.nan)
    valid = (norms[:-1] > 0) & (norms[1:] > 0)
    rho = np.full(len(g) - 1, np.nan)
    rho[valid] = np.sum(g[:-1][valid] * g[1:][valid], axis=1) / (norms[:-1][valid] * norms[1:][valid])
    proxy[1:][valid] = (1 - rho[valid] * norms[1:][valid] / norms[:-1][valid] + epsilon) / eta
    return proxy


def load_sources():
    cohort = json.loads((SOURCE / "cohort.json").read_text())
    histories, sources, turns = [], [], []
    for entry in cohort:
        folder = ROOT / entry["directory"]
        path = folder / "artifacts/attempts/001-local/history.npz"
        receipt = json.loads((folder / "verification.json").read_text())
        assert receipt["passed"] and receipt["history_sha256"] == sha256(path)
        assert receipt["config_sha256"] == sha256(folder / "config.json")
        with np.load(path, allow_pickle=False) as saved:
            history = {key: saved[key] for key in saved.files}
        histories.append(history)
        sources.append(dict(entry, history_sha256=sha256(path), history_bytes=path.stat().st_size,
                            config_sha256=sha256(folder / "config.json"), verification_sha256=sha256(folder / "verification.json")))
        dy = np.diff(history["theta"][:, 1])
        turn = int(np.flatnonzero((dy[:-1] < 0) & (dy[1:] > 0))[0] + 1)
        turns.append(dict(run_id=entry["run_id"], learning_rate=entry["learning_rate"],
                          update=turn, point=history["theta"][turn].tolist()))
    return histories, sources, dict(definition="First strict local minimum of y", indexing="position theta[t] after update t; unchanged position-clock references", points=turns)


def reduce(history, eta):
    g, norms = history["raw_vectors"], history["raw_norm"]
    result = {"updates": np.arange(1, len(g) + 1)}
    proxy = np.full(len(g), np.nan)
    valid = (norms[:-1] > 0) & (norms[1:] > 0) & np.isfinite(history["raw_cosine"][1:])
    proxy[1:][valid] = (1 - history["raw_cosine"][1:][valid] * norms[1:][valid] / norms[:-1][valid]) / eta
    result["kappa_proxy"] = proxy
    u, h = g[:-1], history["hessian"][:-2]
    squared = np.sum(u * u, axis=1)
    delta = np.diff(history["theta"], axis=0)[:-1]
    remainder = g[1:] - u - np.einsum("nij,nj->ni", h, delta)
    fields = dict(kappa_direction=np.einsum("ni,nij,nj->n", u, h, u),
                  optimizer_mismatch=-np.einsum("ni,nij,nj->n", u, h, delta + eta * u) / eta,
                  nonlinear_remainder=-np.sum(u * remainder, axis=1) / eta)
    for name, numerator in fields.items():
        values = np.full(len(g), np.nan)
        values[1:][valid] = numerator[valid] / squared[valid]
        result[name] = values
    for name in ("lambda_min", "lambda_max"):
        result[name] = np.r_[np.nan, history["hessian_" + name][:-2]]
    result["proxy_minus_direction"] = proxy - result["kappa_direction"]
    for name in ("kappa_proxy", "kappa_direction", "lambda_min", "lambda_max", "proxy_minus_direction"):
        result[name + "_mean100"] = rolling(result[name])
    return result


def association(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def summary(values):
    proxy, direction, maximum = (values[k] for k in ("kappa_proxy", "kappa_direction", "lambda_max"))
    valid = np.isfinite(proxy)
    return dict(defined_pairs=int(valid.sum()), undefined_updates=values["updates"][~valid].tolist(),
                pearson_proxy_lambda_raw=association(proxy, maximum),
                pearson_proxy_lambda_mean100=association(values["kappa_proxy_mean100"], values["lambda_max_mean100"]),
                pearson_proxy_direction_raw=association(proxy, direction),
                last100_pair_means={k: float(np.mean(values[k][-100:])) for k in
                    ("kappa_proxy", "kappa_direction", "lambda_max", "optimizer_mismatch", "nonlinear_remainder")},
                proxy_range=[float(np.nanmin(proxy)), float(np.nanmax(proxy))],
                outside_local_spectrum=int(np.sum(valid & ((proxy < values["lambda_min"]) | (proxy > maximum)))),
                mean_absolute_proxy_minus_direction=float(np.nanmean(np.abs(proxy - direction))),
                mean_absolute_proxy_minus_lambda=float(np.nanmean(np.abs(proxy - maximum))))
