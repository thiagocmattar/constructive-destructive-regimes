"""Noise-aware gradient-identity reductions on immutable Rosenbrock paths."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "analyses/014-2026-10-10-rosenbrock-gradient-noise"
REFERENCE = ROOT / "analyses/017-2026-10-12-rosenbrock-kappa-identity"
FIELDS = ("gradient_only", "epsilon", "noise_corrected", "eta_directional",
          "optimizer_mismatch", "nonlinear_remainder")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def rolling(values, width=100):
    values = np.asarray(values, dtype=np.float64)
    result = np.full_like(values, np.nan)
    if len(values) < width:
        return result
    finite = np.isfinite(values)
    sums = np.convolve(np.where(finite, values, 0), np.ones(width), "valid")
    counts = np.convolve(finite.astype(float), np.ones(width), "valid")
    result[width - 1:] = np.where(counts == width, sums / width, np.nan)
    return result


def identity(g, noise):
    g, noise = np.asarray(g, dtype=float), np.asarray(noise, dtype=float)
    assert g.ndim == 2 and g.shape == noise.shape
    norms = np.linalg.norm(g, axis=1)
    squared = np.sum(g[:-1] ** 2, axis=1)
    valid = (norms[:-1] > 0) & (norms[1:] > 0)
    rho = np.full(len(g) - 1, np.nan)
    rho[valid] = np.sum(g[:-1][valid] * g[1:][valid], axis=1) / (norms[:-1][valid] * norms[1:][valid])
    q, eps = np.full(len(g), np.nan), np.full(len(g), np.nan)
    q[1:][valid] = 1 - rho[valid] * norms[1:][valid] / norms[:-1][valid]
    eps[1:][valid] = np.sum(g[:-1][valid] * np.diff(noise, axis=0)[valid], axis=1) / squared[valid]
    return dict(gradient_only=q, epsilon=eps, noise_corrected=q + eps)


def load_sources():
    histories, sources = [], []
    cohort = json.loads((HERE / "cohort.json").read_text())
    for entry in cohort:
        folder = ROOT / entry["directory"]
        path = folder / "artifacts/attempts/001-local/history.npz"
        receipt = json.loads((folder / "verification.json").read_text())
        config = json.loads((folder / "config.json").read_text())
        assert receipt["passed"] and receipt["history_sha256"] == sha256(path)
        assert receipt["config_sha256"] == sha256(folder / "config.json")
        assert config["learning_rate"] == entry["learning_rate"]
        assert config["updates"] == 2000 and config["initial_point"] == [0.0, 1.0]
        assert config["betas"] == [0.9, 0.95] and config["weight_decay"] == 0.1
        assert config["epsilon"] == 1e-8 and config["precision"] == "float64"
        assert config["objective"]["a"] == 1 and config["objective"]["b"] == 100
        assert config.get("sigma", 0) == entry["sigma"]
        if entry["sigma"]:
            assert config["noise_seed"] == 1234
        with np.load(path, allow_pickle=False) as saved:
            history = {key: saved[key] for key in saved.files}
        if not entry["sigma"]:
            history["noise_vectors"] = np.zeros_like(history["raw_vectors"])
            history["signal_vectors"] = history["raw_vectors"].copy()
        assert history["theta"].shape == (2001, 2)
        assert history["raw_vectors"].shape == (2000, 2)
        histories.append(history)
        sources.append(dict(entry, history_sha256=sha256(path), history_bytes=path.stat().st_size,
                            config_sha256=sha256(folder / "config.json"),
                            verification_sha256=sha256(folder / "verification.json")))
    return histories, sources


def hessian(points):
    x, y = np.asarray(points).T
    result = np.empty((len(x), 2, 2))
    result[:, 0, 0] = 2 - 400 * y + 1200 * x * x
    result[:, 0, 1] = result[:, 1, 0] = -400 * x
    result[:, 1, 1] = 200
    return result


def reduce(history, eta):
    g, noise, signal = (history[name + "_vectors"] for name in ("raw", "noise", "signal"))
    result = dict(updates=np.arange(1, len(g) + 1), **identity(g, noise))
    u = g[:-1]
    squared = np.sum(u * u, axis=1)
    valid = np.isfinite(result["gradient_only"][1:])
    h = hessian(history["theta"][:-2])
    delta = np.diff(history["theta"], axis=0)[:-1]
    remainder = np.diff(signal, axis=0) - np.einsum("nij,nj->ni", h, delta)
    numerators = dict(eta_directional=eta * np.einsum("ni,nij,nj->n", u, h, u),
                      optimizer_mismatch=-np.einsum("ni,nij,nj->n", u, h, delta + eta * u),
                      nonlinear_remainder=-np.sum(u * remainder, axis=1))
    for name, numerator in numerators.items():
        values = np.full(len(g), np.nan)
        values[1:][valid] = numerator[valid] / squared[valid]
        result[name] = values
    for name in FIELDS:
        result[name + "_mean100"] = rolling(result[name])
    return result


def summary(values):
    return dict(defined_pairs=int(np.isfinite(values["gradient_only"]).sum()),
                complete_means=int(np.isfinite(values["gradient_only_mean100"]).sum()),
                last100_pair_means={k: float(values[k + "_mean100"][-1]) for k in FIELDS},
                mean100_ranges={k: [float(np.nanmin(values[k + "_mean100"])),
                                    float(np.nanmax(values[k + "_mean100"]))] for k in FIELDS})
