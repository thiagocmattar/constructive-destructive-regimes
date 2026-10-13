"""Run-local deterministic CIFAR MLP, exact streaming geometry and matrix-free Hessian."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh
import torch
from torch import nn


def stamp():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, allow_nan=False) + "\n")


def make_model(input_size=3072, widths=(200, 200), classes=10):
    layers = [nn.Flatten()]
    previous = input_size
    for width in widths:
        layers.extend([nn.Linear(previous, width), nn.Tanh()])
        previous = width
    return nn.Sequential(*layers, nn.Linear(previous, classes))


def flat(values):
    return torch.cat([value.reshape(-1) for value in values])


def loss_sum(prediction, labels, kind):
    if kind == "mse":
        target = torch.nn.functional.one_hot(labels, prediction.shape[1]).to(prediction.dtype)
        return (prediction - target).square().sum() * 0.5
    if kind == "ce":
        return torch.nn.functional.cross_entropy(prediction, labels, reduction="sum")
    raise ValueError(kind)


def batches(x, y, size):
    for start in range(0, len(x), size):
        yield x[start:start + size], y[start:start + size]


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def full_gradient(model, x, y, size, kind):
    model.zero_grad(set_to_none=True)
    loss = torch.zeros((), device=x.device, dtype=torch.float64)
    correct = torch.zeros((), device=x.device, dtype=torch.int64)
    for xb, yb in batches(x, y, size):
        prediction = model(xb)
        batch_loss = loss_sum(prediction, yb, kind) / len(x)
        batch_loss.backward()
        loss += batch_loss.detach().double()
        correct += (prediction.argmax(1) == yb).sum()
    return float(loss), float(correct) / len(x), flat([p.grad.detach() for p in model.parameters()])


@torch.no_grad()
def evaluate(model, x, y, size, kind):
    loss = torch.zeros((), device=x.device, dtype=torch.float64)
    correct = torch.zeros((), device=x.device, dtype=torch.int64)
    for xb, yb in batches(x, y, size):
        prediction = model(xb)
        loss += loss_sum(prediction, yb, kind).double() / len(x)
        correct += (prediction.argmax(1) == yb).sum()
    return {"loss": float(loss), "accuracy": float(correct) / len(x), "examples": len(x)}


def cosine(a, b):
    a, b = a.double(), b.double()
    denominator = float(a.norm() * b.norm())
    return float(a.dot(b)) / denominator if denominator > 0 else None


class Geometry:
    """All coordinates, float64 prefixes/reductions, no temporal truncation."""
    def __init__(self, count, device):
        self.total = torch.zeros(count, dtype=torch.float64, device=device)
        self.previous = None
        self.A = self.energy = self.lag_one = 0.0
        self.count = 0

    @torch.no_grad()
    def consume(self, vector):
        vector = vector.detach().double()
        if not bool(torch.isfinite(vector).all()):
            raise RuntimeError("Nonfinite diagnostic vector.")
        norm = float(vector.norm())
        adjacent = None if self.previous is None else cosine(self.previous, vector)
        dot = 0.0 if self.previous is None else float(self.previous.dot(vector))
        self.A += norm
        self.energy += norm * norm
        self.lag_one += 2 * dot
        self.total += vector
        B = float(self.total.norm())
        ratio = B / self.A if self.A > 0 else None
        if ratio is not None and not (0 <= ratio <= 1 + 1e-10):
            raise RuntimeError("Triangle inequality check failed.")
        if adjacent is not None and abs(adjacent) > 1 + 1e-10:
            raise RuntimeError("Cosine check failed.")
        result = {"norm": norm, "A": self.A, "B": B, "B_over_A": ratio,
                  "cosine": adjacent, "cosine_undefined_reason": ("first_update" if self.previous is None else "zero_norm") if adjacent is None else None,
                  "ratio_undefined_reason": "all_zero_vectors" if ratio is None else None,
                  "energy": self.energy, "all_pair": B * B - self.energy,
                  "lag_one": self.lag_one, "beyond_lag_one": B * B - self.energy - self.lag_one,
                  "A_squared": self.A * self.A, "B_squared": B * B,
                  "triangle_gap": self.A * self.A - B * B}
        self.previous = vector.clone()
        self.count += 1
        return result

    def state_dict(self):
        return {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v for k, v in vars(self).items()}

    def load_state_dict(self, state):
        for key, value in state.items():
            setattr(self, key, value.to(self.total.device).clone() if isinstance(value, torch.Tensor) else value)


def adam_vectors(optimizer, step):
    m, v = [], []
    for group in optimizer.param_groups:
        beta1, beta2 = group["betas"]
        for p in group["params"]:
            state = optimizer.state[p]
            if int(state["step"]) != step:
                raise RuntimeError("Optimizer diagnostic clocks differ.")
            m.append(state["exp_avg"].detach())
            v.append(state["exp_avg_sq"].detach())
    m, v = flat(m).double(), flat(v).double()
    mhat = m / (1 - beta1**step)
    denominator = (v / (1 - beta2**step)).sqrt() + group["eps"]
    return {"momentum": m, "unbiased_momentum": mhat, "adaptive": mhat / denominator}, denominator


class HessianOperator:
    def __init__(self, model, x, y, size, kind, denominator=None):
        self.model, self.x, self.y, self.size, self.kind = model, x, y, size, kind
        self.params = list(model.parameters())
        self.count = sum(p.numel() for p in self.params)
        self.scale = None if denominator is None else denominator.to(x.device, self.params[0].dtype).sqrt()
        if self.scale is not None and not bool((self.scale > 0).all()):
            raise ValueError("Preconditioner must be positive.")
        self.calls = 0

    def product(self, vector):
        self.calls += 1
        vector = vector.to(self.x.device, self.params[0].dtype)
        if self.scale is not None:
            vector = vector / self.scale
        result = torch.zeros_like(vector)
        for xb, yb in batches(self.x, self.y, self.size):
            loss = loss_sum(self.model(xb), yb, self.kind) / len(self.x)
            first = torch.autograd.grad(loss, self.params, create_graph=True)
            dot = flat(first).dot(vector)
            second = torch.autograd.grad(dot, self.params)
            result += flat(second).detach()
        return result if self.scale is None else result / self.scale

    def eigenvalues(self, config):
        start = time.perf_counter()
        dtype = np.float64 if self.params[0].dtype == torch.float64 else np.float32
        def product(array):
            return self.product(torch.from_numpy(np.array(array, copy=True))).cpu().numpy()
        operator = LinearOperator((self.count, self.count), matvec=product, dtype=dtype)
        # Deterministic start does not consume model/training RNG state.
        initial = np.random.default_rng(7).normal(size=self.count).astype(dtype)
        attempts = []
        settings = [{"ncv": config["ncv"], "maxiter": config["maxiter"]}]
        if config.get("fallback"):
            settings.append(config["fallback"])
        for setting in settings:
            try:
                values, vectors = eigsh(operator, k=config["k"], which=config["which"], tol=config["tolerance"],
                                        ncv=min(self.count, setting["ncv"]), maxiter=setting["maxiter"], v0=initial)
                order = np.argsort(values)[::-1]
                values, vectors = values[order], vectors[:, order]
                residuals = []
                for value, vector in zip(values, vectors.T):
                    residual = np.linalg.norm(product(vector) - value * vector)
                    residuals.append(float(residual / max(abs(float(value)), 1e-12)))
                attempts.append(dict(setting, status="passed" if max(residuals) <= config["residual_tolerance"] else "residual_failed", relative_residuals=residuals))
                if attempts[-1]["status"] == "passed":
                    break
            except ArpackNoConvergence:
                attempts.append(dict(setting, status="arpack_no_convergence"))
        if attempts[-1]["status"] != "passed":
            raise RuntimeError(f"Hessian eigensolver failed: {attempts}")
        synchronize(self.x.device)
        if max(residuals) > config["residual_tolerance"]:
            raise RuntimeError(f"Hessian eigenvalue residual failed: {residuals}")
        return {"eigenvalues": [float(v) for v in values], "relative_residuals": residuals,
                "hvp_calls": self.calls, "seconds": time.perf_counter() - start,
                "examples": len(self.x), "which": config["which"], "tolerance": config["tolerance"], "solver_attempts": attempts}


def curvature_event(model, data, config, state_step, optimizer=None):
    options = config["eigenvalues"]
    x, y = data["x_train"][:options["dataset_size"]], data["y_train"][:options["dataset_size"]]
    event = {"state_step": state_step, "utc": stamp(), "hessian": HessianOperator(model, x, y, config["physical_batch_size"], config["loss"]).eigenvalues(options)}
    if optimizer is not None and state_step > 0:
        _, D = adam_vectors(optimizer, state_step)
        beta1 = optimizer.param_groups[0]["betas"][0]
        # Matches the released Adam code exactly. Also expose conventional D scaling.
        P = (1 - beta1**state_step) * D
        preconditioned = HessianOperator(model, x, y, config["physical_batch_size"], config["loss"], P).eigenvalues(options)
        event["paper_preconditioned_hessian"] = preconditioned
        event["D_preconditioned_eigenvalues"] = [(1 - beta1**state_step) * e for e in preconditioned["eigenvalues"]]
    return event


def environment(device):
    return {"utc": stamp(), "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "numpy": np.__version__, "scipy": scipy.__version__,
            "device": str(device), "gpu": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None,
            "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory if str(device).startswith("cuda") else None,
            "matmul_precision": torch.get_float32_matmul_precision(), "tf32": torch.backends.cuda.matmul.allow_tf32}


def memory(device):
    if str(device).startswith("cuda"):
        return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
    return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
