"""Scientific primitives shared by the approved H022 paired runs."""
from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh
from torch import nn


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8", newline="\n")
    temporary.replace(path)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_hash(tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def flat(tensors):
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def fixed_ce(logits, targets, reduction="mean"):
    target = logits.gather(1, targets[:, None])
    mask = F.one_hot(targets, logits.shape[-1]).bool()
    others = (logits - target).masked_fill(mask, -torch.inf)
    # logaddexp is smooth at zero and avoids a manually detached gradient.
    log_others = torch.logsumexp(others, dim=-1)
    loss = torch.logaddexp(torch.zeros_like(log_others), log_others)
    return loss.mean() if reduction == "mean" else loss.sum() if reduction == "sum" else loss


def training_loss(logits, targets, arm):
    if arm == "fixed32":
        return fixed_ce(logits, targets)
    return F.cross_entropy(logits.double() if arm == "ce64" else logits, targets)


class TinyTransformer(nn.Module):
    def __init__(self, width=64, heads=4, ff=256, classes=31):
        super().__init__()
        self.width, self.heads = width, heads
        self.embedding = nn.Embedding(classes + 1, width)
        self.position = nn.Parameter(torch.empty(3, width))
        self.ln1 = nn.LayerNorm(width, eps=1e-5)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.ln2 = nn.LayerNorm(width, eps=1e-5)
        self.fc1 = nn.Linear(width, ff, bias=False)
        self.fc2 = nn.Linear(ff, width, bias=False)
        self.lnf = nn.LayerNorm(width, eps=1e-5)
        self.readout = nn.Linear(width, classes, bias=False)
        self.register_buffer("causal", torch.ones(3, 3, dtype=torch.bool).triu(1), persistent=False)

    def forward(self, tokens):
        x = self.embedding(tokens) + self.position
        q, k, v = self.qkv(self.ln1(x)).reshape(-1, 3, 3, self.heads,
                                               self.width // self.heads).permute(2, 0, 3, 1, 4)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.width // self.heads)
        probabilities = scores.masked_fill(self.causal, -torch.inf).softmax(-1)
        attended = (probabilities @ v).transpose(1, 2).reshape(-1, 3, self.width)
        x = x + self.out(attended)
        x = x + self.fc2(F.relu(self.fc1(self.ln2(x))))
        return self.readout(self.lnf(x)[:, -1])


def initialize(seed, device="cpu", dtype=torch.float32, **kwargs):
    # Frozen initial tensors are prepared on CPU once, then shared between arms.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = TinyTransformer(**kwargs)
        for name, parameter in model.named_parameters():
            if "ln" in name and name.endswith("weight"):
                nn.init.ones_(parameter)
            elif "ln" in name and name.endswith("bias"):
                nn.init.zeros_(parameter)
            else:
                nn.init.normal_(parameter, std=0.02)
    return model.to(device=device, dtype=dtype)


def make_data(device="cpu"):
    a, b = np.meshgrid(np.arange(31), np.arange(31), indexing="ij")
    tokens = np.stack((a.ravel(), b.ravel(), np.full(961, 31)), axis=1)
    labels = ((a + b) % 31).ravel()
    ordering = np.random.default_rng(314159).permutation(961)
    x = torch.tensor(tokens[ordering], dtype=torch.long, device=device)
    y = torch.tensor(labels[ordering], dtype=torch.long, device=device)
    return x[:480], y[:480], x[480:], y[480:]


def configure():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)


def optimizer_for(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-8,
                            weight_decay=1.0, foreach=False, fused=False)


def adam_vectors(model, optimizer, step):
    m = flat([optimizer.state[p]["exp_avg"] for p in model.parameters()])
    v = flat([optimizer.state[p]["exp_avg_sq"] for p in model.parameters()])
    mhat = m / (1 - 0.9 ** step)
    denominator = (v / (1 - 0.98 ** step)).sqrt() + 1e-8
    return m, mhat / denominator, denominator


class Geometry:
    names = ("raw", "momentum", "adaptive", "update")

    def __init__(self, parameter_count, device):
        self.previous = torch.zeros((4, parameter_count), dtype=torch.float64, device=device)
        self.total = torch.zeros_like(self.previous)
        self.A = torch.zeros(4, dtype=torch.float64, device=device)
        self.previous_norm = torch.zeros_like(self.A)
        self.count = 0

    def capture(self, vectors):
        matrix = torch.stack(vectors).double()
        norm = matrix.norm(dim=1)
        cosine = (matrix * self.previous).sum(1) / (norm * self.previous_norm)
        self.total.add_(matrix)
        self.A.add_(norm)
        B = self.total.norm(dim=1)
        result = torch.stack((norm, cosine, self.A, B, B / self.A), dim=1)
        self.previous.copy_(matrix)
        self.previous_norm.copy_(norm)
        self.count += 1
        return result

    def state(self):
        return {"previous": self.previous, "total": self.total, "A": self.A,
                "previous_norm": self.previous_norm, "count": self.count}


def logit_audit(logits, targets):
    z = logits.detach().requires_grad_(True)
    native = torch.autograd.grad(F.cross_entropy(z, targets, reduction="sum"), z)[0]
    z64 = logits.detach().double().requires_grad_(True)
    reference_loss = F.cross_entropy(z64, targets, reduction="sum")
    reference = torch.autograd.grad(reference_loss, z64)[0]
    corrected = torch.autograd.grad(fixed_ce(z, targets, reduction="sum"), z)[0]
    target = native.gather(1, targets[:, None]).squeeze(1)
    target_ref = reference.gather(1, targets[:, None]).squeeze(1)
    target_fixed = corrected.gather(1, targets[:, None]).squeeze(1)
    collapsed = (target == 0) & (target_ref != 0) & (native.max(1).values > 0)
    mask = F.one_hot(targets, logits.shape[-1]).bool()
    shifted = z64.detach() - z64.detach().gather(1, targets[:, None])
    margins = -torch.logsumexp(shifted.masked_fill(mask, -torch.inf), dim=1)
    denominator = native.abs().sum()
    defect = native.sum() / denominator
    relative = (corrected.double() - reference).norm() / reference.norm()
    ref_collapse = ((target_ref == 0) & (reference.max(1).values > 0)).double().mean()
    restored = (collapsed & (target_fixed < 0)).sum() / collapsed.sum()
    return torch.stack((reference_loss.detach() / len(targets), collapsed.double().mean(),
                        defect.double(), relative, ref_collapse, restored.double(),
                        margins.min(), torch.quantile(margins, 0.1), margins.median(),
                        torch.quantile(margins, 0.9), margins.max(),
                        (margins > 16.6).double().mean(), (margins > 36.7).double().mean()))


AUDIT_NAMES = ("common_logits_loss", "native_collapse_fraction", "native_zero_sum_defect",
               "fixed_reference_logit_gradient_relative_error", "reference64_collapse_fraction",
               "collapsed_restored_fraction", "margin_min", "margin_q10", "margin_median",
               "margin_q90", "margin_max", "margin_above_16_6", "margin_above_36_7")


class Hessian:
    """A common fully FP64 objective; retained derivative graph at a frozen state."""
    def __init__(self, model, x, y):
        self.model, self.params = model, tuple(model.parameters())
        self.loss = fixed_ce(model(x), y)
        self.gradient_tensors = torch.autograd.grad(self.loss, self.params, create_graph=True)
        self.gradient = flat(self.gradient_tensors)
        self.count = self.gradient.numel()
        self.calls = 0

    def product(self, vector):
        vector = vector.to(self.gradient)
        result = torch.autograd.grad((self.gradient * vector).sum(), self.params,
                                     retain_graph=True)
        self.calls += 1
        return flat(result).detach()

    def rayleigh(self, vector):
        vector = vector.to(self.gradient)
        norm2 = vector.dot(vector)
        if not bool(norm2 > 0):
            return None
        return float(vector.dot(self.product(vector)) / norm2)

    def largest(self, scaling=None):
        start, before = time.perf_counter(), self.calls
        scale = torch.ones_like(self.gradient) if scaling is None else scaling.to(self.gradient)
        def product(array):
            vector = torch.from_numpy(np.array(array, copy=True)).to(self.gradient)
            return (self.product(vector * scale) * scale).cpu().numpy()
        operator = LinearOperator((self.count, self.count), matvec=product, dtype=np.float64)
        v0 = np.random.default_rng(7).normal(size=self.count)
        attempts = []
        for ncv, maxiter in ((20, 200), (60, 400)):
            try:
                values, vectors = eigsh(operator, k=1, which="LA", tol=1e-3,
                                        ncv=min(self.count, ncv), maxiter=maxiter, v0=v0)
                value, vector = float(values[0]), vectors[:, 0]
                absolute = float(np.linalg.norm(product(vector) - value * vector))
                relative = absolute / max(abs(value), 1e-300)
                attempts.append({"ncv": ncv, "maxiter": maxiter, "relative_residual": relative,
                                 "absolute_residual": absolute})
                if relative <= 0.005:
                    return {"value": value, "relative_residual": relative,
                            "hvp_calls": self.calls - before,
                            "seconds": time.perf_counter() - start, "attempts": attempts,
                            "objective_examples": 480, "precision": "whole-model-fp64", "which": "LA"}
            except ArpackNoConvergence:
                attempts.append({"ncv": ncv, "maxiter": maxiter, "error": "no_convergence"})
        raise RuntimeError("Hessian solve did not qualify: " + json.dumps(attempts))


def inventory(directory):
    root = Path(directory)
    files = [{"path": str(p.relative_to(root)).replace("\\", "/"),
              "bytes": p.stat().st_size, "sha256": sha(p)}
             for p in sorted(root.rglob("*")) if p.is_file() and p.name != "inventory.json"]
    result = {"utc": stamp(), "files": files, "total_bytes": sum(p["bytes"] for p in files)}
    save_json(root / "inventory.json", result)
    return result
