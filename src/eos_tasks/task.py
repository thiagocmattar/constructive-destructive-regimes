"""Task-specific EOS objectives. Training batches never redefine the BN objective."""
from pathlib import Path
import importlib.util
import math
import time

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh

from src.cifar_eos.core import flat, make_model, synchronize


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def model_for(task, cache, vocabulary=None):
    if task == "mlp":
        return make_model()
    if task == "resnet":
        result = module(Path(cache) / "reference/resnet_cifar.py", "eos_resnet").resnet32()
        # Train-mode BN uses exactly the same fixed 1k-group statistics, without
        # mutating irrelevant running buffers during probes/HVPs/evaluation.
        for layer in result.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.track_running_stats = False
                layer.running_mean = layer.running_var = layer.num_batches_tracked = None
        return result
    if task == "transformer":
        return module(Path(cache) / "reference/model.py", "eos_transformer").TransformerModel(
            vocabulary, 200, 2, 200, 2, dropout=0)
    raise ValueError(task)


def configure():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    # Math attention supports second derivatives required by the Hessian.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.mha.set_fastpath_enabled(False)


class Task:
    def __init__(self, name, model, data, physical):
        self.name, self.model, self.data, self.physical = name, model, data, physical
        self.model.train()
        self.n = len(data["x_train"]) // 1000 if name == "resnet" else len(data["x_train"])
        self.units = int((data["y_train"] != -100).sum()) if name == "transformer" else len(data["y_train"])

    def chunks(self, ids=None, split="train"):
        x, y = self.data["x_" + split], self.data["y_" + split]
        if self.name == "resnet":
            groups = range(len(x) // 1000) if ids is None else ids.tolist()
            for i in groups:
                yield x[i*1000:(i+1)*1000], y[i*1000:(i+1)*1000]
        else:
            if ids is not None:
                x, y = x[ids], y[ids]
            for start in range(0, len(x), self.physical):
                yield x[start:start+self.physical], y[start:start+self.physical]

    def loss(self, x, y):
        if self.name == "transformer":
            output = self.model(x.T).transpose(0, 1)
            return torch.nn.functional.nll_loss(output.reshape(-1, output.shape[-1]), y.reshape(-1),
                                                reduction="sum", ignore_index=-100), output
        output = self.model(x)
        if self.name == "mlp":
            target = torch.nn.functional.one_hot(y, 10).to(output.dtype)
            return .5 * (output - target).square().sum(), output
        return torch.nn.functional.cross_entropy(output, y, reduction="sum"), output

    def scale(self, ids):
        if ids is None:
            return 1 / self.units
        # Uniform sampling of objective units remains unbiased even for padded
        # final language chunks: do not normalize by random sampled token count.
        return self.n / (len(ids) * self.units)

    def gradient(self, ids=None):
        self.model.zero_grad(set_to_none=True)
        scale = self.scale(ids)
        loss = torch.zeros((), dtype=torch.float64, device=self.data["x_train"].device)
        for x, y in self.chunks(ids):
            value, _ = self.loss(x, y)
            (value * scale).backward()
            loss += value.detach().double() * scale
        return float(loss), flat([p.grad.detach() for p in self.model.parameters()]).clone()

    @torch.no_grad()
    def evaluate(self, split="train"):
        y = self.data["y_"+split]
        count = int((y != -100).sum()) if self.name == "transformer" else len(y)
        value = torch.zeros((), dtype=torch.float64, device=y.device)
        for xb, yb in self.chunks(split=split):
            loss, _ = self.loss(xb, yb)
            value += loss.detach().double() / count
        return {"loss": float(value), "loss_units": count, "objective_units": len(y)//1000 if self.name == "resnet" else len(y)}

    def hvp(self, vector, ids=None):
        parameters = list(self.model.parameters())
        result = torch.zeros_like(vector)
        # For Hessian subsets use their own token/image-normalized objective,
        # as the paper does, rather than scaling by full training coverage.
        if ids is None:
            denominator = self.units
        elif self.name == "resnet":
            denominator = len(ids)*1000
        elif self.name == "transformer":
            denominator = int((self.data["y_train"][ids] != -100).sum())
        else:
            denominator = len(ids)
        for x, y in self.chunks(ids):
            value, _ = self.loss(x, y)
            first = torch.autograd.grad(value / denominator, parameters, create_graph=True)
            second = torch.autograd.grad(flat(first).dot(vector), parameters)
            result += flat(second).detach()
        return result

    def sharpness(self, ids, options):
        count = sum(p.numel() for p in self.model.parameters())
        device = next(self.model.parameters()).device
        calls = 0
        begin = time.perf_counter()
        def multiply(array):
            nonlocal calls
            calls += 1
            return self.hvp(torch.from_numpy(np.array(array, copy=True)).to(device), ids).cpu().numpy()
        operator = LinearOperator((count, count), matvec=multiply, dtype=np.float32)
        initial = np.random.default_rng(7).normal(size=count).astype(np.float32)
        values, vectors = eigsh(operator, k=options["k"], which="LA", tol=options["tolerance"],
                               ncv=options["ncv"], maxiter=options["maxiter"], v0=initial)
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        residuals = [float(np.linalg.norm(multiply(v)-float(a)*v) / max(abs(float(a)), 1e-12))
                     for a, v in zip(values, vectors.T)]
        if max(residuals) > options["residual_tolerance"]:
            raise RuntimeError(f"Eigenpair residual failed: {residuals}")
        synchronize(device)
        return {"eigenvalues": values.tolist(), "relative_residuals": residuals, "hvp_calls": calls,
                "seconds": time.perf_counter()-begin, "objective_units": self.n if ids is None else len(ids)}
