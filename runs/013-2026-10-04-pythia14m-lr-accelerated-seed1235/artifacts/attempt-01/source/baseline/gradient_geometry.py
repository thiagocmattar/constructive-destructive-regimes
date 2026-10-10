"""Whole-model adjacent cosine and exact streaming prefix sums of recorded gradients."""

from __future__ import annotations

import math
import torch


class GradientGeometry:
    def __init__(self, named_parameters, previous_dtype=torch.float32):
        self.parameters = [(name, p) for name, p in named_parameters if p.requires_grad]
        if not self.parameters or len({id(p) for _, p in self.parameters}) != len(self.parameters):
            raise ValueError("Require a nonempty, unique trainable parameter list.")
        self.names = [name for name, _ in self.parameters]
        self.cumulative = [torch.zeros_like(p, dtype=torch.float64) for _, p in self.parameters]
        self.previous = [torch.zeros_like(p, dtype=previous_dtype) for _, p in self.parameters]
        self.updates = 0
        self.norm_sum = 0.0
        self.previous_norm = None

    @torch.no_grad()
    def consume(self, vectors=None):
        if vectors is None:
            gradients = []
            for name, p in self.parameters:
                if p.grad is None:
                    raise RuntimeError(f"Missing expected gradient: {name}")
                if p.grad.dtype != torch.float32:
                    raise RuntimeError(f"Expected FP32 unscaled gradient: {name}")
                gradients.append(p.grad.detach())
        else:
            gradients = list(vectors)
            if len(gradients) != len(self.parameters):
                raise ValueError("Explicit vector parameter count differs.")
            for (name, p), g in zip(self.parameters, gradients):
                if g.shape != p.shape or g.device != p.device or g.dtype != self.previous[0].dtype:
                    raise ValueError(f"Explicit vector shape/device/dtype differs: {name}")
        energy = torch.stack([torch.sum(g.double().square()) for g in gradients]).sum()
        norm = float(energy.sqrt().item())
        if not math.isfinite(norm):
            raise RuntimeError("Nonfinite raw gradient; cumulative state was not changed.")
        cosine = None
        reason = "first_update" if self.updates == 0 else "zero_gradient_norm"
        if self.updates and norm > 0 and self.previous_norm > 0:
            dot = torch.stack([torch.sum(g.double() * old.double())
                               for g, old in zip(gradients, self.previous)]).sum()
            cosine = float(dot.item()) / (norm * self.previous_norm)
            if abs(cosine) > 1 + 1e-10:
                raise RuntimeError("Cosine outside numerical tolerance.")
            reason = None
        for total, old, g in zip(self.cumulative, self.previous, gradients):
            total.add_(g)
            old.copy_(g)
        self.norm_sum += norm
        self.updates += 1
        self.previous_norm = norm
        sum_norm = float(torch.stack([s.square().sum() for s in self.cumulative]).sum().sqrt().item())
        ratio = sum_norm / self.norm_sum if self.norm_sum > 0 else None
        if ratio is not None and (not math.isfinite(ratio) or ratio > 1 + 1e-10):
            raise RuntimeError("Cumulative ratio outside numerical tolerance.")
        return {
            "gradient_norm": norm,
            "adjacent_cosine": cosine, "adjacent_cosine_undefined_reason": reason,
            "cumulative_norm_sum": self.norm_sum,
            "cumulative_gradient_norm": sum_norm,
            "cumulative_ratio": ratio,
            "cumulative_ratio_undefined_reason": "all_zero_gradients" if ratio is None else None,
        }

    def state_dict(self):
        return {"schema_version": 1, "names": self.names, "updates": self.updates,
                "norm_sum": self.norm_sum, "previous_norm": self.previous_norm,
                "cumulative": [s.detach().cpu().clone() for s in self.cumulative],
                "previous": [p.detach().cpu().clone() for p in self.previous]}

    @torch.no_grad()
    def load_state_dict(self, state):
        if state["schema_version"] != 1 or state["names"] != self.names:
            raise ValueError("Diagnostic checkpoint parameter scope/order differs.")
        for key, destinations in (("cumulative", self.cumulative), ("previous", self.previous)):
            if len(state[key]) != len(destinations):
                raise ValueError("Diagnostic checkpoint parameter count differs.")
            for target, value in zip(destinations, state[key]):
                if target.shape != value.shape or target.dtype != value.dtype:
                    raise ValueError("Diagnostic checkpoint shape/dtype differs.")
                target.copy_(value)
        self.updates = int(state["updates"])
        self.norm_sum = float(state["norm_sum"])
        self.previous_norm = state["previous_norm"]
