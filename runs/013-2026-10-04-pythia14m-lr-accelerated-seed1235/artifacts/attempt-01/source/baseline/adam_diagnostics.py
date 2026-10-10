"""Measure the Adam direction from actual moments, without LR or weight decay."""

import math
import torch


@torch.no_grad()
def adaptive_directions(optimizer, named_parameters, expected_step):
    """FP64 evaluation of bias-corrected preconditioning from FP32 optimizer state.

    Called after optimizer.step(); moments already contain the clipped gradient.
    This diagnostic neither simulates nor changes the optimizer's state.
    """
    directions = {}
    for group in optimizer.param_groups:
        if group.get("amsgrad", False) or group.get("maximize", False):
            raise ValueError("Only the matched ordinary AdamW recipe is supported.")
        beta1, beta2 = group["betas"]
        for p in group["params"]:
            state = optimizer.state[p]
            step = int(state["step"].item())
            if step != expected_step:
                raise ValueError("Adam moment update index differs from diagnostic index.")
            m, v = state["exp_avg"], state["exp_avg_sq"]
            if m.dtype != torch.float32 or v.dtype != torch.float32:
                raise ValueError("Require actual FP32 Adam moments.")
            direction = (m.double() / (1 - beta1 ** step)) / (
                (v.double() / (1 - beta2 ** step)).sqrt() + group["eps"])
            if not bool(torch.isfinite(direction).all()):
                raise RuntimeError("Nonfinite adaptive direction.")
            directions[id(p)] = direction
    scope = [(name, p) for name, p in named_parameters if p.requires_grad]
    if len(directions) != len(scope):
        raise ValueError("Optimizer and measurement parameter scopes differ.")
    return [directions[id(p)] for _, p in scope]


def norm_comparison(raw_norm, adaptive_norm):
    if not math.isfinite(raw_norm) or not math.isfinite(adaptive_norm):
        raise ValueError("Nonfinite norm comparison.")
    return {"adaptive_to_raw_norm_ratio": adaptive_norm / raw_norm if raw_norm > 0 else None,
            "adaptive_to_raw_norm_ratio_undefined_reason": None if raw_norm > 0 else "zero_raw_norm"}
