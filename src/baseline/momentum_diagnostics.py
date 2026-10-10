"""Read Adam's stored first moment without bias correction or preconditioning."""
import math
import torch


@torch.no_grad()
def momentum_vectors(optimizer, named_parameters, expected_step):
    scope = [(name, p) for name, p in named_parameters if p.requires_grad]
    covered = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if covered != {id(p) for _, p in scope}:
        raise ValueError("Optimizer and momentum parameter scopes differ.")
    vectors = []
    for name, p in scope:
        state = optimizer.state[p]
        if int(state["step"].item()) != expected_step:
            raise ValueError("Adam momentum update index differs from diagnostic index.")
        m = state["exp_avg"]
        if m.dtype != torch.float32 or m.shape != p.shape or m.device != p.device:
            raise ValueError(f"Require actual FP32 momentum in parameter coordinates: {name}")
        if not bool(torch.isfinite(m).all()):
            raise RuntimeError("Nonfinite Adam momentum.")
        vectors.append(m.detach())
    return vectors


def momentum_norm_comparison(raw_norm, momentum_norm, adaptive_norm):
    if any(not math.isfinite(v) or v < 0 for v in (raw_norm, momentum_norm, adaptive_norm)):
        raise ValueError("Invalid momentum norm comparison.")
    return {"momentum_to_raw_norm_ratio": momentum_norm / raw_norm if raw_norm > 0 else None,
            "momentum_to_raw_norm_ratio_undefined_reason": None if raw_norm > 0 else "zero_raw_norm",
            "adaptive_to_momentum_norm_ratio": adaptive_norm / momentum_norm if momentum_norm > 0 else None,
            "adaptive_to_momentum_norm_ratio_undefined_reason": None if momentum_norm > 0 else "zero_momentum_norm"}
