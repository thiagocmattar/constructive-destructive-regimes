"""Recipe primitives adapted from sparsity-spillover Run004; no sparsity intervention."""
from __future__ import annotations
from contextlib import nullcontext
from dataclasses import dataclass
import math
from typing import Any

@dataclass
class DynamicLossScaler:
    scale: float = 4096.0
    growth_interval: int = 1000
    hysteresis: int = 2
    minimum_scale: float = 1.0
    growth_tracker: int = 0
    hysteresis_tracker: int = 2

    def __post_init__(self) -> None:
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("Loss scale must be positive and finite.")
        if self.growth_interval <= 0 or self.hysteresis <= 0:
            raise ValueError("Loss-scale growth interval and hysteresis must be positive.")
        if not math.isfinite(self.minimum_scale) or self.minimum_scale <= 0.0:
            raise ValueError("Minimum loss scale must be positive and finite.")
        self.hysteresis_tracker = min(max(1, int(self.hysteresis_tracker)), self.hysteresis)

    def update(self, *, finite: bool) -> dict[str, Any]:
        previous = float(self.scale)
        action = "hold"
        if finite:
            self.growth_tracker += 1
            self.hysteresis_tracker = self.hysteresis
            if self.growth_tracker >= self.growth_interval:
                self.scale *= 2.0
                self.growth_tracker = 0
                action = "grow"
        else:
            self.growth_tracker = 0
            if self.hysteresis_tracker > 1:
                self.hysteresis_tracker -= 1
                action = "overflow_hold"
            else:
                self.scale = max(self.minimum_scale, self.scale / 2.0)
                self.hysteresis_tracker = self.hysteresis
                action = "backoff"
        return {
            "loss_scale_previous": previous,
            "loss_scale": float(self.scale),
            "loss_scale_action": action,
            "loss_scale_growth_tracker": int(self.growth_tracker),
            "loss_scale_hysteresis_tracker": int(self.hysteresis_tracker),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "scale": float(self.scale),
            "growth_interval": int(self.growth_interval),
            "hysteresis": int(self.hysteresis),
            "minimum_scale": float(self.minimum_scale),
            "growth_tracker": int(self.growth_tracker),
            "hysteresis_tracker": int(self.hysteresis_tracker),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        restored = DynamicLossScaler(**state)
        self.__dict__.update(restored.__dict__)

def build_recipe_adamw(model: Any, config: dict[str, Any], *, torch: Any) -> tuple[Any, dict[str, Any]]:
    """Map GPT-NeoX FusedAdam groups to fused PyTorch AdamW."""

    if config.get("optimizer") != "adamw":
        raise ValueError("The Transformers recipe mapping requires AdamW.")
    decay: list[Any] = []
    no_decay: list[Any] = []
    decay_names: list[str] = []
    no_decay_names: list[str] = []
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    seen: set[int] = set()
    for module in model.modules():
        is_normalization = isinstance(module, torch.nn.LayerNorm)
        for local_name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            full_name = parameter_names[id(parameter)]
            if is_normalization or local_name == "bias":
                no_decay.append(parameter)
                no_decay_names.append(full_name)
            else:
                decay.append(parameter)
                decay_names.append(full_name)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if len(seen) != len(trainable) or not decay or not no_decay:
        raise RuntimeError("Recipe weight-decay grouping did not cover every trainable parameter exactly once.")
    betas = tuple(float(value) for value in config["adamw_betas"])
    optimizer = torch.optim.AdamW(
        [
            {"params": decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(config["peak_learning_rate"]),
        betas=betas,
        eps=float(config["adamw_eps"]),
        weight_decay=float(config["weight_decay"]),
        fused=next(model.parameters()).is_cuda,
    )
    metadata = {
        "source_optimizer": "GPT-NeoX v1 Apex/DeepSpeed FusedAdam in AdamW mode",
        "mapped_optimizer": "torch.optim.AdamW(fused=True)",
        "decay_parameter_tensors": len(decay),
        "decay_parameter_elements": sum(parameter.numel() for parameter in decay),
        "no_decay_parameter_tensors": len(no_decay),
        "no_decay_parameter_elements": sum(parameter.numel() for parameter in no_decay),
        "decay_parameter_names": decay_names,
        "no_decay_parameter_names": no_decay_names,
    }
    return optimizer, metadata

def recipe_learning_rate(
    update: int,
    *,
    peak: float,
    max_steps: int,
    warmup_fraction: float,
    minimum: float,
) -> float:
    """LR in effect for an update under GPT-NeoX v1's pre-step scheduler."""

    if not 1 <= update <= max_steps:
        raise ValueError("Update must lie inside the training budget.")
    if not 0.0 < warmup_fraction <= 1.0 or peak <= 0.0 or minimum < 0.0:
        raise ValueError("Invalid recipe learning-rate inputs.")
    scheduler_iteration = update - 1
    warmup_iterations = warmup_fraction * max_steps
    bounded = min(float(scheduler_iteration), max_steps - warmup_iterations)
    if warmup_iterations > 0.0 and scheduler_iteration <= warmup_iterations:
        return peak * bounded / warmup_iterations
    decay_iteration = bounded - warmup_iterations
    value = peak * 0.5 * (math.cos(math.pi * decay_iteration / max_steps) + 1.0)
    return max(value, minimum)

def recipe_attention_context(torch: Any, device: Any):
    if getattr(device, "type", None) != "cuda":
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    return sdpa_kernel(SDPBackend.FLASH_ATTENTION)
