"""Fixed-probe descent accounting and norm-controlled Adam direction substitutions."""
from contextlib import contextmanager
import math

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from baseline.continuation import rng_state, restore_rng
from baseline.data_protocol import target_count, update_microbatches


@contextmanager
def protected_probe(model):
    """Diagnostics preserve RNG, mode and outstanding training gradients."""
    rng, mode = rng_state(), model.training
    parameters = list(model.parameters())
    gradients = [p.grad for p in parameters]
    try:
        model.eval()
        for p in parameters:
            p.grad = None
        with sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        for p, gradient in zip(parameters, gradients):
            p.grad = gradient
        model.train(mode)
        restore_rng(rng)


def fixed_probe(model, tokens, indices, length, microbatch, device, *, gradient=False):
    """Token-weighted FP32 objective; FP64 scalar reduction; no optimizer mutation."""
    total = target_count(len(tokens), indices, length)
    if total <= 0:
        raise ValueError("Empty probe objective")
    loss_sum = 0.0
    with protected_probe(model), torch.set_grad_enabled(gradient):
        for batch, count in update_microbatches(tokens, indices, length, microbatch, device):
            with torch.autocast(device.type, enabled=False):
                loss = model(input_ids=batch, labels=batch, use_cache=False).loss
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite fixed-probe loss")
            weight = count / total
            loss_sum += float(loss.detach()) * weight
            if gradient:
                (loss * weight).backward()
        gradients = None
        if gradient:
            gradients = []
            for p in model.parameters():
                if p.requires_grad:
                    if p.grad is None or not torch.isfinite(p.grad).all():
                        raise ValueError("Missing/nonfinite probe gradient")
                    gradients.append(p.grad.detach().clone())
    return loss_sum, gradients


def vector_norm(vectors):
    return math.sqrt(sum(float(v.double().square().sum()) for v in vectors))


def descent_accounting(before, after, gradients, parameters_before, parameters_after):
    """D=P-R uses the actual rounded parameter displacement, including decay."""
    steps = [q.detach().double() - p.double() for p, q in zip(parameters_before, parameters_after)]
    projection = -sum(float(g.double().mul(s).sum()) for g, s in zip(gradients, steps))
    norm, gradient_norm = vector_norm(steps), vector_norm(gradients)
    denominator = norm * gradient_norm
    return dict(loss_before=before, loss_after=after, actual_decrease=before - after,
                first_order_decrease=projection, nonlinear_residual=projection - (before - after),
                actual_displacement_norm=norm, probe_gradient_norm=gradient_norm,
                descent_cosine=projection / denominator if denominator > 0 else None,
                descent_cosine_undefined_reason=None if denominator > 0 else "zero_norm",
                relative_loss_decrease=(before - after) / before)


def adam_components(optimizer, named_parameters, expected_step, kind):
    """Actual updated moments; current gradient is already globally clipped."""
    if kind not in {"adam", "momentum", "current_preconditioned", "gradient"}:
        raise ValueError(kind)
    groups = {id(p): group for group in optimizer.param_groups for p in group["params"]}
    vectors = []
    for _, p in named_parameters:
        if not p.requires_grad:
            continue
        group, state = groups[id(p)], optimizer.state[p]
        if int(state["step"].item()) != expected_step or p.grad is None:
            raise ValueError("Adam or gradient clock mismatch")
        beta1, beta2 = group["betas"]
        numerator = (state["exp_avg"].double() / (1 - beta1 ** expected_step)
                     if kind in {"adam", "momentum"} else p.grad.double())
        if kind in {"adam", "current_preconditioned"}:
            denominator = (state["exp_avg_sq"].double() / (1 - beta2 ** expected_step)).sqrt() + group["eps"]
            numerator = numerator / denominator
        if not torch.isfinite(numerator).all():
            raise ValueError("Nonfinite counterfactual direction")
        vectors.append(numerator)
    return vectors


@torch.no_grad()
def apply_direction(optimizer, named_parameters, before, vectors, target_adaptive_norm):
    """Identical decay; fixed target norm of adaptive component, before FP32 rounding."""
    named = [(name, p) for name, p in named_parameters if p.requires_grad]
    groups = {id(p): group for group in optimizer.param_groups for p in group["params"]}
    norm = vector_norm(vectors)
    if target_adaptive_norm < 0 or not math.isfinite(target_adaptive_norm):
        raise ValueError("Invalid target step norm")
    if norm == 0 and target_adaptive_norm > 0:
        raise ValueError("Cannot norm-match a zero direction")
    coefficient = target_adaptive_norm / norm if norm else 0.0
    for (_, p), origin, vector in zip(named, before, vectors):
        group = groups[id(p)]
        updated = origin.double() * (1 - group["lr"] * group["weight_decay"]) - coefficient * vector
        p.copy_(updated.to(p.dtype))
    return dict(direction_norm=norm, coefficient=coefficient,
                nominal_adaptive_step_norm=coefficient * norm)


def direction_trials(model, optimizer, before, probe_gradient, probe_loss_before,
                     tokens, probe_indices, config, device, update, *, interventions=True):
    """Counterfactual trials return to the exact accepted state, including RNG."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named]
    accepted = [p.detach().clone() for p in parameters]
    rng = rng_state()
    length, microbatch = config["data"]["sequence_length"], config["causal"]["probe_microbatch"]

    def measure(kind, **extra):
        after, _ = fixed_probe(model, tokens, probe_indices, length, microbatch, device)
        return dict(update=update, candidate=kind, **extra,
                    **descent_accounting(probe_loss_before, after, probe_gradient, before, parameters))

    try:
        rows = [measure("native_adamw" if interventions else "applied_intervention")]
        if not interventions:
            return rows
        adam = adam_components(optimizer, named, update, "adam")
        rates = {group["lr"] for group in optimizer.param_groups}
        if len(rates) != 1:
            raise ValueError("Norm matching requires one shared learning rate")
        target = rates.pop() * vector_norm(adam)
        del adam
        for kind, scale in (("adam", 1.0), ("momentum", 1.0),
                            ("current_preconditioned", 1.0), ("gradient", 1.0),
                            ("adam", .5), ("adam", 2.0)):
            vectors = adam_components(optimizer, named, update, kind)
            if vector_norm(vectors) == 0 and target > 0:
                rows.append(dict(update=update, candidate=kind, scale=scale,
                                 undefined_reason="zero_direction_cannot_match_nonzero_step"))
                continue
            matching = apply_direction(optimizer, named, before, vectors, target * scale)
            del vectors
            label = kind if scale == 1 else f"{kind}_scale_{scale:g}"
            row = measure(label, **matching)
            if label == "adam":
                row["native_reconstruction_loss_gap"] = row["loss_after"] - rows[0]["loss_after"]
                row["native_reconstruction_max_parameter_error"] = max(
                    float((p.detach() - q).abs().max()) for p, q in zip(parameters, accepted))
            rows.append(row)
        return rows
    finally:
        with torch.no_grad():
            for p, saved in zip(parameters, accepted):
                p.copy_(saved)
        restore_rng(rng)


class DisplacementMetrics:
    """Adjacent actual rounded parameter updates; no temporal averaging."""
    def __init__(self):
        self.previous = None

    def consume(self, before, after):
        current = [q.detach().double() - p.double() for p, q in zip(before, after)]
        norm = vector_norm(current)
        previous_norm = vector_norm(self.previous) if self.previous is not None else None
        denominator = norm * previous_norm if previous_norm is not None else 0
        cosine = (sum(float(a.mul(b).sum()) for a, b in zip(current, self.previous)) / denominator
                  if denominator else None)
        self.previous = current
        return dict(norm=norm, adjacent_cosine=cosine,
                    adjacent_cosine_undefined_reason=None if cosine is not None else
                    ("first_update" if previous_norm is None else "zero_norm"))

    def state_dict(self):
        return {"previous": self.previous}

    def load_state_dict(self, state, device):
        self.previous = ([x.to(device) for x in state["previous"]]
                         if state["previous"] is not None else None)
