"""CPU-only CE calculus checks; no model training or scientific run."""

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F


def fixed_ce(z):
    """Single hard target at index zero; preserve differentiable second order."""
    differences = z[1:] - z[0]
    return F.softplus(torch.logsumexp(differences, dim=0), threshold=1000)


def derivatives(loss, z):
    x = z.detach().clone().requires_grad_(True)
    value = loss(x)
    gradient = torch.autograd.grad(value, x)[0]
    hessian = torch.autograd.functional.hessian(loss, z)
    return value.item(), gradient, hessian


def main():
    torch.set_num_threads(1)
    checks = []
    max_hessian_relative_error = 0.0
    for margin in (0.0, 2.0, 12.0, 18.0, 28.0, 40.0):
        z = torch.tensor([margin, 0.0], dtype=torch.float64)
        loss, grad, hessian = derivatives(fixed_ce, z)
        r = torch.exp(torch.tensor(-margin, dtype=torch.float64))
        expected_loss = torch.log1p(r)
        expected_grad = torch.stack((-r / (1 + r), r / (1 + r)))
        expected_hessian = r / (1 + r).square() * torch.tensor(
            [[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float64
        )
        torch.testing.assert_close(torch.tensor(loss, dtype=torch.float64), expected_loss,
                                   rtol=1e-12, atol=0)
        torch.testing.assert_close(grad, expected_grad, rtol=1e-12, atol=0)
        torch.testing.assert_close(hessian, expected_hessian, rtol=1e-12, atol=0)
        error = float((hessian - expected_hessian).abs().max() / expected_hessian.abs().max())
        max_hessian_relative_error = max(max_hessian_relative_error, error)
        shifted = derivatives(fixed_ce, z + 100.0)
        torch.testing.assert_close(shifted[1], grad, rtol=1e-12, atol=0)
        torch.testing.assert_close(shifted[2], hessian, rtol=1e-12, atol=0)
    checks.append("FP64 fixed CE loss/gradient/Hessian and shift invariance at six binary margins")

    native = lambda z: F.cross_entropy(z[None, :], torch.tensor([0]))
    for margin in (0.0, 2.0, 8.0):
        z = torch.tensor([margin, 0.0], dtype=torch.float64)
        a, b = derivatives(native, z), derivatives(fixed_ce, z)
        torch.testing.assert_close(a[1], b[1], rtol=1e-10, atol=1e-16)
        torch.testing.assert_close(a[2], b[2], rtol=1e-10, atol=1e-16)
    checks.append("Native and fixed FP64 derivatives agree below saturation")

    z = torch.tensor([18.0, 0.0], dtype=torch.float32)
    a, b = derivatives(native, z), derivatives(fixed_ce, z)
    assert a[1][0].item() == 0 and a[1][1].item() > 0
    assert b[1][0].item() < 0 and b[1][1].item() > 0
    assert float(b[1].sum()) == 0
    checks.append("Actual CPU FP32 native target-gradient collapse and fixed zero-sum restoration at margin18")
    result = {
        "status": "passed",
        "utc": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU synthetic calculus checks only; no training, no curvature-trajectory evidence",
        "torch": torch.__version__,
        "checks": checks,
        "max_analytic_hessian_relative_error": max_hessian_relative_error,
        "fp32_binary_margin18": {
            "native_loss": a[0], "fixed_loss": b[0],
            "native_gradient": a[1].tolist(), "fixed_gradient": b[1].tolist(),
            "native_autodiff_hessian": a[2].tolist(),
            "fixed_autodiff_hessian": b[2].tolist(),
        },
    }
    destination = Path(__file__).with_name("loss_mechanics.json")
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
