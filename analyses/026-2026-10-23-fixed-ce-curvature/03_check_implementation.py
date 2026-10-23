"""Focused calculus, placement, aggregation, initialization and update checks."""
import json
import os
import sys
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from src.fixed_ce_curvature.core import (Geometry, Hessian, configure, fixed_ce, flat,
    initialize, make_data, optimizer_for, save_json, stamp, tensor_hash)


def main():
    configure()
    checks = []
    for margin in (0., 2., 18., 28., 40.):
        z = torch.tensor([[margin, 0.]], dtype=torch.float64, requires_grad=True)
        loss = fixed_ce(z, torch.tensor([0]))
        g = torch.autograd.grad(loss, z, create_graph=True)[0]
        h = torch.autograd.functional.hessian(lambda w: fixed_ce(w, torch.tensor([0])), z).reshape(2, 2)
        r = torch.exp(torch.tensor(-margin, dtype=torch.float64))
        target = r / (1 + r).square() * torch.tensor([[1., -1.], [-1., 1.]], dtype=torch.float64)
        torch.testing.assert_close(h, target, rtol=1e-12, atol=0)
        assert abs(float(g.detach().sum())) < 1e-20
    checks.append("Production logaddexp fixed-CE double backward matches analytic binary Hessian through margin40")
    a, b = initialize(42), initialize(42)
    assert tensor_hash(a.parameters()) == tensor_hash(b.parameters())
    x, y, xt, yt = make_data()
    assert x.shape == (480, 3) and xt.shape == (481, 3)
    assert len(set(tuple(row) for row in torch.cat((x, xt)).tolist())) == 961
    assert torch.equal((x[:, 0] + x[:, 1]) % 31, y)
    checks.append("Paired CPU initialization and complete disjoint480/481 modular split")
    model = initialize(42, dtype=torch.float64, width=8, heads=2, ff=12)
    hessian = Hessian(model, x[:8], y[:8])
    vector = torch.randn(hessian.count, dtype=torch.float64)
    vector /= vector.norm()
    product = hessian.product(vector)
    parameters = tuple(model.parameters())
    original = flat(parameters).detach().clone()
    def set_vector(value):
        offset = 0
        with torch.no_grad():
            for parameter in parameters:
                parameter.copy_(value[offset:offset + parameter.numel()].view_as(parameter))
                offset += parameter.numel()
    epsilon = 1e-6
    gradients = []
    for sign in (-1, 1):
        set_vector(original + sign * epsilon * vector)
        loss = fixed_ce(model(x[:8]), y[:8])
        gradients.append(flat(torch.autograd.grad(loss, parameters)).detach())
    difference = (gradients[1] - gradients[0]) / (2 * epsilon)
    relative = float((product - difference).norm() / product.norm())
    assert relative < 1e-4, relative
    set_vector(original)
    native = F.cross_entropy(model(x[:8]), y[:8])
    native_gradient = flat(torch.autograd.grad(native, parameters, create_graph=True))
    native_hvp = flat(torch.autograd.grad(native_gradient.dot(vector), parameters))
    torch.testing.assert_close(native_hvp, product, rtol=1e-9, atol=1e-10)
    checks.append("Whole-model FP64 HVP matches native unsaturated CE and central gradient differences")
    geometry = Geometry(3, "cpu")
    values = []
    for v in (torch.tensor([1., 2., 3.]), torch.tensor([-2., 1., 0.])):
        values.append(geometry.capture((v, v, v, v)))
    expected_a = torch.tensor([1., 2., 3.]).double().norm() + torch.tensor([-2., 1., 0.]).double().norm()
    torch.testing.assert_close(values[-1][0, 2], expected_a)
    torch.testing.assert_close(values[-1][0, 3], torch.tensor([-1., 3., 3.]).double().norm())
    checks.append("Streaming FP64 norms, prefix sums and ratios agree with independent explicit vectors")
    model = initialize(42)
    optimizer = optimizer_for(model)
    previous = [p.detach().clone() for p in model.parameters()]
    loss = fixed_ce(model(x[:8]), y[:8])
    loss.backward()
    expected = [p * (1 - 1e-3) - 1e-3 * p0.grad / (p0.grad.abs() + 1e-8)
                for p, p0 in zip(previous, model.parameters())]
    optimizer.step()
    torch.testing.assert_close(flat(model.parameters()), flat(expected), rtol=1e-6, atol=1e-8)
    checks.append("First unfused AdamW step matches independently bias-corrected moments and decoupled decay")
    result = {"status": "passed", "utc": stamp(), "scope": "mechanics-not-training-evidence",
              "checks": checks, "hvp_finite_difference_relative_error": relative,
              "parameters": 53760, "torch": torch.__version__}
    save_json(Path(__file__).with_name("implementation_checks.json"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
