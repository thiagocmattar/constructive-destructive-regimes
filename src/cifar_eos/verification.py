"""Independent mathematical and artifact checks; no scientific conclusions."""
import argparse
import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .core import (Geometry, HessianOperator, adam_vectors, file_hash, flat,
                       full_gradient, loss_sum, make_model, write_json)


def mathematics():
    torch.set_num_threads(2)
    checks = []
    history = torch.tensor([[1., 2., 3.], [-2., 0., 1.], [0., 0., 0.], [4., 1., -2.]], dtype=torch.float64)
    geometry = Geometry(3, "cpu")
    for i, value in enumerate(history):
        result = geometry.consume(value)
        reference = history[:i + 1].numpy()
        np.testing.assert_allclose(result["A"], np.linalg.norm(reference, axis=1).sum(), rtol=1e-14)
        np.testing.assert_allclose(result["B"], np.linalg.norm(reference.sum(0)), rtol=1e-14)
        pair = sum(2 * np.dot(reference[a], reference[b]) for a in range(i + 1) for b in range(a))
        np.testing.assert_allclose(result["all_pair"], pair, atol=2e-13)
    zero = Geometry(3, "cpu")
    assert zero.consume(torch.zeros(3))["B_over_A"] is None
    zero.consume(torch.tensor([1., 0., 0.]))
    result = zero.consume(torch.tensor([-1., 0., 0.]))
    assert result["cosine"] == -1 and result["B_over_A"] == 0
    restored = Geometry(3, "cpu")
    restored.load_state_dict(geometry.state_dict())
    assert restored.consume(torch.tensor([4., 2., 0.])) == geometry.consume(torch.tensor([4., 2., 0.]))
    checks.append("prefix/all-pair/zero-denominator/diagnostic-resume")

    torch.manual_seed(17)
    model = make_model(3, (4,), 2).double()
    x = torch.randn(7, 3, dtype=torch.float64)
    y = torch.tensor([0, 1, 1, 0, 0, 1, 0])
    for kind in ("mse", "ce"):
        reference = copy.deepcopy(model)
        loss_sum(reference(x), y, kind).div(len(x)).backward()
        _, _, accumulated = full_gradient(model, x, y, 3, kind)
        torch.testing.assert_close(accumulated, flat([p.grad for p in reference.parameters()]), rtol=1e-12, atol=1e-14)
    checks.append("MSE/CE full-gradient normalization including short final chunk")

    # Dense autograd Hessian is independent of the run-local matrix-free implementation.
    params = list(model.named_parameters())
    theta = flat([p.detach() for _, p in params]).requires_grad_()
    def objective(v):
        mapping, start = {}, 0
        for name, p in params:
            mapping[name] = v[start:start + p.numel()].reshape(p.shape)
            start += p.numel()
        prediction = torch.func.functional_call(model, mapping, (x,))
        return loss_sum(prediction, y, "mse") / len(x)
    dense = torch.autograd.functional.hessian(objective, theta)
    operator = HessianOperator(model, x, y, 3, "mse")
    probe = torch.randn(len(theta), dtype=torch.float64)
    torch.testing.assert_close(operator.product(probe), dense @ probe, rtol=1e-11, atol=1e-13)
    D = torch.linspace(0.1, 2., len(theta), dtype=torch.float64)
    conditioned = HessianOperator(model, x, y, 3, "mse", D)
    torch.testing.assert_close(conditioned.product(probe), (dense / D.sqrt()[:, None] / D.sqrt()[None, :]) @ probe, rtol=1e-11, atol=1e-13)
    options = {"k": 2, "which": "LA", "tolerance": 1e-10, "ncv": len(theta), "maxiter": 100, "residual_tolerance": 1e-7}
    for op, matrix in ((operator, dense), (conditioned, dense / D.sqrt()[:, None] / D.sqrt()[None, :])):
        computed = op.eigenvalues(options)
        np.testing.assert_allclose(computed["eigenvalues"], torch.linalg.eigvalsh(matrix)[-2:].flip(0), rtol=1e-9, atol=1e-11)
    # Ensure test spectrum is indefinite, so LA differs materially from LM.
    assert torch.linalg.eigvalsh(dense)[0] < 0
    checks.append("indefinite dense Hessian/HVP/LA eigenvalues/preconditioner/residual")

    model = make_model(3, (4,), 2).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99), eps=1e-7, foreach=False)
    full_gradient(model, x.float(), y, 3, "mse")
    gradient = flat([p.grad for p in model.parameters()]).double()
    before = flat([p.detach() for p in model.parameters()]).double()
    optimizer.step()
    vectors, _ = adam_vectors(optimizer, 1)
    torch.testing.assert_close(vectors["momentum"], gradient * 0.1, rtol=1e-6, atol=1e-9)
    torch.testing.assert_close(vectors["unbiased_momentum"], gradient, rtol=1e-6, atol=1e-8)
    expected = gradient / (gradient.abs() + 1e-7)
    torch.testing.assert_close(vectors["adaptive"], expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(flat([p.detach() for p in model.parameters()]).double() - before, -1e-4 * expected, rtol=5e-4, atol=3e-8)
    # Full optimizer + diagnostic recovery yields exactly the same next update.
    state = copy.deepcopy(optimizer.state_dict())
    peer = copy.deepcopy(model)
    peer_optimizer = torch.optim.Adam(peer.parameters(), lr=1e-4, betas=(0.9, 0.99), eps=1e-7, foreach=False)
    peer_optimizer.load_state_dict(state)
    for candidate, opt in ((model, optimizer), (peer, peer_optimizer)):
        full_gradient(candidate, x.float(), y, 3, "mse")
        opt.step()
    assert torch.equal(flat(model.parameters()), flat(peer.parameters()))
    checks.append("actual Adam moments/bias correction/direction/displacement/optimizer resume")
    # Plain constant-step GD displacement must match -eta * raw gradient up to parameter rounding.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    full_gradient(model, x.float(), y, 3, "mse")
    gradient = flat([p.grad for p in model.parameters()]).double()
    before = flat([p.detach() for p in model.parameters()]).double()
    optimizer.step()
    torch.testing.assert_close(flat([p.detach() for p in model.parameters()]).double() - before, -0.01 * gradient, rtol=1e-4, atol=3e-8)
    checks.append("GD update matches raw gradient with float32 rounding")
    return checks


def artifacts(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    for name, identity in manifest["files"].items():
        assert (directory / name).stat().st_size == identity["bytes"]
        assert file_hash(directory / name) == identity["sha256"], name
    rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    summary = json.loads((directory / "summary.json").read_text())
    config = json.loads((directory / "resolved_config.json").read_text())
    assert len(rows) == summary["completed_updates"] - manifest["start_step"]
    for previous, row in zip([None] + rows[:-1], rows):
        assert row["gradient_state_step"] == row["update"] - 1
        assert row["test"]["examples"] == 10000
        for name, metrics in row["streams"].items():
            assert metrics["B_over_A"] is None or 0 <= metrics["B_over_A"] <= 1 + 1e-10
            if previous:
                old = previous["streams"][name]
                np.testing.assert_allclose(metrics["A"], old["A"] + metrics["norm"], rtol=1e-12)
                np.testing.assert_allclose(metrics["energy"], old["energy"] + metrics["norm"]**2, rtol=1e-12)
                if metrics["cosine"] is not None:
                    expected_lag = old["lag_one"] + 2 * old["norm"] * metrics["norm"] * metrics["cosine"]
                    np.testing.assert_allclose(metrics["lag_one"], expected_lag, rtol=1e-10, atol=1e-10)
        if previous and previous["streams"]["raw"]["norm"] > 0 and row["streams"]["raw"]["cosine"] is not None:
            expected = 1 - row["streams"]["raw"]["cosine"] * row["streams"]["raw"]["norm"] / previous["streams"]["raw"]["norm"]
            np.testing.assert_allclose(row["eta_kappa_proxy"], expected, rtol=1e-10, atol=1e-12)
        if "unbiased_momentum" in row["streams"]:
            m, mh = row["streams"]["momentum"], row["streams"]["unbiased_momentum"]
            beta1 = config["optimizers"]["adam"]["betas"][0]
            np.testing.assert_allclose(mh["norm"], m["norm"] / (1 - beta1**row["update"]), rtol=1e-12)
            if m["cosine"] is not None:
                np.testing.assert_allclose(m["cosine"], mh["cosine"], atol=1e-12)
    for event in summary["curvature_samples"]:
        for name in ("hessian", "paper_preconditioned_hessian"):
            if name in event:
                assert max(event[name]["relative_residuals"]) <= config["eigenvalues"]["residual_tolerance"]
                assert event[name]["examples"] == 5000
    return {"files_verified": len(manifest["files"]), "updates_verified": len(rows), "curvature_states": len(summary["curvature_samples"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", type=Path, nargs="*")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"status": "passed", "mathematics": mathematics(), "artifacts": {str(p): artifacts(p) for p in args.directories}}
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
