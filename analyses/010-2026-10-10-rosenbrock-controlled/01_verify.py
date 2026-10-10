"""Independent objective, optimizer, temporal geometry and artifact checks."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import rosen, rosen_der

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "runs/028-2026-10-10-rosenbrock-adam/01_run.py"
spec = importlib.util.spec_from_file_location("rosenbrock_pilot", SOURCE)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
HERE = Path(__file__).resolve().parent


def mechanics():
    rng = np.random.default_rng(173)  # Only randomized verification points, not experiment seeds.
    for point in rng.uniform(-2, 2, size=(40, 2)):
        np.testing.assert_allclose(pilot.objective(point), rosen(point), rtol=1e-14)
        np.testing.assert_allclose(pilot.gradient(point), rosen_der(point), rtol=1e-14)
        tensor = torch.tensor(point, dtype=torch.float64, requires_grad=True)
        loss = (1 - tensor[0]) ** 2 + 100 * (tensor[1] - tensor[0] ** 2) ** 2
        loss.backward()
        np.testing.assert_allclose(pilot.gradient(point), tensor.grad.numpy(), rtol=1e-13)
        h = 1e-5
        finite_difference = [(pilot.objective(point + h * np.eye(2)[j]) -
                              pilot.objective(point - h * np.eye(2)[j])) / (2 * h) for j in range(2)]
        np.testing.assert_allclose(pilot.gradient(point), finite_difference, rtol=2e-8, atol=2e-7)
    np.testing.assert_equal(pilot.objective([1, 1]), 0.0)
    np.testing.assert_array_equal(pilot.gradient([1, 1]), [0, 0])
    for vectors in (np.array([[1., 0], [1, 0]]), np.array([[1., 0], [-1, 0]]),
                    np.array([[1., 0], [0, 1]]), np.array([[0., 0], [0, 0], [2, 0]])):
        geom = pilot.geometry(vectors)
        for i in range(len(vectors)):
            prefix = vectors[:i + 1]
            gram = prefix @ prefix.T
            np.testing.assert_allclose(geom["all_pair"][i], 2 * np.triu(gram, 1).sum(), atol=1e-14)
            np.testing.assert_allclose(geom["lag_one"][i], 2 * np.diagonal(gram, 1).sum(), atol=1e-14)
        assert np.isnan(geom["cosine"][0])
    zero = pilot.geometry(np.zeros((3, 2)))
    assert np.isnan(zero["cosine"]).all() and np.isnan(zero["ratio"]).all()
    opposed = pilot.geometry(np.array([[1., 0], [-1, 0]]))
    assert opposed["ratio"][-1] == 0 and opposed["cosine"][-1] == -1
    return {"objective_gradient_points": 40, "analytic_scipy_autograd_finite_difference": "passed",
            "geometry_aligned_opposite_orthogonal_zero": "passed", "independent_reference": "torch.optim.AdamW"}


def verify_run(entry):
    run_dir = ROOT / entry["directory"]
    output = run_dir / "artifacts/attempts/001-local"
    config = json.loads((run_dir / "config.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["config_sha256"] == pilot.sha256(run_dir / "config.json")
    assert manifest["source_sha256"] == pilot.sha256(SOURCE)
    inventory = json.loads((output / "inventory.json").read_text())
    for item in inventory["files"]:
        path = output / item["path"]
        assert path.stat().st_size == item["size_bytes"] and pilot.sha256(path) == item["sha256"]
    with np.load(output / "history.npz", allow_pickle=False) as saved:
        result = {key: saved[key] for key in saved.files}
    n = config["updates"]
    assert result["theta"].shape == (n + 1, 2)
    assert result["loss"].shape == (n + 1,)
    np.testing.assert_array_equal(result["theta"][0], config["initial_point"])
    np.testing.assert_allclose(result["loss"], rosen(result["theta"].T), rtol=1e-14)
    np.testing.assert_allclose(result["raw_vectors"], rosen_der(result["theta"][:-1].T).T, rtol=1e-13)
    np.testing.assert_array_equal(result["displacement_vectors"], np.diff(result["theta"], axis=0))
    np.testing.assert_allclose(result["displacement_vectors"].sum(axis=0), result["theta"][-1] - result["theta"][0], atol=1e-12)
    recurrence = -config["learning_rate"] * (result["adaptive_vectors"] + config["weight_decay"] * result["theta"][:-1])
    np.testing.assert_allclose(result["displacement_vectors"], recurrence, atol=5e-16, rtol=1e-11)
    beta1 = config["betas"][0]
    np.testing.assert_allclose(np.cumsum(result["momentum_vectors"], axis=0),
        np.cumsum(result["raw_vectors"], axis=0) - beta1 / (1 - beta1) * result["momentum_vectors"], atol=5e-10, rtol=1e-11)
    replay = pilot.simulate(config)
    assert result.keys() == replay.keys()
    for key in result:
        np.testing.assert_array_equal(result[key], replay[key], err_msg=f"deterministic replay: {key}")
    # Independent optimizer fed the captured gradients for the full horizon.
    # This isolates arithmetic correctness from unstable long-horizon trajectory sensitivity.
    parameter = torch.tensor(config["initial_point"], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.AdamW([parameter], lr=config["learning_rate"], betas=tuple(config["betas"]),
                                 eps=config["epsilon"], weight_decay=config["weight_decay"], foreach=False, fused=False)
    max_error = 0.0
    for i in range(n):
        parameter.grad = torch.from_numpy(result["raw_vectors"][i].copy())
        optimizer.step()
        max_error = max(max_error, float(np.max(np.abs(parameter.detach().numpy() - result["theta"][i + 1]))))
        np.testing.assert_allclose(optimizer.state[parameter]["exp_avg"].numpy(), result["momentum_vectors"][i], atol=5e-13, rtol=5e-12)
        np.testing.assert_allclose(optimizer.state[parameter]["exp_avg_sq"].numpy(), result["second_moment"][i], atol=5e-11, rtol=5e-12)
    assert max_error < 2e-11, max_error
    for name in pilot.STREAMS:
        vectors = result[name + "_vectors"]
        assert np.isfinite(vectors).all()
        norm = np.sqrt(np.sum(vectors ** 2, axis=1))
        a = np.add.accumulate(norm)
        b = np.sqrt(np.sum(np.add.accumulate(vectors, axis=0) ** 2, axis=1))
        np.testing.assert_allclose(result[name + "_A"], a, rtol=1e-13)
        np.testing.assert_allclose(result[name + "_B"], b, rtol=1e-13)
        assert np.nanmax(result[name + "_ratio"]) <= 1 + 1e-12
        assert np.nanmax(np.abs(result[name + "_cosine"])) <= 1 + 1e-12
        assert np.isnan(result[name + "_cosine"][0])
        assert np.isnan(result[name + "_cosine"][1:]).sum() == 0  # all pilot vectors nonzero
        for start in (0, n // 2, n - 12):
            sample = vectors[start:start + 12]
            geom = pilot.geometry(sample)
            gram = sample @ sample.T
            np.testing.assert_allclose(geom["all_pair"][-1], 2 * np.triu(gram, 1).sum(), atol=1e-9, rtol=1e-11)
    recorded = json.loads((output / "summary.json").read_text())
    summary = pilot.summarize(config, result)
    for key, value in summary.items():
        assert recorded[key] == value, key
    receipt = {"run_id": config["run_id"], "passed": True, "verified_utc": pilot.stamp(), "updates": n,
               "history_sha256": pilot.sha256(output / "history.npz"), "inventory_sha256": pilot.sha256(output / "inventory.json"),
               "source_sha256": pilot.sha256(SOURCE), "config_sha256": pilot.sha256(run_dir / "config.json"),
               "full_exact_replay": True, "torch_fixed_gradient_updates": n,
               "torch_max_point_absolute_error": max_error, "all_stream_prefixes_and_coverage": True,
               "optimizer_delta_and_telescoping": True, "exact_prefix_momentum_identity": True, "scipy_objective_and_raw_gradient": True,
               "sizes_and_sha256": True, "torch_version": torch.__version__}
    pilot.write_json(run_dir / "verification.json", receipt)
    manifest.update(status="completed", verification_sha256=pilot.sha256(run_dir / "verification.json"))
    pilot.write_json(output / "manifest.json", manifest)
    print(json.dumps({"run": config["run_id"], "passed": True, "torch_max_error": max_error}))
    return receipt


def main():
    report = {"mechanics": mechanics(), "runs": []}
    cohort = json.loads((HERE / "cohort.json").read_text())
    configs = [json.loads((ROOT / entry["directory"] / "config.json").read_text()) for entry in cohort]
    matched = []
    for config in configs:
        common = {key: value for key, value in config.items() if key not in {"run_id", "label", "weight_decay", "betas"}}
        common["beta2"] = config["betas"][1]
        matched.append(common)
    assert all(common == matched[0] for common in matched)
    assert {(config["betas"][0], config["weight_decay"]) for config in configs} == {(0.0, 0.0), (0.9, 0.0), (0.0, 0.1), (0.9, 0.1)}
    report["matched_factorial_contract"] = True
    for entry in cohort:
        report["runs"].append(verify_run(entry))
    report["passed"] = True
    pilot.write_json(HERE / "verification.json", report)


if __name__ == "__main__":
    main()
