"""Independent Hessian checks, complete replay, optimizer and matched controls."""
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import rosen_hess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.controlled import rosenbrock_curvature as experiment

REFERENCE = ROOT / "analyses/010-2026-10-10-rosenbrock-controlled/01_verify.py"
spec = importlib.util.spec_from_file_location("independent_rosenbrock_checks", REFERENCE)
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)
checks.pilot = experiment
checks.SOURCE = Path(experiment.__file__)


def curvature_mechanics():
    points = np.vstack(([[0., 1.], [1., 1.]], np.random.default_rng(901).uniform(-2, 2, (40, 2))))
    for point in points:
        matrix = experiment.hessian(point)
        np.testing.assert_allclose(matrix, rosen_hess(point), atol=1e-12, rtol=1e-13)
        tensor = torch.tensor(point, dtype=torch.float64, requires_grad=True)
        automatic = torch.autograd.functional.hessian(
            lambda z: (1 - z[0]) ** 2 + 100 * (z[1] - z[0] ** 2) ** 2, tensor).numpy()
        np.testing.assert_allclose(matrix, automatic, atol=1e-12, rtol=1e-13)
        h = 1e-5
        finite = np.column_stack([(experiment.gradient(point + h * np.eye(2)[j]) -
                                  experiment.gradient(point - h * np.eye(2)[j])) / (2 * h) for j in range(2)])
        np.testing.assert_allclose(matrix, finite, atol=3e-7, rtol=1e-8)
    np.testing.assert_array_equal(experiment.hessian([0., 1.]), [[-398., 0.], [0., 200.]])
    np.testing.assert_array_equal(np.linalg.eigvalsh(experiment.hessian([0., 1.])), [-398., 200.])
    np.testing.assert_array_equal(experiment.hessian([1., 1.]), [[802., -400.], [-400., 200.]])
    batch = experiment.hessian(points)
    np.testing.assert_array_equal(batch, np.array([experiment.hessian(p) for p in points]))
    tensor = torch.tensor([0.6, 0.2], dtype=torch.float64)
    automatic = torch.autograd.functional.hessian(
        lambda z: (3 - z[0]) ** 2 + 3.5 * (z[1] - z[0] ** 2) ** 2, tensor).numpy()
    np.testing.assert_allclose(experiment.hessian(tensor.numpy(), 3.5), automatic, atol=1e-14)
    return dict(points=42, scipy_autograd_gradient_finite_difference=True,
                indefinite_start_and_known_optimum=True, batch_and_general_b=True)


def main():
    cohort = json.loads((HERE / "cohort.json").read_text())
    configs = [json.loads((ROOT / e["directory"] / "config.json").read_text()) for e in cohort]
    assert [c["learning_rate"] for c in configs] == [0.01, math.sqrt(0.001 * 0.01), 0.001]
    common = [{k: v for k, v in c.items() if k not in {"run_id", "label", "learning_rate"}} for c in configs]
    assert all(c == common[0] for c in common)
    assert common[0]["initial_point"] == [0., 1.] and common[0]["updates"] == 2000
    report = dict(mechanics=checks.mechanics(), curvature_mechanics=curvature_mechanics(),
                  matched_lr_contract=True, runs=[], controls=[],
                  independent_checks_source=REFERENCE.relative_to(ROOT).as_posix(),
                  independent_checks_sha256=experiment.sha256(REFERENCE))
    for entry, config in zip(cohort, configs):
        receipt = checks.verify_run(entry)
        folder = ROOT / entry["directory"]
        output = folder / "artifacts/attempts/001-local"
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["base_source_sha256"] == experiment.sha256(experiment.base.__file__)
        assert manifest["launcher_sha256"] == experiment.sha256(folder / "01_run.py")
        control = ROOT / entry["control_directory"]
        previous_config = json.loads((control / "config.json").read_text())
        excluded = {"run_id", "hypothesis", "label", "curvature"}
        assert {k: v for k, v in config.items() if k not in excluded} == {
            k: v for k, v in previous_config.items() if k not in excluded}
        previous_receipt = json.loads((control / "verification.json").read_text())
        previous_history = control / "artifacts/attempts/001-local/history.npz"
        assert previous_receipt["passed"]
        assert previous_receipt["config_sha256"] == experiment.sha256(control / "config.json")
        assert previous_receipt["history_sha256"] == experiment.sha256(previous_history)
        assert previous_receipt["source_sha256"] == manifest["base_source_sha256"]
        previous_inventory = json.loads((control / "artifacts/attempts/001-local/inventory.json").read_text())
        for item in previous_inventory["files"]:
            path = control / "artifacts/attempts/001-local" / item["path"]
            assert path.stat().st_size == item["size_bytes"] and experiment.sha256(path) == item["sha256"]
        with np.load(output / "history.npz", allow_pickle=False) as saved, np.load(previous_history, allow_pickle=False) as previous:
            for key in previous.files:
                np.testing.assert_array_equal(saved[key], previous[key], err_msg=f"matched control: {key}")
            assert set(saved.files) - set(previous.files) == {
                "hessian", "hessian_eigenvalues", "hessian_lambda_min", "hessian_lambda_max"}
            matrix = saved["hessian"]
            assert matrix.shape == (2001, 2, 2) and matrix.dtype == np.float64 and np.isfinite(matrix).all()
            np.testing.assert_array_equal(matrix, matrix.swapaxes(-1, -2))
            independent = np.array([rosen_hess(point) for point in saved["theta"]])
            np.testing.assert_allclose(matrix, independent, rtol=1e-13, atol=1e-12)
            diagonal1, diagonal2, off = matrix[:, 0, 0], matrix[:, 1, 1], matrix[:, 0, 1]
            center = (diagonal1 + diagonal2) / 2
            radius = np.hypot((diagonal1 - diagonal2) / 2, off)
            closed_form = np.column_stack((center - radius, center + radius))
            np.testing.assert_allclose(saved["hessian_eigenvalues"], closed_form, rtol=1e-11, atol=1e-12)
            np.testing.assert_array_equal(saved["hessian_lambda_min"], saved["hessian_eigenvalues"][:, 0])
            np.testing.assert_array_equal(saved["hessian_lambda_max"], saved["hessian_eigenvalues"][:, 1])
            assert (saved["hessian_lambda_max"] >= saved["hessian_lambda_min"]).all()
        receipt.update(complete_hessian_positions=2001, scipy_hessian_all_points=True,
                       closed_form_eigenvalues_all_points=True, original_arrays_bitwise_identical=True,
                       matched_control=entry["control_directory"], base_source_sha256=manifest["base_source_sha256"],
                       launcher_sha256=manifest["launcher_sha256"])
        experiment.write_json(folder / "verification.json", receipt)
        manifest["verification_sha256"] = experiment.sha256(folder / "verification.json")
        experiment.write_json(output / "manifest.json", manifest)
        report["runs"].append(receipt)
        report["controls"].append(dict(directory=entry["control_directory"],
                                       config_sha256=previous_receipt["config_sha256"],
                                       history_sha256=previous_receipt["history_sha256"],
                                       all_original_arrays_bitwise_identical=True))
    report["passed"] = True
    experiment.write_json(HERE / "verification.json", report)


if __name__ == "__main__":
    main()
