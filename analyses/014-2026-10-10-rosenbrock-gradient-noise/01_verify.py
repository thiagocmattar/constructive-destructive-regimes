"""Verify noise placement, paired draws, all identities and every optimizer update."""
import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import rosen, rosen_der

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.controlled import rosenbrock as exact
from src.controlled import stochastic_rosenbrock as experiment
from src.controlled.noise_geometry import noise_terms

REFERENCE = ROOT / "analyses/010-2026-10-10-rosenbrock-controlled/01_verify.py"
spec = importlib.util.spec_from_file_location("pilot_checks", REFERENCE)
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)
checks.pilot = exact


def identity_error(left, right, scale):
    error = float(np.max(np.abs(left - right) / (1 + np.abs(scale))))
    assert error < 1e-11, error
    return error


def mechanics(cohort):
    report = checks.mechanics()
    for signal, noise in ((np.zeros((4, 2)), np.zeros((4, 2))),
                          (np.array([[1., 0], [1, 0], [0, 1]]), np.array([[1., 0], [-1, 0], [0, -1]])),
                          (np.array([[1., 2], [3, -1], [-2, 4]]), np.array([[.1, -.2], [-.3, .4], [.5, -.6]]))):
        terms = noise_terms(signal, noise)
        for t in range(len(signal)):
            raw = (signal + noise)[:t + 1]
            b2 = np.dot(raw.sum(axis=0), raw.sum(axis=0))
            energy = (raw ** 2).sum()
            interaction = 2 * np.triu(raw @ raw.T, 1).sum()
            np.testing.assert_allclose(b2, terms["signal_B_squared"][t] + terms["noise_B_squared"][t] + terms["prefix_cross"][t], atol=1e-13)
            np.testing.assert_allclose(energy, terms["signal_energy"][t] + terms["noise_energy"][t] + terms["same_step_cross"][t], atol=1e-13)
            np.testing.assert_allclose(interaction, terms["signal_all_pair"][t] + terms["noise_all_pair"][t] + terms["temporal_cross"][t], atol=1e-13)
    report["noise_decomposition_zero_cancellation_mixed"] = True
    report["zero_noise_exact_controls"] = []
    for entry in (item for item in cohort if item["reused"]):
        folder = ROOT / entry["directory"]
        config = json.loads((folder / "config.json").read_text())
        receipt = json.loads((folder / "verification.json").read_text())
        assert receipt["passed"] and receipt["source_sha256"] == exact.sha256(exact.__file__)
        assert receipt["config_sha256"] == exact.sha256(folder / "config.json")
        history = folder / "artifacts/attempts/001-local/history.npz"
        assert receipt["history_sha256"] == exact.sha256(history)
        with np.load(history, allow_pickle=False) as saved:
            zero = experiment.simulate(dict(config, sigma=0., noise_seed=1234))
            for key in saved.files:
                np.testing.assert_array_equal(zero[key], saved[key], err_msg="sigma0:" + key)
        report["zero_noise_exact_controls"].append(dict(run_id=entry["run_id"], history_sha256=exact.sha256(history)))
    return report


def verify_run(entry, standard_noise):
    folder = ROOT / entry["directory"]
    output = folder / "artifacts/attempts/001-local"
    config = json.loads((folder / "config.json").read_text())
    control = json.loads((ROOT / entry["control_directory"] / "config.json").read_text())
    excluded = {"run_id", "hypothesis", "label", "gradient", "sigma", "noise_seed"}
    assert {k: v for k, v in config.items() if k not in excluded} == {k: v for k, v in control.items() if k not in excluded}
    assert config["sigma"] == entry["sigma"] and config["noise_seed"] == 1234
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["source_sha256"] == exact.sha256(experiment.__file__)
    assert manifest["config_sha256"] == exact.sha256(folder / "config.json")
    for dep in manifest["dependencies"]:
        assert dep["sha256"] == exact.sha256(ROOT / dep["path"])
    for item in json.loads((output / "inventory.json").read_text())["files"]:
        p = output / item["path"]
        assert p.stat().st_size == item["size_bytes"] and exact.sha256(p) == item["sha256"]
    with np.load(output / "history.npz", allow_pickle=False) as saved:
        result = {key: saved[key] for key in saved.files}
    n = config["updates"]
    assert n == 2000 and result["theta"].shape == (2001, 2)
    np.testing.assert_array_equal(result["theta"][0], [0., 1.])
    np.testing.assert_array_equal(result["standard_noise_vectors"], standard_noise)
    np.testing.assert_array_equal(result["noise_vectors"], config["sigma"] * standard_noise)
    np.testing.assert_allclose(result["signal_vectors"], rosen_der(result["theta"][:-1].T).T, rtol=1e-13)
    np.testing.assert_allclose(result["loss"], rosen(result["theta"].T), rtol=1e-14)
    np.testing.assert_allclose(result["post_gradient_norm"], np.linalg.norm(rosen_der(result["theta"].T).T, axis=1), rtol=1e-13)
    np.testing.assert_array_equal(result["raw_vectors"], result["signal_vectors"] + result["noise_vectors"])
    replay = experiment.simulate(config)
    assert result.keys() == replay.keys()
    for key in result:
        np.testing.assert_array_equal(result[key], replay[key], err_msg=key)
    np.testing.assert_array_equal(result["displacement_vectors"], np.diff(result["theta"], axis=0))
    np.testing.assert_allclose(result["displacement_vectors"].sum(axis=0), result["theta"][-1] - result["theta"][0], atol=1e-12)
    np.testing.assert_allclose(result["displacement_vectors"], -config["learning_rate"] *
                               (result["adaptive_vectors"] + config["weight_decay"] * result["theta"][:-1]), rtol=1e-10, atol=5e-16)
    beta = config["betas"][0]
    np.testing.assert_allclose(np.cumsum(result["momentum_vectors"], axis=0),
        np.cumsum(result["raw_vectors"], axis=0) - beta / (1 - beta) * result["momentum_vectors"], atol=5e-10, rtol=1e-10)
    max_geometry_error = 0.
    for stream in experiment.STREAMS:
        vectors = result[stream + "_vectors"]
        norms = np.sqrt((vectors ** 2).sum(axis=1))
        prefix = np.add.accumulate(vectors, axis=0)
        b2 = (prefix ** 2).sum(axis=1)
        energy = np.add.accumulate((vectors ** 2).sum(axis=1))
        interaction = 2 * np.add.accumulate(np.einsum("ij,ij->i", prefix - vectors, vectors))
        np.testing.assert_allclose(result[stream + "_norm"], norms, rtol=1e-13)
        np.testing.assert_allclose(result[stream + "_A"], np.add.accumulate(norms), rtol=1e-13)
        np.testing.assert_allclose(result[stream + "_B"], np.sqrt(b2), rtol=1e-13)
        max_geometry_error = max(max_geometry_error, identity_error(result[stream + "_all_pair"], interaction, b2 + energy))
        ratio = np.divide(np.sqrt(b2), np.add.accumulate(norms))
        np.testing.assert_allclose(result[stream + "_ratio"], ratio, rtol=1e-13)
        assert ((ratio >= 0) & (ratio <= 1 + 1e-12)).all()
        cos = np.full(n, np.nan)
        np.divide(np.sum(vectors[:-1] * vectors[1:], axis=1), norms[:-1] * norms[1:], out=cos[1:], where=norms[:-1] * norms[1:] > 0)
        np.testing.assert_allclose(result[stream + "_cosine"], cos, rtol=1e-13, atol=1e-14, equal_nan=True)
        lag = np.r_[0., 2 * np.add.accumulate(np.sum(vectors[:-1] * vectors[1:], axis=1))]
        np.testing.assert_allclose(result[stream + "_lag_one"], lag, rtol=1e-12, atol=1e-8)
        for start in (0, n // 2, n - 12):
            sample = vectors[start:start + 12]
            gram = sample @ sample.T
            geom = exact.geometry(sample)
            np.testing.assert_allclose(geom["all_pair"][-1], 2 * np.triu(gram, 1).sum(), rtol=1e-11, atol=1e-9)
    terms = {key: result["decomposition_" + key] for key in noise_terms(result["signal_vectors"], result["noise_vectors"])}
    b2_scale = terms["signal_B_squared"] + terms["noise_B_squared"] + np.abs(terms["prefix_cross"])
    errors = dict(B_squared=identity_error(result["raw_B_squared"], terms["signal_B_squared"] + terms["noise_B_squared"] + terms["prefix_cross"], b2_scale),
                  energy=identity_error(result["raw_energy"], terms["signal_energy"] + terms["noise_energy"] + terms["same_step_cross"], b2_scale),
                  all_pair=identity_error(result["raw_all_pair"], terms["signal_all_pair"] + terms["noise_all_pair"] + terms["temporal_cross"], b2_scale))
    parameter = torch.tensor(config["initial_point"], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.AdamW([parameter], lr=config["learning_rate"], betas=tuple(config["betas"]),
                                 eps=config["epsilon"], weight_decay=config["weight_decay"], foreach=False, fused=False)
    max_error = 0.
    for i in range(n):
        parameter.grad = torch.from_numpy(result["raw_vectors"][i].copy())
        optimizer.step()
        max_error = max(max_error, float(np.max(np.abs(parameter.detach().numpy() - result["theta"][i + 1]))))
        np.testing.assert_allclose(optimizer.state[parameter]["exp_avg"].numpy(), result["momentum_vectors"][i], atol=5e-13, rtol=5e-12)
        np.testing.assert_allclose(optimizer.state[parameter]["exp_avg_sq"].numpy(), result["second_moment"][i], atol=5e-11, rtol=5e-12)
    assert max_error < 2e-11
    summary = json.loads((output / "summary.json").read_text())
    assert all(summary[k] == v for k, v in exact.summarize(config, result).items())
    rng = np.random.Generator(np.random.PCG64(1234))
    rng.standard_normal((2000, 2))
    assert json.loads((output / "final_state.json").read_text())["rng"] == rng.bit_generator.state
    receipt = dict(run_id=entry["run_id"], passed=True, verified_utc=exact.stamp(),
                   config_sha256=exact.sha256(folder / "config.json"), source_sha256=exact.sha256(experiment.__file__),
                   history_sha256=exact.sha256(output / "history.npz"), inventory_sha256=exact.sha256(output / "inventory.json"),
                   updates=n, full_exact_replay=True, paired_standard_noise=True, noise_placement_and_true_objective=True,
                   independent_torch_updates=n, torch_max_point_absolute_error=max_error,
                   geometry_max_scaled_error=max_geometry_error, decomposition_max_scaled_errors=errors,
                   exact_momentum_prefix_identity=True, decay_displacement_and_telescoping=True,
                   final_rng_state_verified=True, sizes_and_hashes=True, torch_version=torch.__version__)
    exact.write_json(folder / "verification.json", receipt)
    manifest.update(status="completed", verification_sha256=exact.sha256(folder / "verification.json"))
    exact.write_json(output / "manifest.json", manifest)
    print(json.dumps(dict(run=entry["run_id"], verified=True, optimizer_error=max_error)))
    return receipt


def main(mechanics_only=False):
    cohort = json.loads((HERE / "cohort.json").read_text())
    assert sorted(set(item["sigma"] for item in cohort)) == [0., .01, .1, 1., 10.]
    for sigma in (0., .01, .1, 1., 10.):
        assert [item["learning_rate"] for item in cohort if item["sigma"] == sigma] == [1e-2, math.sqrt(1e-3 * 1e-2), 1e-3]
    report = dict(mechanics=mechanics(cohort), runs=[], reference_checks_sha256=exact.sha256(REFERENCE))
    if mechanics_only:
        exact.write_json(HERE / "prelaunch_verification.json", dict(report, mechanics_passed=True, experiment_evidence=False))
        print("Prelaunch mechanics passed; no stochastic attempt executed.")
        return
    draws = np.random.Generator(np.random.PCG64(1234)).standard_normal((2000, 2))
    for entry in cohort:
        if not entry["reused"]:
            report["runs"].append(verify_run(entry, draws))
    report.update(passed=True, new_runs=12, new_updates=24000, sigma_zero_reused=True, one_seed=1234)
    exact.write_json(HERE / "verification.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanics-only", action="store_true")
    main(parser.parse_args().mechanics_only)
