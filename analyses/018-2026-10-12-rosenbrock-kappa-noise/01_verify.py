"""Independent pair algebra, noise pairing, window coverage and provenance checks."""
import json

import numpy as np
from scipy.optimize import rosen_der, rosen_hess

from noise_kappa import HERE, ROOT, REFERENCE, FIELDS, identity, load_sources, rolling, sha256, write_json


def main():
    report = json.loads((HERE / "results.json").read_text())
    histories, sources = load_sources()
    reference = json.loads((REFERENCE / "eta_overlay_data.json").read_text())
    standard = np.random.Generator(np.random.PCG64(1234)).standard_normal((2000, 2))
    checks = []
    for history, source, recorded in zip(histories, sources, report["sources"]):
        assert all(recorded[k] == v for k, v in source.items())
        path = ROOT / recorded["derived_history"]
        assert sha256(path) == recorded["derived_history_sha256"]
        assert path.stat().st_size == recorded["derived_history_bytes"]
        with np.load(path, allow_pickle=False) as saved:
            values = {k: saved[k] for k in saved.files}
        np.testing.assert_array_equal(values["updates"], np.arange(1, 2001))
        g, n, s = (history[k + "_vectors"] for k in ("raw", "noise", "signal"))
        np.testing.assert_array_equal(g, s + n)
        np.testing.assert_array_equal(n, source["sigma"] * standard)
        if source["sigma"]:
            np.testing.assert_array_equal(history["standard_noise_vectors"], standard)
        np.testing.assert_allclose(s, np.array([rosen_der(p) for p in history["theta"][:-1]]), rtol=3e-12, atol=1e-13)
        direct = {k: np.full(2000, np.nan) for k in FIELDS}
        for i in range(1, 2000):
            u, v = g[i - 1], g[i]
            bottom = float(u @ u)
            if bottom == 0 or float(v @ v) == 0:
                continue
            direct["gradient_only"][i] = -float(u @ (v - u)) / bottom
            direct["epsilon"][i] = float(u @ (n[i] - n[i - 1])) / bottom
            direct["noise_corrected"][i] = -float(u @ (s[i] - s[i - 1])) / bottom
            h = rosen_hess(history["theta"][i - 1])
            delta = history["theta"][i] - history["theta"][i - 1]
            eta = source["learning_rate"]
            direct["eta_directional"][i] = eta * float(u @ h @ u) / bottom
            direct["optimizer_mismatch"][i] = -float(u @ h @ (delta + eta * u)) / bottom
            r = s[i] - s[i - 1] - h @ delta
            direct["nonlinear_remainder"][i] = -float(u @ r) / bottom
        for k in FIELDS:
            np.testing.assert_allclose(values[k], direct[k], rtol=4e-10, atol=6e-12, equal_nan=True)
            means = np.r_[np.full(100, np.nan), [sum(float(x) for x in direct[k][i-99:i+1]) / 100 for i in range(100, 2000)]]
            np.testing.assert_allclose(values[k + "_mean100"], means, rtol=5e-10, atol=7e-13, equal_nan=True)
            assert np.isnan(values[k + "_mean100"][:100]).all()
            assert np.isfinite(values[k]).sum() == 1999
            assert np.isfinite(values[k + "_mean100"]).sum() == 1900
        exact = sum(values[k] for k in ("eta_directional", "optimizer_mismatch", "nonlinear_remainder"))
        np.testing.assert_allclose(values["noise_corrected"], exact, rtol=4e-10, atol=2e-11, equal_nan=True)
        error = float(np.nanmax(np.abs(exact - values["noise_corrected"])))
        if not source["sigma"]:
            old = next(r for r in reference["sources"] if r["learning_rate"] == source["learning_rate"])
            for a, b in (("gradient_only", "normalized_raw"), ("gradient_only_mean100", "normalized_mean100")):
                expected = np.array([np.nan if x is None else x for x in old[b]])
                np.testing.assert_allclose(values[a], expected, rtol=1e-12, atol=1e-13, equal_nan=True)
            np.testing.assert_array_equal(values["gradient_only"], values["noise_corrected"])
        checks.append(dict(run_id=source["run_id"], pairs=1999, means_per_metric=1900,
                           noise_and_signal_verified=True, independent_scipy_hessian=True,
                           exact_decomposition_max_absolute_error=error, hashes_and_sizes=True))
    # For a quadratic objective the corrected identity is exactly eta times
    # the Rayleigh quotient under plain SGD, including nonzero additive noise.
    h = np.diag([3., 7.])
    eta = .025
    noise = np.random.Generator(np.random.PCG64(90)).normal(0, .3, (40, 2))
    point = np.array([1., -2.]); gradients = []
    for perturbation in noise:
        gradient = h @ point + perturbation
        gradients.append(gradient)
        point = point - eta * gradient
    gradients = np.array(gradients)
    synthetic = identity(gradients, noise)
    expected = np.array([eta * (u @ h @ u) / (u @ u) for u in gradients[:-1]])
    np.testing.assert_allclose(synthetic["noise_corrected"][1:], expected, rtol=2e-13, atol=2e-14)
    assert np.max(np.abs(synthetic["gradient_only"][1:] - expected)) > .1
    zero = identity([[1., 0], [0., 0], [1., 1], [1., 0]], np.zeros((4, 2)))
    assert np.isnan(zero["noise_corrected"][:3]).all() and np.isfinite(zero["noise_corrected"][3])
    test = rolling([np.nan, 1, 2, 3, np.nan, 5, 6, 7], 3)
    np.testing.assert_allclose(test, [np.nan, np.nan, np.nan, 2, np.nan, np.nan, np.nan, 6], equal_nan=True)
    for row in report["protected_reference_figures"]:
        assert sha256(ROOT / row["path"]) == row["sha256"]
    write_json(HERE / "verification.json", dict(passed=True, source_count=15, pairs=29985,
               full_means_per_metric=28500, fields_verified=list(FIELDS), checks=checks,
               paired_noise_seed=1234, zero_noise_previous_overlay=True,
               synthetic_quadratic_noisy_sgd=True, zero_denominators_undefined=True,
               missing_window_propagation=True, original_figures_unchanged=True,
               script_sha256=sha256(__file__), metric_sha256=sha256(HERE / "noise_kappa.py")))
    print("PASS: 29,985 pairs, 28,500 complete means per metric, all 15 immutable sources, synthetic noisy SGD.")


if __name__ == "__main__":
    main()
