"""Independent mathematics, pair-index, reduction and artifact verification."""
import json

import numpy as np
from scipy.optimize import rosen_der, rosen_hess

from kappa import HERE, ROOT, identity, load_sources, sha256, write_json


def main():
    # Quadratic plain SGD recovers the Rayleigh quotient, not necessarily lambda_max.
    matrix = np.diag([2., 5.]); first = np.array([2., 10.]); eta = .1
    second = (np.eye(2) - eta * matrix) @ first
    estimate = identity([first, second], eta)[1]
    exact = float(first @ matrix @ first / (first @ first))
    np.testing.assert_allclose(estimate, exact, rtol=1e-14)
    assert exact < 5
    np.testing.assert_allclose(identity([first, second], eta, .03)[1], exact + .03 / eta)
    assert np.isnan(identity([[0, 0], first], eta)).all()
    assert np.isnan(identity([first, [0, 0]], eta)).all()
    report = json.loads((HERE / "results.json").read_text())
    assert report["config_sha256"] == sha256(HERE / "config.json")
    config = json.loads((HERE / "config.json").read_text())
    assert report["formula_source_sha256"] == sha256(HERE / "formula_source_snapshot.md")
    for name, expected in report["scripts"].items():
        assert sha256(HERE / name) == expected
    histories, sources, _ = load_sources()
    checks = []
    for h, source, row in zip(histories, sources, report["sources"]):
        assert row["history_sha256"] == source["history_sha256"]
        path = ROOT / row["derived_history"]
        assert sha256(path) == row["derived_history_sha256"] and path.stat().st_size == row["derived_history_bytes"]
        with np.load(path, allow_pickle=False) as saved:
            values = {key: saved[key] for key in saved.files}
        g = h["raw_vectors"]; rate = source["learning_rate"]
        np.testing.assert_allclose(np.stack([rosen_der(point) for point in h["theta"][:-1]]), g, rtol=3e-15, atol=1e-13)
        np.testing.assert_allclose(np.stack([rosen_hess(point) for point in h["theta"]]), h["hessian"], rtol=3e-15, atol=1e-12)
        # Cancellation-resistant direct inner-product form, independent of stored cosines.
        expected = np.r_[np.nan, [-float(a @ (b - a)) / (rate * float(a @ a)) for a, b in zip(g[:-1], g[1:])]]
        np.testing.assert_allclose(values["kappa_proxy"], expected, rtol=2e-10, atol=5e-11, equal_nan=True)
        expected_direction = np.r_[np.nan, [float(a @ rosen_hess(point) @ a / (a @ a)) for a, point in zip(g[:-1], h["theta"][:-2])]]
        np.testing.assert_allclose(values["kappa_direction"], expected_direction, rtol=3e-12, atol=2e-10, equal_nan=True)
        np.testing.assert_array_equal(values["lambda_max"][1:], h["hessian_lambda_max"][:-2])
        np.testing.assert_array_equal(values["updates"], np.arange(1, 2001))
        assert np.isfinite(values["kappa_proxy"][1:]).all() and np.isnan(values["kappa_proxy"][0])
        assert (values["kappa_direction"][1:] >= values["lambda_min"][1:] - 1e-9).all()
        assert (values["kappa_direction"][1:] <= values["lambda_max"][1:] + 1e-9).all()
        error = values["kappa_proxy"] - values["kappa_direction"] - values["optimizer_mismatch"] - values["nonlinear_remainder"]
        assert np.nanmax(np.abs(error)) < 1e-8
        for name in ("kappa_proxy", "kappa_direction", "lambda_min", "lambda_max", "proxy_minus_direction"):
            original = values[name]; mean = np.full(2000, np.nan)
            for i in range(99, 2000):
                window = original[i-99:i+1]
                if np.isfinite(window).all():
                    mean[i] = sum(float(v) for v in window) / 100
            np.testing.assert_allclose(values[name + "_mean100"], mean, rtol=3e-12, atol=1e-10, equal_nan=True)
            assert np.flatnonzero(np.isfinite(mean))[0] == 100
        for name, value in row["last100_pair_means"].items():
            np.testing.assert_allclose(value, sum(float(v) for v in values[name][-100:]) / 100, rtol=2e-13, atol=1e-11)
        for suffix, field in (("raw", "kappa_proxy"), ("mean100", "kappa_proxy_mean100")):
            reference = values["lambda_max" if suffix == "raw" else "lambda_max_mean100"]
            a = values[field]; valid = np.isfinite(a) & np.isfinite(reference)
            a = a[valid] - a[valid].mean(); b = reference[valid] - reference[valid].mean()
            correlation = float(a @ b / np.sqrt((a @ a) * (b @ b)))
            np.testing.assert_allclose(row["pearson_proxy_lambda_" + suffix], correlation, rtol=2e-14)
        checks.append(dict(run_id=row["run_id"], defined_pairs=1999, full_means=1900,
                           decomposition_max_absolute_error=float(np.nanmax(np.abs(error)))))
    write_json(HERE / "verification.json", dict(passed=True, tests=["quadratic SGD directional identity", "normalized noise term", "zero adjacent norms", "all source gradients and Hessians", "direct dot-product identity", "Taylor-point indexing", "Rayleigh spectral bounds", "exact update/nonlinear discrepancy", "independent means and correlations", "source and artifact hashes"], sources=checks, script_sha256=sha256(__file__), results_sha256=sha256(HERE / "results.json")))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
