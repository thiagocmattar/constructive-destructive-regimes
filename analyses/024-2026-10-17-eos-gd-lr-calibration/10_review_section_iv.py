"""Audit retained scalar evidence for the second critical review of Section IV.

No training, figure generation, manuscript edits, or independent-seed inference.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cohort = json.loads((HERE / "cohort.json").read_text())
    expected = {r["run_id"]: r for r in json.loads((HERE / "results.json").read_text())["runs"]}
    records = []
    for r in sorted(cohort["runs"], key=lambda r: (r["task"], r["lr"])):
        path = ROOT / "runs" / r["run"] / "artifacts/attempt-01-spark/production/metrics.jsonl"
        source_hash = sha(path)
        assert source_hash == expected[r["run_id"]]["source_sha256"]["metrics.jsonl"]
        rows = [json.loads(s) for s in path.read_text().splitlines()][:3000]
        assert len(rows) == 3000
        assert [row["update"] for row in rows] == list(range(1, 3001))
        raw = [row["streams"]["raw"] for row in rows]
        field = lambda key: np.asarray([v[key] for v in raw], dtype=float)
        norm, a, b, energy = (field(k) for k in ("norm", "A", "B", "energy"))
        np.testing.assert_allclose(a, np.cumsum(norm), rtol=2e-10)
        np.testing.assert_allclose(energy, np.cumsum(norm**2), rtol=2e-10)
        np.testing.assert_allclose(field("all_pair"), b**2 - energy, rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(field("all_pair"), field("lag_one") + field("beyond_lag_one"), rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(field("B_over_A"), b/a, rtol=2e-10)
        # These arrays index gradient t=1..3000 and loss states 0..2999.
        loss = np.asarray([v.get("train_loss", v.get("batch_objective_estimate")) for v in rows])
        q = np.asarray([v.get("eta_kappa_proxy", v.get("eta_kappa_gradient_proxy")) for v in rows[1:]])
        cosine = field("cosine")[1:]
        np.testing.assert_allclose(q, 1 - cosine * norm[1:]/norm[:-1], rtol=2e-10, atol=2e-10)
        assert np.isfinite(loss).all() and np.isfinite(q).all() and (norm > 0).all()
        # q_t pertains to update t-1. No q_3001 is available for the final update.
        actual = np.diff(loss)
        surrogate = -r["lr"] * norm[:-1]**2 * (2-q) / 2
        sensitivity = []
        for tolerance in (0, 1e-7, 1e-6, 1e-5):
            floor = tolerance * np.maximum(np.abs(loss[:-1]), np.abs(loss[1:]))
            keep = (np.abs(actual) > floor) & (np.abs(surrogate) > floor)
            mismatches = int(np.count_nonzero((actual[keep] > 0) != (surrogate[keep] > 0)))
            sensitivity.append(dict(relative_threshold=tolerance, eligible=int(keep.sum()),
                                    sign_disagreements=mismatches))
        records.append(dict(run_id=r["run_id"], task=r["task"], lr=r["lr"],
            source=path.relative_to(ROOT).as_posix(), source_sha256=source_hash,
            final_R=float(b[-1]/a[-1]),
            final_gradient_sum_displacement=float(r["lr"]*b[-1]),
            final_gradient_sum_path_length=float(r["lr"]*a[-1]),
            final_energy=float(energy[-1]), final_all_pair=float(field("all_pair")[-1]),
            final_lag_one=float(field("lag_one")[-1]),
            final_beyond_lag_one=float(field("beyond_lag_one")[-1]),
            loss_state_2000=float(loss[2000]), final_loss=expected[r["run_id"]]["final_loss"],
            loss_surrogate_pairs=len(actual), sign_sensitivity=sensitivity,
            median_scaled_absolute_loss_residual=float(np.median(np.abs(actual-surrogate)/(r["lr"]*norm[:-1]**2)))))
    assert len(records) == 9
    assert all(r["final_all_pair"] > 0 for r in records)
    assert sum(r["final_lag_one"] < 0 for r in records) == 7
    assert all(r["final_loss"] < r["loss_state_2000"] for r in records)
    for task in ("mlp", "resnet", "transformer"):
        group = [r for r in records if r["task"] == task]
        assert np.all(np.diff([r["final_gradient_sum_displacement"] for r in group]) > 0)
        assert np.all(np.diff([r["final_R"] for r in group]) < 0)
    sources = [HERE/"cohort.json", HERE/"results.json", ROOT/"src/cifar_eos/core.py",
               ROOT/"manuscript/draft/sections/03-methods.tex",
               ROOT/"manuscript/draft/sections/04-gd-experiments.tex",
               ROOT/"manuscript/artifacts/draft/bare_jrnl.pdf"]
    result = dict(status="passed", script_sha256=sha(Path(__file__)),
        sources={p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
        coverage="Nine seed-0 trajectories; 3000 updates each. All-pair fields include factor 2.",
        loss_check="2999 paired transitions per condition: q_t and L(theta[t-1])-L(theta[t-2]), t=2..3000. Final update excluded.",
        limits="Descriptive post-hoc approximation audit, not a forecast or independent samples. Thresholds are sensitivity choices, not calibrated FP32 error bounds. No error cause is isolated.",
        runs=records)
    (HERE/"section_iv_review_checks.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(status="passed", runs=9, prefix_identities=27000, loss_pairs=26991,
                         positive_final_all_pair=9, negative_final_lag_one=7)))


if __name__ == "__main__":
    main()
