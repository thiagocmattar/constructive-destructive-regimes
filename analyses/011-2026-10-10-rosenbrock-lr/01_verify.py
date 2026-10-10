"""Reuse the independent pilot checks and verify the matched LR refinement."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.controlled import rosenbrock as experiment

REFERENCE = ROOT / "analyses/010-2026-10-10-rosenbrock-controlled/01_verify.py"
spec = importlib.util.spec_from_file_location("pilot_checks", REFERENCE)
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)
checks.pilot = experiment
checks.SOURCE = Path(experiment.__file__)


def b_over_a(a, b):
    return np.divide(b, a, out=np.full_like(a, np.nan), where=a > 0)


def main():
    cohort = json.loads((HERE / "cohort.json").read_text())
    configs = [json.loads((ROOT / item["directory"] / "config.json").read_text()) for item in cohort]
    matched = [{key: value for key, value in config.items()
                if key not in {"run_id", "label", "learning_rate"}} for config in configs]
    assert all(config == matched[0] for config in matched)
    assert [config["learning_rate"] for config in configs] == [1e-2, 1e-3, 1e-4]
    assert matched[0]["updates"] == 2000
    assert matched[0]["betas"] == [0.9, 0.95] and matched[0]["weight_decay"] == 0
    # Promotion changes only the opening docstring, preserving the executed pilot.
    original = (ROOT / "runs/028-2026-10-10-rosenbrock-adam/01_run.py").read_bytes()
    assert original.split(b"\n", 1)[1] == checks.SOURCE.read_bytes().split(b"\n", 1)[1]
    cancel = experiment.geometry(np.array([[1., 0.], [-1., 0.]]))
    assert b_over_a(cancel["A"], cancel["B"])[-1] == 0.0
    zero = experiment.geometry(np.zeros((3, 2)))
    assert np.isnan(b_over_a(zero["A"], zero["B"])).all()
    report = {"mechanics": checks.mechanics(), "matched_lr_contract": True,
              "unchanged_promoted_numerics": True, "independent_checks_source": str(REFERENCE.relative_to(ROOT)),
              "independent_checks_sha256": experiment.sha256(REFERENCE), "runs": []}
    prefix = ROOT / "runs/028-2026-10-10-rosenbrock-adam/artifacts/attempts/001-local/history.npz"
    for entry in cohort:
        receipt = checks.verify_run(entry)
        history = ROOT / entry["directory"] / "artifacts/attempts/001-local/history.npz"
        with np.load(history, allow_pickle=False) as result:
            receipt["B_over_A"] = {}
            for stream in ("raw", "momentum"):
                ratio = b_over_a(result[stream + "_A"], result[stream + "_B"])
                assert ratio[0] == 1.0
                assert np.isfinite(ratio).all() and np.min(ratio) >= 0 and np.max(ratio) <= 1 + 1e-12
                np.testing.assert_array_equal(ratio, result[stream + "_ratio"])
                receipt["B_over_A"][stream] = {"finite_updates": len(ratio), "final": float(ratio[-1]),
                                              "minimum": float(ratio.min()), "maximum": float(ratio.max())}
            if entry["learning_rate"] == 1e-2:
                with np.load(prefix, allow_pickle=False) as previous:
                    assert previous.files == result.files
                    for key in result.files:
                        np.testing.assert_array_equal(result[key], previous[key][:len(result[key])], err_msg=key)
                report["exact_Run028_first_2000_updates"] = True
                report["Run028_history_sha256"] = experiment.sha256(prefix)
        # The additional ratio checks belong to the analysis receipt, not the already-hashed run receipt.
        report["runs"].append(receipt)
    report.update(passed=True, B_over_A_zero_denominator_checks=True)
    experiment.write_json(HERE / "verification.json", report)


if __name__ == "__main__":
    main()
