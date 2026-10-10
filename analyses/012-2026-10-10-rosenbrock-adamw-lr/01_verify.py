"""Verify the new decay cohort with the unchanged independent optimizer checks."""
import importlib.util
import json
import math
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


def main():
    cohort = json.loads((HERE / "cohort.json").read_text())
    configs = [json.loads((ROOT / item["directory"] / "config.json").read_text()) for item in cohort]
    common = [{key: value for key, value in config.items()
               if key not in {"run_id", "label", "learning_rate"}} for config in configs]
    assert all(config == common[0] for config in common)
    assert [config["learning_rate"] for config in configs] == [1e-2, math.sqrt(1e-3 * 1e-2), 1e-3]
    assert common[0]["updates"] == 2000 and common[0]["weight_decay"] == 0.1
    assert common[0]["betas"] == [0.9, 0.95]
    report = {"mechanics": checks.mechanics(), "matched_lr_contract": True,
              "independent_checks_source": str(REFERENCE.relative_to(ROOT)),
              "independent_checks_sha256": experiment.sha256(REFERENCE), "runs": [], "no_decay_controls": []}
    for entry, config in zip(cohort, configs):
        receipt = checks.verify_run(entry)
        history = ROOT / entry["directory"] / "artifacts/attempts/001-local/history.npz"
        with np.load(history, allow_pickle=False) as result:
            for stream in ("raw", "momentum"):
                a, b = result[stream + "_A"], result[stream + "_B"]
                ratio = np.divide(b, a, out=np.full_like(a, np.nan), where=a > 0)
                np.testing.assert_array_equal(ratio, result[stream + "_ratio"])
                assert ratio[0] == 1 and np.isfinite(ratio).all() and ((ratio >= 0) & (ratio <= 1 + 1e-12)).all()
            if config["learning_rate"] == 1e-2:
                path = ROOT / "runs/030-2026-10-10-rosenbrock-adamw/artifacts/attempts/001-local/history.npz"
                with np.load(path, allow_pickle=False) as previous:
                    assert previous.files == result.files
                    for key in result.files:
                        np.testing.assert_array_equal(result[key], previous[key][:len(result[key])], err_msg=key)
                report.update(exact_Run030_first_2000_updates=True, Run030_history_sha256=experiment.sha256(path))
        control_slug = {1e-2: "032-2026-10-10-rosenbrock-lr-1e-2", 1e-3: "033-2026-10-10-rosenbrock-lr-1e-3"}.get(config["learning_rate"])
        if control_slug:
            control = ROOT / "runs" / control_slug
            control_config = json.loads((control / "config.json").read_text())
            exclude = {"run_id", "label", "hypothesis", "weight_decay"}
            assert {k: v for k, v in config.items() if k not in exclude} == {k: v for k, v in control_config.items() if k not in exclude}
            assert control_config["weight_decay"] == 0
            control_receipt = json.loads((control / "verification.json").read_text())
            control_history = control / "artifacts/attempts/001-local/history.npz"
            assert control_receipt["passed"] and control_receipt["source_sha256"] == experiment.sha256(checks.SOURCE)
            assert control_receipt["history_sha256"] == experiment.sha256(control_history)
            assert control_receipt["config_sha256"] == experiment.sha256(control / "config.json")
            report["no_decay_controls"].append(dict(run_id=control_config["run_id"], decay_run_id=config["run_id"],
                directory=str(control.relative_to(ROOT)), history_sha256=experiment.sha256(control_history), matched_except_decay=True))
        report["runs"].append(receipt)
    assert len(report["no_decay_controls"]) == 2
    report.update(passed=True, midpoint_no_decay_control=False, B_over_A_matches_recorded_metric=True)
    experiment.write_json(HERE / "verification.json", report)


if __name__ == "__main__":
    main()
