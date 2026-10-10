"""Verify the new paths and the comparison changing only the initial point."""
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
    report = dict(mechanics=checks.mechanics(), matched_lr_contract=True,
                  initial_point=common[0]["initial_point"], runs=[], initial_point_controls=[],
                  independent_checks_source=str(REFERENCE.relative_to(ROOT)),
                  independent_checks_sha256=experiment.sha256(REFERENCE))
    for entry, config in zip(cohort, configs):
        report["runs"].append(checks.verify_run(entry))
        with np.load(ROOT / entry["directory"] / "artifacts/attempts/001-local/history.npz", allow_pickle=False) as saved:
            for stream in ("raw", "momentum"):
                a, b = saved[stream + "_A"], saved[stream + "_B"]
                ratio = np.divide(b, a, out=np.full_like(a, np.nan), where=a > 0)
                np.testing.assert_array_equal(ratio, saved[stream + "_ratio"])
        control = ROOT / entry["control_directory"]
        old_config = json.loads((control / "config.json").read_text())
        excluded = {"run_id", "label", "hypothesis", "initial_point"}
        assert {k: v for k, v in config.items() if k not in excluded} == {
            k: v for k, v in old_config.items() if k not in excluded}
        assert config["initial_point"] != old_config["initial_point"]
        receipt = json.loads((control / "verification.json").read_text())
        history = control / "artifacts/attempts/001-local/history.npz"
        assert receipt["passed"] and receipt["source_sha256"] == experiment.sha256(checks.SOURCE)
        assert receipt["history_sha256"] == experiment.sha256(history)
        assert receipt["config_sha256"] == experiment.sha256(control / "config.json")
        with np.load(history, allow_pickle=False) as previous:
            np.testing.assert_array_equal(previous["theta"][0], old_config["initial_point"])
            old_summary = json.loads((control / "artifacts/attempts/001-local/summary.json").read_text())
            recomputed = experiment.summarize(old_config, previous)
            assert all(old_summary[key] == value for key, value in recomputed.items())
        report["initial_point_controls"].append(dict(
            run_id=old_config["run_id"], new_run_id=config["run_id"],
            directory=entry["control_directory"], initial_point=old_config["initial_point"],
            history_sha256=experiment.sha256(history), config_sha256=receipt["config_sha256"],
            summary_sha256=experiment.sha256(control / "artifacts/attempts/001-local/summary.json"),
            matched_except_initial_point=True))
    report.update(passed=True, B_over_A_matches_recorded_metric=True)
    experiment.write_json(HERE / "verification.json", report)


if __name__ == "__main__":
    main()
