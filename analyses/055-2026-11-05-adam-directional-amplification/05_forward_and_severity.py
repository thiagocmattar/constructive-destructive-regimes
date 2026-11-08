"""Forward-horizon and minimum-rise sweeps requested during the plan review.

Fixed grids, all nine retained arms, no classifier or outcome-selected thresholds.
Every diagnostic is evaluated at state s, before the outgoing update.
"""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("source055", HERE / "01_analyze.py")
SOURCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOURCE)
OUT = HERE / "derived" / "forward_and_severity"
HORIZONS = range(1, 17)
MINIMUM_RISES = (0., 1e-5, .001, .01, .05, .10, .25, .50, 1.)
FORWARD_RISES = (0., .01, .10)
FLAGS = ("negative_cosine", "norm_growth", "reversal_and_growth", "Q_ge2")


def divide(a, b):
    return a / b if b else None


def summarize(arm, states, change, target, horizon, minimum, subset, support):
    if subset == "not_already_rising":
        keep = arm["loss"][states] <= arm["loss"][states - 1]
        states, change = states[keep], change[keep]
    positive = change > 0 if minimum == 0 else change >= minimum
    rho, growth, q = arm["rho"][states], arm["r"][states], arm["Q"][states]
    flags = (rho < 0, growth > 1, (rho < 0) & (growth > 1), q >= 2)
    np.testing.assert_array_equal(q > 2, rho < -1 / growth)
    np.testing.assert_array_equal(q >= 2, (q >= 2) & flags[2])
    # This source has no exact equality; the requested >=2 and theoretical >2 agree.
    assert not np.any(q == 2)
    n, positives = len(states), int(positive.sum())
    row = dict(arm=arm["arm"], task=arm["task"], lr=arm["lr"], tier=arm["tier"],
               target=target, horizon=horizon, minimum_relative_increase=minimum,
               subset=subset, support=support, states=n, positives=positives,
               negatives=n - positives, base_rate=divide(positives, n))
    for name, flag in zip(FLAGS, flags):
        alarms = int(flag.sum())
        tp = int((flag & positive).sum())
        fp, fn = alarms - tp, positives - tp
        tn = n - tp - fp - fn
        row.update({f"{name}_alarms": alarms, f"{name}_tp": tp,
                    f"{name}_fp": fp, f"{name}_fn": fn, f"{name}_tn": tn,
                    f"{name}_recall": divide(tp, positives),
                    f"{name}_precision": divide(tp, alarms),
                    f"{name}_false_positive_rate": divide(fp, n - positives),
                    f"{name}_precision_lift": divide(tp * n, alarms * positives)})
        assert tp + fp + fn + tn == n
    return row


def main():
    arms, sources = SOURCE.load_arms()
    severity, forward = [], []
    equivalent_states = 0
    for arm in arms:
        states = np.arange(1, 1024)
        for timing, start in (("incoming_step", states - 1), ("future_step", states)):
            change = (arm["loss"][start + 1] - arm["loss"][start]) / arm["loss"][start]
            for minimum in MINIMUM_RISES:
                for subset in ("all", "not_already_rising"):
                    # Incoming loss cannot rise in this past-defined non-rising subset.
                    if timing == "incoming_step" and subset != "all":
                        continue
                    severity.append(summarize(arm, states, change, timing, 1, minimum,
                                              subset, "all_available"))
        equivalent_states += len(states)
        for support in ("all_available", "common_16_step"):
            previous_window = {}
            for horizon in HORIZONS:
                last = 1024 - (horizon if support == "all_available" else 16)
                states = np.arange(1, last + 1)
                loss = arm["loss"]
                changes = {
                    "future_step": (loss[states + horizon] - loss[states + horizon - 1]) / loss[states + horizon - 1],
                    "net_change": (loss[states + horizon] - loss[states]) / loss[states],
                    "any_above_current": np.max(np.stack([
                        (loss[states + k] - loss[states]) / loss[states]
                        for k in range(1, horizon + 1)]), axis=0),
                }
                if horizon == 1:
                    np.testing.assert_array_equal(changes["future_step"], changes["net_change"])
                    np.testing.assert_array_equal(changes["future_step"], changes["any_above_current"])
                for target, change in changes.items():
                    for minimum in FORWARD_RISES:
                        for subset in ("all", "not_already_rising"):
                            row = summarize(arm, states, change, target, horizon, minimum,
                                            subset, support)
                            forward.append(row)
                            if target == "any_above_current" and support == "common_16_step":
                                key = (minimum, subset)
                                if key in previous_window:
                                    assert row["positives"] >= previous_window[key]["positives"]
                                    for flag in FLAGS:
                                        assert row[f"{flag}_alarms"] == previous_window[key][f"{flag}_alarms"]
                                        assert row[f"{flag}_tp"] >= previous_window[key][f"{flag}_tp"]
                                previous_window[key] = row
                assert np.all(changes["any_above_current"] >= changes["net_change"])
    # Cross-check the h=1 sweep against the independently verified exhaustive audit.
    with (HERE / "derived/one_step_audit/summary.csv").open(newline="", encoding="utf-8") as stream:
        original = {r["arm"]: r for r in csv.DictReader(stream)
                    if r["timing"] == "outgoing" and float(r["relative_tolerance"]) == 0}
    for row in severity:
        if row["target"] == "future_step" and row["minimum_relative_increase"] == 0 and row["subset"] == "all":
            prior = original[row["arm"]]
            assert row["states"] == int(prior["states"])
            assert row["positives"] == int(prior["increases"])
            for current, old in zip(FLAGS, ("destructive", "growth", "reversing_growth", "Q_ge2")):
                assert row[f"{current}_tp"] == int(prior[f"{old}_increases"])
                assert row[f"{current}_fp"] == int(prior[f"{old}_decreases"])
    assert len(severity) == 243 and len(forward) == 5184
    OUT.mkdir(exist_ok=True)
    SOURCE.write_csv(OUT / "severity.csv", severity)
    SOURCE.write_csv(OUT / "horizons.csv", forward)
    sources[Path(HERE / "01_analyze.py").relative_to(SOURCE.ROOT).as_posix()] = SOURCE.sha(HERE / "01_analyze.py")
    sources[Path(HERE / "derived/one_step_audit/summary.csv").relative_to(SOURCE.ROOT).as_posix()] = SOURCE.sha(HERE / "derived/one_step_audit/summary.csv")
    manifest = dict(
        purpose="Both requested exploratory audits: forward timing and minimum step loss increase",
        date="2026-11-08", horizons=list(HORIZONS), severity_grid=list(MINIMUM_RISES),
        horizon_minimum_rises=list(FORWARD_RISES),
        threshold_units="relative to loss immediately before the target step; net/window targets relative to decision loss L_s",
        threshold_comparison="Strict >0 for any rise; >= minimum for positive minimum magnitudes",
        flags={"negative_cosine": "rho_s<0", "norm_growth": "r_s>1",
               "reversal_and_growth": "rho_s<0 AND r_s>1", "Q_ge2": "Q_s>=2"},
        angle_norm_equivalence="rho_s < -1/r_s iff Q_s>2; no Q_s=2 in the retained states, so its counts equal Q_s>=2",
        targets={"future_step": "(L_(s+h)-L_(s+h-1))/L_(s+h-1)",
                 "net_change": "(L_(s+h)-L_s)/L_s",
                 "any_above_current": "max_(1<=k<=h) (L_(s+k)-L_s)/L_s",
                 "incoming_step": "(L_s-L_(s-1))/L_(s-1); retrospective only"},
        subsets={"all": "all eligible decision states", "not_already_rising": "L_s<=L_(s-1), determined before the outgoing update"},
        support={"all_available": "states 1..1024-h, 9*(1024-h) total before subset filtering",
                 "common_16_step": "same states 1..1008 at every horizon; 9072 total before subset filtering"},
        inference="Descriptive exploration on known single-seed trajectories. No selected winning horizon, fitted threshold, independent-seed CI, or causal/predictive validation.",
        checks=dict(equivalent_angle_norm_states=equivalent_states, severity_rows=len(severity),
                    horizon_rows=len(forward), one_step_matches_independent_audit=True,
                    horizon1_targets_identical=True, common_support_window_monotonicity=True),
        sources=sources,
        generator={"path": Path(__file__).relative_to(SOURCE.ROOT).as_posix(), "sha256": SOURCE.sha(__file__)},
        outputs={name: SOURCE.sha(OUT / name) for name in ("severity.csv", "horizons.csv")},
    )
    SOURCE.dump(OUT / "manifest.json", manifest)
    print(json.dumps(manifest["checks"], indent=2))
    for row in severity:
        if row["tier"] == 2 and row["target"] == "future_step" and row["subset"] == "all" and row["minimum_relative_increase"] in (0., .01, .10, .25, .50):
            print("SEVERITY", row["arm"], row["minimum_relative_increase"], "n+", row["positives"],
                  "recall D/G/DG/Q", *(round(row[f"{f}_recall"], 3) if row[f"{f}_recall"] is not None else None for f in FLAGS))
    for row in forward:
        if row["tier"] == 2 and row["target"] == "any_above_current" and row["horizon"] in (1, 2, 4, 8, 16) and row["minimum_relative_increase"] == .10 and row["subset"] == "not_already_rising" and row["support"] == "common_16_step":
            print("FORWARD", row["arm"], row["horizon"], "n+", row["positives"], "Q recall/precision/lift",
                  *(round(row[f"Q_ge2_{metric}"], 3) if row[f"Q_ge2_{metric}"] is not None else None for metric in ("recall", "precision", "precision_lift")))


if __name__ == "__main__":
    main()
