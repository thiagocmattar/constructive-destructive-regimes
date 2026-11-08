"""Count each future loss rise once when a diagnostic may precede it by 1..16 steps.

This complements the decision-centered horizon analysis. The warning is causal:
at state s, it only remembers flags at states s-w+1,...,s.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("audit055", HERE / "05_forward_and_severity.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)
SOURCE, OUT = AUDIT.SOURCE, AUDIT.OUT


def main():
    arms, sources = SOURCE.load_arms()
    rows = []
    direct_checks = 0
    for arm in arms:
        states = np.arange(16, 1024)
        change = (arm["loss"][states + 1] - arm["loss"][states]) / arm["loss"][states]
        point_flags = (arm["rho"] < 0, arm["r"] > 1,
                       (arm["rho"] < 0) & (arm["r"] > 1), arm["Q"] >= 2)
        prefixes = [np.r_[0, np.cumsum(flag)] for flag in point_flags]
        last_flags = None
        for width in range(1, 17):
            flags = [prefix[states + 1] > prefix[states - width + 1] for prefix in prefixes]
            for point, flag in zip(point_flags, flags):
                expected = [any(point[s-width+1:s+1]) for s in states]
                np.testing.assert_array_equal(flag, expected)
                direct_checks += len(states)
            if last_flags is not None:
                for old, new in zip(last_flags, flags):
                    assert np.all(~old | new)
            last_flags = flags
            for minimum in AUDIT.MINIMUM_RISES:
                for subset in ("all", "not_already_rising"):
                    keep = (np.ones(len(states), dtype=bool) if subset == "all" else
                            arm["loss"][states] <= arm["loss"][states - 1])
                    positive = (change > 0 if minimum == 0 else change >= minimum)[keep]
                    n, positives = len(positive), int(positive.sum())
                    row = dict(arm=arm["arm"], task=arm["task"], lr=arm["lr"], tier=arm["tier"],
                               target="future_step", horizon=width, minimum_relative_increase=minimum,
                               subset=subset, support="common_16_step_history", states=n,
                               positives=positives, negatives=n - positives, base_rate=AUDIT.divide(positives, n))
                    for name, flag in zip(AUDIT.FLAGS, flags):
                        selected = flag[keep]
                        alarms, tp = int(selected.sum()), int((selected & positive).sum())
                        fp, fn = alarms - tp, positives - tp
                        row.update({f"{name}_alarms": alarms, f"{name}_tp": tp, f"{name}_fp": fp,
                                    f"{name}_fn": fn, f"{name}_tn": n - tp - fp - fn,
                                    f"{name}_recall": AUDIT.divide(tp, positives),
                                    f"{name}_precision": AUDIT.divide(tp, alarms),
                                    f"{name}_false_positive_rate": AUDIT.divide(fp, n - positives),
                                    f"{name}_precision_lift": AUDIT.divide(tp * n, alarms * positives)})
                    rows.append(row)
    assert len(rows) == 2592
    SOURCE.write_csv(OUT / "warning_memory.csv", rows)
    sources[Path(HERE / "05_forward_and_severity.py").relative_to(SOURCE.ROOT).as_posix()] = SOURCE.sha(HERE / "05_forward_and_severity.py")
    manifest = dict(date="2026-11-08", purpose="Unique-rise coverage for a causal warning retained over prior states",
                    clock="A width w warning at state s is any pointwise flag in s-w+1,...,s, before update s->s+1",
                    support="Same states 16..1023 for all widths; 1008 decisions per arm, 9072 total",
                    widths=list(range(1, 17)), minimum_rises=list(AUDIT.MINIMUM_RISES),
                    thresholds_and_subsets="Same fixed severity grid and past-defined subsets as 05_forward_and_severity.py",
                    interpretation="Each target update counted once. This is exploratory temporal coverage, not calibrated prediction. Longer memory changes false alarms as well as coverage.",
                    checks=dict(rows=len(rows), direct_window_flag_checks=direct_checks,
                                history_only=True, nested_warning_sets=True), sources=sources,
                    generator={"path": Path(__file__).relative_to(SOURCE.ROOT).as_posix(), "sha256": SOURCE.sha(__file__)},
                    outputs={"warning_memory.csv": SOURCE.sha(OUT / "warning_memory.csv")})
    SOURCE.dump(OUT / "memory_manifest.json", manifest)
    print(json.dumps(manifest["checks"], indent=2))
    for row in rows:
        if row["tier"] == 2 and row["horizon"] in (1, 2, 3, 4, 8, 16) and row["minimum_relative_increase"] == .1 and row["subset"] == "all":
            print(row["arm"], "width", row["horizon"], "Q TP/positives/FP",
                  row["Q_ge2_tp"], row["positives"], row["Q_ge2_fp"])


if __name__ == "__main__":
    main()
