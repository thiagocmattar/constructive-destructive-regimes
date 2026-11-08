"""Audit every one-step loss change, keeping diagnostic availability explicit.

Extends Analysis055 with the 2026-11-08 manuscript-plan review. No fitting,
training, event selection, quiet-state restriction, or threshold optimization.
"""
import csv
import importlib.util
import json
from fractions import Fraction as F
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("analysis055", HERE / "01_analyze.py")
SOURCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOURCE)
OUT = HERE / "derived" / "one_step_audit"
TOLERANCES = (0.0, 1e-5)


def ratio(a, b):
    return a / b if b else None


def summarize(arm, timing, tolerance):
    # State s has s completed updates. Q_s uses gradients at s-1 and s.
    loss = arm["loss"]
    before, after = ((loss[:-2], loss[1:-1]) if timing == "incoming"
                     else (loss[1:-1], loss[2:]))
    relative = (after - before) / before
    up, down = relative > tolerance, relative < -tolerance
    flat = ~(up | down)
    q, rho, growth = arm["Q"][1:], arm["rho"][1:], arm["r"][1:]
    flags = {
        "destructive": rho < 0,
        "growth": growth > 1,
        "reversing_growth": (rho < 0) & (growth > 1),
        "Q_ge2": q >= 2,
        "joint": (q >= 2) & (rho < 0) & (growth > 1),
    }
    # The three-part flag is algebraically redundant, not three confirmations.
    np.testing.assert_array_equal(flags["Q_ge2"], flags["joint"])
    row = dict(arm=arm["arm"], task=arm["task"], lr=arm["lr"],
               tier=arm["tier"], timing=timing, relative_tolerance=tolerance,
               states=len(relative), increases=int(up.sum()),
               decreases=int(down.sum()), unchanged_or_negligible=int(flat.sum()),
               Q_equal2=int((q == 2).sum()))
    for name, flag in flags.items():
        hit, false, neutral = [int((flag & mask).sum()) for mask in (up, down, flat)]
        total = int(flag.sum())
        row.update({f"{name}_total": total, f"{name}_increases": hit,
                    f"{name}_decreases": false, f"{name}_negligible": neutral,
                    f"{name}_recall": ratio(hit, int(up.sum())),
                    f"{name}_precision": ratio(hit, total),
                    f"{name}_decrease_alarm_rate": ratio(false, int(down.sum()))})
    assert row["increases"] + row["decreases"] + row["unchanged_or_negligible"] == 1023
    return row


def verify_against_raw(arm, rows):
    """Independently index boundary states and recompute Q from raw norms/cosine."""
    folder = SOURCE.ROOT / arm["source"]
    boundaries = [json.loads(line) for line in (folder / "boundaries.jsonl").read_text().splitlines()]
    metrics = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
    checked = 0
    for row in rows:
        counts = dict(increases=0, decreases=0, unchanged_or_negligible=0, Q_equal2=0)
        for name in ("destructive", "growth", "reversing_growth", "Q_ge2", "joint"):
            for suffix in ("total", "increases", "decreases", "negligible"):
                counts[f"{name}_{suffix}"] = 0
        for state in range(1, 1024):
            assert metrics[state]["gradient_state"] == state
            assert metrics[state]["update"] == state + 1
            assert metrics[state]["loss_before"] == boundaries[state]["loss"]
            norm = metrics[state]["streams"]["raw"]["norm"]
            previous = metrics[state - 1]["streams"]["raw"]["norm"]
            cosine = metrics[state]["streams"]["raw"]["cosine"]
            growth = norm / previous
            q = 1 - cosine * growth
            # A classification disagreement near 2 must be surfaced, not rounded away.
            assert (q >= 2) == (metrics[state]["projected_gradient_change"] >= 2)
            start = state - 1 if row["timing"] == "incoming" else state
            change = (boundaries[start + 1]["loss"] - boundaries[start]["loss"]) / boundaries[start]["loss"]
            outcome = ("increases" if change > row["relative_tolerance"] else
                       "decreases" if change < -row["relative_tolerance"] else "negligible")
            counts["unchanged_or_negligible" if outcome == "negligible" else outcome] += 1
            counts["Q_equal2"] += int(q == 2)
            for name, flag in dict(destructive=cosine < 0, growth=growth > 1,
                                   reversing_growth=cosine < 0 and growth > 1,
                                   Q_ge2=q >= 2,
                                   joint=q >= 2 and cosine < 0 and growth > 1).items():
                counts[f"{name}_total"] += int(flag)
                counts[f"{name}_{outcome}"] += int(flag)
            checked += 1
        for key, count in counts.items():
            assert count == row[key], (arm["arm"], row["timing"], key, count, row[key])
    return checked


def verify_quadratic_clock():
    # Exact 1-D GD: an overshooting incoming step followed by a shorter descent.
    # Q at the middle state explains the first step, and is not the next outcome.
    x0, incoming_eta, outgoing_eta = F(1), F(3), F(1, 2)
    x1 = x0 - incoming_eta * x0
    x2 = x1 - outgoing_eta * x1
    q1 = 1 - x1 / x0
    incoming = (x1 * x1 - x0 * x0) / 2
    outgoing = (x2 * x2 - x1 * x1) / 2
    assert incoming == incoming_eta * x0 * x0 * (q1 - 2) / 2
    assert q1 > 2 and incoming > 0 and outgoing < 0
    # Reciprocal BB Eq. (6) equals the GD directional secant, exactly.
    step, y = x1 - x0, x1 - x0
    assert q1 / incoming_eta == step * y / (step * step)


def main():
    arms, sources = SOURCE.load_arms()
    rows, examples, checked = [], [], 0
    for arm in arms:
        part = [summarize(arm, timing, tolerance)
                for timing in ("incoming", "outgoing") for tolerance in TOLERANCES]
        checked += verify_against_raw(arm, part)
        rows.extend(part)
        for timing in ("incoming", "outgoing"):
            before, after = ((arm["loss"][:-2], arm["loss"][1:-1]) if timing == "incoming"
                             else (arm["loss"][1:-1], arm["loss"][2:]))
            relative = (after - before) / before
            q = arm["Q"][1:]
            for name, mask in (("missed_increase", (relative > 0) & (q < 2)),
                               ("alarm_before_decrease", (relative < 0) & (q >= 2))):
                candidates = np.flatnonzero(mask)
                if not len(candidates):
                    continue
                # Illustrations only: largest absolute relative counterexample in each group.
                idx = int(candidates[np.argmax(np.abs(relative[candidates]))])
                state = idx + 1
                examples.append(dict(arm=arm["arm"], timing=timing, category=name,
                                     diagnostic_state=state,
                                     change_start_state=state - int(timing == "incoming"),
                                     loss_before=float(before[idx]), loss_after=float(after[idx]),
                                     relative_change=float(relative[idx]), Q=float(q[idx]),
                                     rho=float(arm["rho"][state]), r=float(arm["r"][state])))
    verify_quadratic_clock()
    assert len(rows) == 36 and checked == 36828
    strict = [r for r in rows if r["timing"] == "outgoing" and r["relative_tolerance"] == 0]
    assert sum(r["states"] for r in strict) == 9207
    OUT.mkdir(exist_ok=True)
    SOURCE.write_csv(OUT / "summary.csv", rows)
    SOURCE.dump(OUT / "counterexamples.json", examples)
    sources["analyses/055-2026-11-05-adam-directional-amplification/01_analyze.py"] = SOURCE.sha(HERE / "01_analyze.py")
    manifest = dict(
        purpose="Exhaustive one-step timing audit for the manuscript consolidation plan, 2026-11-08",
        scope="Nine preserved full-objective Adam trajectories; one initialization per model; no new training or fitting",
        clock="state s = s completed updates; Q_s uses raw gradients at s-1 and s, available before update s -> s+1",
        targets={"incoming": "L_s - L_(s-1): retrospective", "outgoing": "L_(s+1) - L_s: pre-update"},
        coverage=dict(arms=9, updates_per_arm=1024, eligible_states_per_arm=1023,
                      eligible_total=9207, unavailable_initial_diagnostics=9,
                      terminal_states_without_outgoing_update=9),
        selection="All states 1..1023; no warmup, quiet-state, event, peak, or magnitude selection",
        relative_tolerances=list(TOLERANCES),
        threshold_semantics="Strict increase uses >0; sensitivity uses relative change >1e-5. Q>=2 is the requested alarm; no exact Q=2 occurs. Q>2 is the exact quadratic-GD positive-rise condition for the incoming step.",
        uncertainty="Counts are temporally dependent states, not independent seeds. No inferential intervals or universal prediction claim.",
        example_selection="Largest absolute relative change per arm/timing/error type, only to illustrate an already exhaustive count",
        checks=dict(raw_recomputed_state_classifications=checked, summary_rows=36,
                    algebraic_joint_redundancy=True, exact_quadratic_clock_and_BB=True),
        strict_outgoing_totals={k: sum(r[k] for r in strict) for k in
                                ("states", "increases", "decreases", "unchanged_or_negligible",
                                 "Q_ge2_increases", "Q_ge2_decreases", "destructive_increases", "growth_increases")},
        sources=sources,
        generator={"path": Path(__file__).relative_to(SOURCE.ROOT).as_posix(), "sha256": SOURCE.sha(__file__)},
        outputs={name: SOURCE.sha(OUT / name) for name in ("summary.csv", "counterexamples.json")},
    )
    SOURCE.dump(OUT / "manifest.json", manifest)
    print(json.dumps({"checks": manifest["checks"], "totals": manifest["strict_outgoing_totals"]}, indent=2))
    for row in strict:
        if row["tier"] == 2:
            print(row["arm"], "up/down", row["increases"], row["decreases"],
                  "before up: rho/r/Q", *(row[f"{name}_increases"] for name in ("destructive", "growth", "Q_ge2")),
                  "before down: rho/r/Q", *(row[f"{name}_decreases"] for name in ("destructive", "growth", "Q_ge2")))


if __name__ == "__main__":
    main()
