"""Audit the three manuscript events and reduce earlier context; no training."""

import csv
import hashlib
import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "analyses/036-2026-11-02-controlled-adam-diagnostics"
RUN = ROOT / "runs/215-2026-11-02-controlled-full-batch-adam"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def recovered_events(losses):
    """Independent fixed-level segmentation, retaining earliest tied peaks."""
    result = []
    t = 0
    while t < len(losses) - 1:
        if losses[t + 1] <= losses[t]:
            t += 1
            continue
        a = t
        c = a + 1
        while c < len(losses) and losses[c] > losses[a]:
            c += 1
        if c == len(losses):
            break
        p = max(range(a + 1, c), key=losses.__getitem__)
        result.append((a, p, c))
        t = c
    return result


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    source_manifest = json.loads((SOURCE / "spike_dataset/manifest.json").read_text())
    pinned = source_manifest["sources"]
    with (SOURCE / "spike_dataset/loss_spikes.csv").open(newline="") as stream:
        events = list(csv.DictReader(stream))
    sources = {}

    def record(path, expected=None):
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Source SHA-256 mismatch: {path}")
        sources[relative(path)] = dict(bytes=path.stat().st_size, sha256=actual)

    for path in (SOURCE / "spike_dataset/loss_spikes.csv",
                 SOURCE / "spike_dataset/manifest.json",
                 SOURCE / "figures/adam_manuscript_spikes.pdf",
                 RUN / "config.json", RUN / "input_manifest.json",
                 RUN / "01_train.py"):
        record(path, pinned.get(relative(path)))

    cases, context, details = [], [], []
    for task, lr in (("mlp", .001), ("resnet", .003), ("transformer", .003)):
        selected = max(
            (e for e in events if e["task"] == task
             and float(e["learning_rate"]) == lr and e["right_censored"] == "False"),
            key=lambda e: float(e["peak_loss_ratio"]),
        )
        folder = ROOT / selected["source"]
        for filename in ("manifest.json", "boundaries.jsonl", "metrics.jsonl"):
            path = folder / filename
            record(path, pinned[relative(path)])
        manifest = json.loads((folder / "manifest.json").read_text())
        if manifest["task"] != task or manifest["lr"] != lr:
            raise ValueError("Condition identity mismatch")
        if manifest["initial_tensor_sha256"] != selected["initial_tensor_sha256"]:
            raise ValueError("Initialization mismatch")
        if manifest["input_manifest_sha256"] != sources[relative(RUN / "input_manifest.json")]["sha256"]:
            raise ValueError("Input identity mismatch")
        if manifest["source_sha256"][relative(RUN / "01_train.py")] != sources[relative(RUN / "01_train.py")]["sha256"]:
            raise ValueError("Training source identity mismatch")
        boundary_rows = read_jsonl(folder / "boundaries.jsonl")
        metric_rows = read_jsonl(folder / "metrics.jsonl")
        assert [r["state"] for r in boundary_rows] == list(range(1025))
        assert [r["gradient_state"] for r in metric_rows] == list(range(1024))
        assert [r["update"] for r in metric_rows] == list(range(1, 1025))
        losses = [r["loss"] for r in boundary_rows]
        for b, m in zip(boundary_rows, metric_rows):
            assert b["loss"] == m["loss_before"]
            assert b["gradient_norm"] == m["streams"]["raw"]["norm"]
        a, p, c = max(recovered_events(losses), key=lambda e: losses[e[1]] / losses[e[0]])
        assert (a, p, c) == tuple(int(selected[k]) for k in ("before_state", "peak_state", "recovery_state"))
        assert losses[p] / losses[a] == float(selected["peak_loss_ratio"])
        f = a - 32
        pre = metric_rows[f - 64:f]
        cap_g = statistics.median(r["streams"]["raw"]["norm"] for r in pre)
        cap_step = statistics.median(lr * r["streams"]["adaptive"]["norm"] for r in pre)
        assert cap_g > 0 and cap_step > 0
        rise = metric_rows[a:p]
        descending = sum(r["gradient_dot_actual_step"] < 0 for r in rise)
        assert descending == p - a
        assert all(r["gradient_adaptive_cosine"] > 0 for r in rise)
        retained = []
        for path in sorted(folder.glob("*.pt")):
            entry = manifest["files"][path.name]
            assert path.stat().st_size == entry["bytes"]
            record(path, entry["sha256"])
            retained.append(dict(path=relative(path), **sources[relative(path)]))
        assert [item["path"].split("/")[-1] for item in retained] == [
            "audit-gradient-0.pt", "audit-gradient-1.pt", "audit-gradient-1024.pt", "final.pt"]
        row = dict(task=task, learning_rate=lr, seed=0, spike_id=selected["spike_id"],
                   before_state=a, onset_state=a + 1, peak_state=p, recovery_state=c,
                   peak_loss_ratio=losses[p] / losses[a], main_fork=f,
                   late_fork=a, reactive_fork=a + 1, horizon_updates=128,
                   main_end=f + 128, history_start=f - 64, history_end=f - 1,
                   raw_gradient_clip_threshold=cap_g, adam_proposal_cap=cap_step,
                   pre64_negative_raw_cosines=sum(r["streams"]["raw"]["cosine"] < 0 for r in pre),
                   pre64_non_descent_adam_directions=sum(r["gradient_adaptive_cosine"] <= 0 for r in pre),
                   rise_updates=p - a, rise_first_order_descent_updates=descending,
                   untreated_rise_gradient_cap_exceedances=sum(r["streams"]["raw"]["norm"] > cap_g for r in rise),
                   untreated_rise_step_cap_exceedances=sum(lr * r["streams"]["adaptive"]["norm"] > cap_step for r in rise),
                   source=selected["source"], source_gpu=selected["gpu"])
        cases.append(row)
        details.append(dict(task=task, retained_states=retained,
                            historical_update_seconds_median=statistics.median(r["seconds"] for r in metric_rows[64:]),
                            main_fork_loss=losses[f], formal_baseline_loss=losses[a], peak_loss=losses[p]))
        for t in range(a - 96, a + 129):
            r = metric_rows[t]
            context.append(dict(task=task, state=t, updates_from_prerise=t - a,
                loss=losses[t], loss_over_formal_baseline=losses[t] / losses[a],
                projected_raw_gradient_change=r["projected_gradient_change"],
                raw_cosine=r["streams"]["raw"]["cosine"],
                momentum_cosine=r["streams"]["moment"]["cosine"],
                gradient_adaptive_cosine=r["gradient_adaptive_cosine"],
                raw_gradient_norm=r["streams"]["raw"]["norm"],
                adam_proposal_norm=lr * r["streams"]["adaptive"]["norm"],
                gradient_dot_applied_step=r["gradient_dot_actual_step"]))

    write_csv(HERE / "cases.csv", cases)
    write_csv(HERE / "extended_context.csv", context)
    review = dict(status="verified_read_only_source_audit", generator=relative(Path(__file__).resolve()),
                  generator_sha256=digest(Path(__file__)), sources=sources,
                  coverage=dict(source_arms=3, loss_states=3075, metric_rows=3072,
                                extended_context_rows=len(context), selected_rise_updates=28,
                                selected_rise_first_order_descent_updates=28),
                  cases=cases, details=details,
                  proposed_work=dict(main_arms=24, main_continuation_updates=24 * 128,
                                     replay_prefix_updates=sum(r["main_fork"] for r in cases),
                                     nominal_main_total_updates=24 * 128 + sum(r["main_fork"] for r in cases)),
                  interpretation="Historical reductions only; thresholds and future intervention arms remain proposals.")
    (HERE / "source_review.json").write_text(json.dumps(review, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(status=review["status"], coverage=review["coverage"],
                          proposed_work=review["proposed_work"]), indent=2))


if __name__ == "__main__":
    main()
