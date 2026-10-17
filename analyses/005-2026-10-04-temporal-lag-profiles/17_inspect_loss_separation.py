"""Describe the user-selected 200-250 interval and flanking training windows."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / "training_size_adam_direction_split_loss_only_n5_n21_manifest.json"
WINDOWS = ((100, 150), (150, 200), (200, 250), (250, 300), (300, 350))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    source = json.loads(SOURCE.read_text())
    assert source["visual_review"]["passed"]
    preserved = {p.name: sha(p) for p in (HERE / "figures").glob("*.pdf")}
    reductions, identities, schedules = [], [], []
    for member in source["sources"]:
        for item in [member["receipt"], *member["files"]]:
            path = ROOT / item["path"]
            assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"]
        paths = {Path(s["path"]).name: ROOT / s["path"] for s in member["files"]}
        identity = json.loads(paths["input_identity.json"].read_text())
        config = json.loads(paths["resolved_config.json"].read_text())
        identities.append((identity, config["seeds"], config["training"]))
        events = [json.loads(line) for line in paths["events.jsonl"].read_text().splitlines()]
        updates = {e["update"]: e for e in events if e["event"] == "update"}
        assert list(updates) == list(range(1, 713))
        assert {e["event"] for e in events} == {"initial_validation", "update", "final_validation"}
        assert all(e["overflow_retries"] == 0 for e in updates.values())
        schedules.append([e["learning_rate"] for e in updates.values()])
        for first, last in WINDOWS:
            selected = [updates[t] for t in range(first, last + 1)]
            assert all(e["sequences"] == 1024 and e["supervised_tokens"] == 2096128 for e in selected)
            loss = np.array([e["loss"] for e in selected])
            assert np.isfinite(loss).all() and (loss > 0).all()
            x = np.arange(first, last + 1, dtype=float)
            centered = x - x.mean()
            slope = float(centered @ np.log(loss) / (centered @ centered))
            np.testing.assert_allclose(slope, np.polyfit(x, np.log(loss), 1)[0], rtol=1e-11, atol=1e-14)
            np.testing.assert_allclose(np.diff(np.log(loss)).mean(), np.log(loss[-1] / loss[0]) / (last - first), atol=1e-14)
            norm = np.array([e["adaptive"]["direction_norm"] for e in selected])
            rho = np.array([e["adaptive"]["adjacent_cosine"] for e in selected])
            previous_norm = np.array([updates[e["update"] - 1]["adaptive"]["direction_norm"] for e in selected])
            rate = np.array([e["learning_rate"] for e in selected])
            reductions.append(dict(size=member["size"], updates=[first, last],
                loss_records=len(selected), loss_differences=last - first,
                initial_loss=float(loss[0]), final_loss=float(loss[-1]),
                endpoint_loss_drop_percent=float(100 * (1 - loss[-1] / loss[0])),
                fitted_log_loss_slope=slope,
                mean_raw_gradient_norm=float(np.mean([e["gradient_norm"] for e in selected])),
                mean_raw_rho=float(np.mean([e["adjacent_cosine"] for e in selected])),
                mean_direction_norm=float(norm.mean()), mean_direction_rho=float(rho.mean()),
                mean_direction_change=float(np.mean(1 - rho * norm / previous_norm)),
                mean_lr_times_direction_norm=float(np.mean(rate * norm)),
                initial_learning_rate=float(rate[0]), final_learning_rate=float(rate[-1]),
                clipping_count=sum(e["gradient_clip_returned_norm"] > config["training"]["gradient_clip_norm"] for e in selected),
                loss_scales=sorted({e["loss_scale"] for e in selected})))
    assert identities[0] == identities[1] == identities[2]
    np.testing.assert_array_equal(schedules[0], schedules[1])
    np.testing.assert_array_equal(schedules[0], schedules[2])
    assert np.all(np.diff(np.array(schedules[0])[149:350]) < 0)
    assert all(sha(HERE / "figures" / name) == digest for name, digest in preserved.items())
    record = dict(script_sha256=sha(__file__), source_manifest=SOURCE.name,
        source_manifest_sha256=sha(SOURCE), sources=source["sources"],
        user_selected_window=[200, 250], windows=list(WINDOWS), reductions=reductions,
        definitions=dict(endpoint_loss_drop="100*(1-L[last]/L[first]); 50 adjacent changes",
            fitted_slope="OLS slope of ln(loss) against update across 51 unsmoothed records",
            geometry="arithmetic descriptive means across all 51 inclusive records; figures unchanged",
            lr_times_direction_norm="norm of adaptive update component eta*d; excludes decay and rounding"),
        checks=dict(accepted_source_hashes_and_bytes=True, matched_data_seed_training_recipe=True,
            identical_lr_history=True, lr_strictly_decreasing_150_350=True,
            no_overflows=True, all_visible_batches_full=True,
            independent_ols_and_log_telescoping=True, previous_figures_unchanged=True),
        preserved_figures=preserved,
        interpretation="Descriptive loss separation in one seed; no causal mechanism or asymptotic convergence established.")
    (HERE / "loss_separation_200_250.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record["checks"], indent=2))
    for r in reductions:
        print(r["size"], r["updates"], "loss drop %", round(r["endpoint_loss_drop_percent"], 3),
              "log slope", round(r["fitted_log_loss_slope"], 7))


if __name__ == "__main__":
    main()
