"""Prepare unexecuted H021 packets; never overwrite an executed input."""
from pathlib import Path
import copy
import hashlib
import json
import shutil

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATE = "2026-10-17"
SIZES = ("14M", "31M", "70M")
SEEDS = (1234, 1235, 1236)
PARENTS = {
    1234: ("002-2026-10-03-pythia14m-adamw-adaptive-geometry", "003-2026-10-03-pythia31m-adamw-scale", "004-2026-10-03-pythia70m-adamw-scale"),
    1235: ("005-2026-10-04-pythia14m-seed1235", "006-2026-10-04-pythia31m-seed1235", "007-2026-10-04-pythia70m-seed1235"),
    1236: ("008-2026-10-04-pythia14m-seed1236", "009-2026-10-04-pythia31m-seed1236", "010-2026-10-04-pythia70m-seed1236"),
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")


def main():
    if (HERE / "cohort.json").exists():
        raise RuntimeError("Prepared cohort exists; review changes explicitly rather than overwrite")
    if any(list((ROOT / "runs").glob(f"{i:03}-*")) for i in range(127, 154)):
        raise RuntimeError("An allocated run ID is already in use")
    # Draw a common probe from the complement of all three online prefixes.
    train_blocks, effective, length = 728374, 1024, 2048
    excluded = np.zeros(train_blocks, dtype=bool)
    orders = {}
    for seed in SEEDS:
        order = np.r_[np.random.default_rng(seed).permutation(train_blocks), train_blocks].astype("<i8")
        orders[seed] = hashlib.sha256(order.tobytes()).hexdigest()
        excluded[order[:350 * effective]] = True
    candidates = np.random.default_rng(104729).permutation(train_blocks)
    probe_indices = candidates[~excluded[candidates]][:128].tolist()
    assert len(probe_indices) == len(set(probe_indices)) == 128
    probe_path = HERE / "probe.json"
    save(probe_path, dict(seed=104729, indices=probe_indices, sequence_length=length,
        supervised_tokens=128 * (length - 1), excluded_online_prefix_updates=350,
        excluded_data_order_seeds=list(SEEDS), selection="first 128 common-complement blocks in seeded permutation"))
    rows, initial_sources, historical = [], [], []
    for seed in SEEDS:
        for size, reference in zip(SIZES, PARENTS[seed]):
            rid = 127 + len(rows)
            name = f"{rid:03}-{DATE}-pythia{size.lower()}-descent-baseline-seed{seed}"
            parent = ROOT / "runs" / reference
            config = json.loads((parent / "config.json").read_text())
            env_path = next(parent.glob("artifacts/*/full-pass/environment.json"))
            initial = env_path.parent / "checkpoints/initial/model"
            initial_files = {p.name: sha(p) for p in initial.iterdir() if p.is_file()}
            assert "model.safetensors" in initial_files
            config["model"].update(frozen_initial_model=f"../../inputs/initials/{size.lower()}-{seed}",
                frozen_initial_files_sha256=initial_files)
            config.update(run_id=f"{rid:03}", name=name[15:], hypothesis="H021")
            config["measurements"]["track_adam_momentum"] = True
            config["causal"] = dict(condition="baseline", baseline_updates=350, schedule_updates=712,
                junctions=[175, 225], branch_updates=50, recovery_every_updates=50,
                measurement_updates=[100, 150, 175, 200, 225, 250, 275, 300, 350],
                probe_path=f"../../analyses/{HERE.name}/probe.json", probe_sha256=sha(probe_path),
                probe_microbatch=1, probe_precision="float32_math_sdpa_tf32_disabled",
                order_sha256=orders[seed], primary_outcome="fixed_probe_log_loss_change_over_50_updates")
            config["execution"] = dict(provider="runpod", gpu="NVIDIA GeForce RTX 4090",
                requested_workers=2, network_volume_gb=200, container_gb_per_worker=30,
                proposed_max_elapsed_hours=12, proposed_max_total_usd=25,
                authorization_record=f"analyses/{HERE.name}/authorization.json",
                monitoring_seconds=300, retrieval_reserve_seconds=5400)
            run = ROOT / "runs" / name
            save(run / "config.json", config)
            shutil.copy2(parent / "model_config.json", run / "model_config.json")
            (run / "prelaunch").mkdir()
            rows.append(dict(run=name, size=size, seed=seed, condition="baseline", reference=reference))
            initial_sources.append(dict(size=size, seed=seed, source=initial.relative_to(ROOT).as_posix(),
                destination=f"inputs/initials/{size.lower()}-{seed}", files=initial_files))
            events = [json.loads(line) for line in (env_path.parent / "events.jsonl").read_text().splitlines()]
            native = [e for e in events if e["event"] == "update" and 11 <= e["update"] <= 350]
            environment = json.loads(env_path.read_text())
            historical.append(dict(size=size, seed=seed, source=env_path.parent.relative_to(ROOT).as_posix(),
                gpu=environment.get("gpu"), mean_update_seconds=float(np.mean([e["update_seconds"] for e in native])),
                median_update_seconds=float(np.median([e["update_seconds"] for e in native]))))
    baselines = list(rows)
    for baseline in baselines:
        for junction in (175, 225):
            rid = 127 + len(rows)
            name = f"{rid:03}-{DATE}-pythia{baseline['size'].lower()}-no-precond-fork{junction}-seed{baseline['seed']}"
            parent = ROOT / "runs" / baseline["run"]
            config = copy.deepcopy(json.loads((parent / "config.json").read_text()))
            config.update(run_id=f"{rid:03}", name=name[15:])
            config["causal"].update(condition="norm_matched_no_preconditioning", junction_update=junction,
                parent_config=f"../{parent.name}/config.json", parent_config_sha256=sha(parent / "config.json"),
                parent_output=f"../{parent.name}/artifacts/attempt-01/full-pass",
                target_step_norm="calendar-matched nominal eta*norm(d) from unmodified parent trajectory")
            run = ROOT / "runs" / name
            save(run / "config.json", config)
            shutil.copy2(parent / "model_config.json", run / "model_config.json")
            (run / "prelaunch").mkdir()
            rows.append(dict(run=name, size=baseline["size"], seed=baseline["seed"],
                             condition=config["causal"]["condition"], parent=parent.name, junction=junction))
    for row in rows:
        run = ROOT / "runs" / row["run"]
        (run / "02_train.py").write_text("from pathlib import Path\nimport sys\nsys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))\nfrom baseline.causal_training import main\nif __name__ == '__main__':\n    main()\n", encoding="utf-8")
        (run / "README.md").write_text(f"# Run{row['run'][:3]}: {row['size']} descent mechanism, seed{row['seed']}\n\nStatus: prepared; spending approval pending. Condition: {row['condition']}.\n\nScientific contract, probe coverage, controls, artifacts, interpretation and\nlaunch bounds: [Analysis025](../../analyses/{HERE.name}/README.md).\nFrozen inputs: config.json and model_config.json; execution has not started.\n", encoding="utf-8")
    save(HERE / "cohort.json", rows)
    save(HERE / "initial_sources.json", initial_sources)
    save(HERE / "historical_timing.json", historical)
    save(HERE / "design.json", dict(hypothesis="H021", sizes=list(SIZES), seed_blocks=list(SEEDS),
        seeds_note="initialization and data-order seed co-vary; match within each size block",
        baselines=9, continuation_branches=18, baseline_updates=350, branch_updates=50,
        branch_junctions=[175, 225], original_schedule_updates=712,
        one_step_candidates=["native_adamw", "adam", "momentum", "current_preconditioned", "gradient", "adam_scale_0.5", "adam_scale_2"],
        primary_continuation="remove coordinate preconditioning while matching unmodified parent's adaptive-step norm schedule; decay unchanged",
        probe_sha256=sha(probe_path), run_ids=[127, 153], status="prepared_pending_spending_approval"))
    print(json.dumps(dict(runs=len(rows), baselines=len(baselines), branches=len(rows)-len(baselines),
        probe_blocks=len(probe_indices), online_probe_overlap=0), indent=2))


if __name__ == "__main__":
    main()
