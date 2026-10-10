"""Allocate the three authorized runs and freeze the shared Spark source packet."""
from pathlib import Path
import copy
import hashlib
import json
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REFERENCE = ROOT / "runs/002-2026-10-03-pythia14m-adamw-adaptive-geometry"
SOURCE_CACHE = ROOT.parent / "sparsity-spillover/data/tokenized/minipile-pythia-14m-full"
STAGE = ROOT / "cache/spark/no-decay-inputs"
RUNS = [("021", "base", .001), ("022", "small", .0001), ("023", "large", .01)]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    STAGE.mkdir(parents=True, exist_ok=False)
    frozen = REFERENCE / "artifacts/attempt-02/full-pass/checkpoints/initial/model"
    shutil.copytree(frozen, STAGE / "initial-model")
    frozen_hashes = {p.name: digest(p) for p in sorted(frozen.iterdir()) if p.is_file()}
    reference_config = json.loads((REFERENCE / "config.json").read_text())
    from sys import path
    path.insert(0, str(ROOT / "src"))
    from baseline.initialization import EXPECTED_ARCHITECTURE
    records = []
    for rid, condition, peak in RUNS:
        name = f"{rid}-2026-10-08-pythia14m-no-decay-{condition}"
        run = ROOT / "runs" / name
        run.mkdir(exist_ok=False)
        (run / "prelaunch").mkdir()
        config = copy.deepcopy(reference_config)
        config.update(run_id=rid, name=name.split("-", 4)[-1], hypothesis="H007", matched_reference_run="002")
        config["model"].update(parameter_count=14067712, expected_architecture=EXPECTED_ARCHITECTURE,
            frozen_initial_model="../../inputs/initial-model", frozen_initial_files_sha256=frozen_hashes)
        config["data"]["tokenizer"] = "EleutherAI/pythia-14m-deduped"
        config["training"].update(peak_learning_rate=peak, minimum_learning_rate=0,
                                 learning_rate_schedule="warmup_constant")
        config["measurements"].update(track_adam_momentum=True,
            momentum_vector="actual_FP32_exp_avg_without_bias_correction",
            momentum_formula="beta1*m_previous+(1-beta1)*globally_clipped_gradient",
            momentum_capture="after_accepted_optimizer_moment_update_before_preconditioning",
            previous_momentum_dtype="float32", momentum_cumulative_dtype="float64",
            momentum_excludes=["bias_correction", "second_moment_preconditioning", "learning_rate", "weight_decay"],
            momentum_norm_comparison="momentum/raw and adaptive/momentum")
        config["measurements"]["vectors"].append("adam_uncorrected_first_moment")
        config["checkpoints"]["recovery_every_updates"] = 100
        config["execution"] = {"provider": "owned_dgx_spark", "hostname": "spark-2277.local",
                               "workers": 1, "maximum_hours": 8, "cohort_maximum_hours": 24,
                               "cloud_gpu_cost_usd": 0, "cohort_additional_storage_allowance_gb": 40}
        save(run / "config.json", config)
        shutil.copyfile(REFERENCE / "model_config.json", run / "model_config.json")
        (run / "01_train.py").write_text('''"""Run-local entry point for the authorized three-stream full pass."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(4)
from baseline.training import main
if __name__ == "__main__":
    main()
''', encoding="utf-8")
        (run / "README.md").write_text(f'''# Run {rid}: Pythia-14M warmup-only LR {peak:g}

Status: preparing; user explicitly authorized full-pass execution on DGX Spark.
Condition: {condition}. Matched initial weights and seed1234 data order with runs021–023.
One complete MiniPile pass,712 updates; FP16 AdamW,batch1024,microbatch4,sequence2048.
Warmup is the existing pre-step1% ramp; first LR is0, plateau from update9 is{peak:g}.
Measure every raw/adaptive/momentum norm,A,B,B/A,adjacent cosine and paired norm ratios.
Momentum is actual uncorrected FP32 Adam exp_avg; its inputs are clipped gradients.
FP64 accumulation/reductions. Both full validation passes and token tails included.
Recovery: initial,midpoint,final,and every100 updates; all three stream buffers,
optimizer,scaler,RNG,data position retained.8h run timeout; shared sequential queue.
Scientific settings and bytes freeze at launch; calibration is separate and reset.
See [the full design/metric contract](../../analyses/{HERE.name}/README.md).
No stability/generalization claim or independent-seed uncertainty is implied.
''', encoding="utf-8")
        records.append({"run_id": rid, "run": name, "condition": condition, "peak_lr": peak})
    save(HERE / "cohort.json", {"runs": records, "status": "prepared", "scope": "full_pass_no_decay_momentum"})
    for split in ("train", "validation"):
        dest = STAGE / split
        dest.mkdir()
        shutil.copyfile(SOURCE_CACHE / split / "metadata.json", dest / "metadata.json")
    shutil.copyfile(SOURCE_CACHE / "validation/tokens.int32.bin", STAGE / "validation/tokens.int32.bin")
    files = list((ROOT / "src/baseline").glob("*.py"))
    for record in records:
        run = ROOT / "runs" / record["run"]
        files += [run / n for n in ("01_train.py", "config.json", "model_config.json")]
    files += [HERE / n for n in ("cohort.json", "01_queue.py", "02_verify.py", "03_status.py")]
    files += [ROOT / "tests/test_adam_momentum.py"]
    inventory = {"files": [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size,
                             "sha256": digest(p)} for p in sorted(files)]}
    save(HERE / "source_inventory.json", inventory)
    files += [HERE / "source_inventory.json"]
    packet = ROOT / "cache/spark/no-decay-source.tar.gz"
    with tarfile.open(packet, "w:gz") as archive:
        for p in files:
            archive.add(p, arcname=p.relative_to(ROOT).as_posix())
        archive.add(STAGE / "initial-model", arcname="inputs/initial-model")
        for split in ("train", "validation"):
            archive.add(STAGE / split, arcname="data/" + split)
    save(HERE / "packet_identity.json", {"bytes": packet.stat().st_size, "sha256": digest(packet)})
    print(json.dumps({"bytes": packet.stat().st_size, "sha256": digest(packet)}))


if __name__ == "__main__":
    main()
