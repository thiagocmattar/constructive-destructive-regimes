"""Freeze six single-seed midpoint continuations and package verified parents."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import shutil
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENTS = {"14M": "005-2026-10-04-pythia14m-seed1235",
           "70M": "007-2026-10-04-pythia70m-seed1235"}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def main():
    if (HERE / "cohort.json").exists():
        raise RuntimeError("Cohort already frozen; do not rewrite scientific inputs.")
    cohort = []
    for size, parent_name in PARENTS.items():
        parent = ROOT / "runs" / parent_name
        base = parent / "artifacts/attempt-01/full-pass"
        original = json.loads((base / "resolved_config.json").read_text())
        checkpoint = base / "checkpoints/midpoint"
        identities = {p.relative_to(checkpoint).as_posix(): sha(p)
                      for p in checkpoint.rglob("*") if p.is_file()}
        order = json.loads((base / "input_identity.json").read_text())["order_sha256"]
        for variant in ("original", "hold", "accelerated"):
            rid = f"{11+len(cohort):03}"
            name = f"{rid}-2026-10-04-pythia{size.lower()}-lr-{variant}-seed1235"
            run = ROOT / "runs" / name
            run.mkdir(exist_ok=False)
            (run / "prelaunch").mkdir()
            config = json.loads(json.dumps(original))
            config.update(run_id=rid, name=name[15:], hypothesis="H004")
            config["resume"] = {"parent_run": parent_name, "parent_attempt": "attempt-01",
                "parent_checkpoint": "full-pass/checkpoints/midpoint", "junction_update": 356,
                "full_updates": 712, "schedule": variant, "remote_checkpoint": "/workspace/parent-checkpoint",
                "parent_config_sha256": sha(parent / "artifacts/attempt-01/source/config.json"),
                "checkpoint_sha256": identities, "order_sha256": order,
                "validation_updates": [445, 534, 623, 712]}
            config["execution"] = {"provider": "runpod", "workers": 1, "cohort_workers": 6,
                "maximum_hours": 2, "cohort_maximum_total_usd": 15,
                "container_gb": 60, "persistent_gb": 40, "retrieval_margin_seconds": 1200,
                "monitoring_interval_seconds": 60, "gpu": "NVIDIA GeForce RTX 5090"}
            config["validation"]["updates"] = [356, 445, 534, 623, 712]
            config["checkpoints"] = {"parent_immutable": True, "recovery_updates": [534, 712]}
            save(run / "config.json", config)
            shutil.copy2(parent / "model_config.json", run / "model_config.json")
            (run / "02_train.py").write_text('''from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from baseline.continuation import main
if __name__ == "__main__": main()
''', encoding="utf-8")
            shutil.copy2(HERE / "verify_continuation.py", run / "03_verify.py")
            setup = (parent / "00_setup_remote.sh").read_text().replace(parent_name, name)
            (run / "00_setup_remote.sh").write_text(setup, encoding="utf-8", newline="\n")
            for filename in ("accept_local_cache.py", "remote_status.py", "deadline-guard.ps1"):
                shutil.copy2(parent / "prelaunch" / filename, run / "prelaunch" / filename)
            rd = "/workspace/constructive/runs/" + name
            (run / "prelaunch/run.sh").write_text(f'''#!/usr/bin/env bash
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
cd {rd}
PYTHON=/opt/baseline-env/bin/python
ATTEMPT=artifacts/attempt-01
mkdir -p "$ATTEMPT"
if [[ "${{1:-}}" == "calibrate" ]]; then
  timeout --signal=TERM 240 "$PYTHON" 02_train.py --config config.json --cache /opt/minipile-cache --output "$ATTEMPT/calibration" --calibrate-updates 3
  echo MECHANICS_COMPLETE
elif [[ "${{1:-}}" == "full-pass" ]]; then
  DEADLINE=$("$PYTHON" -c 'import json; print(json.load(open("prelaunch/lease.json"))["deadline_epoch"])')
  REMAINING=$((DEADLINE - $(date -u +%s) - 1200))
  (( REMAINING > 60 ))
  timeout --signal=TERM "$REMAINING" "$PYTHON" 02_train.py --config config.json --cache /opt/minipile-cache --output "$ATTEMPT/full-pass"
  "$PYTHON" 03_verify.py "$ATTEMPT/full-pass" > "$ATTEMPT/full-pass/verification.json"
  echo CONTINUATION_VERIFIED
else
  exit 2
fi
''', encoding="utf-8", newline="\n")
            (run / "prelaunch/archive.sh").write_text(f'''#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/workspace/constructive
RUN_DIR={rd}
ATTEMPT="$RUN_DIR/artifacts/attempt-01"
cd "$TASK_ROOT"
mkdir -p "$ATTEMPT/source/baseline" "$ATTEMPT/source/prelaunch"
cp "$RUN_DIR"/*.py "$RUN_DIR"/*.json "$RUN_DIR"/*.sh "$ATTEMPT/source/"
cp src/baseline/*.py "$ATTEMPT/source/baseline/"
cp "$RUN_DIR/prelaunch/source-inventory.json" "$RUN_DIR/prelaunch/lease.json" "$ATTEMPT/source/prelaunch/"
cp environment-lock.txt "$ATTEMPT/"
python3 tools/artifact_inventory.py create "$ATTEMPT" --output "$ATTEMPT/inventory.json" > "$ATTEMPT/inventory-creation.tmp"
rm "$ATTEMPT/inventory-creation.tmp"
tar -C "$RUN_DIR/artifacts" -cf - attempt-01 | gzip -1 > /opt/run{rid}-attempt-01.tar.gz
sha256sum /opt/run{rid}-attempt-01.tar.gz > /opt/run{rid}-attempt-01.tar.gz.sha256
stat -c %s /opt/run{rid}-attempt-01.tar.gz > /opt/run{rid}-attempt-01.tar.gz.bytes
echo ARCHIVE_READY
''', encoding="utf-8", newline="\n")
            # Inventory creation writes to stdout, not into the inventoried tree.
            archive_path = run / "prelaunch/archive.sh"
            archive_path.write_text(archive_path.read_text().replace('> "$ATTEMPT/inventory-creation.tmp"\nrm "$ATTEMPT/inventory-creation.tmp"', '> /opt/inventory-creation.json'), encoding="utf-8", newline="\n")
            (run / "README.md").write_text(f'''# Run {rid} Pythia {size} {variant} LR continuation

Status: prepared. H004 first intervention, seed1235. Parent is Run{parent_name[:3]}
attempt01 at update356. Continue identical remaining batches through update712.
The user chose a single seed for the first test and fastest parallel execution;
multiple seeds are deferred to validation. Architecture/weights, moments and Adam
clock, scaler, RNG, raw/adaptive diagnostic states and data position are restored.
The future LR rule alone varies. See config.json for the precise intervention.

Raw accumulated unscaled gradients before clipping, all unique trainable
parameters; FP32 vectors and FP64 cosine/norm/cumulative reductions. Existing
adaptive diagnostics remain secondary and exclude LR/decay. Cumulative geometry
inherits the parent prefix; Adam clocks are never reset. Cosine357 is common
pre-intervention; the primary descriptive interval is358-712. No raw history is
discarded, and final partial update712 remains in terminal reductions.

Complete cached MiniPile coverage over parent plus continuation, no repetition;
full token-weighted validation at356/445/534/623/712, including short tails.
Retain new recovery states at534/712 and immutable parent locally. Global step,
order hash, optimizer/scaler/RNG and diagnostic state persist in recovery files.

Six parallel Secure RTX5090 workers, live price checked before creation, at most
two hours per worker and USD15 aggregate including storage and bounded retries.
60GB container/40GB persistent Pod storage, no network volume. Detached execution,
20-minute retrieval margin, local deadline stop guard and bounded read-only
monitoring. Hash-verify downloaded inventory before worker deletion. Unrelated
resources are preserved. No cloud storage is intentionally retained afterward.

Analysis004 owns the comparison. This is a descriptive single-seed intervention,
conditional on a common trained prefix. LR also scales AdamW weight decay;
alignment changes do not establish convergence or a causal performance mechanism.
''', encoding="utf-8")
            files = [*sorted((ROOT / "src/baseline").glob("*.py")), *sorted(run.glob("*.py")),
                     *sorted(run.glob("*.json")), *sorted(run.glob("*.sh")),
                     *sorted((run / "prelaunch").glob("*.py")), *sorted((run / "prelaunch").glob("*.sh")),
                     ROOT / "tools/artifact_inventory.py", ROOT / "tests/test_continuation.py"]
            save(run / "prelaunch/source-inventory.json", {"schema_version": 1, "files": [
                {"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)} for p in files]})
            files += [run / "prelaunch/source-inventory.json"]
            packet = ROOT / f"cache/runpod/run{rid}-source.tar.gz"
            with tarfile.open(packet, "w:gz") as tar:
                for p in files:
                    tar.add(p, arcname=p.relative_to(ROOT).as_posix())
                cache = ROOT.parent / "sparsity-spillover/data/tokenized/minipile-pythia-14m-full"
                for split in ("train", "validation"):
                    tar.add(cache / split / "metadata.json", arcname=f"input-cache/{split}/metadata.json")
                tar.add(cache / "validation/tokens.int32.bin", arcname="input-cache/validation/tokens.int32.bin")
            save(run / "prelaunch/source-packet.json", {"bytes": packet.stat().st_size, "sha256": sha(packet)})
            save(run / "prelaunch/scientific-freeze.json", {"config_sha256": sha(run / "config.json"),
                "source_inventory_sha256": sha(run / "prelaunch/source-inventory.json"),
                "parent_checkpoint_sha256": identities})
            cohort.append({"run": name, "size": size, "variant": variant, "seed": 1235,
                "parent_run": parent_name, "parent_local": checkpoint.relative_to(ROOT).as_posix(),
                "parent_packet": f"cache/runpod/parent-{size.lower()}-seed1235.tar.gz"})
    save(HERE / "cohort.json", cohort)

    def package_parent(size):
        item = next(r for r in cohort if r["size"] == size)
        source = ROOT / item["parent_local"]
        dest = ROOT / item["parent_packet"]
        with tarfile.open(dest, "w:gz", compresslevel=1) as tar:
            for p in source.rglob("*"):
                if p.is_file():
                    tar.add(p, arcname=p.relative_to(source).as_posix())
        return {"size": size, "path": item["parent_packet"], "bytes": dest.stat().st_size, "sha256": sha(dest)}

    with ThreadPoolExecutor(max_workers=2) as pool:
        packets = list(pool.map(package_parent, PARENTS))
    save(HERE / "parent_packets.json", packets)
    print(json.dumps({"runs": [r["run"] for r in cohort], "parents": packets}, indent=2))


if __name__ == "__main__":
    main()
