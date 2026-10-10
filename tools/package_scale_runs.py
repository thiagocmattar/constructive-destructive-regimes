"""Create minimal source packets, launch records, and inventories for scale runs."""
from pathlib import Path
import hashlib
import json
import tarfile
from datetime import datetime, timezone
import argparse

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'runs/002-2026-10-03-pythia14m-adamw-adaptive-geometry'

def save(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', default='cache/runpod/scale-cohort.json')
    parser.add_argument('--preserve-readme', action='store_true')
    args = parser.parse_args()
    records = json.loads((ROOT / args.cohort).read_text())
    if any((ROOT/'runs'/r['run']/'prelaunch/calibrated-launch.json').exists() for r in records):
        raise RuntimeError('Executed calibration packets are frozen; no files were repackaged.')
    for record in records:
        run = ROOT / 'runs' / record['run']
        rid = run.name[:3]
        rd = '/workspace/constructive/runs/' + run.name
        wrapper = f'''#!/usr/bin/env bash
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
RUN_DIR={rd}
PYTHON=/opt/baseline-env/bin/python
ATTEMPT="$RUN_DIR/artifacts/attempt-01"
cd "$RUN_DIR"
mkdir -p "$ATTEMPT"
if [[ "${{1:-}}" == "calibrate" ]]; then
  timeout --signal=TERM 900 "$PYTHON" 02_train.py --config config.json --cache /opt/minipile-cache --output "$ATTEMPT/calibration" --calibrate-updates 3
  echo CALIBRATION_COMPLETE
elif [[ "${{1:-}}" == "full-pass" ]]; then
  DEADLINE=$("$PYTHON" -c 'import json; print(json.load(open("prelaunch/lease.json"))["deadline_epoch"])')
  REMAINING=$((DEADLINE - $(date -u +%s) - 1800))
  if (( REMAINING < 60 )); then exit 1; fi
  timeout --signal=TERM "$REMAINING" "$PYTHON" 02_train.py --config config.json --cache /opt/minipile-cache --output "$ATTEMPT/full-pass"
  "$PYTHON" 03_verify.py "$ATTEMPT/full-pass" > "$ATTEMPT/full-pass/verification.json"
  echo FULL_PASS_VERIFIED
else
  exit 2
fi
'''
        (run / 'prelaunch/run.sh').write_text(wrapper, encoding='utf-8', newline='\n')
        arch = (OLD / 'prelaunch/archive_attempt02.sh').read_text()
        arch = arch.replace(OLD.name, run.name).replace('attempt-02','attempt-01').replace('run002','run'+rid)
        arch = arch.replace('run_attempt02.sh','run.sh').replace('source-inventory-attempt2.json','source-inventory.json')
        arch = arch.replace('ATTEMPT02_ARCHIVE_READY','ARCHIVE_READY')
        (run / 'prelaunch/archive.sh').write_text(arch, encoding='utf-8', newline='\n')
        config = json.loads((run / 'config.json').read_text())
        if not args.preserve_readme:
          (run / 'README.md').write_text(f'''# Run {rid}: Pythia-{record['parameters']:,} scale baseline

Status: preparing. User explicitly authorized both full passes and Runpod launches.
Hypothesis H002: adjacent alignment and cumulative geometry vary predictably with model scale.

This run varies the pinned Pythia architecture. It matches Run002's random
SmallInit/WangInit recipe, seed 1234, token cache and permutation, 712 AdamW
updates, batch 1024 (microbatch 4), optimizer, LR, precision, and measurement
boundaries. Dimensions change, so initial tensors cannot be identical across sizes.
No pretrained weights are loaded. Tokenizer/cache remain the 14M baseline's bytes.

For each raw gradient and adaptive direction, record adjacent cosine,
A_t=sum ||x_i||, B_t=||sum x_i||, B_t/A_t, and ||x_t||. The direction is
bias-corrected m/sqrt(v), with epsilon, before LR/weight decay; its moments use
the clipped gradient. Both streams use identical parameter ordering. Record
the adaptive/raw norm ratio. This is an optimizer-direction statistic, not
an actual parameter-displacement or causal optimization-benefit measurement.

All training and validation tokens, including short tails, are measured. Main
plots omit the final partial batch (update 712), as previously requested; full
data and terminal statistics retain it. Initial/midpoint/final recovery state
includes both streaming accumulators, optimizer, scaler, RNG, and data position.

Model: `{config['model']['architecture']}` at `{config['model']['revision']}`.
Actual parameters: {record['parameters']:,}. Scientific inputs are frozen before
full-pass execution and retained in the attempt source inventory.

One seed per size is a descriptive comparison. Trends can contradict a proposed
monotone expectation, but three sizes cannot establish a predictive scaling law.
Absolute norms and A/B depend on parameter dimension; compare dimensionless
cosines/ratios alongside requested absolute values. Cross-run analysis belongs
to Analysis001. No new manuscript claim is implied by this run alone.

Execution: one RTX4090, live $0.74/hour, at most six hours including setup,
calibration, full pass and retrieval. Cohort maximum $10 including storage;
60GB container and 40GB persistent volume per worker, removed after size/SHA
acceptance. Thirty-minute retrieval margin; detached process timeout plus local
deadline guard. Guard needs the controller awake/networked. Monitor bounded
read-only snapshots every 45–60 seconds; stale events, nonfinite metrics,
OOM, overflow retries, or deadline fit are warning conditions.
''', encoding='utf-8')
        files = list((ROOT / 'src/baseline').glob('*.py'))
        files += list(run.glob('*.py')) + list(run.glob('*.sh'))
        files += [run/'config.json',run/'model_config.json']
        files += [run/'prelaunch'/n for n in ['run.sh','archive.sh','accept_local_cache.py','remote_status.py','audit_gpu_adam.py']]
        files += [ROOT/'tools/artifact_inventory.py',ROOT/'tools/resolve_scale_initialization.py',ROOT/'tests/test_run002_adaptive.py',OLD/'adam_diagnostics.py']
        identity = {'schema_version':1,'files':[{'path':p.relative_to(ROOT).as_posix(),
            'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)]}
        save(run / 'prelaunch/source-inventory.json', identity)
        files += [run/'prelaunch/source-inventory.json']
        if (run/'prelaunch/lease.json').exists(): files += [run/'prelaunch/lease.json']
        packet = ROOT / f'cache/runpod/run{rid}-source.tar.gz'
        with tarfile.open(packet, 'w:gz') as tar:
            for p in files: tar.add(p, arcname=p.relative_to(ROOT).as_posix())
            cache = ROOT.parent / 'sparsity-spillover/data/tokenized/minipile-pythia-14m-full'
            for split in ['train','validation']:
                tar.add(cache/split/'metadata.json',arcname='input-cache/'+split+'/metadata.json')
            tar.add(cache/'validation/tokens.int32.bin',arcname='input-cache/validation/tokens.int32.bin')
        save(run / 'prelaunch/source-packet.json', {'bytes':packet.stat().st_size,
            'sha256':hashlib.sha256(packet.read_bytes()).hexdigest()})
        print(rid, packet.stat().st_size)

if __name__ == '__main__': main()
