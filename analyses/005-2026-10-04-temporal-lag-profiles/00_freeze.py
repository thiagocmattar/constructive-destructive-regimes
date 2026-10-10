"""Finalize unexecuted packets; never run after calibration or training starts."""
from pathlib import Path
import json,tarfile,hashlib
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
def sha(p):
 with p.open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()
for row in json.loads((HERE/'cohort.json').read_text()):
 run=ROOT/'runs'/row['run']; rid=row['run'][:3]; rd='/workspace/constructive/runs/'+row['run']
 assert not (run/'prelaunch/calibrate-start.json').exists()
 (run/'prelaunch/archive.sh').write_text(f'''#!/usr/bin/env bash
set -euo pipefail
RUN_DIR={rd}
ROOT=/workspace/constructive
PYTHON=/opt/baseline-env/bin/python
ATTEMPT="$RUN_DIR/artifacts/attempt-01"
"$PYTHON" -c 'import json; from pathlib import Path; r=Path("{rd}/artifacts/attempt-01/full-pass"); assert json.loads((r/"verification.json").read_text())["verified"]'
mkdir -p "$ATTEMPT/source/baseline" "$ATTEMPT/source/prelaunch"
cp "$RUN_DIR"/*.py "$RUN_DIR"/*.json "$RUN_DIR"/*.sh "$ATTEMPT/source/"
cp "$ROOT/src/baseline"/*.py "$ATTEMPT/source/baseline/"
cp "$RUN_DIR/prelaunch/source-inventory.json" "$RUN_DIR/prelaunch/lease.json" "$ATTEMPT/source/prelaunch/"
cp "$ROOT/setup.log" "$ROOT/transfer.log" "$ROOT/environment-lock.txt" "$ROOT/input-transfer-receipt.json" "$ROOT/calibration.log" "$ROOT/full-pass.log" "$ATTEMPT/"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$ATTEMPT/hardware.txt"
"$PYTHON" "$ROOT/tools/artifact_inventory.py" create "$ATTEMPT" --output "$ATTEMPT/inventory.json" > /opt/inventory-summary.json
tar -cf - -C "$RUN_DIR/artifacts" attempt-01 | gzip -1 > /opt/run{rid}-attempt-01.tar.gz
sha256sum /opt/run{rid}-attempt-01.tar.gz > /opt/run{rid}-attempt-01.tar.gz.sha256
stat -c '%s' /opt/run{rid}-attempt-01.tar.gz > /opt/run{rid}-attempt-01.tar.gz.bytes
echo ARCHIVE_READY
''',encoding='utf-8',newline='\n')
 files=[*sorted((ROOT/'src/baseline').glob('*.py')),*sorted(run.glob('*.py')),
  *sorted(run.glob('*.json')),*sorted(run.glob('*.sh')),*sorted((run/'prelaunch').glob('*.py')),
  *sorted((run/'prelaunch').glob('*.sh')),ROOT/'tools/artifact_inventory.py',ROOT/'tests/test_temporal_sketch.py']
 inventory={'schema_version':1,'files':[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]}
 (run/'prelaunch/source-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
 files += [run/'prelaunch/source-inventory.json']
 cache=ROOT.parent/'sparsity-spillover/data/tokenized/minipile-pythia-14m-full'
 packet=ROOT/f'cache/runpod/run{rid}-source.tar.gz'
 with tarfile.open(packet,'w:gz') as tar:
  for p in files:tar.add(p,arcname=p.relative_to(ROOT).as_posix())
  if (run/'prelaunch/lease.json').exists():
   tar.add(run/'prelaunch/lease.json',arcname=(run/'prelaunch/lease.json').relative_to(ROOT).as_posix())
  for split in ['train','validation']:tar.add(cache/split/'metadata.json',arcname=f'input-cache/{split}/metadata.json')
  tar.add(cache/'validation/tokens.int32.bin',arcname='input-cache/validation/tokens.int32.bin')
 (run/'prelaunch/source-packet.json').write_text(json.dumps({'bytes':packet.stat().st_size,'sha256':sha(packet)},indent=2)+'\n')
