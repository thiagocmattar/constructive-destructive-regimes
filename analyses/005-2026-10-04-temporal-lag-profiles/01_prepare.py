"""Prepare three frozen single-seed temporal measurement replays."""
from pathlib import Path
import hashlib
import json
import shutil
import tarfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PARENTS=['005-2026-10-04-pythia14m-seed1235','006-2026-10-04-pythia31m-seed1235','007-2026-10-04-pythia70m-seed1235']
def sha(p):
    with Path(p).open('rb') as s: return hashlib.file_digest(s,'sha256').hexdigest()
def save(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8',newline='\n')

def main():
    assert not (HERE/'cohort.json').exists(), 'Do not rewrite frozen inputs'
    rows=[]
    for i,parent_name in enumerate(PARENTS):
        rid=f'{17+i:03}'
        size=['14M','31M','70M'][i]
        name=f'{rid}-2026-10-04-pythia{size.lower()}-temporal-sketch-seed1235'
        parent=ROOT/'runs'/parent_name
        run=ROOT/'runs'/name
        run.mkdir(); (run/'prelaunch').mkdir()
        cfg=json.loads((parent/'config.json').read_text())
        cfg.update(run_id=rid,name=name[15:],hypothesis='H005',matched_reference_run=parent_name[:3])
        cfg['temporal_sketch']={'family':'fixed_signed_CountSketch','dimension':16384,
            'seeds':[71001,71002,71003,71004],'anchor_seed':71999,'anchor_block_updates':16,
            'exact_lags':[2,4,8,16,32,64], 'history_dtype':'float32',
            'calibration_rmse_limit':.02,'calibration_max_abs_limit':.06,
            'raw_norm_denominator':'exact_FP64_recorded_gradient_norm',
            'windows':{'early':[1,128],'middle':[293,420],'late':[585,712]},
            'primary_lags':list(range(1,65)),
            'secondary_max_lag':127,'window_pairs':'both endpoints inside same window',
            'undefined':'exclude zero-norm pairs and report valid count',
            'replay':'same initial SHA/data order/optimizer/schedule as recorded reference; not bitwise trajectory identity'}
        cfg['execution']={'provider':'runpod','gpu':'NVIDIA GeForce RTX 5090','workers':1,
            'cohort_workers':3,'maximum_hours':4,'cohort_maximum_total_usd':15,
            'container_gb':60,'persistent_gb':40,'retrieval_margin_seconds':1200,
            'monitoring_interval_seconds':60}
        save(run/'config.json',cfg)
        shutil.copy2(parent/'model_config.json',run/'model_config.json')
        (run/'02_train.py').write_text('''from pathlib import Path
import sys,json
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from baseline import training
from baseline.gradient_geometry import GradientGeometry
from baseline.temporal_sketch import TemporalRecorder,calibration_report
config=json.loads(Path(sys.argv[sys.argv.index('--config')+1]).read_text())
output=Path(sys.argv[sys.argv.index('--output')+1])
recorders=[]
class RecordingGeometry(GradientGeometry):
    def __init__(self,named_parameters,previous_dtype=torch.float32):
        super().__init__(named_parameters,previous_dtype=previous_dtype)
        self.recorder=None
        if previous_dtype==torch.float32:
            self.recorder=TemporalRecorder(self.parameters,config,output/'temporal')
            recorders.append(self.recorder)
    def consume(self,vectors=None):
        result=super().consume(vectors)
        if self.recorder is not None:
            assert vectors is None
            result.update(self.recorder.consume(result))
        return result
    def state_dict(self):
        result=super().state_dict()
        if self.recorder is not None: result['temporal']=self.recorder.state_dict()
        return result
training.GradientGeometry=RecordingGeometry
if __name__=='__main__':
    training.main()
    assert len(recorders)==1
    report=calibration_report(output/'temporal',recorders[0].count)
    (output/'temporal/calibration.json').write_text(json.dumps(report,indent=2)+'\\n')
    print(json.dumps({'temporal_calibration':report}),flush=True)
''',encoding='utf-8',newline='\n')
        shutil.copy2(parent/'03_verify.py',run/'verify_baseline.py')
        (run/'03_verify.py').write_text('''import sys,json
from pathlib import Path
import numpy as np
from verify_baseline import verify
root=Path(sys.argv[1]); base=verify(root)
config=json.loads((root/'resolved_config.json').read_text())
history=np.load(root/'temporal/raw_sketches.npy',mmap_mode='r')
assert history.shape==(712,4,16384) and history.dtype==np.float32
assert np.isfinite(history).all()
assert json.loads((root/'temporal/sketch_progress.json').read_text())['completed_updates']==712
cal=json.loads((root/'temporal/calibration.json').read_text())
assert cal['qualified'] and cal['completed_updates']==712
assert cal['exact_pairs']>900
print(json.dumps({**base,'temporal_verified':True,'temporal_calibration':cal},indent=2))
''',encoding='utf-8',newline='\n')
        setup=(parent/'00_setup_remote.sh').read_text().replace(parent_name,name)
        (run/'00_setup_remote.sh').write_text(setup,encoding='utf-8',newline='\n')
        for f in ['accept_local_cache.py','remote_status.py','deadline-guard.ps1']:
            text=(parent/'prelaunch'/f).read_text().replace(parent_name,name)
            (run/'prelaunch'/f).write_text(text,encoding='utf-8',newline='\n')
        rd='/workspace/constructive/runs/'+name
        (run/'prelaunch/run.sh').write_text(f'''#!/usr/bin/env bash
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
cd {rd}
PYTHON=/opt/baseline-env/bin/python
ATTEMPT=artifacts/attempt-01
mkdir -p "$ATTEMPT"
if [[ "${{1:-}}" == "calibrate" ]]; then
 timeout --signal=TERM 360 "$PYTHON" 02_train.py --config config.json --cache /opt/minipile-cache --output "$ATTEMPT/calibration" --calibrate-updates 4
 echo MECHANICS_COMPLETE
elif [[ "${{1:-}}" == "full-pass" ]]; then
 DEADLINE=$("$PYTHON" -c 'import json; print(json.load(open("prelaunch/lease.json"))["deadline_epoch"])')
 REMAINING=$((DEADLINE - $(date -u +%s) - 1200))
 (( REMAINING > 60 ))
 timeout --signal=TERM "$REMAINING" "$PYTHON" 02_train.py --config config.json --cache /opt/minipile-cache --output "$ATTEMPT/full-pass"
 "$PYTHON" 03_verify.py "$ATTEMPT/full-pass" > "$ATTEMPT/full-pass/verification.json"
 echo TEMPORAL_VERIFIED
else
 exit 2
fi
''',encoding='utf-8',newline='\n')
        archive=(parent/'prelaunch/archive.sh').read_text().replace(parent_name,name).replace('run005',f'run{rid}')
        (run/'prelaunch/archive.sh').write_text(archive,encoding='utf-8',newline='\n')
        (run/'README.md').write_text(f'''# Run {rid}: Pythia {size} temporal gradient sketches

Status: prepared. H005; first test, seed1235. Replay all712 original-recipe updates
with Run{parent_name[:3]}'s exact initial parameter hash, cache bytes and data order.
No claim of bitwise trajectory equality. New diagnostics leave raw gradients untouched.
Four independent fixed signed CountSketch maps,16384 buckets each, record the
whole-model unscaled accumulated raw gradient every optimizer boundary before
clipping/AdamW. Exact FP64 norms normalize projected inner products. FP32 sketches,
FP64 offline reduction. These are calibrated estimates, not worst-case JL claims.
Exact lag1 every update and randomly anchored longer-lag pairs quantify error.
Diagnostic RNG is isolated; fixed seeds/maps/parameter ordering are recorded.

Full MiniPile pass with both short tails; complete token-weighted initial/final
validation. Initial/midpoint/final recovery states include model, optimizer,
scaler, training RNG, order position, cumulative diagnostics, sketch row position,
and pending exact-gradient anchors. Preserve sketch prefix to resume diagnostics.
Three128-update windows fixed before measurement:1-128,293-420,585-712;
all integer lags1-64 primary,1-127 secondary, only pairs inside each window.
Do not assume these windows are constructive/destructive before inspecting evidence.
Raw-gradient geometry is distinct from AdamW updates and learning quality.

Three parallel RTX5090 workers,USD15 cohort ceiling,four hours each,60GB container
and40GB pod storage,20-minute retrieval margin,detached jobs and local deadline
stop guards. Verify archive sizes and SHA256 before deletion; retain no cloud storage.
Analysis005 owns figures/calibration/interpretation. No multi-seed validation yet.
''',encoding='utf-8',newline='\n')
        files=[*sorted((ROOT/'src/baseline').glob('*.py')),*sorted(run.glob('*.py')),
            *sorted(run.glob('*.json')),*sorted(run.glob('*.sh')),
            *sorted((run/'prelaunch').glob('*.py')),*sorted((run/'prelaunch').glob('*.sh')),
            ROOT/'tools/artifact_inventory.py',ROOT/'tests/test_temporal_sketch.py']
        save(run/'prelaunch/source-inventory.json',{'schema_version':1,'files':[
            {'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]})
        files += [run/'prelaunch/source-inventory.json']
        packet=ROOT/f'cache/runpod/run{rid}-source.tar.gz'
        cache=ROOT.parent/'sparsity-spillover/data/tokenized/minipile-pythia-14m-full'
        with tarfile.open(packet,'w:gz') as tar:
            for p in files: tar.add(p,arcname=p.relative_to(ROOT).as_posix())
            for split in ['train','validation']:
                tar.add(cache/split/'metadata.json',arcname=f'input-cache/{split}/metadata.json')
            tar.add(cache/'validation/tokens.int32.bin',arcname='input-cache/validation/tokens.int32.bin')
        save(run/'prelaunch/source-packet.json',{'bytes':packet.stat().st_size,'sha256':sha(packet)})
        rows.append({'run':name,'size':size,'reference':parent_name})
    save(HERE/'cohort.json',rows)
    save(HERE/'design.json',{'hypothesis':'H005','user_authorized_launch':True,
        'single_seed':1235,'models':['14M','31M','70M'],'varying':'model size and temporal window/lag',
        'matched':'recipe/cache/order/seed/updates/projection family and dimension/window boundaries',
        'questions':['Early monotone decay?','Middle damped/alternating oscillation?','Late loss of correlation?'],
        'exploratory_expectations':True,'claim_scope':'phase-dependent raw temporal gradient alignment',
        'scope_excludes':['stability','causal learning benefit','actual AdamW displacement'],
        'window_length':128,'primary_max_lag':64,'secondary_max_lag':127,
        'projection_uncertainty':'four map profiles and exact pair calibration, not training-seed uncertainty'})
if __name__=='__main__': main()
