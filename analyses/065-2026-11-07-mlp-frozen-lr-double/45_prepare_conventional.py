"""Freeze H050's three requested runs and portable, hashed input packets."""
import hashlib
import json
from pathlib import Path
import tarfile

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
RUNS=[ROOT/'runs/289-2026-11-08-resnet-beta2-reductions',
      ROOT/'runs/290-2026-11-08-transformer-beta2-reductions',
      ROOT/'runs/291-2026-11-08-transformer-gradient-clipping']


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def dump(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8',newline='\n')


def main():
    for run in RUNS:
        run.mkdir(exist_ok=True)
        case='resnet' if run==RUNS[0] else 'transformer'
        beta=run!=RUNS[2]
        cfg=dict(run_id=int(run.name[:3]),analysis_id='065',hypothesis='H050',case=case,seed=0,
            start=256 if case=='resnet' else 544,end=576 if case=='resnet' else 928,
            plot_start=192 if case=='resnet' else 480,base_lr=.003,physical=1000 if case=='resnet' else 64,
            arms=[dict(id='control')]+([dict(id='beta2_'+str(b).replace('.','p'),beta2=b) for b in [.95,.9,.5]]
                if beta else [dict(id='clip_'+str(v).replace('.','p'),clip=v) for v in [.5,1.]]),
            checkpoint_every=32,maximum_worker_seconds=6500,
            comparison='Same frozen Run223 fork; fresh unchanged Adam control. Cross-worker reuse only with identical GPU model/runtime, fork fingerprint and eight-update model/optimizer/history/RNG fingerprint. Otherwise run a same-worker control; never compare unmatched hardware trajectories.',
            beta_semantics='Retain native moments and global step. New beta2 enters v recurrence and native 1-beta2**step correction. No moment reset/rescaling or product-history correction.',
            clipping_semantics='Scale full-objective raw gradient by min(1, threshold / FP64 global L2 norm) before native Adam moment update; no epsilon in clipping factor; preserve raw histories.',
            precision='FP32 model/gradients/moments; FP64 diagnostics; deterministic algorithms; TF32 off; no fused/foreach Adam; no weight decay.',
            objective='Full 50000-image CIFAR10 cross entropy with immutable 1000-image BN groups' if case=='resnet' else 'Full WikiText2 token-weighted negative log likelihood, 2088600 valid tokens, physical batch64, dropout0; full held-out coverage.',
            retention='Complete fork and final/recovery optimizer/model/RNG/raw-gradient history; full first/last update audits; every boundary and update scalar; full validation/test at end. Raw artifacts on owned Spark; compact results local.',
            authorization='User explicitly requested these model extensions and authorized Runpod for fastest execution. Task ceiling USD20 including retries/storage; at most7 GPUs and2h per lease; no unrelated ablations.',
            control_reference_run=290 if not beta else int(run.name[:3]))
        target=run/'config.json'
        if target.exists():assert json.loads(target.read_text(encoding='utf-8'))==cfg
        else:dump(target,cfg)
        source_run=next((ROOT/'runs').glob(('270' if case=='resnet' else '271')+'-*'))
        old=json.loads((source_run/'input_manifest.json').read_text(encoding='utf-8'))['files']
        entries={}
        for f in old:
            if f['path'].startswith(('cache/','src/','runs/223-')):
                entries[f['path']]={k:v for k,v in f.items() if k in ['path','bytes','sha256','local_path']}
            if f['path'].endswith('/inputs/prefix.pt'):
                entry=dict(f,path=run.relative_to(ROOT).as_posix()+'/inputs/prefix.pt');entries[entry['path']]=entry
        for p in list((ROOT/'src/eos_tasks').glob('*.py'))+[HERE/'46_test_conventional.py',HERE/'47_verify_conventional.py',run/'config.json']:
            rel=p.relative_to(ROOT).as_posix();entries[rel]=dict(path=rel,bytes=p.stat().st_size,sha256=sha(p))
        for f in entries.values():
            p=ROOT/f.get('local_path',f['path']); assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],str(p)
        manifest=dict(case=case,run=run.name,files=list(entries.values()))
        target=run/'input_manifest.json'
        if target.exists():assert json.loads(target.read_text(encoding='utf-8'))==manifest,'Executed inputs are immutable'
        else:dump(target,manifest)
        packet=run/'scratch/inputs.tar.gz';packet.parent.mkdir(exist_ok=True)
        with tarfile.open(packet,'w:gz',compresslevel=1) as tar:
            for f in entries.values():tar.add(ROOT/f.get('local_path',f['path']),arcname=f['path'])
            tar.add(target,arcname=target.relative_to(ROOT).as_posix())
        dump(run/'packet.json',dict(path=packet.relative_to(ROOT).as_posix(),bytes=packet.stat().st_size,sha256=sha(packet)))
        print(json.dumps(dict(run=run.name,files=len(entries),packet_bytes=packet.stat().st_size)),flush=True)


if __name__=='__main__':main()
