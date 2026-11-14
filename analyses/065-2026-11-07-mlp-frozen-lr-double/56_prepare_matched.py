"""Freeze the authorized H051 model-specific matrix and portable inputs."""
import hashlib
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = {case: ROOT/'runs'/f'{number}-2026-11-14-{case}-matched-interventions'
        for number, case in [(293, 'mlp'), (294, 'resnet'), (295, 'transformer')]}


def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2) + '\n').encode('utf-8'))


def main():
    base = json.loads((ROOT/'runs/223-2026-11-04-multispike-interventions/config.json').read_text())
    old = json.loads((ROOT/'runs/223-2026-11-04-multispike-interventions/input_manifest.json').read_text())
    arms = [dict(id='control', kind='control', family='control')]
    arms += [dict(id='lr_'+str(q).replace('.', 'p'), kind='lr', family='lr', lr_multiplier=q) for q in [.75, .5, .1]]
    arms += [dict(id='beta2_'+str(q).replace('.', 'p'), kind='beta2', family='beta2', beta2=q) for q in [.95, .9, .5]]
    arms += [dict(id='clip_'+str(q).replace('.', 'p'), kind='clip', family='clipping', clip=q) for q in [.5, 1.]]
    arms += [dict(id='cap_q'+str(int(q*100)), kind='functional', family='cap', scope='trunk', quantile=q) for q in [.25, .5, .75]]
    assert len(arms) == 12
    for case, run in RUNS.items():
        sp = base['cases'][case]
        cfg = dict(run_id=int(run.name[:3]), hypothesis='H051', analysis_id='065', case=case, seed=0,
            fork=sp['start'], end=sp['end'], base_lr=sp['lr'], physical=sp['physical'],
            probe={'mlp':5000, 'resnet':1000, 'transformer':64}[case],
            quantiles=[.25,.5,.75], arms=arms, checkpoint_every=64, maximum_worker_seconds=25200,
            torch_version='2.11.0', cuda_version='13.0' if case=='mlp' else '12.8',
            calibration='Native outgoing states fork-8 through fork-1; np.quantile linear over all unit scores and eight transitions. Frozen before treatment.',
            endpoint='Stop at end, without applying end->end+1; primary mean uses fork+1 through end.',
            controls='Native full prefix and same complete fork/runtime/probe for all arms; exact predecessor replay and eight-update cross-worker fingerprint required.',
            heldout_states=[sp['start'],sp['start']+32,(sp['start']+sp['end'])//2,sp['end']-32,sp['end']],
            beta_semantics='Retain moments/clock; new beta2 enters both native recurrence and 1-beta2**step correction.',
            clip_semantics='Absolute global L2 raw-gradient threshold, FP64 norm, before native Adam; preserve raw histories.',
            cap_semantics='Verified local affine cap on native stored increments; untargeted parameters and Adam moments remain native.',
            precision='FP32 training/storage; FP64 reductions/local scores; deterministic; TF32 off; unfused/non-foreach Adam.',
            retention='Full fork/predecessor, calibration pool, all scalar/probe/update histories, five full held-out evaluations, three full audits/arm, rolling recovery and terminal state. Diagnostic recovery replays from the immutable fork.',
            authorization='User explicitly approved faster option and full delivery on 14 November: Spark MLP +2 RTX4090 ResNet +6 RTX5090 Transformer; total task USD50 including retries/storage, maximum8h execution, no extra initializations or cap-by-LR grid.')
        if (run/'config.json').exists():
            assert json.loads((run/'config.json').read_text()) == cfg
        else: dump(run/'config.json',cfg)
        entries = {}
        for item in old['files']:
            if case not in item['tasks']: continue
            rel = item['path']
            if rel.startswith(('cache/', 'src/', 'runs/223-')):
                path = ROOT/rel
                entries[rel] = dict(path=rel, bytes=path.stat().st_size, sha256=sha(path))
        paths = list((ROOT/'src/eos_tasks').glob('*.py')) + [ROOT/'src/cifar_eos/core.py',
            ROOT/'runs/223-2026-11-04-multispike-interventions/input_manifest.json',run/'config.json',HERE/'57_test_matched.py']
        for path in paths:
            rel = path.relative_to(ROOT).as_posix()
            entries[rel] = dict(path=rel, bytes=path.stat().st_size, sha256=sha(path))
        manifest = dict(case=case, files=list(entries.values()),
            interpretation='All original pinned data/model inputs; shared native runtime from initialization; no trained historical fork substituted.')
        if (run/'input_manifest.json').exists():
            assert json.loads((run/'input_manifest.json').read_text()) == manifest, 'Executed inputs must remain immutable'
        else: dump(run/'input_manifest.json',manifest)
        packet = run/'scratch/inputs.tar.gz'; packet.parent.mkdir(exist_ok=True)
        with tarfile.open(packet,'w:gz',compresslevel=1) as tar:
            for item in manifest['files']: tar.add(ROOT/item['path'],arcname=item['path'])
            tar.add(run/'input_manifest.json',arcname=(run/'input_manifest.json').relative_to(ROOT).as_posix())
        dump(run/'packet.json',dict(path=packet.relative_to(ROOT).as_posix(),bytes=packet.stat().st_size,sha256=sha(packet)))
        (run/'README.md').write_bytes((f'# {case}: matched four-family interventions\n\n'
            f'- H051 / Analysis065; initialization0, states {sp["start"]} through {sp["end"]}; 12 fixed conditions.\n'
            '- User authorized execution and paper deliverables on 14 November 2026.\n'
            '- Scientific configuration and input manifest are frozen at launch. Status/observations are recorded separately.\n'
            '- Worker: `src/eos_tasks/matched_interventions.py`; shared native prefix, eight past calibration updates, unchanged Adam control.\n'
            '- Raw evidence retained on owned Spark after size/SHA-256 verification; compact data and publication PDFs local.\n').encode())
        print(json.dumps(dict(case=case,run=run.name,files=len(entries),packet_bytes=packet.stat().st_size)),flush=True)


if __name__=='__main__': main()
