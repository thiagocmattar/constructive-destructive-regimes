"""Verify retained scientific identities, complete snapshots, and PDF contents."""
import hashlib
import json
from pathlib import Path
import shutil
import fitz
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
ANALYSIS=ROOT/'analyses/063-2026-11-07-architecture-instability-tracing'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text())

def main():
    lease=read(HERE/'cloud_lease.json');receipt=read(HERE/'cloud_retrieval.json')
    assert lease.get('deleted_epoch') and lease.get('verified_retrieval') and receipt['verified']
    cfg=read(HERE/'config.json');source=read(HERE/'input_manifest.json')
    for q in source['files']:
        p=ROOT/q['path']
        if not p.exists() and '/inputs/' in q['path']:
            case=Path(q['path']).parent.name
            base=ROOT/cfg['parent']/'artifacts'/cfg['cases'][case]['parent_worker']/'payload/worker'
            p=base/('prefix/end.pt' if q['path'].endswith('prefix.pt') else 'control/boundaries.jsonl')
        assert p.stat().st_size==q['bytes'] and sha(p)==q['sha256'],str(p)
    raw=HERE/'artifacts/cloud';inventory=read(raw/'inventory.json')
    for q in inventory['files']:
        if q['download']:
            p=raw/q['path'];assert p.stat().st_size==q['bytes'] and sha(p)==q['sha256'],str(p)
    shutil.copyfile(raw/'inventory.json',HERE/'artifact_inventory.json')
    cases={};pdfs={}
    for i,(case,sp) in enumerate(cfg['cases'].items(),1):
        folder=raw/case;manifest=read(folder/'manifest.json')
        states=sorted(int(p.stem) for p in (folder/'checkpoints').glob('*.pt'))
        assert states==list(range(sp['start']-1,sp['end']+2))
        assert manifest['status']=='completed' and manifest['completed_state']==sp['end']+1
        final=torch.load(folder/'end.pt',map_location='cpu',weights_only=False)
        snapshot=torch.load(folder/'checkpoints'/f"{sp['end']+1}.pt",map_location='cpu',weights_only=False)
        model=final['model']
        assert model.keys()==snapshot.keys()
        assert all(torch.equal(model[n],snapshot[n]) for n in model)
        result=read(ANALYSIS/(case+'_results.json'))
        for path,digest in result['sources'].items():assert sha(ROOT/path)==digest
        pdf=ANALYSIS/'figures'/f'{i:02d}-{case}-instability-trace.pdf'
        doc=fitz.open(pdf);assert len(doc)==2
        out=[]
        for page in doc:
            outside=[w for w in page.get_text('words') if w[0]<-.1 or w[1]<-.1 or w[2]>page.rect.width+.1 or w[3]>page.rect.height+.1]
            assert not outside,outside
            out.append(dict(width_pt=page.rect.width,height_pt=page.rect.height,words=len(page.get_text('words')),text_within_page=True))
        assert 'Consecutive' in doc[0].get_text() and 'SVD' in doc[1].get_text()
        pdfs[case]=dict(sha256=sha(pdf),pages=out)
        cases[case]=dict(checkpoints=len(states),states=[states[0],states[-1]],end_snapshot_exact=True,
            elapsed_seconds=manifest['elapsed_seconds'],peak_cuda_bytes=manifest['peak_cuda_bytes'],
            historical_exact=result['historical']['exact'],reduction_checks=result['verification'])
    prior=read(HERE/'infra_attempts/01/cloud_lease.json')
    storage=dict(local_artifacts_bytes=sum(p.stat().st_size for p in (HERE/'artifacts').rglob('*') if p.is_file()),
        local_scratch_bytes=sum(p.stat().st_size for p in (HERE/'scratch').rglob('*') if p.is_file()),
        spark=read(HERE/'spark_closeout.json'),cloud_persistent_storage_retained=False)
    report=dict(status='passed',source_files=len(source['files']),downloaded_files=sum(q['download'] for q in inventory['files']),
        inventory_sha256=sha(HERE/'artifact_inventory.json'),receipt=receipt,cases=cases,pdfs=pdfs,storage=storage,
        provisional_compute_usd=lease['estimated_compute_usd']+prior['estimated_compute_usd'],
        billing_note='Wall-clock estimate from live pod rates; storage and final billing not reconciled. Both owned pods deleted.',
        verification_script_sha256=sha(Path(__file__)))
    (HERE/'verification.json').write_text(json.dumps(report,indent=2)+'\n',newline='\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
