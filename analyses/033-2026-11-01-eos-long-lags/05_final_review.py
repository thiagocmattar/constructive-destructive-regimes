"""Final cross-artifact, frozen-input and cleanup checks; visual review is separate."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

import fitz

HERE=Path(__file__).parent
ROOT=HERE.parents[1]
RUN=ROOT/'runs/206-2026-11-01-eos-long-lags'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    profiles=json.loads((HERE/'results.json').read_text())['profiles']
    source={(p['parent_run_id'],p['anchor'],r['lag']):(p,r) for p in profiles for r in p['rows'][1:]}
    with (HERE/'lag_profiles.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    assert len(rows)==len(source) and len(set((r['parent_run_id'],int(r['anchor']),int(r['lag'])) for r in rows))==len(rows)
    for r in rows:
        p,x=source[(r['parent_run_id'],int(r['anchor']),int(r['lag']))]
        for field in ['cosine','norm','loss']:
            assert (float(r[field]) if r[field] else None)==x[field]
        assert r['primary_eligible']==str(p['primary_eligible']) and r['gpu']==p['gpu']
    report=(HERE/'REPORT.md').read_text(encoding='utf-8')
    table=[line for line in report.splitlines() if line.startswith(('| mlp |','| resnet |','| transformer |'))]
    assert len(table)==27
    lookup={(p['parent_run_id'],p['anchor']):p for p in profiles}
    for line in table:
        cells=[v.strip() for v in line.strip('|').split('|')]
        p=lookup[(cells[1],int(cells[3]))]
        assert int(cells[6])==p['last_lag']
        for cell,key in zip(cells[7:11],['cosine_256','final_cosine','common_256_envelope','final_envelope']):
            assert cell==f'{p[key]:.6f}'
        assert cells[11]==p['stop_reason']
    pdfs=[]
    verification=json.loads((HERE/'verification.json').read_text())
    for figure in verification['figures']:
        path=HERE/figure['path'];assert sha(path)==figure['sha256']
        with fitz.open(path) as doc:
            assert len(doc)==1 and len(doc[0].get_images())==0
            pdfs.append(dict(path=figure['path'],pages=1,width_points=doc[0].rect.width,
                             height_points=doc[0].rect.height,vector_output=True))
    expected=[25*256,sum(p['last_lag'] for p in profiles if p['primary_eligible']),sum(p['last_lag'] for p in profiles),
              sum(p['last_lag']//2-31 for p in profiles if p['primary_eligible'])]
    assert [f['plotted_values'] for f in verification['figures']]==expected
    compiled=[]
    parallel=ROOT/'runs/207-2026-11-01-eos-parallel-resnet-lags'
    handoff=ROOT/'runs/208-2026-11-01-eos-checkpoint-handoffs'
    for path in list(HERE.glob('*.py'))+list(RUN.glob('*.py'))+list(parallel.glob('*.py'))+list(handoff.glob('*.py'))+[ROOT/'src/eos_tasks/lag_profile.py']:
        ast.parse(path.read_text(encoding='utf-8'));compiled.append(path.relative_to(ROOT).as_posix())
    frozen=json.loads((RUN/'input_manifest.json').read_text())
    for item in frozen['files']:
        path=ROOT/item['path'];assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
    parallel_inputs=json.loads((parallel/'input_manifest.json').read_text())
    for item in parallel_inputs['files']:
        path=ROOT/item['path'];assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
    for item in json.loads((handoff/'input_manifest.json').read_text())['files']:
        path=ROOT/item['path'];assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
    captures=json.loads((HERE/'handoff_captures.json').read_text())
    assert sha(HERE/'08_capture_handoffs.py')==captures['program_sha256']
    for capture in captures['captures']:
        assert capture['verified']
        archive=ROOT/f'cache/analysis033-handoff-{capture["parent_run_id"]}-{capture["anchor"]}.tar.gz'
        assert sha(archive)==capture['archive_sha256']
    closeout=json.loads((HERE/'closeout.json').read_text());assert closeout['status']=='complete'
    assert all(p.get('deleted_epoch') for p in closeout['pods']) and closeout['cloud_storage_remaining_gb']==0
    failures=[]
    for pod in closeout['pods']:
        p=subprocess.run(['runpodctl','pod','get',pod['id']],capture_output=True,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=30)
        assert p.returncode and json.loads(p.stderr.splitlines()[0])['code']=='not_found',pod['id']
    inventory=subprocess.run(['runpodctl','pod','list'],capture_output=True,text=True,check=True,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=30)
    assert not [p for p in json.loads(inventory.stdout) if p.get('name','').startswith('cdr-analysis033-')]
    events=[(p['created_epoch'],1) for p in closeout['pods']]+[(p['deleted_epoch'],-1) for p in closeout['pods']]
    n=peak=0
    for _,change in sorted(events):n+=change;peak=max(peak,n)
    assert peak<=12 and closeout['compute_estimate_usd']<30
    visual=json.loads((HERE/'figure_review.json').read_text());assert visual['status']=='passed'
    for item in visual['figures']:
        assert sha(HERE/item['path'])==item['sha256']
    assert len(visual['figures'])==4
    documents=[HERE/'README.md',HERE/'REPORT.md',HERE/'observations/001-long-lag-profiles.md',
               RUN/'README.md',parallel/'README.md',handoff/'README.md']
    links=0
    for document in documents:
        for target in re.findall(r'\]\(([^)]+)\)',document.read_text(encoding='utf-8')):
            if '://' in target:continue
            destination=(document.parent/target.split('#')[0]).resolve()
            assert destination.exists() or destination==(HERE/'final_review.json').resolve(),(document,target)
            links+=1
    record=dict(status='passed',profiles=27,primary_profiles=25,csv_json_values_verified=len(rows),
        report_rows_verified=len(table),pdfs=pdfs,source_files_parsed=compiled,
        frozen_input_files_verified=len(frozen['files']),task_pods_confirmed_absent=len(closeout['pods']),
        task_named_pods_absent_from_live_inventory=True,
        maximum_concurrent_workers=peak,no_task_cloud_storage=True,compute_estimate_within_approved_cap=True,
        local_document_links_verified=links,
        figure_visual_review_passed=True)
    (HERE/'final_review.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(record))


if __name__=='__main__':main()
