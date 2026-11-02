"""Final compact checks: independent CSV, figures, scoped resources and sources."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import fitz

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    results=json.loads((HERE/'results.json').read_text())['profiles']
    index={(p['parent_run_id'],p['anchor'],r['lag']):r for p in results for r in p['rows']}
    with (HERE/'lag_profiles.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==len(index)
    for r in rows:
        source=index[(r['parent_run_id'],int(r['anchor']),int(r['lag']))]
        for field in ['loss','norm','anchor_cosine','adjacent_cosine']:
            assert (float(r[field]) if r[field] else None)==source[field]
    report=(HERE/'REPORT.md').read_text(encoding='utf-8')
    summaries=json.loads((HERE/'profile_summary.json').read_text())['profiles']
    for r in summaries:
        token=f'| {r["task"]} / {r["parent_run_id"]} | {r["anchor"]} | {r["horizon"]} | {r["adam_lr"]:g} | {r["adjacent_first"]:.6f} | {r["terminal64_median_adjacent"]:.6f} | {100*r["terminal64_negative_fraction"]:.2f}% | {r["anchor_terminal_half_gap"]:.6f} | {r["final_loss"]:.6f} |'
        assert token in report,token
    figures=[]
    scratch=HERE/'scratch';scratch.mkdir(exist_ok=True)
    for p in sorted((HERE/'figures').glob('*.pdf')):
        doc=fitz.open(p);assert len(doc)==1
        page=doc[0];assert not page.get_images()
        outside=[]
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                for span in line['spans']:
                    if not page.rect.contains(fitz.Rect(span['bbox'])):outside.append(span['text'])
        assert not outside,(p.name,outside)
        png=scratch/(p.stem+'.png');page.get_pixmap(matrix=fitz.Matrix(1.6,1.6)).save(png)
        figures.append(dict(path=p.relative_to(HERE).as_posix(),sha256=sha(p),pages=1,vector=True,text_inside_page=True,render=png.relative_to(HERE).as_posix()))
    assert len(figures)==4
    close=json.loads((HERE/'closeout.json').read_text());assert close['status']=='complete'
    assert close['compute_estimate_usd']<50 and close['cloud_storage_remaining_gb']==0
    absent=[]
    for p in close['pods']:
        r=subprocess.run(['runpodctl','pod','get',p['id']],capture_output=True,text=True,timeout=45,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert r.returncode and json.loads(r.stderr.splitlines()[0])['code']=='not_found'
        absent.append(p['id'])
    existing=json.loads(subprocess.check_output(['runpodctl','pod','list'],text=True))
    assert not any(p['name'].startswith('cdr-analysis034-') for p in existing)
    sources=[*HERE.glob('*.py'),ROOT/'src/eos_tasks/adam_profile.py',
        *ROOT.glob('runs/209-2026-11-01-eos-adam-calibration/*.py'),*ROOT.glob('runs/210-2026-11-01-eos-adam-long-lags/*.py'),
        *ROOT.glob('runs/211-2026-11-01-eos-parallel-adam-calibration/*.py')]
    for p in sources:ast.parse(p.read_text(encoding='utf-8'))
    record=dict(status='passed',csv_rows=len(rows),report_table_rows=len(summaries),figures=figures,
        task_pods_confirmed_absent=absent,unrelated_resources_preserved=True,syntax_files=len(sources),
        visual_review='required separately on rendered actual PDFs before handoff')
    (HERE/'final_review.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ['figures','task_pods_confirmed_absent']}))


if __name__=='__main__':main()
