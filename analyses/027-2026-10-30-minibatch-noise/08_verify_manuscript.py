"""Independently audit displayed H023 tables, copied PDFs and manuscript build."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
from pypdf import PdfReader

HERE=Path(__file__).parent;ROOT=HERE.resolve().parents[1];DRAFT=ROOT/'manuscript/draft'
OUT=ROOT/'runs/186-2026-10-30-mlp-minibatch-noise/artifacts/attempt-01/production'
BUILD=ROOT/'manuscript/artifacts/draft'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def numeric_rows(p):
    return [[x.strip().removesuffix('\\\\').strip() for x in line.split('&')]
            for line in p.read_text().splitlines() if re.match(r'^\d',line)]


def main():
    assert read(HERE/'verification.json')['status']=='passed'
    results=read(HERE/'results.json');checked=0
    table=numeric_rows(DRAFT/'tables/analysis027_contrasts.tex');assert len(table)==12
    for row in table:
        cp,b=[int(s.replace(',','')) for s in row[:2]]
        for j,field in enumerate(['mean_rho','R','final_loss'],2):
            deltas=[]
            for seed in range(3):
                base=next(r[field] for r in results['moving'] if (r['seed'],r['checkpoint'],r['batch'])==(seed,cp,5000))
                sample=[r[field] for r in results['moving'] if (r['seed'],r['checkpoint'],r['batch'])==(seed,cp,b)]
                assert len(sample)==8;deltas.append(np.mean(sample)-base)
            expected=np.array([np.mean(deltas),min(deltas),max(deltas)])
            displayed=np.array([float(s) for s in re.findall(r'-?\d+\.\d+',row[j])]);assert len(displayed)==3
            np.testing.assert_allclose(displayed,expected,atol=.00050000001 if j==2 else .000050000001,rtol=0);checked+=3
    table=numeric_rows(DRAFT/'tables/analysis027_hvp_errors.tex');assert len(table)==5
    comparisons=0
    for row in table:
        b,count=[int(s.replace(',','')) for s in row[:2]];raw=[];corrected=[]
        for path in OUT.glob(f'moving-*-b{b}-rep*/metrics.jsonl'):
            for r in rows(path):
                if r['previous_hvp_eta_kappa'] is not None:
                    raw.append(abs(r['q']-r['previous_hvp_eta_kappa']))
                    corrected.append(abs(r['q']+r['epsilon']-r['previous_hvp_eta_kappa']))
        assert len(raw)==count;comparisons+=count
        expected=[np.median(raw),np.median(corrected),np.percentile(corrected,90)]
        for cell,value in zip(row[2:],expected):
            if cell==r'$<0.0001$':assert 0<=value<.0001
            else:np.testing.assert_allclose(float(cell),value,atol=.000050000001,rtol=0)
        checked+=3
    assert comparisons==1188
    assets=read(HERE/'manuscript_assets.json');assert len(assets)==7
    for item in assets:assert sha(ROOT/item['source'])==sha(ROOT/item['target'])==item['sha256']
    for item in assets:
        if item['target'].endswith('.pdf'):
            pdf=PdfReader(ROOT/item['target']);assert len(pdf.pages)==1
            assert abs(float(pdf.pages[0].mediabox.width)-7.16*72)<.001
    before=(BUILD/'minibatch-noise-integration/before/03-methods.tex').read_text()
    after=(DRAFT/'sections/03-methods.tex').read_text()
    pattern=r'\\begin\{equation\}(.*?)\\end\{equation\}'
    assert re.findall(pattern,before,re.S)==re.findall(pattern,after,re.S)
    sources=['bare_jrnl.tex','sections/03-methods.tex','sections/04-gd-experiments.tex',
             'sections/05-sgd-noise-experiments.tex','sections/08-sgd-noise-diagnostics.tex']
    for name in sources:assert 'H023_RESULT_' not in (DRAFT/name).read_text()
    log=(BUILD/'bare_jrnl.log').read_text(errors='replace')
    for forbidden in ['Overfull \\hbox','Overfull \\vbox','undefined references','undefined citations','multiply defined','! LaTeX Error']:
        assert forbidden not in log,forbidden
    paper=PdfReader(BUILD/'bare_jrnl.pdf');texts=[p.extract_text() for p in paper.pages]
    assert any('MINIBATCHNOISEANDDISPLACEMENT' in re.sub(r'[^A-Z]','',t.upper()) for t in texts)
    report=dict(status='passed',table_rows=17,displayed_scalars_checked=checked,hvp_pairs=1188,copied_assets=7,
        unchanged_section_iii_equations=len(re.findall(pattern,before,re.S)),pages=len(paper.pages),
        pdf_sha256=sha(BUILD/'bare_jrnl.pdf'),source_sha256={name:sha(DRAFT/name) for name in sources},
        overfull_boxes=0,unresolved_references=0,visual_review='Recorded separately after rendering')
    (HERE/'manuscript_verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(report))


if __name__=='__main__':main()
