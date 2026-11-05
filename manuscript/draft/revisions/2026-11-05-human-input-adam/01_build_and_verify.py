"""Build and render the current manuscript; preserve supplied inputs and figures."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import fitz
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
DRAFT=HERE.parents[1]
ANALYSIS=ROOT/'analyses/057-2026-11-05-adam-response-memory'
SCRATCH=ROOT/'tmp/pdfs/human-input-adam-2026-11-05'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def main():
    subprocess.run(['python',str(ANALYSIS/'02_verify.py')],check=True,capture_output=True)
    subprocess.run(['python',str(HERE/'02_check_evidence.py')],check=True,capture_output=True)
    baseline=read(HERE/'baseline_manifest.json')
    for rel,digest in baseline['figures'].items():assert sha(ROOT/rel)==digest,rel
    preserved=[HERE/'input.md',HERE/'05-adam-diagnostics.tex']
    preserved += [DRAFT/'sections'/n for n in ['03-methods.tex','04-gd-experiments.tex',
        '04-controlled-figures.tex','05-sgd-noise-experiments.tex','06-loss-appendix.tex']]
    for p in preserved:assert sha(p)==baseline['sources'][p.relative_to(ROOT).as_posix()]
    fm=read(ANALYSIS/'figure_manifest.json')
    assert fm['generator_sha256']==sha(ANALYSIS/'03_plot.py')
    assert fm['results_sha256']==sha(ANALYSIS/'results.json')
    for f in fm['figures']:
        assert sha(ROOT/f['path'])==sha(ROOT/f['manuscript_copy'])==f['sha256']
    assert sha(ROOT/fm['table']['path'])==fm['table']['sha256']
    for c in fm['curves']:
        for axis in ['x','y']:
            assert hashlib.sha256(np.asarray(c[axis],dtype='<f8').tobytes()).hexdigest()==c[axis+'_sha256']
    tex=['pdflatex','--disable-installer','-no-shell-escape','-recorder','-interaction=nonstopmode',
        '-halt-on-error','-file-line-error','-output-directory='+str(SCRATCH),'bare_jrnl.tex']
    for i,command in enumerate([tex,['bibtex',str(SCRATCH/'bare_jrnl')],tex,tex],1):
        result=subprocess.run(command,cwd=DRAFT,capture_output=True,timeout=120)
        (SCRATCH/f'build-{i}.txt').write_bytes(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError(f'Build pass {i} failed; see scratch log.')
    log=(SCRATCH/'bare_jrnl.log').read_text(errors='replace')
    issues=[s for s in log.splitlines() if any(w in s.lower() for w in ['overfull','undefined','rerun to get','multiply defined'])]
    pdf=SCRATCH/'bare_jrnl.pdf'
    with fitz.open(pdf) as doc:
        text='\n'.join(p.get_text() for p in doc);assert '??' not in text
        pages=len(doc);page_content=[dict(page=i+1,characters=len(p.get_text())) for i,p in enumerate(doc)]
        (SCRATCH/'manuscript-text.txt').write_text(text,encoding='utf-8')
    subprocess.run(['pdftoppm','-r','110','-png',str(pdf),str(SCRATCH/'review')],check=True,capture_output=True,timeout=120)
    report=dict(status='needs_layout_fix' if issues else 'compiled_pending_visual_review',
        pages=pages,issues=issues,pdf_sha256=sha(pdf),page_content=page_content,
        analysis_verification_sha256=sha(ANALYSIS/'verification.json'),
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in
          [DRAFT/'bare_jrnl.tex',DRAFT/'bibliography.bib',*sorted((DRAFT/'sections').glob('*.tex'))]})
    (HERE/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(report))
    if issues:raise RuntimeError('Resolve compilation issues before delivery.')
    shutil.copyfile(pdf,HERE/'revised-manuscript.pdf')
    shutil.copyfile(pdf,DRAFT/'bare_jrnl.pdf')

if __name__=='__main__':main()
