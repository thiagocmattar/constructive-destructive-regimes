"""Build the current paper, verify preserved evidence, and render every page."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import fitz
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
DRAFT=HERE.parents[1]
ANALYSIS=ROOT/'analyses/056-2026-11-05-gd-paired-displacement'
SCRATCH=ROOT/'tmp/pdfs/human-input-2026-11-05'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    subprocess.run(['python', str(HERE/'02_check_manuscript_evidence.py')], check=True,
                   capture_output=True, timeout=60)
    baseline=read(HERE/'baseline_manifest.json')
    for path,digest in baseline['previous_figure_hashes'].items():
        assert sha(ROOT/path)==digest,path
    for name in ['input.md','04-gd-experiments.tex','04-controlled-figures.tex']:
        path=HERE/name
        assert sha(path)==baseline['source_hashes'][path.relative_to(ROOT).as_posix()]
    for name in ['06-adam-diagnostics.tex','06-loss-appendix.tex','02-related-works.tex']:
        path=DRAFT/'sections'/name
        assert sha(path)==baseline['source_hashes'][path.relative_to(ROOT).as_posix()]
    spec=importlib.util.spec_from_file_location('prior_math_checks',HERE.parent/
        '2026-11-03-original-argument-feedback-feedback/02_build_and_verify.py')
    previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
    math_checks=previous.check_loss_conditions()
    v=read(ANALYSIS/'verification.json');f=read(ANALYSIS/'figure_manifest.json')
    assert v['status']=='passed' and v['results_sha256']==sha(ANALYSIS/'results.json')
    assert f['generator_sha256']==sha(ANALYSIS/'03_plot.py')
    assert f['results_sha256']==sha(ANALYSIS/'results.json')
    for figure in f['figures']:
        assert sha(ROOT/figure['path'])==sha(ROOT/figure['manuscript_copy'])==figure['sha256']
    arrays=np.load(ANALYSIS/'plotted_arrays.npz')
    for curve in f['curves']:
        for axis in ['x','y']:
            values=arrays[curve['tag']+'_'+axis]
            assert len(values)==curve['points']
            assert hashlib.sha256(values.astype('<f8').tobytes()).hexdigest()==curve[axis+'_sha256']
    SCRATCH.mkdir(parents=True,exist_ok=True)
    tex=['pdflatex','--disable-installer','-no-shell-escape','-recorder','-interaction=nonstopmode',
         '-halt-on-error','-file-line-error','-output-directory='+str(SCRATCH),'bare_jrnl.tex']
    for i,command in enumerate([tex,['bibtex',str(SCRATCH/'bare_jrnl')],tex,tex],1):
        process=subprocess.run(command,cwd=DRAFT,capture_output=True,timeout=120)
        (SCRATCH/f'build-{i}.txt').write_bytes(process.stdout+process.stderr)
        if process.returncode:raise RuntimeError(f'Build pass {i} failed: {SCRATCH}/build-{i}.txt')
    log=(SCRATCH/'bare_jrnl.log').read_text(errors='replace')
    issues=[line for line in log.splitlines() if any(token in line.lower() for token in
            ['overfull','undefined','rerun to get','multiply defined'])]
    pdf=SCRATCH/'bare_jrnl.pdf'
    with fitz.open(pdf) as doc:
        text='\n'.join(page.get_text() for page in doc)
        assert '??' not in text
        pages=len(doc)
        (SCRATCH/'manuscript-text.txt').write_text(text,encoding='utf-8')
        page_content=[dict(page=i+1,characters=len(p.get_text()),images=len(p.get_images())) for i,p in enumerate(doc)]
    subprocess.run(['pdftoppm','-r','110','-png',str(pdf),str(SCRATCH/'review')],
                   check=True,capture_output=True,timeout=120)
    for target in [DRAFT/'bare_jrnl.pdf',HERE/'revised-manuscript.pdf']:
        shutil.copyfile(pdf,target)
    active_figures=set()
    for line in (SCRATCH/'bare_jrnl.fls').read_text().splitlines():
        if line.startswith('INPUT '):
            path=Path(line[6:]);path=path if path.is_absolute() else DRAFT/path
            if path.suffix=='.pdf' and path.resolve().is_relative_to((DRAFT/'figures').resolve()):
                active_figures.add(path.resolve().relative_to(ROOT).as_posix())
    report=dict(status='needs_layout_fixes' if issues else 'awaiting_visual_review',pages=pages,
        issues=issues,page_content=page_content,math_checks=math_checks,
        paired_analysis=v,evidence_checks=read(HERE/'evidence_checks.json'),
        active_figures=sorted(active_figures),plotted_points=f['plotted_points'],
        protected_existing_figures=len(baseline['previous_figure_hashes']),
        pdf_sha256=sha(pdf),source_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in
            [DRAFT/'bare_jrnl.tex',DRAFT/'bibliography.bib',*sorted((DRAFT/'sections').glob('*.tex')),
             DRAFT/'tables/analysis056_pairs.tex']},generator_sha256=sha(__file__))
    (HERE/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status','pages','issues','active_figures']},indent=2))


if __name__=='__main__':main()
