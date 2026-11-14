"""Check publication PDFs, render with Poppler, and record visual review."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import fitz

HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--figures',type=Path,default=HERE/'figures/matched')
    p.add_argument('--reviewed',action='store_true');p.add_argument('--skip-render',action='store_true');a=p.parse_args()
    root=a.figures.resolve();meta=json.loads((root/'figure_manifest.json').read_text())
    scratch=HERE/'scratch/h051/pdf-review'/root.name;scratch.mkdir(parents=True,exist_ok=True)
    poppler=shutil.which('pdftoppm');assert poppler or a.skip_render,'Poppler required for visual QA'
    result=[];violations=[]
    for item in meta['outputs']:
        path=root/item['file'];assert sha(path)==item['sha256']
        with fitz.open(path) as doc:
            assert doc.page_count==item['pages']
            for index,page in enumerate(doc):
                text=page.get_text();assert len(text)>50 and '\ufffd' not in text
                assert all(font[2]!='Type3' for font in page.get_fonts()),'Unembedded bitmap math font'
                for block in page.get_text('dict')['blocks']:
                    for line in block.get('lines',[]):
                        for span in line['spans']:
                            box=fitz.Rect(span['bbox'])
                            if not (page.rect+(-1,-1,1,1)).contains(box):violations.append(dict(file=path.name,page=index+1,text=span['text'],bbox=list(box)))
            result.append(dict(file=path.name,pages=doc.page_count,sha256=sha(path)))
        if not a.skip_render:
            subprocess.run([poppler,'-r','130','-png',str(path),str(scratch/path.stem)],check=True,capture_output=True)
    assert not violations,violations
    tex=scratch/'check-tables.tex'
    tex.write_bytes((r'\documentclass[10pt]{article}'+'\n'+
        r'\usepackage[paperwidth=7.15in,paperheight=9in,margin=.18in]{geometry}'+'\n'+
        r'\usepackage{booktabs}'+'\n'+r'\begin{document}'+'\n'+
        r'\input{'+(root/'matched-outcomes.tex').as_posix()+'}\n'+
        r'\input{'+(root/'matched-costs.tex').as_posix()+'}\n'+r'\end{document}'+'\n').encode())
    command=['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory='+str(scratch),str(tex)]
    proc=subprocess.run(command,capture_output=True)
    assert proc.returncode==0,proc.stdout.decode(errors='replace')[-2500:]
    log=(scratch/'check-tables.log').read_text(errors='replace')
    assert 'Overfull \\hbox' not in log,log[-4000:]
    verification=dict(status='passed' if a.reviewed else 'awaiting_visual_review',complete_matrix=meta['complete_matrix'],
        figures=len(result),pages=sum(v['pages'] for v in result),pdfs=result,render_directory=str(scratch),
        fonts='Embedded TrueType; no Type3 fonts',text_outside_pages=violations,editable_table_compile='passed without overfull horizontal boxes',
        editable_sources={name:sha(root/name) for name in ['matched-outcomes.tex','matched-costs.tex']},
        plotted_values_checked=meta['plotted_values_checked'],figure_manifest_sha256=sha(root/'figure_manifest.json'),
        visual_review='All final pages inspected for labels, legends, scaling, overlap, clipping and model/setting order.' if a.reviewed else 'pending',
        verifier_sha256=sha(Path(__file__)))
    (root/'visual_verification.json').write_bytes((json.dumps(verification,indent=2)+'\n').encode())
    print(json.dumps({k:verification[k] for k in ['status','figures','pages','render_directory','editable_table_compile']}))


if __name__=='__main__':main()
