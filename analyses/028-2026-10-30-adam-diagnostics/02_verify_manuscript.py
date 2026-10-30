"""Check the Adam extension's mathematics, evidence tables and compiled manuscript."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import numpy as np
from pypdf import PdfReader

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DRAFT=ROOT/'manuscript/draft'
BUILD=ROOT/'manuscript/artifacts/draft'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cosine(x,y):return float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)))
def ratio(a):return float(np.linalg.norm(np.sum(a,axis=0))/np.linalg.norm(a,axis=1).sum())


def main():
    e=json.loads((HERE/'evidence.json').read_text(encoding='utf-8'))
    assert e['status']=='passed' and e['script_sha256']==digest(HERE/'01_build_evidence.py')
    for name,sha in {**e['source_summaries'],**e['event_sources']}.items():assert digest(ROOT/name)==sha,name
    for name,sha in e['tables'].items():assert digest(DRAFT/'tables'/name)==sha
    section=(DRAFT/'sections/06-adam-diagnostics.tex').read_text(encoding='utf-8')
    stream_table=(DRAFT/'tables/analysis028_streams.tex').read_text(encoding='utf-8')
    table_rows=[line for line in stream_table.splitlines() if line.startswith('$')]
    assert len(table_rows)==7
    for line,rid in zip(table_rows,['022','021','023','024','025','026','027']):
        row=next(r for r in e['streams'] if r['run']==rid)
        values=[float(s.strip().removesuffix('\\\\').strip()) for s in line.split('&')[1:]]
        assert values==[float(f'{row[k]:.3f}') for k in ['raw','momentum','adaptive','validation_loss']]
    table=(DRAFT/'tables/analysis028_schedules.tex').read_text(encoding='utf-8')
    rows=[line for line in table.splitlines() if line.startswith(('Original decay','Held rate','Faster decay'))]
    assert len(rows)==3
    for line,variant in zip(rows,['original','hold','accelerated']):
        targets=[]
        for size in ['14M','70M']:
            row=next(r for r in e['schedules'] if (r['schedule'],r['size'])==(variant,size))
            targets.extend([float(f'{row[k]:.3f}') for k in ['raw','validation_loss']])
        assert [float(s.strip().removesuffix('\\\\').strip()) for s in line.split('&')[1:]]==targets
    high=next(r for r in e['streams'] if r['run']=='023')
    assert high['clipped_updates']==517 and all(r['negative_momentum_pairs']==0 for r in e['streams'])
    for key in ['momentum','late_raw_norm','late_momentum_norm']:assert f'{high[key]:.3f}' in section
    for size in ['14M','70M']:
        group={r['schedule']:r for r in e['schedules'] if r['size']==size}
        assert group['accelerated']['raw']>group['original']['raw']>group['hold']['raw']
        assert group['hold']['validation_loss']<group['original']['validation_loss']<group['accelerated']['validation_loss']
        assert group['accelerated']['positive_recovery']['first_center_update']<group['original']['positive_recovery']['first_center_update']
    assert all(r['positive_recovery']['later_negative_mean'] for r in e['schedules'] if r['size']=='70M' and r['schedule']!='hold')

    math={}
    H=np.diag([1.,4.]);theta=np.array([2.,1.]);g=H@theta
    m=.1*g;v=.05*g*g;d=(m/.1)/(np.sqrt(v/.05)+1e-8)
    noises=[np.array([.1,-.2]),np.array([-.3,.4])]
    errors=[]
    for eta in [.1,1.5]:
        step=-eta*d
        # The same response identity holds if a decay displacement is added.
        for decay in [0.,.02]:
            actual=step-eta*decay*theta
            left=H@(theta+actual)+noises[1]-(H@theta+noises[0])
            right=H@actual+noises[1]-noises[0]
            errors.append(float(np.max(abs(left-right))))
    assert max(errors)<1e-14
    math['quadratic_gradient_response_max_error']=max(errors)
    small=cosine(g,H@(theta-.1*d));large=cosine(g,H@(theta-1.5*d))
    assert small>0 and large<0
    math['fixed_hessian_lr_response']=[small,large]
    current=np.array([1.,0.]);aligned=[.9*current+.1*a*current for a in [1.,10.]]
    assert np.linalg.norm(aligned[1])>np.linalg.norm(aligned[0])
    assert all(cosine(current,a)==1 for a in aligned)
    transverse=[.9*current+.1*np.array([0.,a]) for a in [1.,10.]]
    assert 0<cosine(current,transverse[1])<cosine(current,transverse[0])<1
    math['momentum_rotation_without_sign_reversal']=[cosine(current,a) for a in transverse]
    inputs=[np.array([0.,a]) for a in [2.,20.]]
    clipped=[a/max(1.,np.linalg.norm(a)) for a in inputs]
    np.testing.assert_array_equal(clipped[0],clipped[1])
    math['raw_growth_can_be_removed_by_clipping']=True
    # Positive moment alignment is empirical here, not guaranteed by the recurrence.
    first=.1*np.array([1.,0.]);second=.9*first+.1*np.array([-1.,0.])
    assert cosine(first,second)==-1
    math['momentum_can_reverse']=True
    mseq=np.array([[1.,0.],[0.,2.]])
    corrected=mseq/np.array([.1,.19])[:,None]
    assert cosine(*mseq)==cosine(*corrected) and abs(ratio(mseq)-ratio(corrected))>.01
    math['bias_correction_cosine_invariant_ratio_not_invariant']=True
    directions=np.array([[1.,.1],[.1,1.]])
    assert cosine(*directions)<.2
    math['coordinate_scaling_can_rotate_constant_momentum']=True
    steps=-np.array([.1,.8])[:,None]*directions
    assert abs(ratio(steps)-ratio(directions))>.01
    math['variable_lr_changes_cumulative_retention']=True

    unchanged=['sections/03-methods.tex','sections/04-gd-experiments.tex','sections/05-sgd-noise-experiments.tex',
               'sections/04-results.tex','sections/06-appendix.tex']
    for name in unchanged:
        assert (DRAFT/name).read_bytes()==subprocess.check_output(['git','show',f'c53defa:manuscript/draft/{name}'],cwd=ROOT)
    gd_name='sections/07-gd-diagnostics.tex'
    old_gd=subprocess.check_output(['git','show',f'c53defa:manuscript/draft/{gd_name}'],cwd=ROOT).decode()
    new_gd=(DRAFT/gd_name).read_text(encoding='utf-8')
    table_pattern=r'\\begin\{table\*\}\[!t\].*?\\end\{table\*\}'
    compact=lambda s:re.sub(r'\s+','',s)
    assert [compact(s) for s in re.findall(table_pattern,old_gd,re.S)]==[compact(s) for s in re.findall(table_pattern,new_gd,re.S)]
    assert compact(re.sub(table_pattern,'',old_gd,flags=re.S))==compact(re.sub(table_pattern,'',new_gd,flags=re.S))
    preserved_assets={}
    for relative in ['figures/analysis024','figures/analysis027','tables']:
        for p in (DRAFT/relative).glob('*'):
            if p.is_file() and not p.name.startswith('analysis028_'):
                name=p.relative_to(ROOT).as_posix()
                old=subprocess.check_output(['git','show',f'c53defa:{name}'],cwd=ROOT)
                assert p.read_bytes()==old,name
                preserved_assets[name]=digest(p)
    legacy=(DRAFT/'sections/06-appendix.tex').read_text(encoding='utf-8')
    assert (DRAFT/'sections/06-loss-appendix.tex').read_text(encoding='utf-8')==legacy.split(r'\section{Sample-Level Decomposition')[0].rstrip()+'\n'
    main=(DRAFT/'bare_jrnl.tex').read_text(encoding='utf-8')
    assert r'\input{sections/06-adam-diagnostics}' in main and r'\input{sections/04-results}' not in main
    assert r'\input{sections/06-appendix}' not in main
    assert section.count(r'\begin{equation}')==3 and section.count(r'\begin{table}')==2
    log=(BUILD/'bare_jrnl.log').read_text(encoding='utf-8',errors='replace')
    for term in ['Overfull \\hbox','Overfull \\vbox','undefined references','undefined citations','multiply defined','! LaTeX Error']:
        assert term not in log,term
    aux=(BUILD/'bare_jrnl.aux').read_text(encoding='utf-8')
    assert r'\newlabel{eq:noise-cosine-approx}{{19}' in aux
    texts=[p.extract_text() for p in PdfReader(BUILD/'bare_jrnl.pdf').pages]
    adam_page=next(i+1 for i,t in enumerate(texts) if 'VIDIRECTIONALDIAGNOSTICSUNDERADAM' in re.sub('[^A-Z]','',t.upper()))
    table_page=next(i+1 for i,t in enumerate(texts) if 'TABLE II:' in t and 'TABLE III:' in t)
    last_sgd_figure=max(i+1 for i,t in enumerate(texts) if 'Fig. 4:' in t or 'Fig. 5:' in t)
    assert last_sgd_figure<adam_page<=table_page
    report=dict(status='passed',script_sha256=digest(Path(__file__)),evidence_sha256=digest(HERE/'evidence.json'),
                math_checks=math,table_rows_checked=10,table_cosine_loss_values_checked=40,
                unchanged_sources=unchanged,adam_section_page=adam_page,tables_page=table_page,
                gd_appendix_text_and_tables_preserved=True,preserved_assets=preserved_assets,
                pages=len(texts),overfull_boxes=0,unresolved_references=0,pdf_sha256=digest(BUILD/'bare_jrnl.pdf'),
                section_sha256=digest(DRAFT/'sections/06-adam-diagnostics.tex'))
    (HERE/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:report[k] for k in ['status','table_rows_checked','adam_section_page','tables_page','pages']}))


if __name__=='__main__':main()
