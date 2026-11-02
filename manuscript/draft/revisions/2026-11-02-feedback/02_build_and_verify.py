"""Check revised mathematics/evidence, compile, and render the review manuscript."""
import csv
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import fitz
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DRAFT = HERE.parents[1]
A37 = ROOT/'analyses/037-2026-11-02-sgd-batch-extension'
SCRATCH = ROOT/'tmp/pdfs/manuscript-feedback-2026-11-02'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def arrsha(x):
    return hashlib.sha256(np.asarray(x,dtype='<f8').tobytes()).hexdigest()


def check_math():
    rng=np.random.default_rng(2031); errors=[]; count=0
    for _ in range(30):
        a=rng.normal(size=(4,4)); h=a.T@a+np.eye(4)
        x=rng.normal(size=4); mu=h@x
        k=mu@h@mu/(mu@mu)
        for factor in [.4,1.,1.5,2.5]:
            eta=factor/k; y=x-eta*mu; nu=h@y
            delta=(y@h@y-x@h@x)/2
            prediction=-eta*(mu@mu+mu@nu)/2
            errors.append(abs(delta-prediction));count+=1
            if abs(factor-1)<1e-12: assert abs(mu@nu)<1e-10
            assert (delta>0)==(factor>2)
            rho=mu@nu/(np.linalg.norm(mu)*np.linalg.norm(nu))
            assert (delta>0)==(rho < -np.linalg.norm(mu)/np.linalg.norm(nu))
            if delta>0: assert rho<0 and np.linalg.norm(nu)>np.linalg.norm(mu)
        # Nonzero perpendicular remainder at the line minimum is allowed.
        assert np.linalg.norm(h@(x-mu/k))>1e-8
        # Noisy endpoint formula must match the exact quadratic loss change.
        noise=rng.normal(size=4); eta=.1/k; g=mu+noise; y=x-eta*g
        new_noise=rng.normal(size=4); new_g=h@y+new_noise
        prediction=-eta*(g@g+g@new_g)/2+eta*g@(noise+new_noise)/2
        errors.append(abs((y@h@y-x@h@x)/2-prediction));count+=1
        # Enumerate independent zero-mean noise signs along an evolving path.
        interaction=[]
        for sign in [-1,1]:
            old_noise=sign*noise; old_g=mu+old_noise
            for new_sign in [-1,1]:
                interaction.append(old_g@(new_sign*new_noise-old_noise))
        errors.append(abs(np.mean(interaction)+noise@noise));count+=1
    for sign in [-1,1]:
        mu=np.array([1.,0.,0.]); nu=np.array([sign*.6,.8,0.])
        for scale in [.25,1.,4.]:
            z=np.array([0.,0.,scale]); vals=[]
            for i in [-1,1]:
                for j in [-1,1]:
                    u,v=mu+i*z,nu+j*z
                    vals.append(u@v/(np.linalg.norm(u)*np.linalg.norm(v)))
            expected=(mu@nu)/np.sqrt((mu@mu+scale**2)*(nu@nu+scale**2))
            errors.append(abs(np.mean(vals)-expected));count+=1
    for n in [2,8,32]:
        g=np.eye(n)
        assert abs(np.linalg.norm(g.sum(0))/np.linalg.norm(g,axis=1).sum()-1/np.sqrt(n))<1e-14
    assert np.linalg.norm(np.array([[1.,0.],[-1.,0.]]).sum(0))==0
    assert np.linalg.norm(np.array([1.,0.])-np.eye(2)@np.array([1.,0.]))==0
    assert max(errors)<1e-10
    return dict(quadratic_noise_and_attenuation_checks=count,max_absolute_error=max(errors),
                orthogonal_retention=True,opposing_cancellation=True,zero_gradient_exception=True)


def check_evidence():
    m=read(HERE/'figure_manifest.json')
    assert m['generator_sha256']==sha(HERE/'01_prepare_figures.py')
    for path,digest in m['sources'].items(): assert sha(ROOT/path)==digest,path
    a24=ROOT/'analyses/024-2026-10-17-eos-gd-lr-calibration'
    arrays=np.load(a24/'derived/single_rows_curves.npz')
    a27=read(ROOT/'analyses/027-2026-10-30-minibatch-noise/reduction.json')
    raw37=read(A37/'results.json')['forks']
    sgd=[c for c in m['curves'] if c['tag'].startswith('sgd:')]
    assert {c['tag'] for c in sgd}=={f'sgd:{task}:{state}:mean_rho'
        for task in ['mlp','resnet','transformer'] for state in [0,2000]}
    assert len(sgd)==len(m['sgd_values'])==6
    assert all(r['field']=='mean_rho' for r in m['sgd_values'])
    with (ROOT/'analyses/035-2026-11-02-eos-backward-anchors/lag_profiles.csv').open() as f:
        lag=list(csv.DictReader(f))
    for c in m['curves']:
        tag=c['tag']
        if tag.startswith('gd_'):
            key=tag.split(':')[1]; x,y=arrays[key+'_x'],arrays[key+'_y']
        elif tag.startswith('sgd:'):
            _,task,state,field=tag.split(':')
            batches={'mlp':[32,128,512,2048,5000],'resnet':[1,5,20,50],
                     'transformer':[8,32,128,512,59680]}[task]
            x=np.array(batches)*(1000 if task=='resnet' else 1)
            if task=='mlp':
                y=np.array([[next(r[field] for r in a27['moving'] if
                    (r['seed'],r['checkpoint'],r['batch'])==(seed,int(state),b))
                    for b in batches] for seed in range(3)]).mean(0)
            else:
                y=np.array([[np.mean([r[field] for r in raw37 if
                    (r['task'],r['seed'],r['state'],r['batch'])==(task,seed,int(state),b)])
                    for b in batches] for seed in range(3)]).mean(0)
        else:
            task=tag.split(':')[1]
            rows=sorted([r for r in lag if r['optimizer']=='gd' and r['task']==task and int(r['start'])==2000],key=lambda r:int(r['lag']))
            if tag.startswith('backward_zoom:'): rows=rows[:32]
            x=[int(r['lag']) for r in rows];y=[float(r['cosine']) for r in rows]
        assert len(x)==c['points'] and arrsha(x)==c['x_sha256'] and arrsha(y)==c['y_sha256'],tag
    # Reconstruct each used GD series from the original per-update event files.
    audit=read(a24/'section_iv_review_checks.json')['runs']
    for a in audit:
        all_events=[json.loads(line) for line in (ROOT/a['source']).read_text().splitlines()]
        # The paper's explicitly matched horizon is the first 3,000 updates;
        # the two historical MLP source runs also retain later observations.
        events=[e for e in all_events if e['update']<=3000]
        assert len(events)==3000
        raw=[e['streams']['raw'] for e in events]
        norms=np.array([r['norm'] for r in raw]); cos=np.array([r['cosine'] for r in raw[1:]])
        q=1-cos*norms[1:]/norms[:-1]
        for metric,y in [('rho_smoothed',cos),('gradient_norms_smoothed',norms),('eta_kappa_hat_smoothed',q)]:
            means=np.convolve(y,np.ones(50)/50,mode='valid')
            np.testing.assert_allclose(means,arrays[f'{metric}_run{a["run_id"]}_y'],rtol=2e-12,atol=2e-12)
        np.testing.assert_array_equal([r['B_over_A'] for r in raw],arrays[f'normalized_displacement_run{a["run_id"]}_y'])
        np.testing.assert_array_equal([e.get('train_loss',e.get('batch_objective_estimate')) for e in events],arrays[f'loss_log_run{a["run_id"]}_y'][:-1])
    raw27=read(ROOT/'analyses/027-2026-10-30-minibatch-noise/results.json')['moving']
    for record in m['sgd_values']:
        for i,seed in enumerate(range(3)):
            for j,b in enumerate(record['batches']):
                if record['task']=='mlp':
                    reps=[r[record['field']] for r in raw27 if (r['seed'],r['checkpoint'],r['batch'])==(seed,record['checkpoint'],b)]
                else:
                    reps=[r[record['field']] for r in raw37 if
                        (r['task'],r['seed'],r['state'],r['batch'])==(record['task'],seed,record['checkpoint'],b)]
                assert len(reps)==(1 if b==record['batches'][-1] else 8)
                assert abs(np.mean(reps)-record['initialization_means'][i][j])<1e-14
        block=np.array(record['initialization_means'])
        np.testing.assert_array_equal(block.min(0),record['minimum'])
        np.testing.assert_array_equal(block.max(0),record['maximum'])
    # Independent displayed-table parsing against full-precision reductions.
    results=read(a24/'results.json')['runs'];audit=read(a24/'section_iv_review_checks.json')['runs']
    table_text=(DRAFT/'tables/analysis024_endpoints.tex').read_text()
    rows=[x for x in table_text.splitlines() if ' & ' in x and not x.startswith('Dataset &')]
    assert len(rows)==9
    assert table_text.count(r'\multirow{6}{*}{CIFAR-10}')==1
    assert table_text.count(r'\multirow{3}{*}')==4
    dataset=model=None
    for row in rows:
        cells=[x.strip() for x in row.removesuffix(r'\\').split('&')]
        if cells[0]: dataset=re.fullmatch(r'\\multirow\{[36]\}\{\*\}\{([^}]+)\}',cells[0]).group(1)
        if cells[1]: model=re.fullmatch(r'\\multirow\{3\}\{\*\}\{([^}]+)\}',cells[1]).group(1)
        task={'MLP':'mlp','ResNet-32':'resnet','Transformer':'transformer'}[model]
        assert dataset==('WikiText-2' if task=='transformer' else 'CIFAR-10')
        r=next(r for r in results if r['task']==task and r['lr']==float(cells[2]))
        a=next(a for a in audit if a['run_id']==r['run_id'])
        expected=[r['final_loss'],r['late_1000_cosine_mean'],r['final_B_over_A'],r['late_1000_q_mean'],a['final_gradient_sum_displacement']]
        for cell,val in zip(cells[3:],expected):
            assert abs(float(cell.replace('$^{*}$',''))-val)<=.000050001
        assert ('$^{*}$' in cells[6])==(0<abs(expected[3])<.00005)
    for task in ['mlp','resnet','transformer']:
        cohort=sorted([r for r in results if r['task']==task],key=lambda r:r['lr'])
        assert cohort[-1]['final_loss']==min(r['final_loss'] for r in cohort)
        assert cohort[-1]['final_B_over_A']==min(r['final_B_over_A'] for r in cohort)
        displacements=[next(a['final_gradient_sum_displacement'] for a in audit if a['run_id']==r['run_id']) for r in cohort]
        assert displacements[-1]==max(displacements)
    assert sum(r['late_1000_cosine_mean']<0 for r in results)==7
    assert all(a['final_all_pair']>0 for a in audit)
    for seed in range(3):
        for batch in [32,5000]:
            loss=np.mean([r['loss_change'] for r in raw27 if (r['seed'],r['checkpoint'],r['batch'])==(seed,2000,batch)])
            assert (loss>0)==(batch==32)
    # Check the displayed moving-path example from every retained repetition.
    state_example={}
    for state in [0,2000]:
        state_example[state]={field:float(np.mean([np.mean([r[field] for r in raw37 if
            (r['task'],r['seed'],r['state'],r['batch'])==('transformer',seed,state,8)])
            for seed in range(3)])) for field in ['mean_rho','loss_change']}
        for seed in range(3):
            loss=np.mean([r['loss_change'] for r in raw37 if
                (r['task'],r['seed'],r['state'],r['batch'])==('transformer',seed,state,8)])
            assert (loss>0)==(state==2000)
    text=(DRAFT/'sections/05-sgd-noise-experiments.tex').read_text()
    paragraph=text.split('For the Transformer at $b=8$,',1)[1].split('\n\n',1)[0]
    displayed=re.findall(r'\$(-?\d+\.\d+)\$',paragraph)
    assert f'{state_example[0]["mean_rho"]:.2f}'==f'{state_example[2000]["mean_rho"]:.2f}'
    assert displayed==[f'{state_example[0]["mean_rho"]:.2f}']
    bridge=read(A37/'spark-bridge-acceptance.json')
    coverage=read(A37/'cohort_acceptance.json')
    assert bridge['status']=='passed' and coverage['spark_bridge_acceptance_sha256']==sha(A37/'spark-bridge-acceptance.json')
    appendix=(DRAFT/'sections/07-gd-diagnostics.tex').read_text()
    mantissa,exponent=re.search(r'parameter relative error \$(\d+\.\d+)\\times10\^\{(-?\d+)\}\$',appendix).groups()
    assert f'{float(mantissa)*10**int(exponent):.2e}'==f'{bridge["full_replay_parameter_relative_l2"]:.2e}'
    for seed in range(3):
        full=np.mean([r['loss_change'] for r in raw37 if
            (r['task'],r['seed'],r['state'],r['batch'])==('transformer',seed,2000,59680)])
        assert full<0
    for task,batches in [('mlp',[32,128,512,2048,5000]),('resnet',[1,5,20,50]),
                         ('transformer',[8,32,128,512,59680])]:
        rows=raw27 if task=='mlp' else [r for r in raw37 if r['task']==task]
        state_key='checkpoint' if task=='mlp' else 'state'
        means={b:{field:float(np.mean([np.mean([r[field] for r in rows if
            (r['seed'],r[state_key],r['batch'])==(seed,2000,b)]) for seed in range(3)]))
            for field in ['mean_rho']} for b in batches}
        assert means[batches[-1]]['mean_rho']<means[batches[0]]['mean_rho']<0
    assert np.mean([r['mean_rho'] for r in raw27 if (r['checkpoint'],r['batch'])==(0,32)])<0
    assert np.mean([r['mean_rho'] for r in raw27 if (r['checkpoint'],r['batch'])==(0,5000)])>0
    # The manuscript's all-nine-endpoints statement uses every retained GD profile.
    for task in ['mlp','resnet','transformer']:
        for start in [500,2000,3000]:
            ys=[float(r['cosine']) for r in lag if r['optimizer']=='gd' and r['task']==task and int(r['start'])==start and int(r['lag'])>=193]
            assert len(ys)==64 and min(ys)<0<max(ys)
    for f in m['figures']:
        assert sha(ROOT/f['source'])==sha(ROOT/f['target'])==f['sha256']
        with fitz.open(ROOT/f['target']) as doc:
            assert len(doc)==1 and not doc[0].get_images()
            for font in doc[0].get_fonts(full=True):
                assert font[2]!='Type3' and doc.extract_font(font[0])[3]
    comments=read(HERE/'comments.json')
    assert len(comments['comments'])==28 and sha(HERE/'source-comments.pdf')==comments['source_sha256']
    return dict(curves=len(m['curves']),points=sum(c['points'] for c in m['curves']),
                endpoint_rows=9,displayed_scalars=45,backward_endpoints=9,
                preserved_comments=28,source_hashes=len(m['sources']),vector_figures=4,
                transformer_state_example=state_example,displayed_state_example_scalars=1,
                sgd_curves=len(sgd),sgd_points=sum(c['points'] for c in sgd),
                sgd_initialization_scalars=sum(np.array(r['initialization_means']).size for r in m['sgd_values']),
                displayed_runtime_bridge_scalars=1)


def main():
    SCRATCH.mkdir(parents=True,exist_ok=True)
    record=dict(math=check_math(),evidence=check_evidence(),status='pending_build_and_visual_review')
    commands=[['pdflatex','--disable-installer','-no-shell-escape','-recorder','-interaction=nonstopmode',
               '-halt-on-error','-file-line-error','-output-directory='+str(SCRATCH),'bare_jrnl.tex']]
    commands += [['bibtex',str(SCRATCH/'bare_jrnl')],commands[0],commands[0]]
    for i,command in enumerate(commands,1):
        result=subprocess.run(command,cwd=DRAFT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
        (SCRATCH/f'build-{i}.txt').write_bytes(result.stdout)
        if result.returncode:
            print(result.stdout.decode(errors='replace')[-6000:]);raise RuntimeError(f'Build pass {i} failed')
    log=(SCRATCH/'bare_jrnl.log').read_text(errors='replace')
    problems=[line for line in log.splitlines() if 'Overfull' in line or 'undefined' in line.lower() or 'Rerun to get' in line or 'multiply defined' in line]
    if problems: print('\n'.join(problems));raise AssertionError('Resolve LaTeX warnings')
    aux=(SCRATCH/'bare_jrnl.aux').read_text()
    def page(label):
        return int(re.search(r'\\newlabel\{'+re.escape(label)+r'\}\{\{[^}]*\}\{(\d+)\}',aux).group(1))
    placement={label:page(label) for label in ['fig:gd-loss','fig:gd-alignment-displacement','tab:gd-endpoints','fig:gd-backward','fig:sgd-moving-geometry','sec:adam-diagnostics']}
    assert list(placement.values())==sorted(placement.values()),placement
    record['placement']=placement
    archive=read(ROOT/'manuscript/archive/2026-11-02-unused-figures/manifest.json')
    expected={str((ROOT/p).resolve()) for p in archive['active_figures']}
    present={str(p.resolve()) for p in (DRAFT/'figures').rglob('*') if p.is_file()}
    assert present==expected, 'Keep only compiled figures in draft/figures.'
    recorded=set()
    for line in (SCRATCH/'bare_jrnl.fls').read_text().splitlines():
        if line.startswith('INPUT '):
            p=Path(line[6:])
            if not p.is_absolute(): p=DRAFT/p
            if p.suffix.lower()=='.pdf' and p.resolve().is_relative_to((DRAFT/'figures').resolve()):
                recorded.add(str(p.resolve()))
    assert recorded==expected, 'LaTeX recorder differs from the active figure inventory.'
    for entry in archive['files']:
        p=ROOT/entry['archived']
        assert not (ROOT/entry['original']).exists()
        assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256']
    for tex in (DRAFT/'sections').glob('*.tex'):
        for asset in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',tex.read_text()):
            assert (DRAFT/asset).is_file(), (tex,asset)
    record['figure_archive']=dict(active_figures=len(expected),archived_files=len(archive['files']),
                                 hashes_verified=True,latex_recorder_verified=True)
    pdf=SCRATCH/'bare_jrnl.pdf'
    for target in [DRAFT/'bare_jrnl.pdf',HERE/'revised-manuscript.pdf']:
        shutil.copyfile(pdf,target)
    with fitz.open(pdf) as doc:
        assert '??' not in ''.join(p.get_text() for p in doc)
        record['pages']=len(doc)
        record['page_texts']={str(i+1):p.get_text()[:90] for i,p in enumerate(doc)}
        for i,p in enumerate(doc): p.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(SCRATCH/f'page-{i+1:02}.png')
    for f in read(HERE/'figure_manifest.json')['figures']:
        with fitz.open(ROOT/f['target']) as doc:
            doc[0].get_pixmap(matrix=fitz.Matrix(2,2)).save(SCRATCH/(Path(f['target']).stem+'.png'))
    record.update(status='awaiting_visual_review',pdf_sha256=sha(pdf),build='pdflatex, bibtex, pdflatex, pdflatex; no overfull boxes or unresolved references',
                  source_hashes={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [DRAFT/'bare_jrnl.tex',*sorted((DRAFT/'sections').glob('*.tex'))]},
                  verifier_sha256=sha(__file__))
    (HERE/'verification.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
