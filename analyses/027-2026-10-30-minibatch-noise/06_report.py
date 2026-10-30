"""Compute result statements and tables from verified raw repetitions, without selection."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE=Path(__file__).parent;ROOT=HERE.resolve().parents[1]
OUT=ROOT/'runs/186-2026-10-30-mlp-minibatch-noise/artifacts/attempt-01/production'
BATCHES=[32,128,512,2048,5000];STATES=[0,1000,2000]
EXPECTED_SHA={}


def read(p):return json.loads(p.read_text())
def rows(p):
    content=p.read_bytes();assert hashlib.sha256(content).hexdigest()==EXPECTED_SHA[p.relative_to(OUT).as_posix()]
    return [json.loads(s) for s in content.decode().splitlines() if s.strip()]
def save(p,value):p.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def describe(values):
    a=np.asarray(values,dtype=float)
    return dict(n=len(a),mean=float(a.mean()),minimum=float(a.min()),maximum=float(a.max()),
                median=float(np.median(a)),p90=float(np.percentile(a,90)))


def main():
    verification=read(HERE/'verification.json');assert verification['status']=='passed'
    assert digest(HERE/'results.json')==verification['results_sha256']
    assert digest(OUT/'inventory.json')==verification['production_inventory_sha256']
    EXPECTED_SHA.update({r['path']:r['sha256'] for r in read(OUT/'inventory.json')['files']})
    reduced=read(HERE/'reduction.json');raw=read(HERE/'results.json')
    frozen=reduced['frozen'];moving=reduced['moving'];stochastic=[c for c in frozen if c['batch']!=5000]
    def covered(c,field,target):
        lo,hi=c[field+'_conditional_ci95'];return lo<=target<=hi
    summary=dict(source_results_sha256=digest(HERE/'results.json'),source_reduction_sha256=digest(HERE/'reduction.json'),
        production_inventory_sha256=digest(OUT/'inventory.json'),
        numpy=np.__version__,script_sha256=digest(Path(__file__)),
        frozen=dict(stochastic_cells=len(stochastic),
            variance_relative_error=describe([(c['mean_noise_energy']-c['predicted_variance'])/c['predicted_variance'] for c in stochastic]),
            projection_relative_error=describe([(c['mean_noise_projection']+c['predicted_variance'])/c['predicted_variance'] for c in stochastic]),
            cosine_absolute_error=describe([abs(c['rho_prediction_error']) for c in frozen]),
            norm_cv=describe([c['norm_cv'] for c in stochastic]),
            norm_product_cv=describe([c['norm_product_cv'] for c in stochastic]),
            variance_ci_covers_prediction=sum(covered(c,'mean_noise_energy',c['predicted_variance']) for c in stochastic),
            projection_ci_covers_prediction=sum(covered(c,'mean_noise_projection',-c['predicted_variance']) for c in stochastic),
            cosine_ci_covers_approximation=sum(covered(c,'mean_rho',c['predicted_rho']) for c in stochastic)),
        frozen_cells=[],moving_cells=[],hvp_by_batch=[],hvp_by_state_batch=[],moving_noise_cells=[],reference=[])
    for cp in STATES:
        for b in BATCHES:
            f=[c for c in frozen if c['checkpoint']==cp and c['batch']==b]
            summary['frozen_cells'].append(dict(checkpoint=cp,batch=b,
                rho=describe([c['mean_rho'] for c in f]),predicted_rho=describe([c['predicted_rho'] for c in f]),
                chi=describe([c['predicted_variance']/c['signal_energy'] for c in f]),
                R=describe([c['R'] for c in f])))
            m=[c for c in moving if c['checkpoint']==cp and c['batch']==b]
            summary['moving_cells'].append(dict(checkpoint=cp,batch=b,
                **{k:describe([c[k] for c in m]) for k in ['mean_rho','mean_full_rho','R','actual_R','final_loss','loss_change','absolute_displacement']}))
    probes=[];loss_changes=[]
    for condition in raw['moving']:
        seed,cp,b,rep=(condition[k] for k in ('seed','checkpoint','batch','replicate'))
        records=rows(OUT/f'moving-seed{seed}-state{cp}-b{b}-rep{rep:02d}/metrics.jsonl')
        for row in records:
            if row['taylor_residual'] is not None:
                probes.append(dict(seed=seed,checkpoint=cp,batch=b,replicate=rep,update=row['update'],
                    raw_error=abs(row['q']-row['previous_hvp_eta_kappa']),
                    corrected_error=abs(row['taylor_residual']),hvp=row['previous_hvp_eta_kappa']))
        all_losses=[r['full_loss'] for r in records]+[condition['final_loss']]
        loss_changes.extend(np.diff(all_losses).tolist())
    for b in BATCHES:
        p=[r for r in probes if r['batch']==b]
        summary['hvp_by_batch'].append(dict(batch=b,count=len(p),raw=describe([r['raw_error'] for r in p]),
            corrected=describe([r['corrected_error'] for r in p]),
            corrected_strictly_better=sum(r['corrected_error']<r['raw_error'] for r in p)))
        for cp in STATES:
            p=[r for r in probes if r['batch']==b and r['checkpoint']==cp]
            summary['hvp_by_state_batch'].append(dict(batch=b,checkpoint=cp,count=len(p),
                raw=describe([r['raw_error'] for r in p]),corrected=describe([r['corrected_error'] for r in p])))
    for c in moving:
        V=c['mean_previous_noise_energy'];res=c['mean_noise_identity_residual']
        summary['moving_noise_cells'].append(dict(seed=c['seed'],checkpoint=c['checkpoint'],batch=c['batch'],
            mean_projection=c['mean_noise_projection'],mean_previous_noise_energy=V,residual=res,
            relative_residual=None if V==0 else res/V,ci95=c['mean_noise_identity_residual_conditional_ci95'],
            ci_covers_zero=covered(c,'mean_noise_identity_residual',0)))
    for seed in range(3):
        records=rows(OUT/f'reference-seed{seed}/metrics.jsonl');negative=[r['update'] for r in records if r['cosine'] is not None and r['cosine']<0]
        summary['reference'].append(dict(seed=seed,first_negative=None if not negative else negative[0],negative_pairs=len(negative),
            final_loss=read(OUT/f'reference-seed{seed}/summary.json')['final']['loss']))
    summary['moving_loss']=dict(forks=len(raw['moving']),forks_lower_final_loss=sum(r['loss_change']<0 for r in raw['moving']),
        individual_updates=len(loss_changes),positive_update_loss_changes=sum(x>0 for x in loss_changes))
    summary['maximum_R_roundoff']=max(abs(r['R']-r['actual_R']) for r in raw['moving'])
    summary['reconstruction']=verification['reconstruction']
    save(HERE/'scientific_summary.json',summary)
    tables=HERE/'tables';tables.mkdir(exist_ok=True)
    def fmt(d,digits):return f"${d['mean']:.{digits}f}\\;[{d['minimum']:.{digits}f},\\,{d['maximum']:.{digits}f}]$"
    lines=[r'\begin{tabular}{rrccc}',r'\hline',
        r'GD state & $b$ & $\Delta\overline\rho$ & $\Delta R_{200}$ & $\Delta\mathcal L_{200}$ \\',r'\hline']
    for cp in STATES:
        for b in BATCHES[:-1]:
            records=[r for r in reduced['contrasts_to_full_batch'] if r['checkpoint']==cp and r['batch']==b]
            values=[fmt(describe([r[k] for r in records]),3 if k=='rho_difference' else 4)
                    for k in ('rho_difference','R_difference','final_loss_difference')]
            lines.append(f'{cp:,} & {b:,} & '+ ' & '.join(values)+r' \\')
        lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    (tables/'analysis027_contrasts.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    lines=[r'\begin{tabular}{rrrrr}',r'\hline',r'$b$ & Pairs & Raw med. & Corr. med. & Corr. 90th \\',r'\hline']
    def error_text(x):return r'$<0.0001$' if x<.0001 else f'{x:.4f}'
    for r in summary['hvp_by_batch']:
        values=[error_text(x) for x in (r['raw']['median'],r['corrected']['median'],r['corrected']['p90'])]
        lines.append(f"{r['batch']:,} & {r['count']} & "+' & '.join(values)+r' \\')
    lines.extend([r'\hline',r'\end{tabular}'])
    (tables/'analysis027_hvp_errors.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    save(HERE/'table_manifest.json',dict(source_summary_sha256=digest(HERE/'scientific_summary.json'),
        tables={p.name:digest(p) for p in tables.glob('*.tex')}))
    print(json.dumps({'summary':'scientific_summary.json','tables':2,'hvp_pairs':len(probes),'frozen':summary['frozen'],'moving_loss':summary['moving_loss']}))


if __name__=='__main__':main()
