"""Check displayed numerical claims and retained-curve agreement independently."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
DRAFT=HERE.parents[1]
A=ROOT/'analyses/056-2026-11-05-gd-paired-displacement'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    r=read(A/'results.json');p=np.load(A/'plotted_arrays.npz');a=np.load(A/'arrays.npz')
    gd=' '.join((DRAFT/'sections/04-gd-experiments.tex').read_text(encoding='utf-8').split())
    sgd=' '.join((DRAFT/'sections/05-sgd-noise-experiments.tex').read_text(encoding='utf-8').split())
    counts=0
    highest=['122','120','126']
    ends=[next(e for e in r['endpoints'] if e['run']==rid and e['offset']==0) for rid in highest]
    for field,fmt in [('R','.1%'),('Rperp','.2%'),('C','.3f'),('D','.3f'),('Q','.2f'),('final_loss','.4f')]:
        for e in ends:
            rendered=format(e[field],fmt).replace('%',r'\%')
            assert rendered in gd,(field,rendered)
            counts+=1
    for rid in highest:
        e0,e1=[e for e in r['endpoints'] if e['run']==rid]
        assert max(abs(e0[k]-e1[k]) for k in ['C','D'])<.004
        for offset in [0,1]:
            w=next(w for w in r['loss_windows'] if w['run']==rid and w['offset']==offset and w['window_start']==2000)
            assert format(w['decreases']/w['pairs']*100,'.1f')+r'\%' in gd
            assert w['sum_change']<0
            counts+=1
        w=next(w for w in r['loss_windows'] if w['run']==rid and w['offset']==0 and w['window_start']==2000)
        assert f"{-w['sum_change']:.4f}" in gd
        counts+=1
    assert all(e['Q']>1 for e in r['endpoints'])
    assert sum(t['lag_one']<0 for t in r['trajectories'])==7
    assert all(s['mean_odd_first32']<0 and s['mean_even_first32']>0 for s in r['lag_summary'])
    assert all(s['odd_negative_first32']==s['even_positive_first32']==16 for s in r['lag_summary'] if s['task']=='transformer')
    for task in ['mlp','resnet','transformer']:
        rates=[t for t in r['trajectories'] if t['task']==task]
        assert min(rates,key=lambda t:t['final_loss'])==max(rates,key=lambda t:t['lr'])
        for state in [0,2000]:
            cells=[c for c in r['sgd_cells'] if c['task']==task and c['state']==state]
            for c in [cells[0],cells[-1]]:
                value=f"{c['mean']['mean_rho']:.3f}"
                # Initial MLP smallest batch is covered by the sign statement instead.
                if (task,state,c['batch'])!=('mlp',0,32):
                    assert value in sgd,value
                    counts+=1
            if state==2000:
                assert all(small>large for small,large in zip(cells[0]['seed_means']['mean_rho'],cells[-1]['seed_means']['mean_rho']))
    for c in r['sgd_cells']:
        if c['task']=='mlp' and c['state']==0 and c['batch']<=512:
            assert all(v<0 for v in c['seed_means']['mean_rho'])
        if c['task']=='transformer' and c['batch']==8:
            assert f"{c['mean']['loss_change']:+.3f}".replace('+','') in sgd

    # Compare retained arrays with the earlier verified figure inputs.
    oldbase=ROOT/'analyses/024-2026-10-17-eos-gd-lr-calibration'
    oldmanifest=read(oldbase/'single_rows_figure_manifest.json')
    assert sha(ROOT/oldmanifest['arrays'])==oldmanifest['arrays_sha256']
    old=np.load(ROOT/oldmanifest['arrays'])
    matches=0;maximum=0.
    for t in r['trajectories']:
        rid=t['run']
        for tag,oldtag in [('loss','loss_log'),('rho','rho_smoothed'),('norm','gradient_norms_smoothed'),('R','b_over_a'),('q','eta_kappa_hat_smoothed')]:
            key=f'{oldtag}_run{rid}_y'
            if key not in old:
                # The old registry calls its ratio row normalized_displacement.
                key=f'normalized_displacement_run{rid}_y' if tag=='R' else key
            assert key in old,(tag,key,list(old.files)[:12])
            values=p[f'{tag}_{rid}_y']
            np.testing.assert_allclose(values,old[key],rtol=3e-11,atol=3e-11)
            maximum=max(maximum,float(np.max(np.abs(values-old[key]))));matches+=len(values)
        for offset in [0,1]:
            vals=a[f'{rid}_C{offset}']*a[f'{rid}_D{offset}']
            np.testing.assert_allclose(vals,p[f'R_{rid}_y'][1::2],rtol=2e-15)
    table=(DRAFT/'tables/analysis056_pairs.tex').read_text()
    for e in r['endpoints']:
        for field in ['C','D','Rperp']:
            assert f'{e[field]:.4f}' in table
    report=dict(status='passed',displayed_scalar_checks=counts,retained_curve_points=matches,
                max_retained_curve_difference=maximum,three_sweeps_covering_nine_conditions=True,
                endpoint_and_lag_claims=True,results_sha256=sha(A/'results.json'),
                script_sha256=sha(__file__))
    (HERE/'evidence_checks.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
