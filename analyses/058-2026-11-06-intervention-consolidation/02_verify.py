"""Independent read-back of every displayed scalar and source/PDF identity."""
from pathlib import Path
import csv
import hashlib
import json
import math

import fitz
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def records(rel):
    return [json.loads(s) for s in (ROOT/rel).read_text().splitlines() if s.strip()]


def main():
    m=json.loads((HERE/'manifest.json').read_text())
    assert sha(HERE/'01_figures.py')==m['generator_sha256']
    assert sha(HERE/'plotted_values.csv')==m['plotted_values_sha256']
    for rel,identity in m['sources'].items():
        p=ROOT/rel
        assert p.stat().st_size==identity['bytes'] and sha(p)==identity['sha256'],rel
    raw={};identities=0;loss_summary={}
    for key,d in m['series'].items():
        if d['kind']=='full_objective':
            prefix=records(d['prefix']);branch=records(d['branch'])
            assert prefix[-1]['state']==branch[0]['state']==d['fork']
            # A cross-hardware boundary can change the recomputed gradient/loss.
            # Preserve the qualified branch boundary rather than overwriting it.
            full=prefix[:-1]+branch
            assert [r['state'] for r in full]==list(range(d['end']+1))
            raw[key]={r['state']:dict(r) for r in full}
            for i,r in enumerate(full):
                prev=full[i-1]['raw_gradient_norm'] if i else None
                if prev and r.get('rho') is not None and i!=d['fork']:
                    expected=(prev-r['rho']*r['raw_gradient_norm'])/prev
                    assert math.isclose(expected,r['eta_kappa_hat'],rel_tol=1e-7,abs_tol=2e-7),(key,i)
                    identities+=1
            if d['metrics']:
                for r in records(d['metrics']):
                    raw[key][r['state']]['optimizer_input_norm']=r['optimizer_input_norm']
            losses=[r['loss'] for r in branch[1:]]
            loss_summary[key]=dict(first_state=d['fork']+1,last_state=d['end'],n=len(losses),
                mean=float(np.mean(losses)),last32_mean=float(np.mean(losses[-32:])),
                maximum=max(losses),final=losses[-1])
        else:
            prefix=[r for r in records(d['prefix']) if r['event']=='update' and r['update']<=356]
            branch=[r for r in records(d['branch']) if r['event']=='update']
            full=prefix+branch
            assert [r['update'] for r in full]==list(range(1,713))
            raw[key]={r['update']:r for r in full}
            assert full[-1]['supervised_tokens']<full[-2]['supervised_tokens']
    with (HERE/'plotted_values.csv').open() as f:values=list(csv.DictReader(f))
    grouped={i:[] for i in range(len(m['curves']))}
    for r in values:grouped[int(r['curve'])].append((int(r['state']),float(r['value'])))
    # Read the already-audited original plot tables independently of the generator.
    old_map={
        'mlp.control':(44,'mlp','unchanged'), 'mlp.half':(44,'mlp','fixed_half'),
        'resnet.control':(44,'resnet','unchanged'),'resnet.half':(44,'resnet','fixed_half'),
        'transformer.control':(44,'transformer','unchanged'),'transformer.half':(44,'transformer','fixed_half'),
        'mlp.cosinehalf':(47,None,'cosine32_half'),'mlp.cosine75':(47,None,'cosine32_three_quarters'),
        'mlp.half_return':(45,None,'fixed_half_return075'),'mlp.cosine_return':(46,None,'abrupt075'),
        'mlp.cosinerise':(46,None,'cosine70_rise'),'mlp.twostage':(50,None,'two_stage_decay'),
        'mlp.clip1':(52,None,'global_clip1'),'mlp.clip05':(52,None,'global_clip05'),
        'resnet.spark_control':(54,None,'Unclipped'),'resnet.clip1':(54,None,'Clipping at 1')}
    old={};old_sources={}
    for n in {v[0] for v in old_map.values()}:
        folder=next((ROOT/'analyses').glob(f'{n:03d}-*'))
        p=folder/('figure_values.csv' if n==44 else 'plotted_values.csv')
        old_sources[p.relative_to(ROOT).as_posix()]=sha(p)
        with p.open() as f:
            for r in csv.DictReader(f):
                k=(n,r.get('case'),r['series'],r['metric'],int(r['state']))
                val=float(r['value'])
                if k in old:assert old[k]==val
                old[k]=val
    raw_checks=old_checks=0
    for i,c in enumerate(m['curves']):
        d=m['series'][c['series']];points=grouped[i]
        assert len(points)==c['points'] and len(set(x for x,_ in points))==len(points)
        xx,yy=zip(*points)
        for axis,arr in [('x',xx),('y',yy)]:
            assert hashlib.sha256(np.asarray(arr,dtype='<f8').tobytes()).hexdigest()==c[axis+'_sha256']
        for state,value in points:
            r=raw[c['series']][state];metric=c['metric']
            if d['kind']=='full_objective':expected=r[metric]
            elif metric=='rho':expected=r['adjacent_cosine']
            elif metric=='raw_gradient_norm':expected=r['gradient_norm']
            elif metric=='applied_lr':expected=r['learning_rate']
            elif metric=='eta_kappa_hat':
                prev=raw[c['series']][state-1]['gradient_norm']
                expected=(prev-r['adjacent_cosine']*r['gradient_norm'])/prev
            else:expected=r[metric]
            if metric=='applied_lr':expected/=d['base_lr']
            assert math.isclose(value,expected,rel_tol=2e-14,abs_tol=2e-14),(c,state,value,expected)
            raw_checks+=1
            if c['series'] in old_map:
                n,case,arm=old_map[c['series']]
                assert value==old[(n,case,arm,metric,state)],(c,state)
                old_checks+=1
        if c['terminal_interval_closed']:
            assert c['metric']=='applied_lr' and max(xx)==d['end']-1
    pdfs=[]
    for name,o in m['outputs'].items():
        p=ROOT/o['path'];assert sha(p)==o['sha256']
        with fitz.open(p) as doc:
            assert len(doc)==1 and not doc[0].get_images()
            assert abs(doc[0].rect.width-7.16*72)<.1
            for word in doc[0].get_text('words'):assert doc[0].rect.contains(fitz.Rect(word[:4]))
            for font in doc[0].get_fonts(full=True):assert font[2]!='Type3' and doc.extract_font(font[0])[3]
        if 'copied_from' in o:assert p.read_bytes()==(ROOT/o['copied_from']).read_bytes()
        pdfs.append(name)
    assert len(pdfs)==7 and raw_checks==m['points']==69354
    # Confirm requested panel/metric selections, rather than only array values.
    f2=[c for c in m['curves'] if c['figure'].startswith('02-')]
    assert len(f2)==24 and {c['metric'] for c in f2}=={'loss','eta_kappa_hat','raw_gradient_norm','applied_lr'}
    assert {c['series'] for c in f2 if c['column']==2}=={'mlp.half','mlp.half_return'}
    f1=[c for c in m['curves'] if c['figure'].startswith('01-')]
    assert all('cosine' not in c['series'] for c in f1)
    report=dict(status='passed',raw_scalar_checks=raw_checks,unchanged_original_plot_values=old_checks,
        projected_change_identities=identities,source_files=len(m['sources']),pdfs=pdfs,
        original_plot_tables=old_sources,figure00='byte-identical to verified Analysis055 original',
        notes=['Pythia includes raw tail update 712 and undefined first cosine/q is omitted.',
               'ResNet clipping uses fresh same-Spark control; shaded prefix is historical RTX4090.',
               'No smoothing; no raw vector, training state or scientific input modified.'])
    (HERE/'verification.json').write_text(json.dumps(report,indent=2)+'\n',newline='\n')
    (HERE/'loss_summary.json').write_text(json.dumps(loss_summary,indent=2)+'\n',newline='\n')
    print(json.dumps({k:report[k] for k in ['status','raw_scalar_checks','unchanged_original_plot_values','projected_change_identities','source_files']}))


if __name__=='__main__':main()
