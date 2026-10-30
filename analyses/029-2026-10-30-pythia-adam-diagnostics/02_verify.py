"""Independent scalar, plotting-array and PDF checks for Analysis029."""
from pathlib import Path
import hashlib
import json
import math
import re
import subprocess
import fitz
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    result=json.loads((HERE/'results.json').read_text(encoding='utf-8'))
    assert result['script_sha256']==sha(HERE/'01_analyze_plot.py')
    for name,digest in result['sources'].items():assert sha(ROOT/name)==digest,name
    assert result['plot_arrays_sha256']==sha(HERE/'plot_arrays.npz')
    cache={}
    for name in result['sources']:
        if name.endswith('events.jsonl'):
            events=[json.loads(s) for s in (ROOT/name).read_text(encoding='utf-8').splitlines()]
            cache[name]=events
    data={}
    metadata={(r['cohort'],r['run']):r for r in result['records']}
    errors=[]
    for key,r in metadata.items():
        raw=cache[r['event_source']]
        updates=[e for e in raw if e['event']=='update']
        if r['cohort']=='schedule':
            parent=metadata['size',r['parent_run']]
            updates=[e for e in cache[parent['event_source']] if e['event']=='update'][:356]+updates
        assert len(updates)==712 and [e['update'] for e in updates]==list(range(1,713))
        a={k:[] for k in ['loss','lr','tokens','g_norm','g_cos','g_R','d_norm','d_cos','d_R','q']}
        if 'momentum' in updates[0]:
            for k in ['m_norm','m_cos','m_R','input_m_cos','input_memory_ratio']:a[k]=[]
        for i,e in enumerate(updates):
            for k,field in [('loss','loss'),('lr','learning_rate'),('tokens','supervised_tokens'),
                            ('g_norm','gradient_norm'),('g_cos','adjacent_cosine'),('g_R','cumulative_ratio')]:
                a[k].append(float(e[field]) if e[field] is not None else np.nan)
            for k,field in [('d_norm','direction_norm'),('d_cos','adjacent_cosine'),('d_R','cumulative_ratio')]:
                value=e['adaptive'][field];a[k].append(float(value) if value is not None else np.nan)
            if not i:a['q'].append(np.nan)
            else:
                before=updates[i-1]['gradient_norm'];after=e['gradient_norm']
                overlap=before*after*e['adjacent_cosine']
                a['q'].append((before*before-overlap)/(before*before))
                assert (a['q'][-1]>1)==(e['adjacent_cosine']<0)
            if 'momentum' in e:
                m=e['momentum']
                for k,field in [('m_norm','momentum_norm'),('m_cos','adjacent_cosine'),('m_R','cumulative_ratio')]:
                    a[k].append(float(m[field]) if m[field] is not None else np.nan)
                if not i:
                    a['input_m_cos'].append(np.nan);a['input_memory_ratio'].append(np.nan)
                else:
                    prev=updates[i-1]['momentum']['momentum_norm'];now=m['momentum_norm']
                    # Resolve the current moment into components parallel/perpendicular
                    # to its predecessor; invert the EMA in this two-dimensional plane.
                    parallel=(now*m['adjacent_cosine']-.9*prev)/.1
                    perpendicular=now*math.sqrt(1-m['adjacent_cosine']**2)/.1
                    input_norm=math.hypot(parallel,perpendicular)
                    a['input_m_cos'].append(parallel/input_norm)
                    a['input_memory_ratio'].append(.1*input_norm/(.9*prev))
                    expected=e['gradient_norm']*min(1,1/(e['gradient_clip_returned_norm']+1e-6))
                    errors.append(abs(input_norm/expected-1))
        a={k:np.array(v) for k,v in a.items()}
        for window,stats in r['windows'].items():
            first,last={'full':(2,712),'full_batches':(2,711),'late':(612,711),'affected':(358,712)}[window]
            for metric,value in stats.items():
                np.testing.assert_allclose(np.median(a[metric][first-1:last]),value,rtol=2e-12,atol=1e-12)
        final=[e for e in raw if e['event'] in ['final_validation','validation']][-1]
        assert final['loss']==r['validation'][-1]['loss'] and final['supervised_tokens']==693329
        data[key]=a
    assert max(errors)<5e-6 and len(errors)==4977

    arrays=np.load(HERE/'plot_arrays.npz',allow_pickle=False)
    finite_points=0
    for spec in result['curves']:
        ys=[]
        for rid in spec['runs']:
            key=(spec['cohort'],rid);a=data[key];metric=spec['metric']
            if metric=='validation':
                val=metadata[key]['validation'];x=np.array([v['update'] for v in val]);y=np.array([v['loss'] for v in val])
            else:
                first,last=spec['first'],spec['last'];x=np.arange(first,last+1)
                raw=a[metric][first-1:last];y=raw.copy()
                if spec['smooth']:
                    y[:]=np.nan
                    for i in range(24,len(y)):
                        sample=raw[i-24:i+1]
                        if np.isfinite(sample).all():
                            weights=a['tokens'][first-1+i-24:first+i] if metric=='loss' else np.ones(25)
                            y[i]=math.fsum(float(v*w) for v,w in zip(sample,weights))/math.fsum(weights)
            ys.append(y)
        agg=spec['aggregation']
        expected=ys[0] if agg=='individual' else getattr(np.array(ys),agg)(axis=0)
        np.testing.assert_array_equal(arrays[spec['key']+'_x'],x)
        np.testing.assert_allclose(arrays[spec['key']+'_y'],expected,rtol=2e-12,atol=1e-12,equal_nan=True)
        finite_points+=int(np.isfinite(expected).sum())

    # Direct-vector checks, including clipped inputs, independently establish
    # that the scalar reconstruction recovers the intended comparison.
    rng=np.random.default_rng(1234);m=np.zeros(29);max_cos_error=0.;max_norm_error=0.
    for _ in range(100):
        g=rng.normal(size=29)*float(rng.uniform(.1,5));h=g*min(1,1/(np.linalg.norm(g)+1e-6))
        new=.9*m+.1*h
        if np.linalg.norm(m)>0:
            p,n=np.linalg.norm(m),np.linalg.norm(new);c=np.dot(m,new)/(p*n)
            parallel=(n*c-.9*p)/.1;perpendicular=n*np.sqrt(1-c*c)/.1
            norm=np.hypot(parallel,perpendicular)
            max_norm_error=max(max_norm_error,abs(norm-np.linalg.norm(h)))
            max_cos_error=max(max_cos_error,abs(parallel/norm-np.dot(h,m)/(np.linalg.norm(h)*p)))
        m=new
    assert max_norm_error<1e-12 and max_cos_error<1e-12
    # A fixed Hessian can give opposite alignment signs solely through step length.
    H=np.diag([1.,4.]);theta=np.array([2.,1.]);g=H@theta;direction=np.ones(2)
    response=[]
    for eta in [.1,1.5]:
        future=H@(theta-eta*direction)
        response.append(float(np.dot(g,future)/(np.linalg.norm(g)*np.linalg.norm(future))))
    assert response[0]>0>response[1]

    pdfs=[]
    for name,info in result['figures'].items():
        p=ROOT/info['path'];assert sha(p)==info['sha256'] and info['data_bounds_within_axes']
        with fitz.open(p) as doc:
            assert len(doc)==1
            for block in doc[0].get_text('blocks'):
                assert (doc[0].rect+(-.5,-.5,.5,.5)).contains(fitz.Rect(block[:4])),(name,block[:4])
        fonts=subprocess.run(['pdffonts',str(p)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True).stdout.decode()
        flags=re.findall(r'\s(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$',fonts,re.M)
        assert flags and all(f[0]=='yes' for f in flags)
        pdfs.append(dict(name=name,sha256=sha(p),pages=1,all_fonts_embedded=True,text_bounds_ok=True))
    report=dict(status='passed',script_sha256=sha(Path(__file__)),results_sha256=sha(HERE/'results.json'),
                source_event_files=len(cache),records=len(data),curves_checked=len(result['curves']),
                finite_plot_values_checked=finite_points,reconstructed_pairs=len(errors),
                max_clip_norm_relative_error=max(errors),synthetic_cosine_max_error=max_cos_error,
                synthetic_input_norm_max_error=max_norm_error,fixed_hessian_lr_cosines=response,pdfs=pdfs)
    (HERE/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:report[k] for k in ['status','source_event_files','curves_checked','finite_plot_values_checked','reconstructed_pairs']}))


if __name__=='__main__':main()
