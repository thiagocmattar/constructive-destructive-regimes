"""Independent frozen-layer checks and matched temporal reductions, using all examples."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
RUN=ROOT/'runs/244-2026-11-07-mlp-freeze-first-linear'
RAW=RUN/'artifacts/spark/payload/worker'
PREVIOUS=HERE.parent/'063-2026-11-07-architecture-instability-tracing'
spec=importlib.util.spec_from_file_location('common',PREVIOUS/'01_reduce.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s]
def clean(values):return [float(x) if np.isfinite(x) else None for x in values]


def gram(path,raw_gradient=False):
    a=np.load(path,mmap_mode='r');flat=a.reshape(len(a),-1)
    g=np.zeros((52,52),dtype=np.float64)
    for start in range(0,flat.shape[1],32768):
        # Delta449..500 needs states449..501; raw g449..500 is used directly.
        block=np.asarray(flat[2:54 if raw_gradient else 55,start:start+32768],dtype=np.float64)
        d=block if raw_gradient else np.diff(block,axis=0)
        assert len(d)==52;g+=d@d.T
    return g


def main():
    receipt=json.loads((RUN/'retrieval.json').read_text());assert receipt['verified']
    inventory=json.loads((RAW/'inventory.json').read_text())
    for item in inventory['files']:
        p=RAW/item['path'];assert p.stat().st_size==item['bytes'] and r.sha(p)==item['sha256'],str(p)
    for f in ['qualification.json','freeze_verification.json']:
        assert json.loads((RAW/f).read_text())['status']=='passed'
    original=json.loads((PREVIOUS/'mlp_results.json').read_text())
    event_control=json.loads((PREVIOUS/'mlp_450_500_results.json').read_text())
    assert original['verification']['status']==event_control['verification']['status']=='passed'
    for relative,digest in event_control['sources'].items():assert r.sha(ROOT/relative)==digest,relative
    boundary={q['state']:q for q in rows(RAW/'boundaries.jsonl')}
    control_boundary={q['state']:q for q in original['boundaries']}
    parent=ROOT/'runs/223-2026-11-04-multispike-interventions/artifacts/spark/payload/worker'
    native_control={q['state']:q for q in rows(parent/'prefix/boundaries.jsonl')[:-1]+rows(parent/'control/boundaries.jsonl')}
    for t in range(448,801):
        assert native_control[t]['loss']==control_boundary[t]['loss']
        np.testing.assert_allclose(native_control[t]['raw_gradient_norm'],
            original['curves']['raw_gradient']['change_rms'][t-448]*np.sqrt(656810),rtol=1e-12,atol=1e-12)
    # Independent pointwise recovery from the actual full raw-gradient vectors.
    frozen_g=np.asarray(np.load(RAW/'raw_gradient.npy',mmap_mode='r'),dtype=np.float64)
    prior_g=np.asarray(np.load(ROOT/'runs/240-2026-11-07-mlp-hidden-change-svd/artifacts/spark/payload/worker/gradients.npy',mmap_mode='r')[:55],dtype=np.float64)
    for arm,vectors,records in [('control',prior_g,native_control),('freeze',frozen_g,boundary)]:
        for idx in range(1,len(vectors)):
            state=447+idx;g,previous=vectors[idx],vectors[idx-1]
            q=1-float(g@previous)/float(previous@previous)
            np.testing.assert_allclose(q,records[state]['eta_kappa_hat'],rtol=1e-9,atol=1e-10)
            np.testing.assert_allclose(np.linalg.norm(g),records[state]['raw_gradient_norm'],rtol=1e-12,atol=1e-12)
    rep={q['state']:q['streams'] for q in rows(RAW/'representations.jsonl')}
    updates=rows(RAW/'updates.jsonl')
    assert list(boundary)==list(range(802)) and list(rep)==list(range(447,801))
    assert [q['state'] for q in updates]==list(range(448,801))
    assert all(q['lr']==.001 and q['layers']['1']['actual_increment_norm']==0 for q in updates)
    for t in range(448,801):
        for n in ['preactivation1','hidden1']:
            assert rep[t][n]['change_rms']==0 and rep[t][n]['rho'] is None
    for name in ['preactivation1','hidden1']:
        a=np.load(RAW/(name+'.npy'),mmap_mode='r')
        assert all(np.array_equal(a[1],a[i]) for i in range(2,len(a))),name

    # Control gradients restricted to the same remaining 42,210 parameters.
    prior_g_path=ROOT/'runs/240-2026-11-07-mlp-hidden-change-svd/artifacts/spark/payload/worker/gradients.npy'
    previous_g=np.load(prior_g_path,mmap_mode='r')
    ga=np.asarray(previous_g[:,614600:],dtype=np.float64)
    norms=np.linalg.norm(ga,axis=1);rho=np.sum(ga[1:]*ga[:-1],axis=1)/(norms[1:]*norms[:-1])
    control_active={t:dict(norm=float(norms[t-447]),rho=float(rho[t-448])) for t in range(448,801)}

    names=[s['name'] for s in original['stages']]
    with threadpool_limits(limits=4):
        grams={n:gram(RAW/(n+'.npy'),n=='raw_gradient') for n in names}
        fits={n:r.decompose(g) for n,g in grams.items()}
    event={};checks={}
    for name in names:
        fit=fits[name];g=grams[name];p=next(s['coordinates'] for s in original['stages'] if s['name']==name)
        cosine=r.cosine(g);rms=np.sqrt(np.maximum(np.diag(g)[1:],0)/p)
        expected=[boundary[t]['rho'] if name=='raw_gradient' else rep[t][name]['rho'] for t in range(450,501)]
        expected=np.array([np.nan if q is None else q for q in expected])
        np.testing.assert_allclose(cosine,expected,rtol=1e-9,atol=1e-10,equal_nan=True)
        expected_rms=[boundary[t]['raw_gradient_norm']/np.sqrt(p) if name=='raw_gradient' else rep[t][name]['change_rms'] for t in range(450,501)]
        np.testing.assert_allclose(rms,expected_rms,rtol=1e-9,atol=1e-12)
        independent=np.linalg.svd(g[1:,1:],compute_uv=False)
        np.testing.assert_allclose(fit['values'],independent,rtol=1e-6,atol=max(independent[0],1)*1e-10)
        defined=np.isfinite(cosine);energy=fit['energy'];alpha=fit['coeff'][:,0]
        event[name]=dict(coordinates=p,rho=clean(cosine),change_rms=rms.tolist(),alpha=alpha.tolist(),
            spectrum=fit['fractions'].tolist() if energy else None,energy=energy,
            top1=float(fit['fractions'][0]) if energy else None,top10=float(fit['fractions'][:10].sum()) if energy else None,
            rank95=int(np.searchsorted(np.cumsum(fit['fractions']),.95)+1) if energy else 0,
            negative=int(np.sum(cosine<0)),defined=int(defined.sum()),
            leading_sign_flips=int(np.sum(alpha[1:]*alpha[:-1]<0)) if energy else None)
        checks[name]=dict(cosine_and_rms_match_streaming=True,independent_gram_singular_values=True,
            gram_reconstruction_relative_error=float(np.linalg.norm(fit['coeff']@fit['coeff'].T-g[1:,1:])/energy) if energy else 0.)
    long_curves={};metrics=[]
    for a,b in [(450,500),(448,800)]:
        tt=np.arange(a,b+1)
        for arm in ['control','freeze']:
            loss=np.array([(control_boundary if arm=='control' else boundary)[int(t)]['loss'] for t in tt])
            change=np.diff(loss);peak=int(tt[np.argmax(loss)])
            motion=np.array([original['curves']['output']['change_rms'][t-448] if arm=='control' else rep[t]['output']['change_rms'] for t in tt])
            rr=np.array([original['curves']['output']['rho'][t-448] if arm=='control' else rep[t]['output']['rho'] for t in tt])
            metrics.append(dict(window=f'{a}-{b}',arm=arm,states=len(tt),loss_mean=float(loss.mean()),loss_max=float(loss.max()),
                peak_state=peak,loss_final=float(loss[-1]),positive_loss_variation=float(np.maximum(change,0).sum()),
                largest_upward_loss_change=float(max(0,change.max())),positive_loss_transitions=int(np.sum(change>0)),
                output_negative=int(np.sum(rr<0)),output_rms_max=float(motion.max()),output_rms_energy=float(np.sum(motion**2))))
    t=list(range(448,801))
    for arm in ['control','freeze']:
        long_curves[arm]=dict(loss=[(control_boundary if arm=='control' else boundary)[i]['loss'] for i in t],
            eta_kappa_hat=[(native_control if arm=='control' else boundary)[i]['eta_kappa_hat'] for i in t],
            raw_gradient_norm=[(native_control if arm=='control' else boundary)[i]['raw_gradient_norm'] for i in t],
            gradient_rho=[original['curves']['raw_gradient']['rho'][i-448] if arm=='control' else boundary[i]['rho'] for i in t],
            active_gradient_rho=[control_active[i]['rho'] if arm=='control' else boundary[i]['active_gradient_rho'] for i in t],
            active_gradient_norm=[control_active[i]['norm'] if arm=='control' else boundary[i]['active_gradient_norm'] for i in t],
            layers={n:dict(rho=[original['curves'][n]['rho'][i-448] if arm=='control' else rep[i][n]['rho'] for i in t],
                change_rms=[original['curves'][n]['change_rms'][i-448] if arm=='control' else rep[i][n]['change_rms'] for i in t]) for n in names if n!='raw_gradient'})
    sources=[RUN/'config.json',RUN/'input_manifest.json',RUN/'retrieval.json',RAW/'inventory.json',
             PREVIOUS/'mlp_results.json',PREVIOUS/'mlp_450_500_results.json',PREVIOUS/'01_reduce.py',prior_g_path,
             parent/'prefix/boundaries.jsonl',parent/'control/boundaries.jsonl']
    result=dict(status='verified',states=t,primary_states=list(range(450,501)),metrics=metrics,curves=long_curves,
        event=dict(control=event_control,freeze=event),fork=448,
        point474=dict(control=control_boundary[474]['loss'],freeze=boundary[474]['loss']),
        parameter_count=dict(frozen=614600,total=656810,active=42210),
        qualification=json.loads((RAW/'qualification.json').read_text()),freeze_verification=json.loads((RAW/'freeze_verification.json').read_text()),
        independent_verification=checks,gradient_diagnostics_verification=dict(status='passed',
            vector_checked_states=[448,501],arms=2,full_gradient_norm_and_eta_kappa_hat=True),
        sources={p.relative_to(ROOT).as_posix():r.sha(p) for p in sources},generator_sha256=r.sha(Path(__file__)))
    r.dump(HERE/'comparison.json',result)
    with (HERE/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]),lineterminator='\n');writer.writeheader();writer.writerows(metrics)
    (HERE/'artifacts').mkdir(exist_ok=True);np.savez(HERE/'artifacts/freeze_event_grams.npz',**grams)
    print(json.dumps(dict(metrics=metrics,point474=result['point474'],event={n:{k:v for k,v in s.items() if k in ['negative','defined','top1','top10','rank95','leading_sign_flips']} for n,s in event.items()},checks=checks),indent=2))


if __name__=='__main__':main()
