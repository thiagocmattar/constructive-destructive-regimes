"""Independent mathematical and raw-evidence checks; synthetic cases are tests."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(name): return list(csv.DictReader((HERE/name).open(encoding='utf-8')))


def mathematics():
    rng=np.random.default_rng(571105)
    counts={'quadratic_steps':0,'nonquadratic_steps':0,'memory_recurrence':0,'exact_alternation':0}
    worst=0.
    for angle in [0., .23, .71, 1.13]:
        rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        h=rotation@np.diag([.5,7.])@rotation.T
        for beta in [0.,.5,.9]:
            x=rotation@np.array([1.,1.])/np.sqrt(7.5)
            assert np.isclose(.5*x@h@x,.5)
            m=np.zeros(2);v=np.zeros(2)
            for t in range(1,101):
                g=h@x; m=beta*m+(1-beta)*g;v=.999*v+.001*g*g
                mh=m/(1-beta**t);P=1/(np.sqrt(v/(1-.999**t))+1e-8)
                d=P*mh;eta=.013;c=(1-beta)/((1+beta)*(1-beta**t));e=mh-c*g
                following=h@(x-eta*d)
                z=1-(g@following)/(g@g)
                lam=c*(P*g)@(h@g)/(g@g)
                memory=(P*e)@(h@g)/(g@g)
                error=abs(z/eta-lam-memory)/max(1.,abs(z/eta),abs(lam),abs(memory))
                worst=max(worst,error); assert error<1e-10
                if beta==0: assert np.linalg.norm(e)==0 and abs(memory)==0
                x-=eta*d;counts['quadratic_steps']+=1
    for beta in [0.,.5,.9,.99]:
        a=(1-beta)/(1+beta);m=np.zeros(7);prev_g=None;prev_r=None
        g0=rng.normal(size=7);ma=np.zeros(7)
        for t in range(1,101):
            g=rng.normal(size=7)*np.exp(rng.normal())
            m=beta*m+(1-beta)*g;r=m-a*g
            if t>=2:
                np.testing.assert_allclose(r,beta*prev_r+beta*a*(g+prev_g),rtol=2e-12,atol=2e-12)
                counts['memory_recurrence']+=1
            alt=(-1)**(t-1)*g0;ma=beta*ma+(1-beta)*alt
            np.testing.assert_allclose(ma,a*(1-(-beta)**t)*alt,rtol=1e-12,atol=1e-12)
            counts['exact_alternation']+=1;prev_r=r;prev_g=g
    nodes,weights=np.polynomial.legendre.leggauss(4)
    for _ in range(100):
        b=rng.normal(size=(5,5));h=b.T@b+.2*np.eye(5);q=.3
        x=rng.normal(size=5);g=h@x+q*x**3;H=h+np.diag(3*q*x*x)
        P=np.exp(rng.normal(size=5));mh=rng.normal(size=5);d=P*mh;eta=.03
        c=.0527;e=mh-c*g;lam=c*(P*g)@(H@g)/(g@g);memory=(P*e)@(H@g)/(g@g)
        endpoint=x-eta*d;gg=h@endpoint+q*endpoint**3;z=1-g@gg/(g@g)
        Hbar=sum(w/2*(h+np.diag(3*q*(x-(u+1)/2*eta*d)**2)) for u,w in zip(nodes,weights))
        finite=g@(Hbar-H)@d/(g@g)
        np.testing.assert_allclose(z/eta,lam+memory+finite,rtol=2e-11,atol=2e-11)
        lipschitz=6*q*max(np.max(np.abs(x)),np.max(np.abs(endpoint)))
        bound=eta*np.linalg.norm(H@g)*np.linalg.norm(P*e)/(g@g)+lipschitz*eta**2*np.linalg.norm(d)**2/(2*np.linalg.norm(g))
        assert abs(z-eta*lam)<=bound+1e-12
        # Secant convergence checked against an independent endpoint evaluation.
        errors=[]
        for alpha in [1.,.5,.25,1e-4]:
            trial=x-alpha*eta*d;gt=h@trial+q*trial**3
            value=g@(g-gt)/(alpha*eta*(g@g))
            errors.append(abs(value-g@H@d/(g@g)))
        assert errors[-1]<max(errors[:3])+1e-10
        counts['nonquadratic_steps']+=1
    return dict(**counts,maximum_relative_quadratic_identity_error=worst)


def main():
    result=json.loads((HERE/'results.json').read_text())
    for rel,digest in result['sources'].items():assert sha(ROOT/rel)==digest,rel
    for rel,digest in result['outputs'].items():assert sha(HERE/rel)==digest,rel
    raw_results=json.loads((ROOT/'runs/215-2026-11-02-controlled-full-batch-adam/results.json').read_text())
    curves={};events={};phase_checks=0
    for condition in raw_results['conditions']:
        arm=f"{condition['task']}:{condition['lr']:g}";folder=ROOT/condition['source']
        boundaries=[json.loads(s) for s in (folder/'boundaries.jsonl').read_text().splitlines()]
        metrics=[json.loads(s) for s in (folder/'metrics.jsonl').read_text().splitlines()]
        loss=np.array([b['loss'] for b in boundaries]);curves[arm]=(loss,boundaries,metrics)
        found=[];start=0
        # Independently segment each loss history without reading event boundaries.
        while start<1024:
            if loss[start+1]<=loss[start]:start+=1;continue
            c=start+1
            while c<=1024 and loss[c]>loss[start]:c+=1
            end=min(c,1024);p=start+1+int(np.argmax(loss[start+1:c]))
            found.append(dict(a=start,p=p,end=end,censored=c==1025,ratio=float(loss[p]/loss[start])))
            start=end
        events[arm]=found
    assert sum(len(es) for es in events.values())==812
    incidence=rows('event_incidence.csv')
    for row in incidence:
        loss,_,_=curves[row['arm']];selected=[e for e in events[row['arm']] if e['ratio']>1+float(row['threshold'])]
        rec=[e for e in selected if not e['censored']]
        occupied={s for e in selected for s in range(e['a'],e['end'])}
        assert len(rec)==int(row['recovered'])
        assert len(selected)-len(rec)==int(row['censored'])
        assert len(occupied)==int(row['occupied_updates'])
        excess=sum(sum(max(0.,loss[s]-loss[e['a']]) for s in range(e['a']+1,e['end']+1)) for e in selected)
        np.testing.assert_allclose(excess,float(row['observed_loss_excess_sum']),rtol=1e-12,atol=1e-12)
    features=rows('event_features.csv')
    retained=list(csv.DictReader((ROOT/'analyses/036-2026-11-02-controlled-adam-diagnostics/spike_dataset/loss_spikes.csv').open()))
    event_map={e['spike_id']:e for e in retained}
    for row in features:
        e=event_map[row['spike_id']];_,bs,ms=curves[row['arm']];a=int(e['before_state']);p=int(e['peak_state'])
        np.testing.assert_allclose(float(row['A']),bs[p]['gradient_norm']/bs[a]['gradient_norm'],rtol=1e-14)
        assert float(row['rho'])==ms[a]['streams']['raw']['cosine']
        np.testing.assert_allclose(float(row['log_loss_ratio']),np.log(bs[p]['loss']/bs[a]['loss']),rtol=1e-14)
    for row in rows('event_phase_statistics.csv'):
        e=event_map[row['spike_id']];loss,_,ms=curves[row['arm']]
        a,p,c=[int(e[k]) for k in ['before_state','peak_state','recovery_state']]
        states=range(a,p) if row['phase']=='rise' else range(p,c)
        assert len(states)==int(row['updates'])
        assert sum(ms[s]['gradient_actual_step_cosine']<0 for s in states)==int(row['locally_descending'])
        assert sum(loss[s+1]>loss[s] for s in states)==int(row['loss_increasing'])
        phase_checks+=len(states)
    for row in rows('cross_alignments.csv'):
        _,_,ms=curves[row['arm']];m=ms[int(row['state'])]
        for dest,source in [('gradient_moment','gradient_current_moment_cosine'),('gradient_direction','gradient_adaptive_cosine'),('gradient_dot_actual_step','gradient_dot_actual_step')]:
            assert float(row[dest])==m[source]
    regression_checks=0
    for f in json.loads((HERE/'regressions.json').read_text()):
        test=[r for r in features if r['arm']==f['arm']]
        train=test if f['scope']=='in_sample' else [r for r in features if r['task']==f['task'] and r['arm']!=f['arm']]
        def matrix(es):
            table=[]
            for e in es:
                a=float(e['A']);rho=float(e['rho'])
                v={'A':a,'rho':rho,'rho*A':rho*a,'logA':np.log(a),'rho*logA':rho*np.log(a)}
                table.append([1]+[v[k] for k in f['model'].split('+')])
            return np.array(table)
        if f['status']!='defined':continue
        x=matrix(train);y=np.array([float(r['log_loss_ratio']) for r in train])
        # QR solution is independent of the generator's SVD least-squares fit.
        q,r=np.linalg.qr(x,mode='reduced');coef=np.linalg.solve(r,q.T@y)
        np.testing.assert_allclose(coef,f['coefficients'],rtol=1e-8,atol=1e-10)
        yy=np.array([float(r['log_loss_ratio']) for r in test]);res=yy-matrix(test)@coef
        np.testing.assert_allclose(np.mean(res**2),f['mse'],rtol=1e-9,atol=1e-12)
        regression_checks+=1
    report=dict(status='passed',source_hashes=len(result['sources']),outputs=len(result['outputs']),
        independent_loss_segmentations=9,event_boundaries=812,incidence_rows=len(incidence),
        severity_events=len(features),phase_update_checks=phase_checks,cross_alignment_states=9216,
        independent_QR_fits=regression_checks,mathematics=mathematics(),
        results_sha256=sha(HERE/'results.json'),verifier_sha256=sha(HERE/'02_verify.py'))
    (HERE/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(report))


if __name__=='__main__':main()
