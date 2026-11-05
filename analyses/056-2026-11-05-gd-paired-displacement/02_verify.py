"""Independent vector, high-precision scalar and aggregation checks."""
import importlib.util
import json
import math
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('analysis056', HERE/'01_analyze.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def main():
    r = m.read(HERE/'results.json')
    assert r['generator_sha256'] == m.sha(HERE/'01_analyze.py')
    for p, digest in r['sources'].items():
        assert m.sha(m.ROOT/p) == digest, p
    assert m.sha(HERE/'arrays.npz') == r['arrays_sha256']
    arrays = np.load(HERE/'arrays.npz')
    rng = np.random.default_rng(561105)
    examples = [np.ones((20, 3)), np.tile([[1.,0.],[-1.,0.]], (10,1)),
                np.eye(20), np.zeros((20, 3)),
                np.tile([[1.,.1],[-1.,.1]], (10,1))]
    examples += [rng.normal(size=(20, 7)) * rng.lognormal(size=(20, 1)) for _ in range(60)]
    for g in examples:
        norms = np.linalg.norm(g, axis=1)
        den = norms[1:]*norms[:-1]
        rho = np.r_[np.nan, np.divide(np.sum(g[1:]*g[:-1],axis=1), den,
                     out=np.zeros_like(den), where=den>0)]
        B = np.linalg.norm(np.cumsum(g, axis=0), axis=1)
        even, lengths, partitions = m.partitions(norms, rho, B)
        np.testing.assert_allclose(lengths, np.linalg.norm(g[:-1]+g[1:],axis=1), atol=2e-7, rtol=2e-12)
        for offset, (C,D,H) in enumerate(partitions):
            for j,T in enumerate(even):
                segments = [g[i]+g[i+1] for i in range(offset,T-1,2)]
                if offset:
                    segments = [g[0], *segments, g[T-1]]
                expected = sum(np.linalg.norm(v) for v in segments)
                assert abs(H[j]-expected) < 2e-7
                np.testing.assert_allclose(np.sum(segments,axis=0),g[:T].sum(axis=0), atol=1e-12)
                if expected == 0:
                    assert np.isnan(D[j]) and B[T-1] == 0
                elif norms[:T].sum():
                    assert abs(C[j]*D[j]-B[T-1]/norms[:T].sum()) < 1e-12
        if not norms.sum():
            assert all(np.isnan(p[0]).all() for p in partitions)
    checks, largest_decimal_error = 0, 0.
    for run in r['trajectories']:
        rid=run['run']; norms=arrays[f'{rid}_norm']; rho=arrays[f'{rid}_rho']
        A=arrays[f'{rid}_A']; B=arrays[f'{rid}_B']; energy=arrays[f'{rid}_energy']
        assert run['displacement_relative_error'] < 1e-5, run
        with localcontext() as ctx:
            ctx.prec = 45
            expected=[]
            for a,b,c in zip(norms[:-1],norms[1:],rho[1:]):
                a,b,c=map(lambda x:Decimal(str(x)),[a,b,c])
                squared=a*a+b*b+2*a*b*c
                assert squared>=0
                expected.append(float(squared.sqrt()))
        actual=arrays[f'{rid}_pair_lengths']
        largest_decimal_error=max(largest_decimal_error,float(np.max(np.abs(actual-expected))))
        np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=2e-11)
        for offset in [0,1]:
            endpoints=np.arange(2,3001,2)
            # Independent sequential sum, rather than the vectorized prefix formula.
            H=[]; total=0.
            for T in endpoints:
                if offset==0:
                    total+=expected[T-2]; H.append(total)
                else:
                    if T>2: total+=expected[T-3]
                    H.append(norms[0]+total+norms[T-1])
            np.testing.assert_allclose(arrays[f'{rid}_H{offset}'],H,rtol=2e-12,atol=2e-12)
        np.testing.assert_allclose(arrays[f'{rid}_Q'],(arrays[f'{rid}_R']/arrays[f'{rid}_Rperp'])**2,rtol=2e-15)
        assert run['all_pair']>0
        loss=arrays[f'{rid}_loss']
        for w in [w for w in r['loss_windows'] if w['run']==rid]:
            pairs=[loss[s+2]-loss[s] for s in range(w['covered_start'],w['covered_end'],2)]
            assert len(pairs)==w['pairs']
            assert sum(v<0 for v in pairs)==w['decreases']
            assert sum(v>0 for v in pairs)==w['increases']
            assert abs(math.fsum(pairs)-(loss[w['covered_end']]-loss[w['covered_start']]))<1e-13
        checks+=len(expected)

    # Independent block-first aggregation of every displayed SGD cell.
    forks=m.read(m.A037/'results.json')['forks']
    forks += [dict(f, task='mlp', state=f['checkpoint']) for f in
              m.read(m.ROOT/'analyses/027-2026-10-30-minibatch-noise/results.json')['moving']
              if f['checkpoint'] in [0,2000]]
    assert len(forks)==546
    for c in r['sgd_cells']:
        for metric in ['mean_rho','R','loss_change']:
            means=[]
            for seed in range(3):
                values=[f[metric] for f in forks if f['task']==c['task'] and f['state']==c['state']
                        and f['batch']==c['batch'] and f['seed']==seed]
                assert len(values)==c['repetitions_per_seed']
                means.append(math.fsum(values)/len(values))
            np.testing.assert_allclose(means,c['seed_means'][metric],rtol=2e-13,atol=2e-13)
            assert abs(math.fsum(means)/3-c['mean'][metric])<2e-13
            assert min(means)==c['minimum'][metric] or abs(min(means)-c['minimum'][metric])<2e-13
            assert abs(max(means)-c['maximum'][metric])<2e-13
    report=dict(status='passed', sources=len(r['sources']), synthetic_vector_cases=len(examples),
        scalar_pairs=checks, max_decimal_pair_norm_error=largest_decimal_error,
        sgd_trajectories=len(forks), sgd_cells=len(r['sgd_cells']),
        backward_pairs=r['coverage']['backward_pairs'],
        max_actual_displacement_relative_error=max(v['displacement_relative_error'] for v in r['trajectories']),
        results_sha256=m.sha(HERE/'results.json'), verifier_sha256=m.sha(__file__))
    m.save(HERE/'verification.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
