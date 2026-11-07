"""Standardized exact-Gram SVD and architecture tables for both fixed probes."""
import csv
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
RUN=ROOT/'runs/243-2026-11-07-architecture-instability-tracing'

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',newline='\n')
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x]
def cosine(g):
    norm=np.sqrt(np.maximum(np.diag(g),0));den=norm[1:]*norm[:-1]
    return np.divide(np.diag(g,1),den,out=np.full(len(den),np.nan),where=den>0)
def decompose(g):
    fitted=(g[1:,1:]+g[1:,1:].T)/2
    values,u=np.linalg.eigh(fitted);order=np.argsort(values)[::-1];values=values[order];u=u[:,order]
    assert values.min()>=-1e-9*max(values.max(),1)
    values=np.maximum(values,0);energy=values.sum();rank=int(np.sum(values>values[0]*1e-10)) if energy else 0
    coeff=u*np.sqrt(values)[None,:]
    for k in range(len(values)):
        if coeff[np.argmax(abs(coeff[:,k])),k]<0:coeff[:,k]*=-1;u[:,k]*=-1
    if energy:
        fractions=values/energy
        np.testing.assert_allclose(np.trace(fitted),energy,rtol=1e-9,atol=1e-7)
        np.testing.assert_allclose(coeff@coeff.T,fitted,rtol=1e-7,atol=1e-7*max(np.max(abs(fitted)),1))
    else:fractions=np.zeros(len(values))
    return dict(values=values,u=u,coeff=coeff,fractions=fractions,rank=rank,energy=energy)

def main(case_filter=None):
    cfg=json.loads((RUN/'config.json').read_text());summary={}
    for case,sp in cfg['cases'].items():
        if case_filter and case!=case_filter:continue
        raw=RUN/'artifacts/cloud'/case
        stages=json.loads((raw/'stages.json').read_text());g=dict(np.load(raw/'grams.npz'));levels=dict(np.load(raw/'levels.npz'))
        target=np.load(raw/'target_readout.npz');target_meta=json.loads((raw/'target_readout_verification.json').read_text());assert target_meta['status']=='verified'
        stages['output.correct_label_log_probability']=dict(shape=target_meta['shape'],coordinates=target_meta['shape'][0])
        g['output.correct_label_log_probability']=target['gram'];levels['output.correct_label_log_probability']=target['levels']
        boundary=rows(raw/'boundaries.jsonl');lookup={r['state']:r for r in boundary};times=np.arange(sp['start'],sp['end']+1)
        fits={name:decompose(g[name]) for name in stages};output=fits['output.log_probabilities'];records=[];curves={}
        for number,(name,meta) in enumerate(stages.items()):
            f=fits[name];rho=cosine(g[name]);assert len(rho)==len(times)
            defined=np.isfinite(rho);out_rho=cosine(g['output.log_probabilities']);both=defined&np.isfinite(out_rho)
            corr=float(np.corrcoef(rho[both],out_rho[both])[0,1]) if both.sum()>1 and np.std(rho[both])>0 and np.std(out_rho[both])>0 else None
            k=min(10,f['rank'],output['rank']); overlap=float(np.sum((f['u'][:,:k].T@output['u'][:,:k])**2)/k) if k else None
            out_capture=float(np.trace(f['u'][:,:k].T@g['output.log_probabilities'][1:,1:]@f['u'][:,:k])/output['energy']) if k and output['energy'] else None
            rms=np.sqrt(np.maximum(np.diag(g[name])[1:],0)/meta['coordinates'])
            alpha=f['coeff'][:,0]
            record=dict(index=number,name=name,shape=meta['shape'],coordinates=meta['coordinates'],negative=int(np.sum(rho<0)),defined=int(defined.sum()),
                top1=float(f['fractions'][0]) if f['energy'] else None,top10=float(f['fractions'][:10].sum()) if f['energy'] else None,
                rank95=int(np.searchsorted(np.cumsum(f['fractions']),.95)+1) if f['energy'] else 0,
                rank99=int(np.searchsorted(np.cumsum(f['fractions']),.99)+1) if f['energy'] else 0,
                leading_sign_flips=int(np.sum(alpha[1:]*alpha[:-1]<0)),correlation_with_output=corr,
                temporal_top10_overlap=overlap,output_energy_captured=out_capture,change_rms_min=float(rms.min()),change_rms_max=float(rms.max()),
                vector_kind='raw gradient' if name=='raw_gradient' else 'representation displacement',
                activation_rms_min=float(np.sqrt(levels[name]/meta['coordinates']).min()),
                activation_rms_max=float(np.sqrt(levels[name]/meta['coordinates']).max()))
            records.append(record);curves[name]=dict(rho=rho.tolist(),change_rms=rms.tolist(),alpha=alpha.tolist(),
                activation_rms=np.sqrt(levels[name]/meta['coordinates']).tolist(),spectrum=f['fractions'].tolist())
            if name=='raw_gradient':
                np.testing.assert_allclose(rho,[lookup[int(t)]['rho'] for t in times],rtol=1e-7,atol=1e-8)
                np.testing.assert_allclose(np.sqrt(np.diag(g[name])[1:]),[lookup[int(t)]['raw_gradient_norm'] for t in times],rtol=1e-7)
        # JSON null is the explicit undefined-cosine representation.
        for row in curves.values():row['rho']=[x if np.isfinite(x) else None for x in row['rho']]
        with (HERE/(case+'_stages.csv')).open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(records[0]),lineterminator='\n');writer.writeheader();writer.writerows(records)
        manifest=json.loads((raw/'manifest.json').read_text())
        historical=json.loads((raw/'historical_comparison.json').read_text())
        historical['worker_description_original']=historical['interpretation']
        historical['interpretation']='Same saved fork, unchanged policy on '+manifest['gpu']+'; historical agreement measured, not assumed. Original worker description retained separately.'
        full_loss=np.array([lookup[int(t)]['loss'] for t in times]);probe_loss=np.array([lookup[int(t)]['probe_loss'] for t in times])
        result=dict(case=case,states=times.tolist(),stages=records,curves=curves,boundaries=boundary,
            manifest=manifest,historical=historical,
            loss_summary=dict(probe_full_correlation=float(np.corrcoef(full_loss,probe_loss)[0,1]) if np.std(full_loss)*np.std(probe_loss)>0 else None),
            sources={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in raw.glob('*') if p.is_file()},generator_sha256=sha(Path(__file__)),
            verification=dict(PSD=True,trace=True,full_Gram_reconstruction=True,raw_gradient_boundary_cosines=True))
        if case=='transformer':
            expected=200*g['encoder.call0.output'];observed=g['pos_encoder.call0.output']
            result['verification']['embedding_positional_gram_relative_error']=float(np.linalg.norm(observed-expected)/np.linalg.norm(expected))
            assert result['verification']['embedding_positional_gram_relative_error']<.001
        dump(HERE/(case+'_results.json'),result)
        summary[case]=dict(stages=len(stages),states=len(times),peak=max((lookup[int(t)] for t in times),key=lambda r:r['loss']),
            output=next(q for q in records if q['name']=='output.log_probabilities'),
            target_readout=next(q for q in records if q['name']=='output.correct_label_log_probability'))
    dump(HERE/'summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['resnet','transformer']);a=p.parse_args();main(a.case)
