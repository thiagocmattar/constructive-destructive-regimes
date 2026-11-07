"""Compare cumulative MLP interventions with the unchanged Adam baseline."""
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
RUNS={'double':ROOT/'runs/245-2026-11-07-mlp-frozen-lr-double',
      'high':ROOT/'runs/246-2026-11-07-mlp-frozen-lr-0p02'}
PARENT=ROOT/'runs/244-2026-11-07-mlp-freeze-first-linear/artifacts/spark/payload/worker'
OLD=HERE.parent/'064-2026-11-07-mlp-freeze-first-linear'


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s]
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',newline='\n')


def verify_dense(raw,high,rep):
    names=['preactivation1','hidden1','preactivation2','hidden2','output'];checks={}
    # Independent FP64 coordinate blocks: all53 native states499..551.
    for name in names:
        a=np.load(raw/(name+'.npy'),mmap_mode='r');flat=a.reshape(len(a),-1);assert len(a)==53
        norm2=np.zeros(52);dots=np.zeros(51)
        for start in range(0,flat.shape[1],32768):
            delta=np.diff(np.asarray(flat[:,start:start+32768],dtype=np.float64),axis=0)
            norm2+=np.sum(delta*delta,axis=1);dots+=np.sum(delta[1:]*delta[:-1],axis=1)
        den=np.sqrt(norm2[1:]*norm2[:-1]);rho=np.divide(dots,den,out=np.full(51,np.nan),where=den>0)
        rms=np.sqrt(norm2[1:]/flat.shape[1])
        expected=np.array([np.nan if rep[t][name]['rho'] is None else rep[t][name]['rho'] for t in range(500,551)])
        np.testing.assert_allclose(rho,expected,rtol=1e-9,atol=1e-10,equal_nan=True)
        np.testing.assert_allclose(rms,[rep[t][name]['change_rms'] for t in range(500,551)],rtol=1e-9,atol=1e-12)
        checks[name]=dict(rows=51,independent_cosine_and_rms=True,zero_energy=bool(norm2.sum()==0))
    gradients=np.load(raw/'raw_gradient.npy',mmap_mode='r')
    for i in range(1,len(gradients)):
        t=499+i;g=np.asarray(gradients[i],dtype=np.float64);p=np.asarray(gradients[i-1],dtype=np.float64)
        for values,old,keynorm,keyrho in [(g,p,'raw_gradient_norm','rho'),(g[614600:],p[614600:],'active_gradient_norm','active_gradient_rho')]:
            gn=np.linalg.norm(values);pn=np.linalg.norm(old);rho=float(values@old)/(gn*pn)
            np.testing.assert_allclose([gn,rho],[high[t][keynorm],high[t][keyrho]],rtol=1e-9,atol=1e-10)
        q=1-float(g@p)/float(p@p);np.testing.assert_allclose(q,high[t]['eta_kappa_hat'],rtol=1e-9,atol=1e-10)
    return dict(representations=checks,gradient_states_checked=[500,551],raw_and_active_gradient_and_eta=True)


def main():
    previous=json.loads((OLD/'comparison.json').read_text());assert previous['status']=='verified'
    for name,digest in previous['sources'].items():assert sha(ROOT/name)==digest,name
    low={q['state']:q for q in rows(PARENT/'boundaries.jsonl')}
    names=['preactivation1','hidden1','preactivation2','hidden2','output']
    run_records={};sources=[OLD/'comparison.json',PARENT/'boundaries.jsonl',PARENT/'representations.jsonl']
    for arm,run in RUNS.items():
        raw=run/'artifacts/spark/payload/worker'
        assert json.loads((run/'retrieval.json').read_text())['verified']
        assert json.loads((run/'verification.json').read_text())['status']=='passed'
        inventory=json.loads((raw/'inventory.json').read_text())['files']
        for item in inventory:
            p=raw/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],str(p)
        high={q['state']:q for q in rows(raw/'boundaries.jsonl')}
        rep={q['state']:q['streams'] for q in rows(raw/'representations.jsonl')}
        assert list(high)==list(range(500,802)) and list(rep)==list(range(500,801))
        for key in ['loss','raw_gradient_norm','rho','eta_kappa_hat','active_gradient_norm','active_gradient_rho']:
            np.testing.assert_allclose(low[500][key],high[500][key],rtol=0,atol=1e-12)
        cfg=json.loads((run/'config.json').read_text());updates=rows(raw/'updates.jsonl')
        assert len(updates)==301 and all(q['lr']==cfg['intervention_lr'] and q['layers']['1']['actual_increment_norm']==0 for q in updates)
        for t in rep:
            for n in names[:2]:assert rep[t][n]['change_rms']==0 and rep[t][n]['rho'] is None
        run_records[arm]=dict(boundaries=high,representations=rep,lr=cfg['intervention_lr'],
            independent_verification=verify_dense(raw,high,rep),
            qualification=json.loads((raw/'qualification.json').read_text()),
            freeze_verification=json.loads((raw/'freeze_verification.json').read_text()))
        sources += [run/'config.json',run/'input_manifest.json',run/'retrieval.json',run/'verification.json',
                    raw/'inventory.json',raw/'boundaries.jsonl',raw/'representations.jsonl']
    curves={'control':copy.deepcopy(previous['curves']['control']),
            'freeze':copy.deepcopy(previous['curves']['freeze']),
            'double':copy.deepcopy(previous['curves']['freeze']),
            'high':copy.deepcopy(previous['curves']['freeze'])}
    states=list(range(448,801));mapping={'gradient_rho':'rho','raw_gradient_norm':'raw_gradient_norm',
        'eta_kappa_hat':'eta_kappa_hat','loss':'loss','active_gradient_norm':'active_gradient_norm','active_gradient_rho':'active_gradient_rho'}
    for arm in curves:
        curves[arm]['lr']=[run_records[arm]['lr'] if arm in run_records and t>=500 else .001 for t in states]
        for t in range(500,801):
            if arm in run_records:
                high=run_records[arm]['boundaries'];rep=run_records[arm]['representations']
                for key,source in mapping.items():curves[arm][key][t-448]=high[t][source]
                for name in names:
                    for key in ['rho','change_rms']:curves[arm]['layers'][name][key][t-448]=rep[t][name][key]
            elif arm=='freeze':
                for key,source in mapping.items():np.testing.assert_allclose(curves[arm][key][t-448],low[t][source],rtol=1e-12,atol=1e-12)
    assert previous['states']==states
    # All interventions share the original state448 and the full freeze-only prefix.
    for key in mapping:
        np.testing.assert_allclose(curves['control'][key][0],curves['freeze'][key][0],rtol=0,atol=1e-12)
        for arm in RUNS:np.testing.assert_array_equal(curves['freeze'][key][:53],curves[arm][key][:53])
    metrics=[]
    for a,b in [(450,500),(500,550),(500,800),(448,800)]:
        for arm in curves:
            t=np.arange(a,b+1);loss=np.array(curves[arm]['loss'])[t-448];change=np.diff(loss)
            motion=np.array(curves[arm]['layers']['output']['change_rms'])[t-448]
            rho=np.array(curves[arm]['layers']['output']['rho'])[t-448]
            metrics.append(dict(window=f'{a}-{b}',arm=arm,states=len(t),loss_mean=float(loss.mean()),loss_max=float(loss.max()),
                peak_state=int(t[np.argmax(loss)]),loss_final=float(loss[-1]),positive_loss_variation=float(np.maximum(change,0).sum()),
                largest_upward_loss_change=float(max(0,change.max())),positive_loss_transitions=int(np.sum(change>0)),
                output_negative=int(np.sum(rho<0)),output_rms_max=float(motion.max()),output_rms_energy=float(np.sum(motion**2))))
    # Preserve previously verified baseline/freeze summaries on their original windows.
    for old in previous['metrics']:
        current=next(m for m in metrics if (m['window'],m['arm'])==(old['window'],old['arm']))
        for key in old:
            if isinstance(old[key],(float,int)):np.testing.assert_allclose(current[key],old[key],rtol=1e-12,atol=1e-12)
            else:assert current[key]==old[key]
    baseline_comparisons=[]
    for a,b in [(450,500),(500,550),(500,800),(448,800)]:
        window=f'{a}-{b}';reference=next(m for m in metrics if m['window']==window and m['arm']=='control')
        for arm in ['freeze','double','high']:
            intervention=next(m for m in metrics if m['window']==window and m['arm']==arm)
            baseline_comparisons.append(dict(window=window,intervention=arm,reference='control',
                percent_change={key:100*(intervention[key]/reference[key]-1) for key in
                    ['loss_mean','loss_max','loss_final','positive_loss_variation','output_rms_max','output_rms_energy']}))
    layer_summary=[]
    for arm in curves:
        for name in names:
            values=curves[arm]['layers'][name];rho=np.array([np.nan if x is None else x for x in values['rho'][52:103]])
            rms=np.array(values['change_rms'][52:103]);assert len(rms)==51
            layer_summary.append(dict(arm=arm,name=name,negative=int(np.sum(rho<0)),defined=int(np.isfinite(rho).sum()),
                rms_max=float(rms.max()),rms_energy=float(np.sum(rms**2))))
    result=dict(status='verified',states=states,primary_states=list(range(500,551)),curves=curves,metrics=metrics,layers=layer_summary,
        baseline_comparisons=baseline_comparisons,
        analysis_scope='One ongoing MLP intervention analysis; 064 is the retained first-stage source, 065 is the current workspace.',
        intervention_sets={'control':[], 'freeze':['Freeze first-linear weights and biases from state448'],
            'double':['Freeze first-linear weights and biases from state448','Adam LR0.002 at state500'],
            'high':['Freeze first-linear weights and biases from state448','Adam LR0.02 at state500 (10x I2)']},
        run_verifications={arm:{k:v for k,v in record.items() if k not in ['boundaries','representations']} for arm,record in run_records.items()},
        sources={p.relative_to(ROOT).as_posix():sha(p) for p in sources},generator_sha256=sha(Path(__file__)))
    dump(HERE/'comparison.json',result)
    with (HERE/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]),lineterminator='\n');writer.writeheader();writer.writerows(metrics)
    print(json.dumps(dict(metrics=metrics,verified_arms=list(run_records)),indent=2))


if __name__=='__main__':main()
