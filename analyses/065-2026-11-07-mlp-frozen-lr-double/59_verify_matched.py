"""Independent H051 coverage, recurrence, stored-update and gradient audits."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.eos_tasks import representation_continuation as r
from src.eos_tasks import representation_policy as policy
w=r.w


def rows(path): return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def close(a,b,rtol=1e-9,atol=1e-11):
    if a is None or b is None:
        assert a is None and b is None,(a,b)
        return
    assert math.isclose(a,b,rel_tol=rtol,abs_tol=atol),(a,b)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--shared',type=Path,required=True)
    a=p.parse_args();cfg=json.loads((a.run/'config.json').read_text());manifest=json.loads((a.out/'manifest.json').read_text())
    assert manifest['status']=='completed'
    start,end=cfg['fork'],cfg['end'];begin=time.monotonic()
    w.configure();torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    c=w.prepare(cfg['case'],json.loads((ROOT/'runs/223-2026-11-04-multispike-interventions/config.json').read_text()))
    layout=json.loads((a.shared/'layout.json').read_text())
    calibration=json.loads((a.shared/'thresholds.json').read_text())
    assert calibration['outgoing_states']==list(range(start-8,start))
    arrays=np.load(a.shared/'calibration.npz')
    for i,name in enumerate(calibration['module_order']):
        values=arrays[str(i)];assert values.shape[0]==8 and np.isfinite(values).all() and (values>=0).all()
        for q in cfg['quantiles']:
            close(calibration['modules'][name]['functional'][str(q)],float(np.quantile(values,q,method='linear')),rtol=0,atol=0)
    expected_splits={'mlp':{'test':10000},'resnet':{'test':10000},'transformer':{'valid':217630,'test':245550}}[cfg['case']]
    counts=dict(arms=0,updates=0,boundaries=0,evaluations=0,audits=0,exact_gradient_replays=0,probe_replays=0,cap_audits=0,
                terminal_gradient_replays=0,terminal_heldout_split_replays=0)
    results=[]
    for name in manifest['arms']:
        folder=a.out/name;arm=next(v for v in cfg['arms'] if v['id']==name)
        status=json.loads((folder/'status.json').read_text());assert status['status']=='completed' and not status['mechanics_only']
        assert status['completed_state']==end and status['updates']==end-start
        boundaries=rows(folder/'boundaries.jsonl');updates=rows(folder/'updates.jsonl');evaluations=rows(folder/'evaluations.jsonl')
        assert [v['state'] for v in boundaries]==list(range(start,end+1))
        assert [v['state'] for v in updates]==list(range(start,end))
        assert [v['update'] for v in updates]==list(range(start+1,end+1))
        assert [v['state'] for v in evaluations]==cfg['heldout_states']
        for e in evaluations:
            assert set(e['splits'])==set(expected_splits)
            for split,count in expected_splits.items():
                assert e['splits'][split]['loss_units']==count and math.isfinite(e['splits'][split]['loss'])
        for b,u in zip(boundaries,updates):
            assert math.isfinite(b['loss']) and b['loss']>=0
            close(b['raw_gradient_norm'],u['raw_gradient_norm'])
            close(u['lr'],cfg['base_lr']*arm.get('lr_multiplier',1.))
            close(u['beta2'],arm.get('beta2',.999))
            expected=min(1.,arm['clip']/b['raw_gradient_norm']) if arm.get('clip') and b['raw_gradient_norm'] else 1.
            close(u['clip_factor'],expected)
            close(u['optimizer_input_norm'],b['raw_gradient_norm']*expected,rtol=3e-7)
            for rep in b['representations'].values():
                assert rep['coordinates']>0 and rep['change_rms'] is not None and rep['change_rms']>=0
                assert rep['rho'] is None or -1.000000000001<=rep['rho']<=1.000000000001
            if b['mse_identity'] is not None: assert abs(b['mse_identity']['residual'])<1e-10
            if arm['kind']=='functional':
                assert set(u['gates'])==set(layout['targets'])
                for n,g in u['gates'].items():
                    close(g['threshold'],calibration['modules'][n]['functional'][str(arm['quantile'])])
                    if g.get('disabled_zero_reference'):continue
                    proposed=np.array(g['proposed_scores']);actual=np.array(g['actual_scores']);rounding=np.array(g['rounding_bound'])
                    factors=np.minimum(1.,np.divide(g['threshold'],proposed,out=np.ones_like(proposed),where=proposed>0))
                    np.testing.assert_allclose(g['factors'],factors,rtol=1e-12,atol=1e-12)
                    assert np.all(actual<=g['threshold']+rounding+1e-10)
        audits=sorted(folder.glob('audit-*.pt'))
        assert [int(p.stem.split('-')[1]) for p in audits]==[start,start+32,end-1]
        for path in audits:
            saved=torch.load(path,map_location='cpu',weights_only=False)
            s=saved['state'];assert saved['before']['state']==s and saved['after']['state']==s+1
            assert torch.equal(saved['raw_gradient'],saved['before']['current_gradient'])
            r.restore(c,saved['before'])
            loss,g=c['task'].gradient()
            assert torch.equal(g.cpu(),saved['raw_gradient']),f'{name}/{s}: full gradient replay differs'
            close(loss,boundaries[s-start]['loss'],rtol=0,atol=0)
            raw=g.double();old=saved['before']['previous_gradient'].double();u=updates[s-start];b=boundaries[s-start]
            close(float(raw.norm()),b['raw_gradient_norm'])
            close(policy.cosine(raw.cpu(),old),b['rho'])
            denominator=float(old.dot(old))
            close(float(old.dot(old-raw.cpu()))/denominator if denominator else None,b['eta_kappa_hat'])
            factor=min(1.,arm['clip']/float(raw.norm())) if arm.get('clip') and bool(raw.norm()) else 1.
            expected_input=g*factor
            torch.testing.assert_close(expected_input.cpu(),saved['optimizer_input'],rtol=2e-7,atol=1e-10)
            before_params={n:p.detach().clone() for n,p in c['task'].model.named_parameters()}
            rng=w.rng()
            values,inputs,probe_loss=policy.capture(c['task'],cfg['case'],cfg['probe'],layout['targets'] if arm['kind']=='functional' else [])
            w.equal(rng,w.rng());close(probe_loss,b['probe_loss'],rtol=0,atol=0)
            if cfg['case']!='mlp':
                labels=c['task'].data['y_train'][:cfg['probe']]
                if cfg['case']=='transformer':labels=labels[labels!=-100]
                values['correct_label_log_probability']=values['output'].gather(1,labels.reshape(-1,1)).squeeze(1)
                close(float(-values['correct_label_log_probability'].double().mean()),probe_loss,rtol=3e-6,atol=2e-6)
            for stage,value in values.items():
                assert value.numel()==b['representations'][stage]['coordinates']
                close(float(value.double().square().mean().sqrt()),b['representations'][stage]['activation_rms'],rtol=0,atol=0)
            counts['probe_replays']+=1
            for p in c['task'].model.parameters():p.grad.mul_(factor)
            c['opt'].step()
            assert torch.equal(w.flat(c['task'].model.parameters()).detach().cpu(),saved['proposed_parameters'])
            # No cap may modify moments or clocks after the native proposal.
            w.equal(w.cpu(c['opt'].state_dict()),saved['after']['optimizer'])
            proposed=w.flat(c['task'].model.parameters()).detach().double()-w.flat(before_params.values()).double()
            actual=w.flat(saved['after']['model'][n] for n,_ in c['task'].model.named_parameters()).double()-w.flat(before_params.values()).double().cpu()
            close(float(proposed.norm()),u['proposed_norm'])
            close(float(actual.norm()),u['actual_norm'])
            close(float(raw.cpu().dot(actual)),u['raw_gradient_dot_actual'])
            if arm['kind']=='functional':
                changed=set()
                for n in layout['targets']:
                    module,weight,bias=policy.affine_parameters(c['task'].model,n)
                    wn=n+('.in_proj_weight' if isinstance(module,torch.nn.MultiheadAttention) else '.weight')
                    bn=n+('.in_proj_bias' if isinstance(module,torch.nn.MultiheadAttention) else '.bias')
                    changed.add(wn)
                    if bias is not None:changed.add(bn)
                    dw=weight.double()-before_params[wn].double();db=None if bias is None else bias.double()-before_params[bn].double()
                    scores=policy.affine_scores(module,inputs[n],dw,db)
                    threshold=calibration['modules'][n]['functional'][str(arm['quantile'])]
                    factors=torch.where(scores>0,(threshold/scores).clamp(max=1),torch.ones_like(scores)) if threshold else torch.ones_like(scores)
                    expected=(before_params[wn].double()+dw*factors.reshape(-1,*([1]*(dw.ndim-1)))).float().cpu()
                    assert torch.equal(expected,saved['after']['model'][wn])
                    if db is not None:
                        assert torch.equal((before_params[bn].double()+db*factors).float().cpu(),saved['after']['model'][bn])
                for n,p in c['task'].model.named_parameters():
                    if n not in changed:assert torch.equal(p.detach().cpu(),saved['after']['model'][n])
                counts['cap_audits']+=1
            else:
                for n,p in c['task'].model.named_parameters(): assert torch.equal(p.detach().cpu(),saved['after']['model'][n])
            counts['audits']+=1;counts['exact_gradient_replays']+=1
            del saved,inputs,values,raw,actual,proposed,before_params
        final=torch.load(folder/'end.pt',map_location='cpu',weights_only=False)
        assert final['state']==end
        for state in final['optimizer']['state'].values(): assert int(state['step'])==end
        for value in final['model'].values(): assert torch.isfinite(value).all()
        r.restore(c,final)
        terminal_loss,terminal_gradient=c['task'].gradient()
        close(terminal_loss,boundaries[-1]['loss'],rtol=0,atol=0)
        assert torch.equal(terminal_gradient.cpu(),final['current_gradient'])
        counts['terminal_gradient_replays']+=1
        for split in expected_splits:
            rng=w.rng();observed=c['task'].evaluate(split);w.equal(rng,w.rng())
            expected=evaluations[-1]['splits'][split]
            assert observed==expected,(name,split,observed,expected)
            counts['terminal_heldout_split_replays']+=1
        del final,terminal_gradient
        counts['arms']+=1;counts['updates']+=len(updates);counts['boundaries']+=len(boundaries);counts['evaluations']+=len(evaluations)
        results.append(dict(arm=name,status='passed',updates=len(updates),audits=len(audits),final_loss=boundaries[-1]['loss']))
    report=dict(status='passed',case=cfg['case'],counts=counts,arms=results,elapsed_seconds=time.monotonic()-begin,
        verifier_sha256=w.sha(Path(__file__)),input_manifest_sha256=w.sha(a.run/'input_manifest.json'),
        eight_step_fingerprint=json.loads((a.out/'qualification.json').read_text())['eight_step_fingerprint'])
    w.write(a.out/'verification.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
