"""Audit exact streaming products and retained vectors; reduce and plot all lags."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
RUN = ROOT/'runs/206-2026-11-01-eos-long-lags'
spec = importlib.util.spec_from_file_location('streaming_worker', RUN/'01_worker.py')
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def figure(profiles, cfg, filename, *, common=False, flagged=False):
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    fig, axes = plt.subplots(3,3,figsize=(11.4,7.8),sharey=True)
    colors = ['#0072B2','#D55E00','#009E73']
    counts = 0
    titles = dict(mlp='CIFAR-10 / tanh MLP',resnet='CIFAR-10 / ResNet-32',transformer='WikiText-2 / Transformer')
    for col, task in enumerate(['mlp','resnet','transformer']):
        lrs = sorted(set(p['lr'] for p in profiles if p['task']==task))
        for row, anchor in enumerate(cfg['anchors']):
            ax = axes[row,col]
            selected = [p for p in profiles if p['task']==task and p['anchor']==anchor and (p['primary_eligible'] or flagged)]
            handles=[]
            for p in sorted(selected,key=lambda p:p['lr']):
                values = [r for r in p['rows'] if r['lag']>0 and (not common or r['lag']<=256)]
                x=np.array([r['lag'] for r in values]);y=np.array([np.nan if r['cosine'] is None else r['cosine'] for r in values])
                color=colors[lrs.index(p['lr'])]
                alpha=1.0 if p['primary_eligible'] else .35
                for parity, style in [(0,'-'),(1,'--')]:
                    mask=x%2==parity
                    line,=ax.plot(x[mask],y[mask],style,color=color,alpha=alpha,lw=1.1)
                    np.testing.assert_array_equal(line.get_xdata(),x[mask])
                    np.testing.assert_array_equal(line.get_ydata(),y[mask])
                    counts+=int(mask.sum())
                    if mask.any() and not common:
                        ax.plot(x[mask][-1],y[mask][-1],'.',color=color,alpha=alpha,ms=3)
                label=f'LR {p["lr"]:g}'+(' [unqualified]' if not p['primary_eligible'] else '')
                handles.append(Line2D([],[],color=color,lw=1.4,alpha=alpha,label=label))
            ax.axhline(0,color='#888888',lw=.6)
            ax.set_ylim(-1.06,1.06);ax.set_yticks([-1,-.5,0,.5,1]);ax.grid(alpha=.16)
            if common:
                ax.set_xlim(1,256);ax.set_xticks([1,64,128,192,256])
            else:
                ax.set_xscale('log',base=2);ax.set_xlim(1,cfg['stages'][-1])
                ax.set_xticks([1,4,16,64,256,1024,2048]);ax.set_xticklabels(['1','4','16','64','256','1024','2048'])
                ax.axvline(256,color='#777777',lw=.65,ls=':')
            ax.set_title(f'{titles[task]} · anchor {anchor}',fontsize=9)
            ax.legend(handles=handles,loc='best',fontsize=6.7,framealpha=.85)
            if col==0:ax.set_ylabel('Anchored gradient cosine')
            if row==2:ax.set_xlabel('Forward lag (GD updates)')
    title='Exact full-gradient cosine: '+('common lag-256 coverage' if common else 'longer horizons')
    fig.suptitle(title+(' · includes flagged reconstructions' if flagged else ' · primary profiles'),fontsize=12,y=.985)
    subtitle='Solid = even lags; dashed = odd lags. Lines connect measured points within each parity; no smoothing.'
    foot='One anchored pair per lag, seed 0. '+('All shown profiles share this range.' if common else 'Each curve ends at its declared adaptive stop; dotted vertical line = common lag 256.')
    fig.text(.5,.947,subtitle,ha='center',fontsize=8)
    fig.text(.5,.012,foot,ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.035,1,.93])
    (HERE/'figures').mkdir(exist_ok=True);(HERE/'scratch').mkdir(exist_ok=True)
    pdf=HERE/'figures'/filename
    fig.savefig(pdf);fig.savefig(HERE/'scratch'/pdf.with_suffix('.png').name,dpi=135)
    plt.close(fig)
    return dict(path=pdf.relative_to(HERE).as_posix(),sha256=sha(pdf),plotted_values=counts,artist_arrays_verified=True)


def main():
    cfg=json.loads((RUN/'config.json').read_text())
    for item in json.loads((RUN/'input_manifest.json').read_text())['files']:
        p=ROOT/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],item['path']
    profiles=[]; audits=[];cpu_errors=[];norm_errors=[]
    raw_files=raw_bytes=0
    for worker_id in range(cfg['resource_limits']['workers']):
        base=RUN/f'artifacts/attempt-01/worker{worker_id}'
        receipt=json.loads((HERE/f'retrieval-worker{worker_id}.json').read_text())
        assert receipt['verified'] and sha(base/'inventory.json')==receipt['inventory_sha256']
        for item in json.loads((base/'inventory.json').read_text())['files']:
            p=(base/item['path']).resolve();assert p.is_relative_to(base.resolve())
            assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],str(p)
            raw_files+=1;raw_bytes+=item['bytes']
        manifest=json.loads((base/'manifest.json').read_text());assert manifest['status']=='complete'
        for job in [j for j in cfg['jobs'] if j['worker']==worker_id]:
            out=base/f'{job["parent_run_id"]}-{job["anchor"]}'
            if not (out/'result.json').exists():
                raise AssertionError('Missing requested window: '+str(out))
            result=json.loads((out/'result.json').read_text())
            assert result['primary_eligible']==job['primary_eligible']
            assert result['last_lag']>=256, 'Common coverage not completed'
            rows=result['rows'];assert [r['lag'] for r in rows]==list(range(result['last_lag']+1))
            assert rows[:11]==job['prefix_rows']
            assert result['last_state']==job['anchor']+result['last_lag']
            anchor=np.load(ROOT/job['anchor_gradient'])
            for r in rows[11:]:
                assert r['state']==job['anchor']+r['lag']
                if r['cosine'] is not None:
                    error=abs(r['cosine']-r['cpu']['cosine']);cpu_errors.append(error)
                    norm_error=abs(r['norm']/r['cpu']['norm']-1);norm_errors.append(norm_error)
                    assert error<cfg['cosine_audit_tolerance'] and norm_error<cfg['cosine_audit_tolerance']
                    assert -1-1e-12<=r['cosine']<=1+1e-12
            for name,lag in [('audit-first.npy',11),('audit-previous.npy',result['last_lag']-1),('audit-current.npy',result['last_lag'])]:
                vector=np.load(out/name)
                recomputed=worker.cpu_products(anchor,vector)
                r=rows[lag]
                error=None if r['cosine'] is None else abs(recomputed['cosine']-r['cosine'])
                assert error is None or error<cfg['cosine_audit_tolerance']
                audits.append(dict(parent_run_id=job['parent_run_id'],anchor=job['anchor'],lag=lag,
                    path=(out/name).relative_to(ROOT).as_posix(),cosine_error=error,
                    norm_relative_error=abs(recomputed['norm']/r['norm']-1)))
            state=torch.load(out/'end.pt',map_location='cpu',weights_only=False)
            assert state['step']==result['last_state'] and not state['optimizer']['state']
            group=state['optimizer']['param_groups'][0]
            assert group['lr']==job['lr'] and group['momentum']==0 and group['weight_decay']==0
            assert 'torch_rng' in state and 'cuda_rng' in state and result['last_gradient_unapplied']
            del state
            result['source_result']=(out/'result.json').relative_to(ROOT).as_posix()
            result['source_result_sha256']=sha(out/'result.json')
            result['final_envelope']=worker.envelope(rows,cfg['envelope_width'])
            result['common_256_envelope']=worker.envelope(rows[:257],cfg['envelope_width'])
            result['cosine_256']=rows[256]['cosine']
            result['final_cosine']=rows[-1]['cosine']
            profiles.append(result)
    profiles.sort(key=lambda p:({'mlp':0,'resnet':1,'transformer':2}[p['task']],p['lr'],p['anchor']))
    assert len(profiles)==27 and sum(p['primary_eligible'] for p in profiles)==25
    save(HERE/'results.json',dict(config_sha256=sha(RUN/'config.json'),profiles=profiles))
    fields=['parent_run_id','task','lr','anchor','primary_eligible','lag','state','cosine','norm','loss','source']
    count=0
    with (HERE/'lag_profiles.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for p in profiles:
            for r in p['rows'][1:]:
                writer.writerow({k:({**p,**r,'source':p['source_result']}).get(k) for k in fields});count+=1
    figures=[figure(profiles,cfg,'common_lag256.pdf',common=True),
             figure(profiles,cfg,'long_lag_profiles.pdf'),
             figure(profiles,cfg,'all_profiles_with_reconstruction_flags.pdf',flagged=True)]
    save(HERE/'verification.json',dict(status='passed',profiles=27,primary_profiles=25,
        cosines=count,new_cosines=len(cpu_errors),maximum_cpu_gpu_cosine_error=max(cpu_errors),
        maximum_cpu_gpu_norm_relative_error=max(norm_errors),retained_full_vector_audits=audits,
        raw_files=raw_files,raw_bytes=raw_bytes,source_prefix_preserved=True,
        complete_common_lag256=True,all_end_recovery_states_verified=True,figures=figures))
    summary=[]
    for p in profiles:
        summary.append({k:p[k] for k in ['parent_run_id','task','lr','anchor','primary_eligible','last_lag',
            'initial_reversal','cosine_256','final_cosine','common_256_envelope','final_envelope','stop_reason','elapsed_seconds']})
    save(HERE/'profile_summary.json',dict(profiles=summary))
    print(json.dumps(dict(status='verified',profiles=len(profiles),cosines=count,new_cosines=len(cpu_errors),
        raw_bytes=raw_bytes,maximum_cpu_gpu_cosine_error=max(cpu_errors))))


if __name__=='__main__':main()
