"""Independent scalar/vector audits and publication figures for Adam forks."""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent
CAL=ROOT/'runs/209-2026-11-01-eos-adam-calibration'
PROD=ROOT/'runs/210-2026-11-01-eos-adam-long-lags'
sys.path.insert(0,str(ROOT))
from src.eos_tasks.lag_profile import cpu_products, envelope
from src.eos_tasks.adam_profile import select_rate


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def inputs(run):
    manifest=json.loads((run/'input_manifest.json').read_text())
    for r in manifest['files']:
        p=ROOT/r['path']
        assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],r['path']
    return len(manifest['files'])


def receipts(run_id,expected=None):
    roots={};records=[]
    for path in sorted(HERE.glob(f'retrieval-{run_id}-*.json')):
        r=json.loads(path.read_text());root=ROOT/r['artifact_root']
        assert r['verified'] and sha(root/'inventory.json')==r['inventory_sha256']
        inv=json.loads((root/'inventory.json').read_text())
        for item in inv['files']:
            p=root/item['path']
            assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],str(p)
        records.append(r)
        if r['scientific_status']=='complete':roots[r['worker']]=root
    if expected is not None:assert len(roots)==expected
    return roots,records


def vector_check(root,arm):
    rows=arm['rows'];bylag={r['lag']:r for r in rows};last=rows[-1]['lag']
    assert [r['lag'] for r in rows]==list(range(last+1))
    anchor=np.load(root.parent/'anchor.npy',mmap_mode='r')
    checks=[]
    for lag,path in [(0,root/'gradient-0.npy'),(1,root/'gradient-1.npy'),(2,root/'gradient-2.npy'),
            (last-1,root/'gradient-previous.npy'),(last,root/'gradient-current.npy')]:
        value=np.load(path,mmap_mode='r');actual=cpu_products(anchor,value)
        assert value.dtype==np.float32 and np.isfinite(value).all()
        row=bylag[lag]
        assert abs(actual['norm']/row['norm']-1)<1e-10
        assert abs(actual['cosine']-row['anchor_cosine'])<1e-10
        checks.append(dict(lag=lag,anchor_error=actual['cosine']-row['anchor_cosine']))
    for lag,a,b in [(1,'gradient-0.npy','gradient-1.npy'),(2,'gradient-1.npy','gradient-2.npy'),
            (last,'gradient-previous.npy','gradient-current.npy')]:
        actual=cpu_products(np.load(root/a,mmap_mode='r'),np.load(root/b,mmap_mode='r'))
        assert abs(actual['cosine']-bylag[lag]['adjacent_cosine'])<1e-10
    cp=torch.load(root/'end.pt',map_location='cpu',weights_only=False)
    assert cp['lag']==last and cp['last_gradient_unapplied']
    assert cp['step']==int(root.parent.name.split('-')[-1])+last
    states=list(cp['optimizer']['state'].values())
    assert states and all(int(s['step'])==last for s in states)
    group=cp['optimizer']['param_groups'][0]
    assert set(cp['optimizer']['state'])==set(group['params'])
    for state in states:
        for field in ['exp_avg','exp_avg_sq']:
            assert state[field].dtype==torch.float32 and bool(torch.isfinite(state[field]).all())
        assert bool((state['exp_avg_sq']>=0).all())
    assert group['lr']==arm['lr'] and group['betas']==(.9,.999) and group['eps']==1e-8
    assert group['weight_decay']==0 and not group['amsgrad'] and not group['foreach'] and not group['fused']
    return checks


def main():
    parallel=ROOT/'runs/211-2026-11-01-eos-parallel-adam-calibration'
    counts=dict(calibration_input_files=inputs(CAL),production_input_files=inputs(PROD),parallel_input_files=inputs(parallel))
    calroots,calreceipts=receipts('209');roots,prodreceipts=receipts('210',17)
    for path in HERE.glob('calibration-scalar-worker*.json'):
        scalar=json.loads(path.read_text());root=calroots[scalar['worker']]
        for r in scalar['results']:
            p=root/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
    parallelroots,parallelreceipts=receipts('211',10)
    cfg=json.loads((PROD/'config.json').read_text())
    frozen_source={r['path']:r['sha256'] for r in json.loads((ROOT/'runs/206-2026-11-01-eos-long-lags/input_manifest.json').read_text())['files']}
    for job in cfg['jobs']:assert sha(ROOT/job['anchor_gradient'])==frozen_source[job['anchor_gradient']]
    canonical={};caldirectories={}
    for worker,root in calroots.items():
        if worker in [1,2]:continue
        for path in root.glob('*-*/result.json'):
            r=json.loads(path.read_text());key=(r['parent_run_id'],r['anchor']);canonical[key]=r
            for a in r['arms']:caldirectories[(*key,a['lr'])]=path.parent/f'lr-{a["lr"]:g}'
    for root in parallelroots.values():
        for path in root.glob('*-*/result.json'):
            r=json.loads(path.read_text());key=(r['parent_run_id'],r['anchor'])
            if key not in canonical:canonical[key]=dict(r,arms=[])
            canonical[key]['arms'].extend(r['arms'])
            for a in r['arms']:caldirectories[(*key,a['lr'])]=path.parent/f'lr-{a["lr"]:g}'
    assert len(canonical)==27
    cal=[];profiles=[];audits=[];repeats=[]
    max_product_error=0.
    for job in cfg['jobs']:
        key=f'{job["parent_run_id"]}-{job["anchor"]}'
        prior=canonical[(job['parent_run_id'],job['anchor'])];cal.append(prior)
        assert sorted(a['lr'] for a in prior['arms'])==[1e-5,3e-5,1e-4,3e-4,1e-3],key
        assert all(a['status']=='complete' and a['last_lag']==64 and len(a['rows'])==65 for a in prior['arms']),key
        directory=roots[job['worker']]/key
        result=json.loads((directory/'result.json').read_text())
        assert len(result['arms'])==1
        arm=result['arms'][0];rate=job['rates'][0]
        assert arm['lr']==rate and arm['status']=='complete' and arm['last_lag']==job['horizon']
        assert len(arm['rows'])==job['horizon']+1
        assert result['initial_tensor_sha256']==job['tensor_sha256']
        for label,source in [('calibration',prior),('production',result)]:
            for a in source['arms']:
                path=caldirectories[(job['parent_run_id'],job['anchor'],a['lr'])] if label=='calibration' else directory/f'lr-{a["lr"]:g}'
                boundary=json.loads((path.parent/'boundary_checks.json').read_text())
                assert boundary['source_anchor_relative_error']<1e-5
                assert boundary['fixed_tensor_sha256']==job['tensor_sha256']
                if 'compute_chunk_relative_error' in boundary:
                    assert boundary['compute_chunk_relative_error']<1e-5 and abs(boundary['compute_chunk_loss_error'])<1e-5
                if a['status']=='complete':
                    audits.append(dict(phase=label,key=key,lr=a['lr'],checks=vector_check(path,a)))
                for r in a['rows']:
                    assert math.isfinite(r['loss']) and r['norm']>0
                    for field,cpu_key in [('anchor_cosine','anchor_cpu'),('adjacent_cosine','adjacent_cpu')]:
                        if r[field] is None:assert field=='adjacent_cosine' and r['lag']==0;continue
                        assert abs(r[field])<=1+1e-12
                        e=abs(r[field]-r[cpu_key]['cosine']);max_product_error=max(max_product_error,e)
                        assert e<1e-10
        same=next(a for a in prior['arms'] if a['lr']==rate)
        repeat=dict(key=key,maximum_loss_error=0.,maximum_cosine_error=0.,maximum_norm_relative_error=0.)
        for before,after in zip(same['rows'],arm['rows']):
            repeat['maximum_loss_error']=max(repeat['maximum_loss_error'],abs(before['loss']-after['loss']))
            repeat['maximum_norm_relative_error']=max(repeat['maximum_norm_relative_error'],abs(before['norm']/after['norm']-1))
            for field in ['anchor_cosine','adjacent_cosine']:
                if before[field] is not None:
                    repeat['maximum_cosine_error']=max(repeat['maximum_cosine_error'],abs(before[field]-after[field]))
        # Identical saved tensors define matched starts. CUDA reduction drift along
        # evolving trajectories is reported; it is not silently called bitwise replay.
        repeat['within_1e_minus5_every_boundary']=repeat['maximum_loss_error']<1e-5 and repeat['maximum_cosine_error']<1e-5 and repeat['maximum_norm_relative_error']<1e-5
        assert abs(same['rows'][0]['loss']-arm['rows'][0]['loss'])<1e-5
        assert abs(same['rows'][0]['norm']/arm['rows'][0]['norm']-1)<1e-5
        repeats.append(repeat)
        profiles.append(dict(parent_run_id=job['parent_run_id'],task=job['task'],anchor=job['anchor'],
            parent_gd_lr=job['parent_gd_lr'],adam_lr=rate,primary_eligible=job['primary_eligible'],
            horizon=job['horizon'],source_checkpoint=job['checkpoint'],initial_tensor_sha256=job['tensor_sha256'],
            artifact_root=directory.relative_to(ROOT).as_posix(),rows=arm['rows'],boundary_checks=result['boundary_checks']))
    decision=json.loads((HERE/'calibration_decision.json').read_text())
    for name in ['mlp','resnet','transformer']:
        actual=select_rate([r for r in cal if r['task']==name],[1e-5,3e-5,1e-4,3e-4,1e-3])
        expected=decision['decisions'][name]
        assert actual['selected_lr']==expected['selected_lr'] and actual['primary_checkpoints']==expected['primary_checkpoints']
        for a,b in zip(actual['candidates'],expected['candidates']):
            assert a['lr']==b['lr'] and a['eligible']==b['eligible'] and a['checkpoints']==b['checkpoints']
            assert abs(a['mean_log_terminal_loss_ratio']-b['mean_log_terminal_loss_ratio'])<1e-12
    gd=json.loads((ROOT/'analyses/033-2026-11-01-eos-long-lags/results.json').read_text())['profiles']
    bykey={(r['parent_run_id'],r['anchor']):r for r in gd}
    summaries=[];flat=[]
    for p in profiles:
        old=bykey[(p['parent_run_id'],p['anchor'])]
        assert old['last_lag']==p['horizon'] and old['primary_eligible']==p['primary_eligible']
        rows=p['rows'];adj=np.array([r['adjacent_cosine'] for r in rows[1:]])
        final=rows[-1];gdlast=old['rows'][-1]
        summaries.append(dict(parent_run_id=p['parent_run_id'],task=p['task'],parent_gd_lr=p['parent_gd_lr'],
            anchor=p['anchor'],adam_lr=p['adam_lr'],primary_eligible=p['primary_eligible'],horizon=p['horizon'],
            initial_loss=rows[0]['loss'],final_loss=final['loss'],gd_final_loss=gdlast['loss'],
            loss_ratio=final['loss']/rows[0]['loss'],gd_loss_ratio=gdlast['loss']/old['rows'][0]['loss'],
            adjacent_first=adj[0],adjacent_last=adj[-1],negative_fraction=float(np.mean(adj<0)),
            terminal64_negative_fraction=float(np.mean(adj[-64:]<0)),terminal64_median_adjacent=float(np.median(adj[-64:])),
            anchor_last=final['anchor_cosine'],gd_anchor_last=gdlast['cosine'],
            anchor_terminal_half_gap=envelope([dict(lag=r['lag'],cosine=r['anchor_cosine']) for r in rows]),
            gd_anchor_terminal_half_gap=envelope(old['rows'])))
        for r,oldrow in zip(rows,old['rows']):
            assert r['lag']==oldrow['lag']
            flat.append(dict(parent_run_id=p['parent_run_id'],task=p['task'],parent_gd_lr=p['parent_gd_lr'],
                anchor=p['anchor'],adam_lr=p['adam_lr'],primary_eligible=p['primary_eligible'],**{k:r[k] for k in
                ['lag','state','loss','norm','anchor_cosine','adjacent_cosine']},gd_anchor_cosine=oldrow['cosine'],gd_loss=oldrow['loss']))
    with (HERE/'lag_profiles.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    save(HERE/'results.json',dict(profiles=profiles))
    save(HERE/'profile_summary.json',dict(profiles=summaries))
    counts.update(profiles=len(profiles),primary=sum(p['primary_eligible'] for p in profiles),
        production_rows=len(flat),production_cosines=2*sum(p['horizon'] for p in profiles),
        calibration_arms=sum(len(r['arms']) for r in cal),calibration_rows=sum(len(a['rows']) for r in cal for a in r['arms']),
        vector_audits=len(audits),transferred_files=sum(r['files'] for r in calreceipts+prodreceipts),
        transferred_bytes=sum(r['bytes'] for r in calreceipts+parallelreceipts+prodreceipts))
    counts['transferred_files']=sum(r['files'] for r in calreceipts+parallelreceipts+prodreceipts)
    assert counts['calibration_arms']==135 and counts['calibration_rows']==8775
    assert counts['production_rows']==36123 and counts['production_cosines']==72192 and counts['vector_audits']==162
    counts['calibration_prefixes_within_1e_minus5']=sum(r['within_1e_minus5_every_boundary'] for r in repeats)
    save(HERE/'verification.json',dict(status='passed',counts=counts,maximum_cpu_gpu_cosine_error=max_product_error,
        vector_audits=audits,calibration_production_repetition=repeats,selection_recomputed=True,
        source_gd_sha256=sha(ROOT/'analyses/033-2026-11-01-eos-long-lags/results.json')))
    figures(profiles,cal,decision,bykey)
    print(json.dumps(counts))


def figures(profiles,cal,decision,gd):
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'path.simplify':False,'axes.spines.top':False,
        'axes.spines.right':False,'savefig.bbox':'tight'})
    target=HERE/'figures';target.mkdir(exist_ok=True)
    names=['mlp','resnet','transformer'];titles=['CIFAR-10 / MLP','CIFAR-10 / ResNet-32','WikiText-2 / Transformer']
    colors=['#247ba0','#e28b24','#9d4b9e']
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    for ax,name,title in zip(axes,names,titles):
        for r in [r for r in cal if r['task']==name and r['primary_eligible']]:
            arms=sorted(r['arms'],key=lambda a:a['lr'])
            rates=[a['lr'] for a in arms]
            ratios=[np.mean([z['loss'] for z in a['rows'][-16:]])/a['rows'][0]['loss'] for a in arms]
            ax.plot(rates,ratios,color='#247ba0',alpha=.28,lw=.8,marker='.',ms=3)
        selected=decision['decisions'][name]['selected_lr']
        candidates=decision['decisions'][name]['candidates']
        ax.plot([r['lr'] for r in candidates],[math.exp(r['mean_log_terminal_loss_ratio']) for r in candidates],
            color='.15',lw=1.3,marker='o',ms=4,label='geometric mean across states')
        invalid=[r for r in candidates if not r['eligible']]
        if invalid:ax.scatter([r['lr'] for r in invalid],[math.exp(r['mean_log_terminal_loss_ratio']) for r in invalid],
            marker='x',s=40,color='#ba3b3b',zorder=5,label='fails declared safeguard')
        ax.axvline(selected,color='#ba3b3b',ls='--',label=f'selected {selected:g}')
        ax.axhline(1,color='.6',lw=.6);ax.set_xscale('log');ax.set_yscale('log')
        ax.set_title(title);ax.set_xlabel('Adam learning rate');ax.legend(fontsize=8)
    axes[0].set_ylabel('Mean loss at updates 49–64 / initial loss')
    fig.suptitle('Training-only calibration; each line is one primary retained state')
    fig.savefig(target/'calibration.pdf');plt.close(fig)
    for mode,filename,cap in [('anchored','anchored_profiles',None),('adjacent','adjacent_profiles',None),('adjacent','adjacent_first128',128)]:
        fig,axes=plt.subplots(3,3,figsize=(13,9.3),layout='constrained')
        for i,name in enumerate(names):
            parent_rates=sorted({p['parent_gd_lr'] for p in profiles if p['task']==name})
            rate=decision['decisions'][name]['selected_lr']
            for j,anchor in enumerate([500,2000,3000]):
                ax=axes[i,j]
                for p in [p for p in profiles if p['task']==name and p['anchor']==anchor]:
                    color=colors[parent_rates.index(p['parent_gd_lr'])];flag='' if p['primary_eligible'] else ' [U]'
                    rows=[r for r in p['rows'] if (cap is None or r['lag']<=cap) and (mode=='anchored' or r['lag']>0)]
                    field='anchor_cosine' if mode=='anchored' else 'adjacent_cosine'
                    alpha=1 if p['primary_eligible'] else .4
                    ax.plot([r['lag'] for r in rows],[r[field] for r in rows],color=color,lw=.7,alpha=alpha,zorder=3,
                        label=f'source GD LR {p["parent_gd_lr"]:g}{flag}',rasterized=False)
                    if mode=='anchored':
                        old=gd[(p['parent_run_id'],p['anchor'])]['rows']
                        ax.plot([r['lag'] for r in old],[r['cosine'] for r in old],color=color,lw=.55,
                            ls='--',alpha=.35,zorder=1,rasterized=False)
                ax.axhline(0,color='.5',lw=.6);ax.set_ylim(-1.04,1.04)
                if cap is None:
                    ax.set_xscale('symlog',linthresh=16);ax.set_xlim(0,2048)
                    ax.set_xticks([0,16,64,256,512,2048]);ax.set_xticklabels(['0','16','64','256','512','2048'])
                else:ax.set_xlim(1,cap)
                ax.set_title(f'{titles[i]} · anchor {anchor}\nAdam LR {rate:g}',fontsize=9)
                if j==0:ax.set_ylabel('cos(g₀, gₖ)' if mode=='anchored' else 'cos(gₖ₋₁, gₖ)')
                if i==2:ax.set_xlabel('Updates after retained anchor (k)')
                ax.legend(loc='best',fontsize=6.5,framealpha=.85)
        fig.suptitle('Exact anchored raw gradients: Adam solid, prior GD dashed' if mode=='anchored'
            else ('Exact adjacent raw gradients along Adam: first 128 updates' if cap else 'Exact adjacent raw gradients along Adam: matched GD horizons'))
        fig.supxlabel('[U]: unqualified historical reconstruction, excluded from primary summaries and LR selection. Raw values; no smoothing.',fontsize=8)
        fig.savefig(target/(filename+'.pdf'));plt.close(fig)
    save(HERE/'figure_sources.json',dict(figures=[dict(path=p.relative_to(HERE).as_posix(),sha256=sha(p),
        generator='02_analyze.py',generator_sha256=sha(HERE/'02_analyze.py'),
        source_results_sha256=sha(HERE/'results.json'),calibration_sha256=sha(HERE/'calibration_decision.json'),
        gd_reference_sha256=sha(ROOT/'analyses/033-2026-11-01-eos-long-lags/results.json'),
        vector=True) for p in sorted(target.glob('*.pdf'))]))


if __name__=='__main__':main()
