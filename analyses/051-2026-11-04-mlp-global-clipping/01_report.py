"""Verified same-Spark MLP global clipping comparison; no smoothing."""
import csv, hashlib, importlib.util, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
BASE=ROOT/'runs/223-2026-11-04-multispike-interventions/artifacts/spark/payload/worker'
NEW=ROOT/'runs/228-2026-11-04-mlp-global-clipping/artifacts/spark/payload/worker'
def rows(p): return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    checked=0
    for source in [BASE,NEW]:
        receipt=json.loads((source.parent.parent/'receipt.json').read_text()); assert receipt['verified']
        assert sha(source.parent.parent/'output.tar.gz')==receipt['sha256']
        for item in json.loads((source.parent/'transfer_inventory.json').read_text())['files']:
            p=source.parent/item['path']; assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']; checked+=1
    manifest=json.loads((NEW/'manifest.json').read_text()); assert manifest['status']=='completed'
    qualification=json.loads((NEW/'qualification.json').read_text()); assert qualification['fork_gradient_exact'] and qualification['two_update_tensor_replay_exact']
    prefix=rows(BASE/'prefix/boundaries.jsonl')[:-1]
    paths={'unchanged':BASE/'control/boundaries.jsonl','global_clip1':NEW/'global_clip1/boundaries.jsonl'}
    series={name:prefix+rows(p) for name,p in paths.items()}
    assert all([r['state'] for r in v]==list(range(801)) for v in series.values())
    metrics={name:rows(p.parent/'metrics.jsonl') for name,p in paths.items()}
    for x in metrics['global_clip1']:
        expected=min(1.,1./x['raw_gradient_norm']) if x['raw_gradient_norm'] else 1.
        assert x['clip_factor']==expected and x['applied_lr']==.001 and x['decay_increment_norm']==0
        assert x['optimizer_input_norm']<=1+2e-7
        assert abs(x['optimizer_input_norm']-x['raw_gradient_norm']*expected)<2e-7
    for t in range(519):
        for key in ['loss','rho','raw_gradient_norm','eta_kappa_hat','parameter_norm']: assert series['global_clip1'][t][key]==series['unchanged'][t][key]
    geometry_checks=0
    for data in series.values():
        for t in range(512,801):
            assert np.isclose(data[t]['eta_kappa_hat'],1-data[t]['rho']*data[t]['raw_gradient_norm']/data[t-1]['raw_gradient_norm'],rtol=1e-12,atol=1e-12)
            geometry_checks+=1
    spec=importlib.util.spec_from_file_location('events',ROOT/'analyses/036-2026-11-02-controlled-adam-diagnostics/02_extract_loss_spikes.py'); events=importlib.util.module_from_spec(spec); spec.loader.exec_module(events)
    summary={}
    for name,data in series.items():
        loss=np.array([r['loss'] for r in data]); exc=[]
        for e in events.excursions(loss):
            ratio=float(loss[e['peak_state']]/loss[e['before_state']])
            if e['peak_state']>512 and ratio>1.1: exc.append(dict(e,ratio=ratio,peak_loss=float(loss[e['peak_state']])))
        mm=metrics[name]
        summary[name]=dict(mean_513_800=float(loss[513:].mean()),last32_mean=float(loss[-32:].mean()),final_loss=float(loss[-1]),max_loss_513_800=float(loss[513:].max()),events_count=len(exc),events_gt10=exc,min_rho=min(r['rho'] for r in data[513:]),max_raw_gradient_norm=max(r['raw_gradient_norm'] for r in data[513:]),max_optimizer_input_norm=max(r['optimizer_input_norm'] for r in mm),mean_actual_step_norm=float(np.mean([r['actual_increment_norm'] for r in mm])),max_actual_step_norm=max(r['actual_increment_norm'] for r in mm))
    effects={metric:100*(summary['global_clip1'][metric]/summary['unchanged'][metric]-1) for metric in ['mean_513_800','last32_mean','final_loss','max_loss_513_800']}
    clipped=[m for m in metrics['global_clip1'] if m['clip_factor']<1]
    clipping=dict(updates=288,clipped_updates=len(clipped),fraction=len(clipped)/288,first_clipped_state=clipped[0]['state'],min_factor=min(m['clip_factor'] for m in clipped))
    panels=[('loss','Training loss (log scale)'),('rho',r'Raw gradient $\rho_t$'),('raw_gradient_norm',r'Raw gradient norm $\|g_t\|$'),('eta_kappa_hat',r'$\eta\widehat\kappa_t$'),('applied_lr','Applied LR / base LR')]
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(5,1,figsize=(9,11),sharex=True,layout='constrained'); values=[]
    for ax,(metric,ylabel) in zip(axes,panels):
        for name,color,label in [('unchanged','#777777','Unchanged baseline'),('global_clip1','#0072B2','Global norm clipping C = 1.0')]:
            data=[r for r in series[name] if 448<=r['state']<=800 and (metric!='applied_lr' or r['state']<800)]
            xx=[r['state'] for r in data]; yy=[r[metric] if metric!='applied_lr' else r[metric]/.001 for r in data]
            values.extend(dict(series=name,metric=metric,state=x,value=y) for x,y in zip(xx,yy))
            if metric=='applied_lr': xx+=[800]; yy+=[yy[-1]]
            ax.plot(xx,yy,color=color,lw=.95,ls='-',label=label,drawstyle='steps-post' if metric=='applied_lr' else 'default')
        ax.axvspan(448,512,color='#eeeeee',zorder=-3); ax.axvline(512,color='#444444',lw=.8)
        ax.grid(alpha=.14); ax.set_ylabel(ylabel); ax.set_xlim(448,800); ax.spines[['top','right']].set_visible(False)
    axes[0].set_yscale('log'); axes[0].legend(loc='upper left',fontsize=9)
    axes[0].text(.01,.04,'Shared prefix | clipping begins at state 512',transform=axes[0].transAxes,fontsize=8)
    axes[4].set_ylim(.94,1.06); axes[4].set_yticks([1]); axes[4].text(.5,.2,'Both LR curves coincide: base LR stays 0.001',ha='center',transform=axes[4].transAxes,fontsize=9)
    axes[4].set_xlabel('Completed optimizer updates'); axes[4].set_xticks([448,512,560,608,656,704,752,800])
    fig.suptitle('MLP / CIFAR-10 | global gradient clipping before Adam\nSeed 0 | full 5,000-image objective | same Spark / exact fork replay',fontsize=13)
    fig.supxlabel('Raw diagnostics shown; only optimizer input is capped. No hardware change at the fork.\nClipped 125 / 288 updates (43.4%); minimum gradient scale 0.253.',fontsize=9)
    pdf=HERE/'figures/mlp_global_clipping.pdf'; fig.savefig(pdf); plt.close(fig)
    import fitz
    doc=fitz.open(pdf); assert len(doc)==1 and not doc[0].get_images(); doc[0].get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(HERE/'scratch/mlp_global_clipping.png')
    for r in values:
        val=series[r['series']][r['state']][r['metric']]
        if r['metric']=='applied_lr': val/=.001
        assert val==r['value']
    with (HERE/'plotted_values.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    result=dict(status='completed',summary=summary,clipping_vs_unchanged_pct=effects,clipping=clipping,qualification=qualification,source_file_checks=checked,geometry_checks=geometry_checks,plotted_scalar_checks=len(values),pdf_sha256=sha(pdf),sources=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in [BASE/'prefix/boundaries.jsonl',*paths.values()]])
    (HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({k:v for k,v in result.items() if k!='summary'},indent=2));print(json.dumps({n:{k:v for k,v in s.items() if k!='events_gt10'} for n,s in summary.items()},indent=2))
if __name__=='__main__': main()
