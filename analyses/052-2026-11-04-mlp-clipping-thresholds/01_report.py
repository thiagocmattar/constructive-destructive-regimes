"""Verified same-Spark MLP global clipping comparison; no smoothing."""
import csv, hashlib, importlib.util, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
BASE=ROOT/'runs/223-2026-11-04-multispike-interventions/artifacts/spark/payload/worker'
OLD=ROOT/'runs/228-2026-11-04-mlp-global-clipping/artifacts/spark/payload/worker'
NEW=ROOT/'runs/229-2026-11-04-mlp-clipping-half/artifacts/spark/payload/worker'
def rows(p): return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    checked=0
    for source in [BASE,OLD,NEW]:
        receipt=json.loads((source.parent.parent/'receipt.json').read_text()); assert receipt['verified']
        assert sha(source.parent.parent/'output.tar.gz')==receipt['sha256']
        for item in json.loads((source.parent/'transfer_inventory.json').read_text())['files']:
            p=source.parent/item['path']; assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']; checked+=1
    manifest=json.loads((NEW/'manifest.json').read_text()); assert manifest['status']=='completed'
    qualification=json.loads((NEW/'qualification.json').read_text()); assert qualification['fork_gradient_exact'] and qualification['two_update_tensor_replay_exact']
    prefix=rows(BASE/'prefix/boundaries.jsonl')[:-1]
    paths={'unchanged':BASE/'control/boundaries.jsonl','global_clip1':OLD/'global_clip1/boundaries.jsonl','global_clip05':NEW/'global_clip05/boundaries.jsonl'}
    series={name:prefix+rows(p) for name,p in paths.items()}
    assert all([r['state'] for r in v]==list(range(801)) for v in series.values())
    metrics={name:rows(p.parent/'metrics.jsonl') for name,p in paths.items()}
    for x in metrics['global_clip05']:
        expected=min(1.,.5/x['raw_gradient_norm']) if x['raw_gradient_norm'] else 1.
        assert x['clip_factor']==expected and x['applied_lr']==.001 and x['decay_increment_norm']==0
        assert x['optimizer_input_norm']<=.5+2e-7
        assert abs(x['optimizer_input_norm']-x['raw_gradient_norm']*expected)<2e-7
    for t in range(514):
        for key in ['loss','rho','raw_gradient_norm','eta_kappa_hat','parameter_norm']: assert series['global_clip05'][t][key]==series['unchanged'][t][key]
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
    effects={name:{metric:100*(summary[name][metric]/summary['unchanged'][metric]-1) for metric in ['mean_513_800','last32_mean','final_loss','max_loss_513_800']} for name in ['global_clip1','global_clip05']}
    clipping={}
    for name in ['global_clip1','global_clip05']:
        clipped=[m for m in metrics[name] if m['clip_factor']<1]
        clipping[name]=dict(updates=288,clipped_updates=len(clipped),fraction=len(clipped)/288,first_clipped_state=clipped[0]['state'],min_factor=min(m['clip_factor'] for m in clipped))
    panels=[('loss','Training loss (log scale)'),('rho',r'Raw gradient $\rho_t$'),('raw_gradient_norm','Gradient norm'),('eta_kappa_hat',r'$\eta\widehat\kappa_t$'),('applied_lr','Applied LR / base LR')]
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(5,2,figsize=(13,11),sharex='col',layout='constrained');values=[]
    for col,(treatment,cap) in enumerate([('global_clip1',1.),('global_clip05',.5)]):
        for ax,(metric,ylabel) in zip(axes[:,col],panels):
            for name,color,label in [('unchanged','#777777','Unchanged baseline'),(treatment,'#0072B2',f'Global norm clipping C = {cap}')]:
                data=[r for r in series[name] if 448<=r['state']<=800 and (metric!='applied_lr' or r['state']<800)]
                xx=[r['state'] for r in data];yy=[r[metric] if metric!='applied_lr' else r[metric]/.001 for r in data]
                values.extend(dict(column=col,series=name,metric=metric,state=x,value=y) for x,y in zip(xx,yy))
                if metric=='applied_lr':xx+=[800];yy+=[yy[-1]]
                ax.plot(xx,yy,color=color,lw=.85,ls='-',label=label,drawstyle='steps-post' if metric=='applied_lr' else 'default')
            if metric=='raw_gradient_norm':
                mm=metrics[treatment];assert [m['state'] for m in mm]==list(range(512,800))
                ax.plot([m['state'] for m in mm],[m['optimizer_input_norm'] for m in mm],color='#009E73',lw=1.2,ls='-',label='After clipping')
                values.extend(dict(column=col,series=treatment,metric='optimizer_input_norm',state=m['state'],value=m['optimizer_input_norm']) for m in mm)
                handles,_=ax.get_legend_handles_labels();ax.legend(handles,['Baseline: raw','Clipping run: raw','Clipping run: after clipping'],loc='upper left',fontsize=7)
            ax.axvspan(448,512,color='#eeeeee',zorder=-3);ax.axvline(512,color='#444444',lw=.8)
            ax.grid(alpha=.14);ax.set_ylabel(ylabel);ax.set_xlim(448,800);ax.spines[['top','right']].set_visible(False)
        axes[0,col].set_yscale('log');axes[0,col].legend(loc='upper left',fontsize=8);axes[0,col].set_title(f'Clip threshold {cap} | independent panel ranges')
        axes[4,col].set_ylim(.94,1.06);axes[4,col].set_yticks([1]);axes[4,col].text(.5,.2,'Both LR curves coincide at base LR 0.001',ha='center',transform=axes[4,col].transAxes,fontsize=8)
        axes[4,col].set_xlabel('Completed optimizer updates');axes[4,col].set_xticks([448,512,584,656,728,800])
    fig.suptitle('MLP / CIFAR-10 | gradient clipping before Adam: 1.0 versus 0.5\nEach compared with unchanged LR | seed 0 | full 5,000-image objective | same Spark',fontsize=13)
    fig.supxlabel('Shaded: shared prefix. Fork at512; exact replay verified, no hardware change. Solid curves, no smoothing.\nGreen: recorded post-clipping Adam input, states512-799. Rho and eta kappa hat remain raw.',fontsize=9)
    pdf=HERE/'figures/mlp_clipping_thresholds.pdf';fig.savefig(pdf);plt.close(fig)
    import fitz
    doc=fitz.open(pdf);assert len(doc)==1 and not doc[0].get_images();doc[0].get_pixmap(matrix=fitz.Matrix(1.3,1.3)).save(HERE/'scratch/mlp_clipping_thresholds.png')
    for r in values:
        val=metrics[r['series']][r['state']-512][r['metric']] if r['metric']=='optimizer_input_norm' else series[r['series']][r['state']][r['metric']]
        if r['metric']=='applied_lr':val/=.001
        assert val==r['value']
    with (HERE/'plotted_values.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    result=dict(status='completed',summary=summary,vs_unchanged_pct=effects,clipping=clipping,qualification=qualification,source_file_checks=checked,geometry_checks=geometry_checks,plotted_scalar_checks=len(values),pdf_sha256=sha(pdf),sources=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in [BASE/'prefix/boundaries.jsonl',*paths.values(),OLD/'global_clip1/metrics.jsonl',NEW/'global_clip05/metrics.jsonl']])
    (HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['summary','sources']},indent=2));print(json.dumps({n:{k:v for k,v in a.items() if k!='events_gt10'} for n,a in summary.items()},indent=2))
if __name__=='__main__':main()
