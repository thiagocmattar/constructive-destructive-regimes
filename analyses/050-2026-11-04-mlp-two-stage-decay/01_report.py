"""Run227 two-stage decay versus unchanged and single-decay controls."""
import csv,hashlib,importlib.util,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BASE=ROOT/'runs/223-2026-11-04-multispike-interventions/artifacts/spark/payload/worker'
SINGLE=ROOT/'runs/226-2026-11-04-mlp-cosine75/artifacts/spark/payload/worker'
NEW=ROOT/'runs/227-2026-11-04-mlp-two-stage-decay/artifacts/spark/payload/worker'
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    checked=0
    for source in [BASE,SINGLE,NEW]:
        assert json.loads((source.parent.parent/'receipt.json').read_text())['verified']
        for item in json.loads((source.parent/'transfer_inventory.json').read_text())['files']:
            p=source.parent/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];checked+=1
    manifest=json.loads((NEW/'manifest.json').read_text());assert manifest['status']=='completed'
    prefix=rows(BASE/'prefix/boundaries.jsonl')[:-1]
    paths={'unchanged':BASE/'control/boundaries.jsonl','single_decay075':SINGLE/'cosine32_three_quarters/boundaries.jsonl','two_stage_decay':NEW/'cosine32_075_then_05/boundaries.jsonl'}
    series={name:prefix+rows(p) for name,p in paths.items()}
    assert all([r['state'] for r in v]==list(range(801)) for v in series.values())
    for t in range(652):
        for key in ['loss','rho','raw_gradient_norm','eta_kappa_hat','parameter_norm']:assert series['two_stage_decay'][t][key]==series['single_decay075'][t][key]
    spec=importlib.util.spec_from_file_location('events',ROOT/'analyses/036-2026-11-02-controlled-adam-diagnostics/02_extract_loss_spikes.py');events=importlib.util.module_from_spec(spec);spec.loader.exec_module(events)
    summary={}
    for name,data in series.items():
        loss=np.array([r['loss'] for r in data]);exc=[]
        for e in events.excursions(loss):
            ratio=float(loss[e['peak_state']]/loss[e['before_state']])
            if e['peak_state']>650 and ratio>1.1:exc.append(dict(e,ratio=ratio,peak_loss=float(loss[e['peak_state']])))
        summary[name]=dict(mean_513_800=float(loss[513:].mean()),mean_651_800=float(loss[651:].mean()),mean_683_800=float(loss[683:].mean()),last32_mean=float(loss[-32:].mean()),final_loss=float(loss[-1]),max_loss_651_800=float(loss[651:].max()),events_count_after650=len(exc),events_gt10_after650=exc,min_rho_683_800=min(r['rho'] for r in data[683:]),max_gradient_norm_683_800=max(r['raw_gradient_norm'] for r in data[683:]))
    effects={name:{metric:100*(summary['two_stage_decay'][metric]/summary[name][metric]-1) for metric in ['mean_513_800','mean_651_800','mean_683_800','last32_mean','final_loss','max_loss_651_800']} for name in ['unchanged','single_decay075']}
    metrics=[('loss','Training loss'),('rho',r'Raw gradient $\rho_t$'),('raw_gradient_norm',r'Raw gradient norm $\|g_t\|$'),('eta_kappa_hat',r'$\eta\widehat\kappa_t$'),('applied_lr','Applied LR / base LR')]
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(5,2,figsize=(11.5,10.7),sharex='col',layout='constrained');values=[]
    for col,(control,title,label) in enumerate([('unchanged','Compared with unchanged baseline LR','Unchanged baseline'),('single_decay075','Compared with one decay to 0.75x','Cosine32 to 0.75x; then held')]):
        for ax,(metric,ylabel) in zip(axes[:,col],metrics):
            for name,color,legend in [(control,'#777777',label),('two_stage_decay','#009E73','Cosine32 to 0.75x; then to 0.5x')]:
                v=[r for r in series[name] if 448<=r['state']<=800 and (metric!='applied_lr' or r['state']<800)]
                xx=[r['state'] for r in v];yy=[r[metric] if metric!='applied_lr' else r[metric]/.001 for r in v]
                values.extend(dict(column=control,series=name,metric=metric,state=x,value=y) for x,y in zip(xx,yy))
                if metric=='applied_lr':xx=xx+[800];yy=yy+[yy[-1]]
                ax.plot(xx,yy,color=color,lw=.95,ls='-',label=legend,drawstyle='steps-post' if metric=='applied_lr' else 'default')
            ax.axvspan(448,512,color='#eeeeee',zorder=-3);ax.axvspan(650,682,color='#009E73',alpha=.045,zorder=-3)
            for t,c in [(512,'#444444'),(650,'#009E73'),(682,'#009E73')]:ax.axvline(t,color=c,lw=.65,alpha=.65)
            ax.grid(alpha=.14);ax.set_ylabel(ylabel);ax.set_xlim(448,800);ax.spines[['top','right']].set_visible(False)
        axes[0,col].set_yscale('log');axes[0,col].set_title(title,fontsize=11);axes[0,col].legend(loc='lower left',fontsize=8,framealpha=.95)
        axes[4,col].set_xlabel('Completed optimizer updates');axes[4,col].set_ylim(.45,1.06);axes[4,col].set_xticks([448,512,580,650,682,740,800])
    fig.suptitle('MLP / CIFAR-10 | cosine32 to 0.75x, then cosine32 to 0.5x\nDecays start at 512 and 650 | base LR 0.001 | seed 0 | same Spark',fontsize=13)
    (HERE/'figures').mkdir(exist_ok=True);(HERE/'scratch').mkdir(exist_ok=True)
    pdf=HERE/'figures/mlp_two_stage_decay_paired.pdf';fig.savefig(pdf);plt.close(fig)
    import fitz
    doc=fitz.open(pdf);assert len(doc)==1 and not doc[0].get_images();doc[0].get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(HERE/'scratch/mlp_two_stage_decay_paired.png')
    for r in values:
        v=series[r['series']][r['state']][r['metric']]
        if r['metric']=='applied_lr':v/=.001
        assert r['value']==v
    with (HERE/'plotted_values.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    result=dict(status='completed',summary=summary,two_stage_vs_comparator_pct=effects,source_file_checks=checked,plotted_scalar_checks=len(values),pdf_sha256=sha(pdf),sources=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in [BASE/'prefix/boundaries.jsonl',*paths.values()]])
    (HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
