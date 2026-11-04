"""Run224 source-verified paired MLP return-to-.75 comparison; no training."""
import csv,hashlib,importlib.util,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'runs/223-2026-11-04-multispike-interventions/artifacts/spark/payload/worker'
NEW=ROOT/'runs/224-2026-11-04-mlp-lr-return/artifacts/spark/payload/worker'
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    checked=0
    for src in [OLD,NEW]:
        assert read(src.parent.parent/'receipt.json')['verified']
        for item in read(src.parent/'transfer_inventory.json')['files']:
            p=src.parent/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];checked+=1
    assert read(NEW/'manifest.json')['status']=='completed'
    prefix=rows(OLD/'prefix/boundaries.jsonl')[:-1]
    data={'unchanged':prefix+rows(OLD/'control/boundaries.jsonl')}
    arms=['fixed_half','cosine32_half']
    for arm in arms:
        data[arm]=prefix+rows(OLD/arm/'boundaries.jsonl')
        data[arm+'_return075']=prefix+rows(NEW/(arm+'_return075')/'boundaries.jsonl')
        for state in range(651):
            for metric in ['loss','raw_gradient_norm','rho','eta_kappa_hat','parameter_norm']:
                assert data[arm][state][metric]==data[arm+'_return075'][state][metric]
    for name,v in data.items():assert [r['state'] for r in v]==list(range(801))
    spec=importlib.util.spec_from_file_location('events',ROOT/'analyses/036-2026-11-02-controlled-adam-diagnostics/02_extract_loss_spikes.py');events=importlib.util.module_from_spec(spec);spec.loader.exec_module(events)
    summary={}
    for name,v in data.items():
        loss=np.array([r['loss'] for r in v]);exc=[]
        for e in events.excursions(loss):
            ratio=float(loss[e['peak_state']]/loss[e['before_state']])
            if e['peak_state']>650 and ratio>1.1:exc.append(dict(e,ratio=ratio))
        summary[name]=dict(mean_513_800=float(loss[513:].mean()),mean_651_800=float(loss[651:].mean()),last32_mean=float(loss[-32:].mean()),final_loss=float(loss[-1]),max_loss_651_800=float(loss[651:].max()),events_gt10_after650=exc,events_count_after650=len(exc),min_rho_651_800=min(r['rho'] for r in v[651:]),max_gradient_norm_651_800=max(r['raw_gradient_norm'] for r in v[651:]))
    comparisons={}
    for arm in arms:
        comparisons[arm]={metric:100*(summary[arm+'_return075'][metric]/summary[arm][metric]-1) for metric in ['mean_651_800','last32_mean','final_loss']}
    metrics=[('loss','Training loss'),('rho',r'Raw gradient $\rho_t$'),('raw_gradient_norm',r'Raw gradient norm $\|g_t\|$'),('eta_kappa_hat',r'$\eta\widehat\kappa_t$'),('applied_lr','Applied LR / base LR')]
    plt.rcParams.update({'font.size':10,'axes.titlesize':12,'pdf.fonttype':42})
    fig,axes=plt.subplots(5,2,figsize=(11.5,10.7),sharex='col',layout='constrained');values=[]
    for col,(arm,title) in enumerate(zip(arms,['Immediate half from 512','Cosine32 to half from 512'])):
        for row,(metric,label) in enumerate(metrics):
            ax=axes[row,col]
            for name,color,width,legend in [('unchanged','#888888',.85,'Unchanged base LR'),(arm,'#0072B2',1.0,'Held at 0.5x'),(arm+'_return075','#CC3377',1.0,'Return to 0.75x at 650')]:
                v=[r for r in data[name] if 448<=r['state']<=800 and (metric!='applied_lr' or r['state']<800)]
                xx=[r['state'] for r in v];yy=[r[metric] if metric!='applied_lr' else r[metric]/.001 for r in v]
                values.extend(dict(column=arm,series=name,metric=metric,state=x,value=y) for x,y in zip(xx,yy))
                if metric=='applied_lr':xx=xx+[800];yy=yy+[yy[-1]]
                ax.plot(xx,yy,color=color,lw=width,ls='-',label=legend,drawstyle='steps-post' if metric=='applied_lr' else 'default')
            ax.axvspan(448,512,color='#eeeeee',zorder=-3)
            ax.axvline(512,color='#444444',lw=.7);ax.axvline(650,color='#CC3377',lw=.8,alpha=.65)
            ax.grid(alpha=.14);ax.set_ylabel(label);ax.set_xlim(448,800);ax.spines[['top','right']].set_visible(False)
            if row==0:ax.set_yscale('log');ax.set_title(title);ax.legend(loc='lower left',fontsize=8,framealpha=.95)
            if row==4:ax.set_xlabel('Completed optimizer updates');ax.set_ylim(.45,1.06);ax.set_xticks([448,512,576,650,720,800])
    fig.suptitle('MLP / CIFAR-10 | LR return at 650 | base LR = 0.001 | seed 0\nAll trajectories on Spark; exact replay through 650; solid unsmoothed observations',fontsize=13)
    (HERE/'figures').mkdir(exist_ok=True);(HERE/'scratch').mkdir(exist_ok=True)
    pdf=HERE/'figures/mlp_lr_return_paired.pdf';fig.savefig(pdf);plt.close(fig)
    import fitz
    doc=fitz.open(pdf);assert len(doc)==1 and len(doc[0].get_images())==0
    doc[0].get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(HERE/'scratch/mlp_lr_return_paired.png')
    for r in values:
        expected=data[r['series']][r['state']][r['metric']]
        if r['metric']=='applied_lr':expected/=.001
        assert r['value']==expected
    with (HERE/'plotted_values.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    sources=[OLD/'prefix/boundaries.jsonl',OLD/'control/boundaries.jsonl']+[p/arm/'boundaries.jsonl' for p,arm in [(OLD,'fixed_half'),(OLD,'cosine32_half'),(NEW,'fixed_half_return075'),(NEW,'cosine32_half_return075')]]
    result=dict(status='completed',summary=summary,return_vs_held_half_pct=comparisons,source_file_checks=checked,plotted_scalar_checks=len(values),pdf_sha256=sha(pdf),source_files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in sources])
    (HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
