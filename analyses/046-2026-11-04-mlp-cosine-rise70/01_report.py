"""Source-verified Run225 cosine70-rise comparison with completed matched paths."""
import csv,hashlib,importlib.util,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BASE=ROOT/'runs/223-2026-11-04-multispike-interventions/artifacts/spark/payload/worker'
ABRUPT=ROOT/'runs/224-2026-11-04-mlp-lr-return/artifacts/spark/payload/worker'
NEW=ROOT/'runs/225-2026-11-04-mlp-cosine-rise70/artifacts/spark/payload/worker'
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    checked=0
    for src in [BASE,ABRUPT,NEW]:
        assert read(src.parent.parent/'receipt.json')['verified']
        for item in read(src.parent/'transfer_inventory.json')['files']:
            p=src.parent/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];checked+=1
    manifest=read(NEW/'manifest.json');assert manifest['status']=='completed'
    prefix=rows(BASE/'prefix/boundaries.jsonl')[:-1]
    source={'unchanged':BASE/'control','held_half':BASE/'cosine32_half','abrupt075':ABRUPT/'cosine32_half_return075','cosine70_rise':NEW/'cosine32_half_rise70'}
    data={name:prefix+rows(p/'boundaries.jsonl') for name,p in source.items()}
    for name,v in data.items():assert [r['state'] for r in v]==list(range(801))
    for state in range(651):
        for key in ['loss','raw_gradient_norm','rho','eta_kappa_hat','parameter_norm']:
            assert data['cosine70_rise'][state][key]==data['held_half'][state][key]==data['abrupt075'][state][key]
    spec=importlib.util.spec_from_file_location('events',ROOT/'analyses/036-2026-11-02-controlled-adam-diagnostics/02_extract_loss_spikes.py');events=importlib.util.module_from_spec(spec);spec.loader.exec_module(events)
    summary={}
    for name,v in data.items():
        loss=np.array([r['loss'] for r in v]);exc=[]
        for e in events.excursions(loss):
            ratio=float(loss[e['peak_state']]/loss[e['before_state']])
            if e['peak_state']>650 and ratio>1.1:exc.append(dict(e,ratio=ratio,peak_loss=float(loss[e['peak_state']]),peak_lr_ratio=v[e['peak_state']]['applied_lr']/.001 if e['peak_state']<800 else None))
        summary[name]=dict(mean_513_800=float(loss[513:].mean()),mean_651_800=float(loss[651:].mean()),mean_651_720=float(loss[651:721].mean()),mean_721_800=float(loss[721:].mean()),last32_mean=float(loss[-32:].mean()),final_loss=float(loss[-1]),max_loss_651_800=float(loss[651:].max()),events_gt10_after650=exc,events_count_after650=len(exc),min_rho_651_800=min(r['rho'] for r in v[651:]),max_gradient_norm_651_800=max(r['raw_gradient_norm'] for r in v[651:]))
    comparisons={name:{metric:100*(summary['cosine70_rise'][metric]/summary[name][metric]-1) for metric in ['mean_651_800','mean_721_800','last32_mean','final_loss','max_loss_651_800']} for name in ['unchanged','held_half','abrupt075']}
    metrics=[('loss','Training loss'),('rho',r'Raw gradient $\rho_t$'),('raw_gradient_norm',r'Raw gradient norm $\|g_t\|$'),('eta_kappa_hat',r'$\eta\widehat\kappa_t$'),('applied_lr','Applied LR / base LR')]
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'pdf.fonttype':42})
    fig,axes=plt.subplots(5,3,figsize=(15,10.5),sharex='col',layout='constrained');values=[]
    for col,(control,title,label) in enumerate([('unchanged','Compared with unchanged base LR','Unchanged'),('held_half','Compared with cosine32 held at half','Held at 0.5x'),('abrupt075','Compared with abrupt rise at 650','Abrupt return to 0.75x')]):
        for row,(metric,ylabel) in enumerate(metrics):
            ax=axes[row,col]
            for name,color,width,legend in [(control,'#777777',.95,label),('cosine70_rise','#009E73',1.0,'Cosine70 rise to 0.75x')]:
                v=[r for r in data[name] if 448<=r['state']<=800 and (metric!='applied_lr' or r['state']<800)]
                xx=[r['state'] for r in v];yy=[r[metric] if metric!='applied_lr' else r[metric]/.001 for r in v]
                values.extend(dict(column=control,series=name,metric=metric,state=x,value=y) for x,y in zip(xx,yy))
                if metric=='applied_lr':xx=xx+[800];yy=yy+[yy[-1]]
                ax.plot(xx,yy,color=color,lw=width,ls='-',label=legend,drawstyle='steps-post' if metric=='applied_lr' else 'default')
            ax.axvspan(448,512,color='#eeeeee',zorder=-3);ax.axvspan(650,720,color='#009E73',alpha=.045,zorder=-3)
            for state,color in [(512,'#444444'),(650,'#009E73'),(720,'#009E73')]:ax.axvline(state,color=color,lw=.6,alpha=.6)
            ax.grid(alpha=.14);ax.set_ylabel(ylabel);ax.set_xlim(448,800);ax.spines[['top','right']].set_visible(False)
            if row==0:ax.set_yscale('log');ax.set_title(title);ax.legend(loc='lower left',fontsize=8,framealpha=.95)
            if row==4:ax.set_xlabel('Completed optimizer updates');ax.set_ylim(.45,1.06);ax.set_xticks([448,512,580,650,720,800])
    fig.suptitle('MLP / CIFAR-10 | cosine32 down, hold to 650, cosine70 up to 0.75x at 720\nSeed 0; base LR 0.001; same Spark; exact shared state at 650; raw unsmoothed curves',fontsize=13)
    (HERE/'figures').mkdir(exist_ok=True);(HERE/'scratch').mkdir(exist_ok=True)
    pdf=HERE/'figures/mlp_cosine70_rise_paired.pdf';fig.savefig(pdf);plt.close(fig)
    import fitz
    doc=fitz.open(pdf);assert len(doc)==1 and len(doc[0].get_images())==0
    doc[0].get_pixmap(matrix=fitz.Matrix(1.3,1.3)).save(HERE/'scratch/mlp_cosine70_rise_paired.png')
    for r in values:
        expected=data[r['series']][r['state']][r['metric']]
        if r['metric']=='applied_lr':expected/=.001
        assert r['value']==expected
    with (HERE/'plotted_values.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    result=dict(status='completed',summary=summary,cosine70_vs_comparator_pct=comparisons,source_file_checks=checked,plotted_scalar_checks=len(values),pdf_sha256=sha(pdf),source_files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in [BASE/'prefix/boundaries.jsonl']+[p/'boundaries.jsonl' for p in source.values()]])
    (HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
