"""Verified paired MLP/ResNet previews for each approved Run223 intervention."""
import csv,hashlib,importlib.util,json,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];RUN=ROOT/'runs/223-2026-11-04-multispike-interventions'
CASES={'mlp':dict(folder='spark',start=512,end=800,plot_start=448,base=.001,title='MLP | LR 0.001\nSpark GB10 | 5,000 training images'),'resnet':dict(folder='duqoihms8w1gle',start=256,end=576,plot_start=192,base=.003,title='ResNet-32 | LR 0.003\nRTX 4090 | 50,000 training images')}
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    observed={};checks=0;sources=[];summary=[]
    spec=importlib.util.spec_from_file_location('events',ROOT/'analyses/036-2026-11-02-controlled-adam-diagnostics/02_extract_loss_spikes.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    for case,sp in CASES.items():
        base=RUN/'artifacts'/sp['folder'];source=base/'payload/worker'
        assert json.loads((base/'receipt.json').read_text())['verified']
        for item in json.loads((base/'payload/transfer_inventory.json').read_text())['files']:
            p=base/'payload'/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];checks+=1
        manifest=json.loads((source/'manifest.json').read_text());assert manifest['status']=='completed' and manifest['control']=='fresh_same_worker'
        assert json.loads((source/'qualification/qualification.json').read_text())['status']=='passed'
        prefix=rows(source/'prefix/boundaries.jsonl')[:-1]
        series={name:prefix+rows(source/folder/'boundaries.jsonl') for name,folder in [('unchanged','control'),('fixed_half','fixed_half'),('cosine32_half','cosine32_half')]}
        observed[case]=series;control=np.array([r['loss'] for r in series['unchanged']])[sp['start']+1:]
        for name,data in series.items():
            assert [r['state'] for r in data]==list(range(sp['end']+1))
            assert data[sp['start']]['loss']==series['unchanged'][sp['start']]['loss']
            losses=np.array([r['loss'] for r in data]);tail=losses[sp['start']+1:];events=[]
            for e in ev.excursions(losses):
                ratio=float(losses[e['peak_state']]/losses[e['before_state']])
                if e['peak_state']>sp['start'] and ratio>1.1:events.append(dict(e,ratio=ratio))
            summary.append(dict(case=case,arm=name,mean_loss=float(tail.mean()),mean_change_pct=float(100*(tail.mean()/control.mean()-1)),last32_mean=float(tail[-32:].mean()),last32_change_pct=float(100*(tail[-32:].mean()/control[-32:].mean()-1)),final_loss=float(tail[-1]),events_gt10=len(events),censored=sum(e['right_censored'] for e in events),events=events))
        for folder in ['prefix','control','fixed_half','cosine32_half']:
            p=source/folder/'boundaries.jsonl';sources.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)))
    metrics=[('loss','Training loss'),('rho',r'Raw gradient $\rho_t$'),('raw_gradient_norm',r'Raw gradient norm $\|g_t\|$'),('eta_kappa_hat',r'$\eta\widehat\kappa_t$'),('applied_lr','Applied LR / base LR')]
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42});values=[];pdfs=[]
    for arm,title,color in [('fixed_half','Immediate and permanent half LR','#0072B2'),('cosine32_half','Cosine32 down to half LR, then held','#009E73')]:
        fig,axes=plt.subplots(5,2,figsize=(11.5,10.8),sharex='col',layout='constrained')
        for col,(case,sp) in enumerate(CASES.items()):
            for ax,(metric,label) in zip(axes[:,col],metrics):
                for name,c,legend in [('unchanged','#777777','Unchanged baseline'),(arm,color,'Intervention')]:
                    data=[r for r in observed[case][name] if sp['plot_start']<=r['state']<=sp['end'] and (metric!='applied_lr' or r['state']<sp['end'])]
                    xx=[r['state'] for r in data];yy=[r[metric] if metric!='applied_lr' else r[metric]/sp['base'] for r in data]
                    values.extend(dict(figure=arm,case=case,series=name,metric=metric,state=x,value=y) for x,y in zip(xx,yy))
                    if metric=='applied_lr':xx=xx+[sp['end']];yy=yy+[yy[-1]]
                    ax.plot(xx,yy,color=c,lw=.95,ls='-',label=legend,drawstyle='steps-post' if metric=='applied_lr' else 'default')
                ax.axvspan(sp['plot_start'],sp['start'],color='#eeeeee',zorder=-3);ax.axvline(sp['start'],color='#444444',lw=.7)
                ax.grid(alpha=.14);ax.set_ylabel(label);ax.set_xlim(sp['plot_start'],sp['end']);ax.spines[['top','right']].set_visible(False)
            axes[0,col].set_yscale('log');axes[0,col].set_title(sp['title'],fontsize=11);axes[0,col].legend(loc='lower left',fontsize=8,framealpha=.95)
            axes[4,col].set_xlabel('Completed optimizer updates');axes[4,col].set_ylim(.45,1.06)
            axes[4,col].set_xticks([448,512,576,650,720,800] if case=='mlp' else [192,256,320,384,448,512,576])
        fig.suptitle(title+'\nSeed 0 | fresh same-worker controls | solid unsmoothed curves',fontsize=13)
        (HERE/'figures').mkdir(exist_ok=True);(HERE/'scratch').mkdir(exist_ok=True)
        pdf=HERE/'figures'/('mlp_resnet_'+arm+'.pdf');fig.savefig(pdf);plt.close(fig)
        import fitz
        doc=fitz.open(pdf);assert len(doc)==1 and not doc[0].get_images();doc[0].get_pixmap(matrix=fitz.Matrix(1.4,1.4)).save(HERE/'scratch'/('mlp_resnet_'+arm+'.png'))
        pdfs.append(dict(path=pdf.relative_to(ROOT).as_posix(),sha256=sha(pdf)))
    for r in values:
        v=observed[r['case']][r['series']][r['state']][r['metric']]
        if r['metric']=='applied_lr':v/=CASES[r['case']]['base']
        assert r['value']==v
    with (HERE/'mlp_resnet_values.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    result=dict(status='MLP_and_ResNet_complete_Transformer_pending',summary=summary,source_file_checks=checks,plotted_scalar_checks=len(values),sources=sources,pdfs=pdfs)
    (HERE/'mlp_resnet_preliminary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in x.items() if k!='events'} for x in summary],indent=2));print('Files checked',checks,'; plotted values',len(values))
if __name__=='__main__':main()
