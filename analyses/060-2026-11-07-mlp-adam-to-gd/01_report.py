"""Full-window Adam-to-GD comparison from authenticated retained evidence."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUN=ROOT/'runs/233-2026-11-07-mlp-adam-to-gd'
BASE=ROOT/'runs/223-2026-11-04-multispike-interventions/artifacts/spark/payload/worker'
RUN234=ROOT/'runs/234-2026-11-07-mlp-gd-learning-rates'
RUN235=ROOT/'runs/235-2026-11-07-mlp-gd-lr0p03'
NEW=RUN/'artifacts/spark/payload/worker/gd_lr0p02'
SOURCES={}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p):
    SOURCES[p.relative_to(ROOT).as_posix()]=sha(p)
    return [json.loads(s) for s in p.read_text().splitlines()]


def main():
    for run in [RUN,RUN234,RUN235]:
        source=run/'verification_final.json'
        assert json.loads(source.read_text())['status']=='passed'
        SOURCES[source.relative_to(ROOT).as_posix()]=sha(source)
    audit_path=RUN234/'failure_audit.json'
    audit=json.loads(audit_path.read_text())
    assert audit['state']==547 and audit['loss']=='inf' and audit['status']=='verified_nonfinite_boundary'
    SOURCES[audit_path.relative_to(ROOT).as_posix()]=sha(audit_path)
    (HERE/'figures').mkdir(exist_ok=True)
    scratch=ROOT/'tmp/pdfs/analysis060'
    scratch.mkdir(parents=True,exist_ok=True)
    prefix=rows(BASE/'prefix/boundaries.jsonl')[:-1]
    paths={'adam':BASE/'control','gd_lr0p02':NEW,
        'gd_lr0p03':RUN235/'artifacts/spark/payload/worker/gd_lr0p03',
        'gd_lr0p04':RUN234/'artifacts/spark/payload/worker/gd_lr0p04',
        'gd_lr0p08':RUN234/'artifacts/spark/payload/worker/gd_lr0p08'}
    series={name:prefix+rows(path/'boundaries.jsonl') for name,path in paths.items()}
    for name,data in series.items():
        assert [r['state'] for r in data]==list(range(547 if name=='gd_lr0p08' else 801))
    event_source=ROOT/'analyses/036-2026-11-02-controlled-adam-diagnostics/02_extract_loss_spikes.py'
    SOURCES[event_source.relative_to(ROOT).as_posix()]=sha(event_source)
    spec=importlib.util.spec_from_file_location('events',event_source)
    events=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(events)
    summary={}
    for name,data in series.items():
        loss=np.array([r['loss'] for r in data])
        selected=[]
        for event in events.excursions(loss):
            ratio=float(loss[event['peak_state']]/loss[event['before_state']])
            if event['peak_state']>512 and ratio>1.1:
                selected.append(dict(event,peak_ratio=ratio))
        p=paths[name]/'status.json'
        SOURCES[p.relative_to(ROOT).as_posix()]=sha(p)
        status=json.loads(p.read_text())
        complete=status['status']=='completed'
        summary[name]=dict(mean_loss=float(loss[513:].mean()) if complete else None,
            last32_mean=float(loss[-32:].mean()) if complete else None,
            final_loss=float(loss[-1]) if complete else None,
            maximum_loss=float(loss[513:].max()) if complete else None,events_gt10=len(selected),
            recovered_gt10=sum(not e['right_censored'] for e in selected),
            censored_gt10=sum(e['right_censored'] for e in selected),events=selected,
            loss_increasing_updates=int(np.sum(np.diff(loss[512:])>0)),
            test_loss=status['evaluation']['test']['loss'] if complete else None,
            elapsed_seconds=status['elapsed_seconds'],execution_status=status['status'],
            completed_updates=status['updates'],last_finite_state=len(loss)-1,last_finite_loss=float(loss[-1]),
            loss_comparison_updates=len(loss)-513,
            failed_state=None if complete else status['completed_state'])
    effects={name:{key:100*(summary[name][key]/summary['adam'][key]-1)
             for key in ['mean_loss','last32_mean','final_loss','maximum_loss','test_loss']}
             for name in ['gd_lr0p02','gd_lr0p03','gd_lr0p04']}
    # Preserve every original numerical reduction, including event details.
    old=json.loads((HERE/'archive/2026-11-07-single-rate/results.json').read_text())
    archive=HERE/'archive/2026-11-07-single-rate'
    for filename,key in [('01_report.py','generator_sha256'),('01-adam-to-gd-cifar10-mlp.pdf','pdf_sha256'),
                         ('plotted_values.csv','plotted_values_sha256')]:
        assert sha(archive/filename)==old[key]
    for old_name,new_name in [('adam','adam'),('gd','gd_lr0p02')]:
        for key,value in old['summary'][old_name].items():
            assert summary[new_name][key]==value,(new_name,key)
    prior=HERE/'archive/2026-11-07-rates-0p02-0p04-0p08'
    prior_result=json.loads((prior/'results.json').read_text())
    for filename,key in [('01_report.py','generator_sha256'),('01-adam-to-gd-cifar10-mlp.pdf','pdf_sha256'),
                         ('plotted_values.csv','plotted_values_sha256')]:
        assert sha(prior/filename)==prior_result[key]
    for name,value in prior_result['summary'].items():
        assert summary[name]==value,name
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','font.size':9,
        'axes.labelsize':9.5,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'path.simplify':False})
    fig,axes=plt.subplots(5,3,figsize=(7.16,7.1),sharex=True)
    fig.subplots_adjust(left=.10,right=.98,bottom=.10,top=.87,hspace=.22,wspace=.35)
    values=[]
    metrics=['loss','eta_kappa_hat','rho','raw_gradient_norm','applied_lr']
    labels=['Training loss',r'$\eta\widehat\kappa_t$',r'$\rho_t$',r'$\|g_t\|$','Applied LR']
    rates=[.02,.03,.04]
    for col,lr in enumerate(rates):
        arm='gd_lr'+str(lr).replace('.','p')
        for row,(metric,label) in enumerate(zip(metrics,labels)):
            ax=axes[row,col]
            for name,color,title in [('adam','#777777','Continued Adam (LR 0.001)'),(arm,'#0072B2','Switch to GD (column LR)')]:
                selected=[r for r in series[name] if 448<=r['state']<=800 and (metric!='applied_lr' or r['state']<800)]
                x,y=[r['state'] for r in selected],[r[metric] for r in selected]
                values.extend(dict(column_lr=lr,series=name,metric=metric,state=a,value=b) for a,b in zip(x,y))
                if metric=='applied_lr':
                    x=x+[800];y=y+[y[-1]]
                line,=ax.plot(x,y,color=color,lw=.75,label=title,drawstyle='steps-post' if metric=='applied_lr' else 'default')
                np.testing.assert_array_equal(line.get_xdata(),x)
                np.testing.assert_array_equal(line.get_ydata(),y)
            ax.axvspan(448,512,color='.92',zorder=-5,lw=0)
            ax.axvline(512,color='.45',lw=.55)
            ax.set(xlim=(448,800))
            if col==0:
                ax.set_ylabel(label)
            ax.grid(alpha=.15,lw=.45)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            if metric in ['loss','raw_gradient_norm']:
                ax.set_yscale('log')
            if metric=='rho':
                ax.set(ylim=(-1.08,1.08),yticks=[-1,0,1])
            if metric=='applied_lr':
                ax.set(ylim=(0,.044),yticks=[0,.02,.04])
        axes[0,col].set_title(f'GD learning rate {lr:g}',fontsize=9.5,pad=6)
        axes[-1,col].set_xlabel(r'Parameter state $s$')
        axes[-1,col].set_xticks([448,512,640,800])
    # Match each row's range across all three requested, complete branches.
    for row in [0,1,3]:
        low=min(axes[row,c].get_ylim()[0] for c in range(3))
        high=max(axes[row,c].get_ylim()[1] for c in range(3))
        for c in range(3):
            axes[row,c].set_ylim(low,high)
    handles,legend=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='upper center',bbox_to_anchor=(.54,.943),frameon=False,ncol=2,fontsize=8)
    fig.suptitle('CIFAR-10 / MLP: Adam to plain GD at state 512',y=.984,fontsize=11)
    fig.text(.5,.025,'Loss and gradient norm: log scales. Each row uses the same y range across all three learning rates.',
        ha='center',fontsize=7.2)
    fig.canvas.draw()
    bbox=fig.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans)
    assert bbox.x0>=0 and bbox.y0>=0 and bbox.x1<=fig.bbox.width and bbox.y1<=fig.bbox.height
    pdf=HERE/'figures/01-adam-to-gd-cifar10-mlp.pdf'
    fig.savefig(pdf,metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)
    with fitz.open(pdf) as doc:
        assert len(doc)==1 and not doc[0].get_images()
        for font in doc[0].get_fonts(full=True):
            assert font[2]!='Type3' and doc.extract_font(font[0])[3]
        for word in doc[0].get_text('words'):
            assert doc[0].rect.contains(fitz.Rect(word[:4]))
        doc[0].get_pixmap(matrix=fitz.Matrix(2,2)).save(scratch/'adam-to-gd.png')
    for r in values:
        assert r['value']==series[r['series']][r['state']][r['metric']]
    previous=list(csv.DictReader((HERE/'archive/2026-11-07-single-rate/plotted_values.csv').open()))
    first=[r for r in values if r['column_lr']==.02]
    assert len(previous)==len(first)
    for a,b in zip(previous,first):
        assert a['series']==('gd' if b['series']=='gd_lr0p02' else b['series'])
        assert a['metric']==b['metric'] and int(a['state'])==b['state'] and float(a['value'])==b['value']
    with (HERE/'plotted_values.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(values[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(values)
    result=dict(status='completed',summary=summary,gd_vs_adam_pct=effects,
        source_sha256=SOURCES,generator_sha256=sha(Path(__file__)),pdf_sha256=sha(pdf),
        plotted_values_sha256=sha(HERE/'plotted_values.csv'),plotted_scalar_checks=len(values),original_scalar_checks=len(previous),smoothing=None,
        displayed_lrs=rates,historical_incomplete_branch='gd_lr0p08',failed_state=547,failed_loss='inf',
        note='Figure shows complete0.02/0.03/0.04 branches. Historical0.08 failure and summary retained; no full-window statistics for that incomplete branch.')
    (HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(summary={k:{a:b for a,b in v.items() if a!='events'} for k,v in summary.items()},effects=effects,plotted_values=len(values)),indent=2))


if __name__=='__main__':
    main()
