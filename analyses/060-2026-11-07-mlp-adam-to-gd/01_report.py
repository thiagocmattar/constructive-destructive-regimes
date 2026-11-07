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
NEW=RUN/'artifacts/spark/payload/worker/gd_lr0p02'
SOURCES={}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p):
    SOURCES[p.relative_to(ROOT).as_posix()]=sha(p)
    return [json.loads(s) for s in p.read_text().splitlines()]


def main():
    assert json.loads((RUN/'verification_final.json').read_text())['status']=='passed'
    (HERE/'figures').mkdir(exist_ok=True)
    scratch=ROOT/'tmp/pdfs/analysis060'
    scratch.mkdir(parents=True,exist_ok=True)
    prefix=rows(BASE/'prefix/boundaries.jsonl')[:-1]
    paths={'adam':BASE/'control','gd':NEW}
    series={name:prefix+rows(path/'boundaries.jsonl') for name,path in paths.items()}
    assert all([r['state'] for r in data]==list(range(801)) for data in series.values())
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
        summary[name]=dict(mean_loss=float(loss[513:].mean()),last32_mean=float(loss[-32:].mean()),
            final_loss=float(loss[-1]),maximum_loss=float(loss[513:].max()),events_gt10=len(selected),
            recovered_gt10=sum(not e['right_censored'] for e in selected),
            censored_gt10=sum(e['right_censored'] for e in selected),events=selected,
            loss_increasing_updates=int(np.sum(np.diff(loss[512:])>0)),
            test_loss=status['evaluation']['test']['loss'],elapsed_seconds=status['elapsed_seconds'])
    effects={key:100*(summary['gd'][key]/summary['adam'][key]-1)
             for key in ['mean_loss','last32_mean','final_loss','maximum_loss','test_loss']}
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','font.size':9,
        'axes.labelsize':9.5,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'path.simplify':False})
    fig,axes=plt.subplots(5,1,figsize=(7.16,7.4),sharex=True)
    fig.subplots_adjust(left=.12,right=.98,bottom=.07,top=.89,hspace=.19)
    values=[]
    for ax,metric,label in zip(axes,['loss','eta_kappa_hat','rho','raw_gradient_norm','applied_lr'],
            ['Training loss',r'$\eta\widehat\kappa_t$',r'$\rho_t$',r'$\|g_t\|$','Applied LR']):
        for name,color,title in [('adam','#777777','Continued Adam, LR 0.001'),('gd','#0072B2','Switch to GD, LR 0.02')]:
            selected=[r for r in series[name] if 448<=r['state']<=800 and (metric!='applied_lr' or r['state']<800)]
            x,y=[r['state'] for r in selected],[r[metric] for r in selected]
            values.extend(dict(series=name,metric=metric,state=a,value=b) for a,b in zip(x,y))
            if metric=='applied_lr':
                x=x+[800];y=y+[y[-1]]
            line,=ax.plot(x,y,color=color,lw=.85,label=title,drawstyle='steps-post' if metric=='applied_lr' else 'default')
            np.testing.assert_array_equal(line.get_xdata(),x)
            np.testing.assert_array_equal(line.get_ydata(),y)
        ax.axvspan(448,512,color='.92',zorder=-5,lw=0)
        ax.axvline(512,color='.45',lw=.6)
        ax.set(xlim=(448,800),ylabel=label)
        ax.grid(alpha=.15,lw=.45)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        if metric=='loss':
            ax.set_yscale('log')
        if metric=='rho':
            ax.set(ylim=(-1.08,1.08),yticks=[-1,0,1])
        if metric=='applied_lr':
            ax.set(ylim=(0,.022),yticks=[.001,.01,.02])
    axes[0].legend(loc='lower left',bbox_to_anchor=(0,1.03),frameon=False,ncol=2,fontsize=8)
    axes[-1].set_xlabel(r'Parameter state $s$')
    axes[-1].set_xticks([448,512,576,640,704,768,800])
    fig.suptitle('CIFAR-10 / MLP: Adam to plain GD at state 512',y=.98,fontsize=11)
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
    with (HERE/'plotted_values.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(values[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(values)
    result=dict(status='completed',summary=summary,gd_vs_adam_pct=effects,
        source_sha256=SOURCES,generator_sha256=sha(Path(__file__)),pdf_sha256=sha(pdf),
        plotted_values_sha256=sha(HERE/'plotted_values.csv'),plotted_scalar_checks=len(values),smoothing=None)
    (HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(summary={k:{a:b for a,b in v.items() if a!='events'} for k,v in summary.items()},effects=effects,plotted_values=len(values)),indent=2))


if __name__=='__main__':
    main()
