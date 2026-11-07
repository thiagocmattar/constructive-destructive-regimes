"""Three paired beta2 columns using the Analysis058 Figure04 row contract."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, LogLocator, FuncFormatter, NullFormatter
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUN=ROOT/'runs/232-2026-11-07-mlp-beta2-reduction'
BASE=ROOT/'runs/223-2026-11-04-multispike-interventions/artifacts/spark/payload/worker'
NEW=RUN/'artifacts/spark/payload/worker'
SOURCES={}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p):
    SOURCES[p.relative_to(ROOT).as_posix()]=sha(p)
    return [json.loads(s) for s in p.read_text().splitlines()]


def main():
    assert json.loads((RUN/'verification_final.json').read_text())['status']=='passed'
    (HERE/'figures').mkdir(exist_ok=True)
    scratch=ROOT/'tmp/pdfs/analysis059'
    scratch.mkdir(parents=True,exist_ok=True)
    prefix=rows(BASE/'prefix/boundaries.jsonl')[:-1]
    data={'control':prefix+rows(BASE/'control/boundaries.jsonl')}
    betas={'control':.999,'beta2_0p95':.95,'beta2_0p9':.9,'beta2_0p5':.5}
    for name in list(betas)[1:]:
        data[name]=prefix+rows(NEW/name/'boundaries.jsonl')
    for name,series in data.items():
        assert [r['state'] for r in series]==list(range(801))
        for r in series:
            r['plotted_beta2']=.999 if r['state']<512 else betas[name]
    # Use the established fixed-reference event detector on complete prefix+response histories.
    event_source=ROOT/'analyses/036-2026-11-02-controlled-adam-diagnostics/02_extract_loss_spikes.py'
    SOURCES[event_source.relative_to(ROOT).as_posix()]=sha(event_source)
    spec=importlib.util.spec_from_file_location('events',event_source)
    events=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(events)
    summaries={}
    for name,series in data.items():
        losses=np.array([r['loss'] for r in series])
        excursions=[]
        for event in events.excursions(losses):
            ratio=float(losses[event['peak_state']]/losses[event['before_state']])
            if event['peak_state']>512 and ratio>1.1:
                excursions.append(dict(event,peak_ratio=ratio))
        status_path=(BASE/'control' if name=='control' else NEW/name)/'status.json'
        SOURCES[status_path.relative_to(ROOT).as_posix()]=sha(status_path)
        status=json.loads(status_path.read_text())
        summaries[name]=dict(beta2=betas[name],mean_loss=float(losses[513:].mean()),
            last32_mean=float(losses[-32:].mean()),final_loss=float(losses[-1]),
            maximum_loss=float(losses[513:].max()),events_gt10=len(excursions),
            recovered_gt10=sum(not e['right_censored'] for e in excursions),
            censored_gt10=sum(e['right_censored'] for e in excursions),events=excursions,
            test_loss=status['evaluation']['test']['loss'],elapsed_seconds=status['elapsed_seconds'])
    for name in list(betas)[1:]:
        summaries[name]['change_pct']={k:100*(summaries[name][k]/summaries['control'][k]-1)
            for k in ['mean_loss','last32_mean','final_loss','maximum_loss','test_loss']}
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix',
        'font.size':8.5,'axes.labelsize':9,'axes.titlesize':9,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'pdf.fonttype':42,'path.simplify':False})
    fig,axs=plt.subplots(5,3,figsize=(7.16,6.35),sharex='col')
    fig.subplots_adjust(left=.09,right=.986,bottom=.075,top=.865,hspace=.18,wspace=.30)
    metrics=['loss','eta_kappa_hat','rho','raw_gradient_norm','plotted_beta2']
    labels=['Training loss',r'$\eta\widehat\kappa_t$',r'$\rho_t$',r'$\|g_t\|$',r'$\beta_2$']
    values=[]
    for col,name in enumerate(list(betas)[1:]):
        for row,(metric,label) in enumerate(zip(metrics,labels)):
            ax=axs[row,col]
            for key,color in [('control','#777777'),(name,'#009E73')]:
                selected=[r for r in data[key] if 448<=r['state']<=800 and (metric!='plotted_beta2' or r['state']<800)]
                x=[r['state'] for r in selected]
                y=[r[metric] for r in selected]
                values.extend(dict(column=col,series=key,metric='beta2' if metric=='plotted_beta2' else metric,state=a,value=b) for a,b in zip(x,y))
                if metric=='plotted_beta2':
                    x=x+[800];y=y+[y[-1]]
                line,=ax.plot(x,y,color=color,lw=.75,drawstyle='steps-post' if metric=='plotted_beta2' else 'default')
                np.testing.assert_array_equal(line.get_xdata(),x)
                np.testing.assert_array_equal(line.get_ydata(),y)
            ax.axvspan(448,512,color='.92',zorder=-5,lw=0)
            ax.axvline(512,color='.45',lw=.6)
            ax.set_xlim(448,800)
            ax.grid(alpha=.15,lw=.45)
            ax.tick_params(length=2.3,width=.6,pad=2)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4,integer=True))
            ax.margins(y=.10)
            if metric=='loss':
                ax.set_yscale('log')
                ax.yaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5),numticks=5))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
                ax.yaxis.set_minor_formatter(NullFormatter())
            if metric=='rho':
                ax.set(ylim=(-1.08,1.08),yticks=[-1,0,1])
                ax.axhline(0,color='.7',lw=.4)
            if metric=='plotted_beta2':
                ax.set(ylim=(.45,1.03),yticks=[.5,.75,1])
            if col==0:
                ax.set_ylabel(label)
            if row==0:
                ax.set_title(f'({chr(97+col)}) '+rf'$\beta_2={betas[name]:g}$',pad=32)
                ax.legend(handles=[Line2D([],[],color='#777777',lw=1,label=r'Unchanged $\beta_2=0.999$'),
                    Line2D([],[],color='#009E73',lw=1,label=rf'Reduced $\beta_2={betas[name]:g}$')],
                    loc='lower left',bbox_to_anchor=(-.015,1.015),frameon=False,fontsize=7.2,
                    handlelength=1.8,labelspacing=.12,borderaxespad=0)
            if row==4:
                ax.set_xlabel(r'Parameter state $s$')
    fig.text(.54,.978,r'CIFAR-10 / MLP: reduce $\beta_2$ at state 512; constant LR 0.001',ha='center',va='top',fontsize=9.5)
    fig.canvas.draw()
    bbox=fig.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans)
    assert bbox.x0>=0 and bbox.y0>=0 and bbox.x1<=fig.bbox.width and bbox.y1<=fig.bbox.height
    pdf=HERE/'figures/01-beta2-reductions-cifar10-mlp.pdf'
    fig.savefig(pdf,metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)
    with fitz.open(pdf) as doc:
        assert len(doc)==1 and not doc[0].get_images()
        for font in doc[0].get_fonts(full=True):
            assert font[2]!='Type3' and doc.extract_font(font[0])[3]
        for word in doc[0].get_text('words'):
            assert doc[0].rect.contains(fitz.Rect(word[:4]))
        doc[0].get_pixmap(matrix=fitz.Matrix(2,2)).save(scratch/'beta2-reductions.png')
    for r in values:
        key='plotted_beta2' if r['metric']=='beta2' else r['metric']
        assert r['value']==data[r['series']][r['state']][key]
    with (HERE/'plotted_values.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(values[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(values)
    result=dict(status='completed',summary=summaries,source_sha256=SOURCES,
        generator_sha256=sha(Path(__file__)),plotted_values_sha256=sha(HERE/'plotted_values.csv'),
        plotted_scalar_checks=len(values),pdf_sha256=sha(pdf),
        layout_reference='Analysis058 Figure04; final row replaced by beta2',smoothing=None,
        coverage='states448..800; outcomes513..800; beta2 applied at512..799 with last interval closed at800')
    (HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({name:{k:v for k,v in s.items() if k!='events'} for name,s in summaries.items()},indent=2))
    print(f'PASS: {len(values)} plotted scalars and one vector PDF.')


if __name__=='__main__':
    main()
