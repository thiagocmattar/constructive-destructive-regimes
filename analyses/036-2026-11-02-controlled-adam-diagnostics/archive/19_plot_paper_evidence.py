"""Four focused vector figures with Analysis024 typography and audited arrays."""
import argparse
import csv
import importlib.util
import shutil
from pathlib import Path

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter
import numpy as np
from sklearn.metrics import precision_recall_curve

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE/'paper_evidence'
spec = importlib.util.spec_from_file_location('temporal', HERE/'15_peak_temporal_analysis.py')
temporal = importlib.util.module_from_spec(spec); spec.loader.exec_module(temporal)
COLORS = ['#0072B2', '#D55E00', '#009E73']
TITLES = ['CIFAR-10 / MLP', 'CIFAR-10 / ResNet-32', 'WikiText-2 / Transformer']
WINDOW = 8
KEYS = ['loss', 'projected', 'raw_cosine', 'gradient', 'momentum']


def style():
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','font.size':9,
        'axes.labelsize':9.5,'axes.titlesize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'pdf.fonttype':42,
        'path.simplify':False})


def mean_curve(y):
    valid = np.flatnonzero(np.isfinite(y))
    assert np.all(np.diff(valid)==1)
    return valid[WINDOW-1:], np.convolve(y[valid], np.ones(WINDOW)/WINDOW, mode='valid')


def main(accept=False):
    manifest_path=OUT/'figure_manifest.json'
    if accept:
        m=temporal.read(manifest_path)
        for p,digest in m['outputs'].items(): assert temporal.sha(ROOT/p)==digest
        m['visual_review']='passed: all four Poppler renders inspected for readable labels, exact axes, legends, reference lines and unclipped observations'
        temporal.dump(manifest_path,m);return
    arms,sources=temporal.load_arms()
    evidence=temporal.read(OUT/'evidence_results.json')
    original=temporal.read(HERE/'peak_temporal_results.json')
    for p in [OUT/'evidence_results.json',HERE/'peak_temporal_results.json',HERE/'peak_forecasts.csv']:
        sources[p.relative_to(ROOT).as_posix()]=temporal.sha(p)
    # Preserve the previously integrated figure versions without touching manuscript copies.
    archive=HERE/'figures/archive/2026-11-02-before-evidence-package'
    archive.mkdir(parents=True,exist_ok=True)
    for name in ['adam_manuscript_overview.pdf','adam_manuscript_spikes.pdf']:
        path=HERE/'figures'/name
        if not (archive/name).exists(): shutil.copyfile(path,archive/name)
    m=dict(generator=Path(__file__).relative_to(ROOT).as_posix(),generator_sha256=temporal.sha(__file__),
        sources=sources,window=WINDOW,clock='Parameter state t; vectors available before update t+1.',
        style_source='analyses/024-2026-10-17-eos-gd-lr-calibration/07_plot_single_rows.py',
        curves=[],outputs={},dimensions={},visual_review='pending',
        archived_outputs={p.relative_to(ROOT).as_posix():temporal.sha(p) for p in archive.glob('*.pdf')})
    style()

    def plot(ax,x,y,kind,task,key,lr=None,normalization=1.,window=None,role='line',**kwargs):
        x,y=np.asarray(x),np.asarray(y)
        artist,=ax.plot(x,y,**kwargs)
        np.testing.assert_array_equal(artist.get_xdata(),x)
        np.testing.assert_array_equal(artist.get_ydata(),y)
        m['curves'].append(dict(figure=kind,task=task,lr=lr,key=key,normalization=normalization,
            window=window,role=role,x=x.tolist(),y=y.tolist()))

    def decorate(axes,titles,top_pad=8):
        for row in range(4):
            for col in range(3):
                ax=axes[row,col]
                letter=f'({chr(97+row*3+col)})'
                if row==0: ax.set_title(letter+' '+titles[col],pad=top_pad)
                else: ax.text(0,1.035,letter,transform=ax.transAxes,va='bottom',fontsize=9)
                ax.grid(alpha=.16,lw=.5);ax.tick_params(length=2.5,width=.6)
                ax.margins(y=.1)
                if row==2:
                    ax.set(ylim=(-1.06,1.06),yticks=[-1,0,1])
                    ax.axhline(0,color='.45',ls=(0,(4,3)),lw=.7)
                if row==1: ax.axhline(2,color='.45',ls=(0,(4,3)),lw=.7)
                if row in [0,3]:
                    ax.set_yscale('log');ax.yaxis.set_minor_formatter(NullFormatter())
                    ax.yaxis.set_major_locator(LogLocator(base=10,numticks=4))
        for ax,label in zip(axes[:,0],['Training loss',r'$\eta\widehat\kappa_t$',
                                       r'$\rho_t$ (raw)', 'Relative norm']): ax.set_ylabel(label)

    def save(fig,name):
        fig.canvas.draw()
        bbox=fig.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans)
        assert bbox.x0>=-1 and bbox.y0>=-1 and bbox.x1<=fig.bbox.width+1 and bbox.y1<=fig.bbox.height+1,(name,bbox.bounds)
        path=HERE/'figures'/name
        fig.savefig(path,metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
        with fitz.open(path) as doc:
            assert len(doc)==1 and not doc[0].get_images()
            for font in doc[0].get_fonts(full=True): assert doc.extract_font(font[0])[3]
            for word in doc[0].get_text('words'):
                assert word[0]>=0 and word[1]>=0 and word[2]<=doc[0].rect.width+.01 and word[3]<=doc[0].rect.height+.01
            m['dimensions'][name]=[doc[0].rect.width/72,doc[0].rect.height/72]
        m['outputs'][path.relative_to(ROOT).as_posix()]=temporal.sha(path)

    fig,axes=plt.subplots(4,3,figsize=(7.16,4.9),sharex='col')
    for col,task in enumerate(temporal.TASKS):
        for color,a in zip(COLORS,[a for a in arms if a['task']==task]):
            for key,row in zip(KEYS,[0,1,2,3,3]):
                values=a[key];divisor=float(values[0]) if row==3 else 1.
                if key=='loss': x=np.arange(len(values));y=values
                else: x,y=mean_curve(values/divisor)
                if key=='projected':
                    valid=np.flatnonzero(np.isfinite(values))
                    plot(axes[row,col],valid,values[valid],'overview',task,key,a['lr'],role='raw_context',
                         color=color,alpha=.22,lw=.35,zorder=1)
                plot(axes[row,col],x,y,'overview',task,key,a['lr'],normalization=divisor,
                     window=None if key=='loss' else WINDOW,color=color,lw=.9,zorder=2,
                     ls='--' if key=='momentum' else '-',label=rf'$\eta={a["lr"]:g}$' if row==0 else None)
        axes[0,col].legend(loc='lower center',bbox_to_anchor=(.5,1.015),ncol=3,
            frameon=False,fontsize=7.6,handlelength=1.0,columnspacing=.5,handletextpad=.25,borderaxespad=0)
        for ax in axes[:,col]: ax.set(xlim=(0,1024),xticks=[0,256,512,768,1024])
        axes[-1,col].set_xlabel(r'Parameter state $t$')
    decorate(axes,TITLES,top_pad=26)
    for ax in axes[1]:
        lower,upper=ax.get_ylim()
        ax.set_yticks([v for v in [-2,0,2,4] if lower<=v<=upper])
        ax.set_ylim(lower,upper)
    fig.legend(handles=[Line2D([],[],color='.25',lw=1,label='Gradient'),
                        Line2D([],[],color='.25',lw=1,ls='--',label='Momentum')],
               loc='lower center',bbox_to_anchor=(.55,.005),ncol=2,frameon=False,fontsize=8)
    fig.subplots_adjust(left=.105,right=.975,top=.88,bottom=.14,hspace=.48,wspace=.31)
    save(fig,'adam_manuscript_overview.pdf')

    fig,axes=plt.subplots(4,3,figsize=(7.16,4.55),sharex='col')
    titles=[]
    for col,e in enumerate(original['selected_examples']):
        a=next(a for a in arms if a['id']==e['arm'])
        x=np.arange(e['start_state'],e['end_state']+1)
        titles.append(TITLES[col]+'\n'+rf'$\eta={a["lr"]:g}$')
        for key,row in zip(KEYS,[0,1,2,3,3]):
            divisor=float(a[key][x[0]]) if row==3 else 1.
            plot(axes[row,col],x,a[key][x]/divisor,'spikes',a['task'],key,a['lr'],normalization=divisor,
                color=COLORS[0] if key=='gradient' else COLORS[1] if key=='momentum' else COLORS[2],
                lw=.85,ls='--' if key=='momentum' else '-',marker='o',ms=1.7,markeredgewidth=0)
        for ax in axes[:,col]:
            ax.axvline(e['episode_onset'],color='.65',lw=.7,ls=':',zorder=0)
            ax.axvline(e['peak_state'],color='.4',lw=.7,ls='--',zorder=0)
            ax.set_xlim(x[0]-.5,x[-1]+.5)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4,integer=True))
        axes[0,col].axhline(e['starting_loss'],color='.65',lw=.55,ls=':')
        axes[-1,col].set_xlabel(r'Parameter state $t$')
    decorate(axes,titles)
    for ax in axes[0]:
        ax.yaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5),numticks=4))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
    axes[-1,0].legend(handles=[Line2D([],[],color=COLORS[0],label='Gradient'),
        Line2D([],[],color=COLORS[1],ls='--',label='Momentum')],frameon=False,fontsize=7.5,
        loc='upper right',handlelength=1.1,borderpad=.1)
    fig.subplots_adjust(left=.105,right=.975,top=.88,bottom=.105,hspace=.48,wspace=.31)
    save(fig,'adam_manuscript_spikes.pdf')

    fig,axes=plt.subplots(4,3,figsize=(7.16,4.8),sharex=True)
    titles=[]
    for col,task in enumerate(temporal.TASKS):
        selected=[e for e in evidence['aligned_events'] if e['task']==task]
        x=np.arange(-8,9)
        titles.append(TITLES[col]+f'\n{len(selected)} large episodes')
        for key,row in zip(KEYS,[0,1,2,3,3]):
            color=COLORS[0] if key=='gradient' else COLORS[1] if key=='momentum' else COLORS[2]
            matrix=np.array([e[key] for e in selected])
            for e,y in zip(selected,matrix):
                plot(axes[row,col],x,y,'events',task,key,e['lr'],role=f'episode_{e["onset"]}',
                    color=color,alpha=.22,lw=.55,ls='--' if key=='momentum' else '-')
            plot(axes[row,col],x,np.median(matrix,axis=0),'events',task,key,selected[0]['lr'],
                role='median',color=color,lw=1.3,ls='--' if key=='momentum' else '-')
        for ax in axes[:,col]:
            ax.axvline(0,color='.45',lw=.7,ls=':');ax.set(xlim=(-8,8),xticks=[-8,-4,0,4,8])
        axes[-1,col].set_xlabel('Updates relative to loss onset')
    decorate(axes,titles)
    axes[0,0].set_ylabel('Loss / baseline')
    for ax in axes[0]:
        ax.axhline(1.1,color='.6',lw=.6,ls='--')
        ax.yaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5),numticks=4))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
    fig.legend(handles=[Line2D([],[],color=COLORS[2],alpha=.3,lw=.7,label='Individual episode'),
        Line2D([],[],color=COLORS[2],lw=1.3,label='Median'),
        Line2D([],[],color=COLORS[0],label='Gradient'),
        Line2D([],[],color=COLORS[1],ls='--',label='Momentum')],ncol=4,loc='lower center',
        bbox_to_anchor=(.55,0),frameon=False,fontsize=7.5,handlelength=1.25,columnspacing=1.)
    fig.subplots_adjust(left=.105,right=.975,top=.885,bottom=.14,hspace=.48,wspace=.31)
    save(fig,'adam_episode_onsets.pdf')

    rows=list(csv.DictReader((HERE/'peak_forecasts.csv').open(newline='',encoding='utf-8')))
    fig,axes=plt.subplots(1,3,figsize=(7.16,2.55),sharex=True,sharey=True)
    for col,task in enumerate(temporal.TASKS):
        subset=[r for r in rows if r['task']==task]
        y=np.array([int(r['target']) for r in subset])
        for name,color,label in [('loss_history',COLORS[0],'Loss history'),
                                 ('projected_change',COLORS[1],r'+ $\eta\widehat\kappa_t$')]:
            p=np.array([float(r['prob_'+name]) for r in subset])
            precision,recall,_=precision_recall_curve(y,p)
            ap=evidence['primary_forecast']['per_task'][task][name]['average_precision']
            plot(axes[col],recall,precision,'forecast',task,name,color=color,lw=1.,
                 drawstyle='steps-post',label=f'{label} (AP {ap:.3f})')
        axes[col].axhline(y.mean(),color='.45',ls=':',lw=.7)
        axes[col].set(title=f'({chr(97+col)}) {TITLES[col]}',xlabel='Recall',xlim=(0,1),ylim=(0,1.02),
                      xticks=[0,.5,1],yticks=[0,.5,1])
        axes[col].grid(alpha=.16,lw=.5)
        axes[col].legend(frameon=False,fontsize=7.5,loc='upper right',handlelength=1.,borderpad=.2)
    axes[0].set_ylabel('Precision')
    fig.subplots_adjust(left=.095,right=.975,top=.80,bottom=.21,wspace=.30)
    save(fig,'adam_forward_precision_recall.pdf')
    temporal.dump(manifest_path,m)
    print(dict(figures=m['outputs'],curves=len(m['curves']),points=sum(len(c['x']) for c in m['curves'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accept-reviewed',action='store_true')
    main(parser.parse_args().accept_reviewed)
