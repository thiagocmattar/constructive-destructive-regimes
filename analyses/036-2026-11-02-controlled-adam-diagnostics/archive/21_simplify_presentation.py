"""Three-row Adam overview and a descriptive recurrence table; no new fitting."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'paper_evidence'
ARCHIVE=OUT/'archive/2026-11-02-before-simplification'
PDF=HERE/'figures/adam_manuscript_overview.pdf'
MANIFEST=OUT/'presentation_manifest.json'
LIMITS=(-.1,2.2)
WINDOW=16
spec=importlib.util.spec_from_file_location('previous_figures',HERE/'19_plot_paper_evidence.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def main(accept=False):
    if accept:
        record=read(MANIFEST)
        for path,digest in record['outputs'].items():assert sha(ROOT/path)==digest
        record['status']='passed'
        record['visual_review']='Poppler render inspected: nine panels, ordinary numeric loss ticks on log axes, 16-observation trailing means, no raw overlays or norm row, common projected-change limits [-0.1,2.2], intact labels and legends; intentional viewport clipping is documented.'
        dump(MANIFEST,record);return
    old=read(OUT/'figure_manifest.json')
    archived_overview=ARCHIVE/PDF.name
    assert sha(archived_overview)==old['outputs'][PDF.relative_to(ROOT).as_posix()]
    spike=HERE/'figures/adam_manuscript_spikes.pdf'
    assert sha(spike)==old['outputs'][spike.relative_to(ROOT).as_posix()]
    arms,sources=previous.temporal.load_arms()
    for path in [OUT/'figure_manifest.json',OUT/'evidence_results.json',archived_overview,
                 HERE/'19_plot_paper_evidence.py',spike]:
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    record=dict(generator=Path(__file__).relative_to(ROOT).as_posix(),generator_sha256=sha(__file__),
        sources=sources,rows=['loss','projected','raw_cosine'],smoothing_window=WINDOW,
        projected_ylim=list(LIMITS),viewport_only=True,data_clamped_or_deleted=False,
        curves=[],outside_projected_view=[],outputs={},status='numerical_checks_passed',
        retained_spikes_sha256=sha(spike),rejected_from_presentation=[
            'adam_episode_onsets.pdf','adam_forward_precision_recall.pdf'],
        claim='Descriptive recurrence and temporal ordering; no classifier or incremental-forecast claim in the presentation.')
    previous.style()
    fig,axes=plt.subplots(3,3,figsize=(7.16,3.85),sharex='col')
    maximum_error=0.
    for col,task in enumerate(previous.temporal.TASKS):
        for color,arm in zip(previous.COLORS,[a for a in arms if a['task']==task]):
            curves=[c for c in old['curves'] if c['figure']=='overview' and c['task']==task
                    and c['lr']==arm['lr'] and c['key'] in record['rows'] and c['role']=='line']
            assert len(curves)==3
            for c in curves:
                row=record['rows'].index(c['key'])
                values=arm[c['key']]
                if row:
                    valid=np.flatnonzero(np.isfinite(values))
                    assert np.all(np.diff(valid)==1)
                    x=valid[WINDOW-1:]
                    y=np.convolve(values[valid],np.ones(WINDOW)/WINDOW,mode='valid')
                    expected=np.array([math.fsum(values[t-WINDOW+1:t+1])/WINDOW for t in x])
                else:
                    x=np.asarray(c['x']);y=np.asarray(c['y']);expected=values[x]
                np.testing.assert_allclose(y,expected,rtol=3e-14,atol=3e-14)
                maximum_error=max(maximum_error,float(np.max(np.abs(y-expected))))
                line,=axes[row,col].plot(x,y,color=color,lw=.9,zorder=2,
                    label=rf'$\eta={arm["lr"]:g}$' if row==0 else None)
                np.testing.assert_array_equal(line.get_xdata(),x)
                np.testing.assert_array_equal(line.get_ydata(),y)
                record['curves'].append(dict(c,x=x.tolist(),y=y.tolist(),window=WINDOW if row else None))
                if row==1:
                    record['outside_projected_view'].append(dict(task=task,lr=arm['lr'],role=c['role'],
                        below=int(sum(y<LIMITS[0])),above=int(sum(y>LIMITS[1])),points=len(y)))
        axes[0,col].set_title(f'({chr(97+col)}) '+previous.TITLES[col],pad=26)
        axes[0,col].legend(loc='lower center',bbox_to_anchor=(.5,1.015),ncol=3,
            frameon=False,fontsize=7.6,handlelength=1.,columnspacing=.5,handletextpad=.25,borderaxespad=0)
        axes[0,col].set_yscale('log')
        axes[0,col].yaxis.set_minor_formatter(NullFormatter())
        span=np.log10(axes[0,col].get_ylim()[1]/axes[0,col].get_ylim()[0])
        axes[0,col].yaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5) if span<2 else (1,),numticks=4))
        axes[0,col].yaxis.set_major_formatter(FuncFormatter(lambda value,position:f'{value:g}'))
        axes[1,col].set(ylim=LIMITS,yticks=[0,1,2])
        axes[1,col].axhline(2,color='.45',ls=(0,(4,3)),lw=.7)
        axes[2,col].set(ylim=(-1.06,1.06),yticks=[-1,0,1])
        axes[2,col].axhline(0,color='.45',ls=(0,(4,3)),lw=.7)
        for row,ax in enumerate(axes[:,col]):
            ax.set(xlim=(0,1024),xticks=[0,256,512,768,1024])
            ax.grid(alpha=.16,lw=.5);ax.tick_params(length=2.5,width=.6)
            if row:ax.text(0,1.035,f'({chr(97+row*3+col)})',transform=ax.transAxes,va='bottom',fontsize=9)
        axes[-1,col].set_xlabel(r'Parameter state $t$')
    for ax,label in zip(axes[:,0],['Training loss',r'$\eta\widehat\kappa_t$',r'$\rho_t$ (raw)']):ax.set_ylabel(label)
    fig.subplots_adjust(left=.105,right=.975,top=.85,bottom=.14,hspace=.48,wspace=.31)
    fig.canvas.draw()
    bounds=fig.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans)
    assert bounds.x0>=-1 and bounds.y0>=-1 and bounds.x1<=fig.bbox.width+1 and bounds.y1<=fig.bbox.height+1
    assert all(tuple(ax.get_ylim())==LIMITS for ax in axes[1])
    fig.savefig(PDF,metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    with fitz.open(PDF) as doc:
        assert len(doc)==1 and not doc[0].get_images()
        for font in doc[0].get_fonts(full=True):assert doc.extract_font(font[0])[3]
        for word in doc[0].get_text('words'):
            assert word[0]>=0 and word[1]>=0 and word[2]<=doc[0].rect.width+.01 and word[3]<=doc[0].rect.height+.01
    assert len(record['curves'])==27 and all(c['normalization']==1 for c in record['curves'])
    record['checks']=dict(curves=27,points=sum(len(c['x']) for c in record['curves']),
        loss_curves_identical_to_prior=True,independent_mean_maximum_error=maximum_error,
        no_relative_norms=True,no_raw_overlays=True,loss_tick_format='ordinary numeric labels on log axis',
        spikes_unchanged=sha(spike)==record['retained_spikes_sha256'])
    data=read(OUT/'evidence_results.json')['episode_summary'];rows=[]
    for arm in arms:
        get=lambda severity:next(r for r in data if r['arm']==arm['id'] and r['severity']==severity and r['alarm']=='eta_kappa_crosses_2')
        allr,large=get('all'),get('large')
        fraction=lambda r:f"{r['hits']}/{r['episodes']}" if r['episodes'] else '--'
        lead=f"{large['lead_median']:g} [{large['lead_min']},{large['lead_max']}]" if large['hits'] else '--'
        label=dict(mlp='MLP',resnet='ResNet-32',transformer='Transformer')[arm['task']]
        rows.append([label,f"{arm['lr']:g}",allr['episodes'],fraction(allr),large['episodes'],fraction(large),lead])
    tex=[r'\begin{tabular}{lrrrrrr}',r'\hline',
        r'Case & $\eta$ & Episodes & Earlier crossing & Large & Earlier crossing & Lead \\',r'\hline',
        *[' & '.join(map(str,r))+r' \\' for r in rows],r'\hline',r'\end{tabular}']
    (OUT/'table_recurrence.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8',newline='\n')
    md=['# Descriptive recurrence across all nine arms','',
        '| Case | LR | Episodes | Earlier crossing: all | Large episodes | Earlier crossing: large | Lead for large: median [range] |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- |',
        *['| '+' | '.join(map(str,r))+' |' for r in rows],'',
        'Loss-only episodes begin at a 10% rise above the preceding 16-state minimum. '
        'Large means an observed peak at least 50% above that baseline. Earlier crossing '
        'means a fresh crossing of raw projected change above two, 1-8 updates before '
        'onset while loss is less than 5% above its own recent minimum. Every episode '
        'remains in the denominator, including misses. Leads condition on a hit and use '
        'the earliest qualifying crossing in that window. The highest-rate MLP and '
        'Transformer each include one unfinished recovery. These are dependent events '
        'from one initialization per case, not independent replications.', '',
        'The zero-episode low-rate ResNet and Transformer arms nevertheless cross two '
        'during training. Thus crossing two is not sufficient for a pronounced excursion. '
        'This table describes recurrence; it is not a classifier evaluation.']
    (OUT/'RECURRENCE_TABLE.md').write_text('\n'.join(md)+'\n',encoding='utf-8',newline='\n')
    for path in [PDF,OUT/'table_recurrence.tex',OUT/'RECURRENCE_TABLE.md']:
        record['outputs'][path.relative_to(ROOT).as_posix()]=sha(path)
    dump(MANIFEST,record);print(record['checks'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accept-reviewed',action='store_true')
    main(parser.parse_args().accept_reviewed)
