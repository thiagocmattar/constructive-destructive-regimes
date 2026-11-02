"""State-aligned, unsmoothed peak micro views for all three controlled tasks."""
import argparse
import importlib.util
from pathlib import Path

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('peak_analysis',HERE/'15_peak_temporal_analysis.py')
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)
PDF=HERE/'figures/adam_peak_micro_three_models.pdf'
RENDER=HERE/'scratch/adam_peak_micro_three_models.png'
MANIFEST=HERE/'peak_micro_three_models_manifest.json'
NAMES=['training_loss_log','eta_kappa_hat_raw','rho_raw','rho_momentum','rho_adam',
       'raw_gradient_norm_log','momentum_norm_log','R_t']


def main(accept_reviewed=False):
    if accept_reviewed:
        record=analysis.read(MANIFEST)
        assert record['generator_sha256']==analysis.sha(__file__)
        for path,digest in record['sources'].items():assert analysis.sha(ROOT/path)==digest
        assert analysis.sha(PDF)==record['output_sha256'] and RENDER.exists()
        record.update(status='passed',visual_review=(
            'All27 panels visually reviewed: three selected peaks, exact individual points, '
            'signed loss differences, log loss/norms, local retention axes, legible labels '
            'and source-clock/footer explanations without overlap or clipping.'),
            render=RENDER.relative_to(ROOT).as_posix(),render_sha256=analysis.sha(RENDER))
        analysis.dump(MANIFEST,record);print('Visual review accepted.');return
    result=analysis.read(HERE/'peak_temporal_results.json')
    assert result['generator_sha256']==analysis.sha(HERE/'15_peak_temporal_analysis.py')
    arms,sources=analysis.load_arms()
    assert sources==result['sources']
    for path in [HERE/'15_peak_temporal_analysis.py',HERE/'peak_temporal_results.json']:
        sources[path.relative_to(ROOT).as_posix()]=analysis.sha(path)
    original=analysis.read(HERE/'full_unsmoothed_manifest.json')
    sources[original['output']]=analysis.sha(ROOT/original['output'])
    assert sources[original['output']]==original['output_sha256']
    record=dict(generator=Path(__file__).relative_to(ROOT).as_posix(),
        generator_sha256=analysis.sha(__file__),sources=sources,smoothing='None in any row.',
        example_selection='Highest LR per task; completed loss-only episode with largest peak/baseline ratio.',
        clock='Loss and R_t at state t; g/m/adaptive diagnostics consume g_t before the next parameter update.',
        historical_mapping='Original gradient/momentum/adaptive curve x becomes state x-1; loss and actual R_t keep x.',
        retention='Actual-update prefix from initialization through state t, without a window reset.',
        examples=result['selected_examples'],curves=[],checks={})
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix',
        'font.size':10,'axes.labelsize':10,'axes.titlesize':10,'xtick.labelsize':9,'ytick.labelsize':9,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
        'pdf.fonttype':42,'path.simplify':False})
    fig,axes=plt.subplots(9,3,figsize=(10.5,13.5),sharex='col')
    titles=['CIFAR-10 / MLP','CIFAR-10 / ResNet-32','WikiText-2 / Transformer']
    labels=['Training loss',r'$L_t-L_{t-1}$',r'$\eta\widehat\kappa_t$'+'\n(raw gradients)',
        r'$\rho_t$'+'\n(raw gradients)',r'$\rho_t$'+'\n(momentum)',r'$\rho_t$'+'\n(Adam)',
        r'$\|g_t\|$',r'$\|m_t\|$',r'$R_t$'+'\n(Adam)']
    for col,example in enumerate(result['selected_examples']):
        arm=next(a for a in arms if a['id']==example['arm'])
        start,end,peak,marker=[example[k] for k in ['start_state','end_state','peak_state','marker_state']]
        states=np.arange(start,end+1)
        values=[arm[k][states] for k in ['loss','projected','raw_cosine','moment_cosine','adam_cosine',
                                       'gradient','momentum','retention']]
        for name,yy,row in zip(NAMES,values,[0,2,3,4,5,6,7,8]):
            prior=next(c for c in original['curves'] if c['task']==arm['task'] and
                       c['lr']==arm['lr'] and c['row']==name)
            oldx=np.asarray(prior['x']);oldy=np.asarray(prior['y'])
            aligned=oldx if name in ['training_loss_log','R_t'] else oldx-1
            mask=(aligned>=start)&(aligned<=end)
            np.testing.assert_array_equal(states,aligned[mask])
            np.testing.assert_array_equal(yy,oldy[mask])
            assert np.isfinite(yy).all() and prior['smoothing_window'] is None
            artist,=axes[row,col].plot(states,yy,color='#009E73',lw=.8,marker='o',ms=2.3,markeredgewidth=0)
            np.testing.assert_array_equal(artist.get_xdata(),states)
            np.testing.assert_array_equal(artist.get_ydata(),yy)
            record['curves'].append(dict(task=arm['task'],lr=arm['lr'],row=name,x=states.tolist(),
                y=yy.tolist(),smoothing_window=None,exact_original_values=True,
                original_index_shift=0 if name in ['training_loss_log','R_t'] else -1))
        diffs=np.diff(arm['loss'][states])
        bars=axes[1,col].bar(states[1:],diffs,width=.72,linewidth=0,
                            color=np.where(diffs>0,'#D55E00','#0072B2'))
        np.testing.assert_array_equal([bar.get_height() for bar in bars],diffs)
        np.testing.assert_allclose(diffs.sum(),arm['loss'][end]-arm['loss'][start],rtol=0,atol=2e-15)
        record['curves'].append(dict(task=arm['task'],lr=arm['lr'],row='loss_difference',
            x=states[1:].tolist(),y=diffs.tolist(),smoothing_window=None))
        axes[0,col].set_title(titles[col]+'\n'+rf'Adam, $\eta={arm["lr"]:g}$; peak {peak}',pad=18)
        axes[0,col].axhline(arm['loss'][start],color='.5',lw=.6,ls=':')
        axes[1,col].axhline(0,color='.45',lw=.6)
        axes[2,col].axhline(2,color='.45',lw=.6,ls=':')
        for row in [3,4,5]:
            axes[row,col].set(ylim=(-1.08,1.08),yticks=[-1,0,1])
            axes[row,col].axhline(0,color='.55',lw=.6)
        for row in [0,6,7]:
            assert np.all(axes[row,col].lines[0].get_ydata()>0)
            axes[row,col].set_yscale('log')
            axes[row,col].yaxis.set_minor_formatter(NullFormatter())
        axes[0,col].yaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5),numticks=5))
        axes[0,col].yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
        axes[8,col].yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:.3f}'))
        ticks=list(range(start,end+1,20 if end-start>50 else 10))
        ticks=sorted(set(ticks+[peak,end]))
        axes[-1,col].set(xticks=ticks,xlabel=r'Training state $t$')
        for row in range(9):
            ax=axes[row,col]
            ax.axvline(peak,color='.45',lw=.7,ls='--',zorder=0)
            ax.axvline(marker,color='.75',lw=.7,ls='--',zorder=0)
            ax.set_xlim(start-.6,end+.6)
            ax.grid(axis='y',alpha=.15,lw=.5)
            ax.tick_params(length=3,width=.6)
            panel=row*3+col
            letter=chr(97+panel) if panel<26 else 'aa'
            ax.text(0,1.04,f'({letter})',transform=ax.transAxes,va='bottom',fontsize=9)
    for ax,label in zip(axes[:,0],labels):ax.set_ylabel(label)
    fig.subplots_adjust(left=.085,right=.985,top=.935,bottom=.10,hspace=.40,wspace=.28)
    fig.text(.535,.052,'All observations unsmoothed. Light dashed: before loss crosses starting value; dark dashed: peak.',
             ha='center',fontsize=9)
    fig.text(.535,.032,r'$g_t,m_t$ and adaptive diagnostics are aligned to state $t$, before the next parameter update.',
             ha='center',fontsize=9)
    fig.text(.535,.012,r'$R_t$ retains actual updates through state $t$. Windows end at the first loss below their starting value.',
             ha='center',fontsize=9)
    fig.savefig(PDF,metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    doc=fitz.open(PDF);assert len(doc)==1
    page=doc[0]
    for word in page.get_text('words'):
        assert word[0]>=0 and word[1]>=0 and word[2]<=page.rect.width+.01 and word[3]<=page.rect.height+.01
    assert len(record['curves'])==27
    record.update(status='numerical_and_pdf_checks_passed',visual_review='pending_poppler_render',
                  output=PDF.relative_to(ROOT).as_posix(),output_sha256=analysis.sha(PDF))
    record['checks']=dict(original_curves_exact=24,panels=27,source_clock_mapping_verified=True,
        loss_differences_and_telescope_verified=True,pdf_text_inside_page=True,
        original_points=sum(len(c['x']) for c in record['curves'] if c['row']!='loss_difference'),
        difference_points=sum(len(c['x']) for c in record['curves'] if c['row']=='loss_difference'))
    analysis.dump(MANIFEST,record)
    print(record['checks'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accept-reviewed',action='store_true')
    main(parser.parse_args().accept_reviewed)
