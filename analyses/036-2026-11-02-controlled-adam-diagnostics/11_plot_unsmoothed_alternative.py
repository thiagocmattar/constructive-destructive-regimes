"""Plot all raw points for the exact cohort frozen in the current preview manifest."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
TASKS=['mlp','resnet','transformer']
TITLES=['CIFAR-10 / MLP','CIFAR-10 / ResNet-32','WikiText-2 / Transformer']
COLORS=['#0072B2','#D55E00','#009E73']


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main(source_manifest='completed_preview_manifest.json',output_stem='adam_diagnostics_completed_unsmoothed',
         manifest_name='completed_unsmoothed_manifest.json'):
    basefile=HERE/source_manifest
    base=json.loads(basefile.read_text())
    configfile=ROOT/'runs/215-2026-11-02-controlled-full-batch-adam/config.json'
    config=json.loads(configfile.read_text());updates=config['updates']
    # Results.json can advance while training continues. The preview's raw arrays
    # and original immutable histories define this paired display comparison.
    histories={p:h for p,h in base['sources'].items() if '/artifacts/' in p}
    for p,h in histories.items():assert sha(ROOT/p)==h
    assert sha(configfile)==base['sources'][configfile.relative_to(ROOT).as_posix()]
    manifest=dict(generator=Path(__file__).relative_to(ROOT).as_posix(),generator_sha256=sha(__file__),
        source_preview=basefile.relative_to(ROOT).as_posix(),source_preview_sha256=sha(basefile),
        source_histories=histories,completed_counts=base['completed_counts'],
        omitted_conditions=base['omitted_conditions'],smoothing='None in any row.',
        same_cohort_as_preview=True,rows=base['rows'],retention_streams=['actual'],
        clock=base['clock'],curves=[],limitations=base['limitations'][:2]+[
            'Every finite raw observation is shown. Undefined initial adjacent quantities are omitted.',
            'Dense oscillations can obscure individual traces at this full-horizon scale.'])
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix',
        'font.size':9,'axes.titlesize':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
        'pdf.fonttype':42,'path.simplify':False})
    fig,axes=plt.subplots(8,3,figsize=(7.16,12.1),sharex=True)
    extrema=[]
    for c in base['curves']:
        row=base['rows'].index(c['row']);col=TASKS.index(c['task'])
        color=COLORS[config['learning_rates_by_task'][c['task']].index(c['lr'])]
        x=np.asarray(c['source_x']);y=np.asarray([np.nan if v is None else v for v in c['source_y']])
        valid=np.isfinite(y);x=x[valid];y=y[valid]
        exponent=int(np.floor(np.log10(c['lr'])));coefficient=int(round(c['lr']/10**exponent))
        label=rf'$\eta={"" if coefficient==1 else str(coefficient)+r"\times"}10^{{{exponent}}}$'
        artist,=axes[row,col].plot(x,y,color=color,lw=.85,label=label if row==0 else None)
        np.testing.assert_array_equal(artist.get_xdata(),x);np.testing.assert_array_equal(artist.get_ydata(),y)
        manifest['curves'].append(dict(task=c['task'],lr=c['lr'],row=c['row'],stream=c['stream'],
            smoothing_window=None,x=x.tolist(),y=y.tolist()))
        if row==1:extrema.extend(y)
    for col,task in enumerate(TASKS):
        axes[0,col].set_title(TITLES[col],pad=30)
        if base['completed_counts'][task]:
            axes[0,col].legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=3,
                fontsize=7,frameon=False,columnspacing=.7,handlelength=1.1,handletextpad=.25)
        for row in [0,5,6]:
            axes[row,col].set_yscale('log');axes[row,col].yaxis.set_minor_formatter(NullFormatter())
        limits=(min(axes[r,col].get_ylim()[0] for r in [5,6]),max(axes[r,col].get_ylim()[1] for r in [5,6]))
        for row in [5,6]:axes[row,col].set_ylim(limits)
        axes[0,col].yaxis.set_major_locator(LogLocator(base=10,numticks=4))
        axes[0,col].yaxis.set_major_formatter(FuncFormatter(lambda value,position:f'{value:g}'))
        for row in [2,3,4]:
            axes[row,col].set(ylim=(-1.08,1.08),yticks=[-1,0,1]);axes[row,col].axhline(0,color='.5',lw=.6,zorder=0)
        axes[1,col].axhline(2,color='.35',lw=.7,ls='--',zorder=0)
        axes[7,col].set(ylim=(0,1.04),yticks=[0,.5,1],xlabel='Training step')
        for row in range(8):
            ax=axes[row,col];ax.set(xlim=(0,updates),xticks=[0,256,512,768,1024])
            ax.text(0,1.035,f'({chr(97+row*3+col)})',transform=ax.transAxes,va='bottom',fontsize=8)
            ax.tick_params(length=3,width=.6);ax.grid(axis='y',alpha=.15,lw=.5)
    span=max(extrema)-min(extrema);limits=(min(-.08,min(extrema)-.04*span),max(2.18,max(extrema)+.04*span))
    for ax in axes[1]:ax.set(ylim=limits,yticks=[-2,0,2])
    labels=['Training loss',r'$\eta\widehat\kappa_t$'+'\n(raw gradients)',r'$\rho_t$'+'\n(raw gradients)',
        r'$\rho_t$'+'\n(momentum)',r'$\rho_t$'+'\n(Adam)',r'$\|g_t\|$',r'$\|m_t\|$',r'$R_t$'+'\n(Adam)']
    for ax,label in zip(axes[:,0],labels):ax.set_ylabel(label)
    fig.subplots_adjust(left=.105,right=.975,top=.925,bottom=.10,hspace=.34,wspace=.29)
    counts=base['completed_counts']
    fig.text(.545,.045,f'Completed runs only: MLP {counts["mlp"]}/3, ResNet-32 {counts["resnet"]}/3, Transformer {counts["transformer"]}/3.',ha='center',fontsize=8)
    fig.text(.545,.018,'All eight rows unsmoothed; every finite recorded observation is shown.',ha='center',fontsize=8)
    output=HERE/'figures'/(output_stem+'.pdf')
    fig.savefig(output,metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    assert len(manifest['curves'])==len(base['curves'])
    manifest.update(status='source_and_artist_checks_passed',output=output.relative_to(ROOT).as_posix(),
        output_sha256=sha(output),projected_change_ylim=limits)
    (HERE/manifest_name).write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=manifest['output'],curves=len(manifest['curves']),smoothing='none')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest',default='completed_preview_manifest.json')
    parser.add_argument('--output-stem',default='adam_diagnostics_completed_unsmoothed')
    parser.add_argument('--manifest-name',default='completed_unsmoothed_manifest.json')
    args=parser.parse_args();main(args.source_manifest,args.output_stem,args.manifest_name)
