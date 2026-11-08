"""Manuscript-style H049 curves: all conditions, plus seed-resolved diagnostics."""
import hashlib
import json
from pathlib import Path
import string
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, LogFormatterSciNotation, MaxNLocator
import numpy as np

HERE=Path(__file__).resolve().parent
FIGURES=HERE/'figures';FIGURES.mkdir(exist_ok=True)
COLORS={25:'#D55E00',50:'#0072B2',75:'#009E73',90:'#CC79A7',99:'#7B4F12'}
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
    'mathtext.fontset':'stix','font.size':8,'axes.titlesize':9,'axes.labelsize':8,
    'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'pdf.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
    'xtick.major.width':.65,'ytick.major.width':.65,'lines.linewidth':.85})


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    summary=json.loads((HERE/'cap_lr_summary.json').read_text());assert summary['status']=='passed'
    source=HERE/'cap_lr_curves.npz';assert sha(source)==summary['curves_sha256'];data=np.load(source)
    x=data['states'];test_x=data['test_states'];verified=[];values_checked=0
    def y(seed,arm,key):return data[f's{seed}/{arm}/{key}']
    def draw(ax,xv,yv,**kwargs):
        nonlocal values_checked
        line,=ax.plot(xv,yv,**kwargs)
        np.testing.assert_array_equal(line.get_xdata(),xv);np.testing.assert_array_equal(line.get_ydata(),yv)
        values_checked+=len(yv);return line
    def decorate(ax,row,col,log=False):
        ax.set_xlim(448,1152);ax.set_xticks([448,624,800,976,1152]);ax.grid(alpha=.17,linewidth=.45)
        if log:
            ax.set_yscale('log');ax.yaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5),numticks=6))
            ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10,labelOnlyBase=False))
        else:ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.text(.01,1.025,f'({string.ascii_lowercase[row*3+col]})',transform=ax.transAxes,fontsize=8)
    def legend(fig):
        handles=[Line2D([],[],color='black',label='Uncapped, same LR'),Line2D([],[],color='.55',linestyle='--',label='Original baseline (1x)')]
        handles += [Line2D([],[],color=v,label=f'q{q}') for q,v in COLORS.items()]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.98),ncol=4,frameon=False,columnspacing=1.3,handlelength=2.)
    # The overview shows means only: individual seed spikes are retained below.
    fig,axes=plt.subplots(2,3,figsize=(7.16,4.65),sharex=True)
    for col,lr in enumerate([1,2,4]):
        for row,key in enumerate(['loss','test_loss']):
            ax=axes[row,col];xv=x if row==0 else test_x
            for q,color in COLORS.items():
                values=np.mean([y(s,f'cap_q{q}_lr{lr}',key) for s in range(3)],axis=0)
                draw(ax,xv,values,color=color)
            if lr!=1:draw(ax,xv,np.mean([y(s,'adam_lr1',key) for s in range(3)],axis=0),color='.55',linestyle='--',linewidth=.8)
            draw(ax,xv,np.mean([y(s,f'adam_lr{lr}',key) for s in range(3)],axis=0),color='black',linewidth=.95)
            decorate(ax,row,col,log=True)
        axes[0,col].set_title(rf'$\eta={.001*lr:.3f}$ ({lr}$\times$)',pad=14)
        axes[1,col].set_xlabel(r'Completed optimizer updates $s$')
    axes[0,0].set_ylabel('Full training objective loss')
    axes[1,0].set_ylabel('Full test objective loss')
    legend(fig)
    fig.text(.5,.018,'MLP / CIFAR-10; arithmetic mean of 3 seeds. Independent log scales. Fixed 1x cap budgets; test every 32 updates.',ha='center',fontsize=7)
    fig.subplots_adjust(left=.10,right=.985,top=.78,bottom=.13,hspace=.32,wspace=.32)
    path=FIGURES/'12-distributed-cap-quantile-lr-mlp.pdf';fig.savefig(path,metadata={'Title':'MLP distributed cap: quantile x learning rate','Creator':'40_plot_cap_lr.py'});plt.close(fig)
    verified.append(dict(path=path.relative_to(HERE).as_posix(),sha256=sha(path),pages=1))
    # One page per initialization. No time decimation and no smoothing.
    path=FIGURES/'13-distributed-cap-lr-seed-diagnostics.pdf'
    labels=[r'Training loss',r'$\eta\widehat\kappa_{s+1}$',r'$\rho_{s+1}$',r'$\|g_{s+1}\|$',
            r'Output $r^{F}_{s+1}$',r'$\gamma_{s+1}$ (step retention)',
            'Hidden units capped']
    keys=['loss','eta_kappa_hat','rho','raw_gradient_norm','representation_output_change_rms','retained_norm_ratio','modified_fraction']
    with PdfPages(path,metadata={'Title':'H049 per-seed diagnostics','Creator':'40_plot_cap_lr.py'}) as pdf:
        for seed in range(3):
            fig,axes=plt.subplots(7,3,figsize=(7.16,10.15),sharex=True)
            for col,lr in enumerate([1,2,4]):
                for row,key in enumerate(keys):
                    ax=axes[row,col]
                    for q,color in COLORS.items():draw(ax,x,y(seed,f'cap_q{q}_lr{lr}',key),color=color,linewidth=.7)
                    if lr!=1:draw(ax,x,y(seed,'adam_lr1',key),color='.55',linestyle='--',linewidth=.7)
                    draw(ax,x,y(seed,f'adam_lr{lr}',key),color='black',linewidth=.75)
                    decorate(ax,row,col,log=row in [0,3,4])
                    if row==1:ax.axhline(2,color='.5',linestyle=':',linewidth=.6)
                    if row==2:ax.set_ylim(-1.05,1.05);ax.set_yticks([-1,0,1]);ax.axhline(0,color='.5',linestyle=':',linewidth=.6)
                    if row in [5,6]:ax.set_ylim(-.04,1.04);ax.set_yticks([0,.5,1])
                axes[0,col].set_title(rf'$\eta={.001*lr:.3f}$ ({lr}$\times$)',pad=14)
                axes[-1,col].set_xlabel(r'Completed optimizer updates $s$')
            for row,label in enumerate(labels):axes[row,0].set_ylabel(label)
            legend(fig)
            fig.text(.5,.020,f'MLP / CIFAR-10, initialization {seed}. All 705 states; no smoothing. Raw-gradient diagnostics; outgoing output movement.',ha='center',fontsize=7)
            fig.subplots_adjust(left=.115,right=.985,top=.895,bottom=.07,hspace=.32,wspace=.32)
            pdf.savefig(fig);plt.close(fig)
    verified.append(dict(path=path.relative_to(HERE).as_posix(),sha256=sha(path),pages=3))
    result=dict(status='passed',source_sha256=sha(source),script_sha256=sha(Path(__file__)),plotted_values_verified=values_checked,
        figures=verified,overview='Arithmetic seed means; no smoothing/time decimation. Test lines connect all 23 measured states.',
        diagnostics='All 3 seeds separately, all 705 states, linear eta-kappa-hat. Per-column y axes independent.')
    (HERE/'cap_lr_plot_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result))


if __name__=='__main__':main()
