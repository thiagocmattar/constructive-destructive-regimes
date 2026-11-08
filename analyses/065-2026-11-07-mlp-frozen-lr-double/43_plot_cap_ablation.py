"""H049 cap-quantile ablation: connected seed means, sample SD, shared loss axis."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np

HERE=Path(__file__).resolve().parent
TARGET=HERE/'figures/14-distributed-cap-quantile-ablation-mlp.pdf'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    source=HERE/'cap_lr_summary.json';summary=json.loads(source.read_text())
    curves=HERE/'cap_lr_curves.npz';assert sha(curves)==summary['curves_sha256']
    data=np.load(curves);np.testing.assert_array_equal(data['states'],np.arange(448,1153))
    aggregate={r['arm']:r for r in summary['aggregate']};checked={}
    for lr in [1,2,4]:
        for name in [*[f'cap_q{q}_lr{lr}' for q in [25,50,75,90,99]],f'adam_lr{lr}']:
            seeds=np.array([data[f's{s}/{name}/loss'].mean() for s in range(3)])
            saved=[next(m['mean_loss'] for m in summary['metrics'] if m['arm']==name and m['seed']==s) for s in range(3)]
            np.testing.assert_allclose(seeds,saved,rtol=1e-14,atol=0)
            mean=float(seeds.mean());sd=float(seeds.std(ddof=1))
            np.testing.assert_allclose([mean,sd],[aggregate[name]['mean_loss_mean'],aggregate[name]['mean_loss_sd']],rtol=1e-14,atol=0)
            assert mean-sd>0
            checked[name]=dict(seed_means=seeds.tolist(),mean=mean,sd=sd)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'mathtext.fontset':'stix','font.size':8,'axes.titlesize':9,'axes.labelsize':8,
        'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65})
    fig,axes=plt.subplots(1,3,figsize=(7.16,2.95),sharey=True)
    baseline=checked['adam_lr1'];blue='#0072B2';caps=np.arange(5);uncapped_x=5.3
    def errors(ax,x,means,sds,**style):
        artist=ax.errorbar(x,means,yerr=sds,capsize=3,elinewidth=.85,capthick=.85,**style)
        np.testing.assert_array_equal(artist.lines[0].get_ydata(),means)
        for segment,xi,mean,sd in zip(artist.lines[2][0].get_segments(),x,means,sds):
            np.testing.assert_allclose(segment,[[xi,mean-sd],[xi,mean+sd]],rtol=1e-14,atol=0)
    for col,lr in enumerate([1,2,4]):
        ax=axes[col]
        ax.axhspan(baseline['mean']-baseline['sd'],baseline['mean']+baseline['sd'],color='.6',alpha=.18,zorder=0)
        ax.axhline(baseline['mean'],color='.45',linestyle='--',linewidth=.8,zorder=1)
        records=[checked[f'cap_q{q}_lr{lr}'] for q in [25,50,75,90,99]]
        errors(ax,caps,np.array([r['mean'] for r in records]),np.array([r['sd'] for r in records]),
               color=blue,marker='o',markersize=4,linewidth=1.1,zorder=3)
        native=checked[f'adam_lr{lr}']
        errors(ax,np.array([uncapped_x]),np.array([native['mean']]),np.array([native['sd']]),
               color='black',marker='D',markersize=4,linestyle='none',zorder=4)
        ax.set_yscale('log');ax.set_ylim(.0105,.34)
        ax.yaxis.set_major_locator(FixedLocator([.015,.03,.06,.12,.24]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_:f'{y:g}'));ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks([*caps,uncapped_x],['q25','q50','q75','q90','q99','No cap'])
        ax.set_xlim(-.35,5.9);ax.set_xlabel('Cap quantile')
        ax.set_title(rf'({chr(97+col)}) $\eta={.001*lr:.3f}$ ({lr}$\times$)',pad=8)
        ax.grid(axis='y',alpha=.2,linewidth=.5)
    axes[0].set_ylabel('Mean training loss')
    for ax in axes[1:]:assert ax.get_ylim()==axes[0].get_ylim()
    fig.legend(handles=[Line2D([],[],color=blue,marker='o',markersize=4,label='Distributed cap'),
        Line2D([],[],color='black',marker='D',linestyle='none',markersize=4,label='Uncapped, same LR'),
        Line2D([],[],color='.45',linestyle='--',label='Original Adam (1x)')],
        loc='upper center',bbox_to_anchor=(.5,.985),ncol=3,frameon=False,columnspacing=2)
    fig.text(.5,.035,'MLP / CIFAR-10, states 448-1152. Points and band: mean '+r'$\pm$'+' sample SD across 3 seeds. Shared log Y axis.',
             ha='center',fontsize=7)
    fig.subplots_adjust(left=.09,right=.985,bottom=.235,top=.77,wspace=.17)
    fig.savefig(TARGET,metadata={'Title':'MLP distributed-cap quantile ablation','Creator':Path(__file__).name});plt.close(fig)
    result=dict(status='passed',source_summary_sha256=sha(source),source_curves_sha256=sha(curves),
        script_sha256=sha(Path(__file__)),pdf_sha256=sha(TARGET),conditions=18,seeds_per_condition=3,
        statistics='Per-seed arithmetic mean over 705 loss states, then mean and sample SD (ddof=1) across seeds.',
        shared_y_limits=[.0105,.34],y_scale='log',cap_spacing='Categorical; connecting lines guide the eye.',checks=checked)
    (HERE/'cap_ablation_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(status='passed',conditions=18,seed_means_checked=54,means_and_errorbar_endpoints_checked=True,pdf=str(TARGET))))


if __name__=='__main__':main()
