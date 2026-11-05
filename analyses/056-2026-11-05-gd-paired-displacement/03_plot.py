"""Publication views of the verified GD reduction; retain earlier artwork."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DRAFT=ROOT/'manuscript/draft'
TASKS=['mlp','resnet','transformer']
TITLES=['CIFAR-10 / MLP','CIFAR-10 / ResNet-32','WikiText-2 / Transformer']
COLORS=['#0072B2','#D55E00','#009E73']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    result=json.loads((HERE/'results.json').read_text())
    verification=json.loads((HERE/'verification.json').read_text())
    assert verification['status']=='passed' and verification['results_sha256']==sha(HERE/'results.json')
    a=np.load(HERE/'arrays.npz')
    curves=[]; figures=[]; plot_arrays={}
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','font.size':9,
        'axes.labelsize':9,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'pdf.fonttype':42,
        'path.simplify':False})

    def line(ax,x,y,tag,**kwargs):
        x=np.asarray(x);y=np.asarray(y)
        assert len(x)==len(y) and np.isfinite(y).all()
        artist,=ax.plot(x,y,**kwargs)
        np.testing.assert_array_equal(artist.get_xdata(),x)
        np.testing.assert_array_equal(artist.get_ydata(),y)
        plot_arrays[tag+'_x']=x;plot_arrays[tag+'_y']=y
        curves.append(dict(tag=tag,points=len(x),x_sha256=hashlib.sha256(x.astype('<f8').tobytes()).hexdigest(),
                           y_sha256=hashlib.sha256(y.astype('<f8').tobytes()).hexdigest()))

    def save(fig,name):
        folder=HERE/'figures';folder.mkdir(exist_ok=True)
        pdf=folder/(name+'.pdf')
        fig.savefig(pdf,metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
        destination=DRAFT/'figures/analysis056';destination.mkdir(exist_ok=True)
        shutil.copyfile(pdf,destination/pdf.name)
        scratch=HERE/'scratch';scratch.mkdir(exist_ok=True)
        subprocess.run(['pdftoppm','-r','130','-singlefile','-png',str(pdf),str(scratch/name)],
                       check=True,capture_output=True,timeout=60)
        figures.append(dict(path=pdf.relative_to(ROOT).as_posix(),sha256=sha(pdf),
            manuscript_copy=(destination/pdf.name).relative_to(ROOT).as_posix()))

    def setup(rows,height):
        fig,axes=plt.subplots(rows,3,figsize=(7.16,height),squeeze=False)
        for i,ax in enumerate(axes.flat):
            ax.grid(axis='y',alpha=.14,lw=.5);ax.tick_params(length=3,width=.6)
            if i>=3:
                ax.text(0,1.035,f'({chr(97+i)})',transform=ax.transAxes,va='bottom',fontsize=8)
        for col,title in enumerate(TITLES):axes[0,col].set_title(f'({chr(97+col)})  {title}',pad=25)
        return fig,axes

    def rate_legend(ax):
        ax.legend(loc='lower center',bbox_to_anchor=(.5,1.015),ncol=3,fontsize=7,frameon=False,
                  columnspacing=.65,handlelength=1.1,handletextpad=.25,borderaxespad=0)

    def smooth(v):return np.convolve(v,np.ones(50)/50,'valid')

    fig,axes=setup(2,3.55)
    for col,task in enumerate(TASKS):
        for color,r in zip(COLORS,[r for r in result['trajectories'] if r['task']==task]):
            rid=r['run'];loss=a[f'{rid}_loss'];q=a[f'{rid}_q']
            line(axes[0,col],np.arange(3001),loss,'loss_'+rid,color=color,lw=.8,label=rf'$\eta={r["lr"]:g}$')
            line(axes[1,col],np.arange(51,3001),smooth(q[1:]),'q_'+rid,color=color,lw=.9)
            source=ROOT/r['source'];spectra=[json.loads(v) for v in source.with_name('curvature.jsonl').read_text().splitlines()]
            spectra=[v for v in spectra if v['state_step']<=3000]
            x=np.array([v['state_step']+2 for v in spectra])
            y=np.array([(v['hessian'] if 'hessian' in v else v)['eigenvalues'][0]*r['lr'] for v in spectra])
            line(axes[1,col],x,y,'spectrum_'+rid,color=color,lw=.8,ls='--',marker='o',ms=2,mfc='white',mew=.6)
        for ax in axes[:,col]:ax.set(xlim=(0,3002),xticks=[0,1000,2000,3000])
        ax=axes[0,col];ax.set(yscale='log',xlabel=r'Parameter state $s$');rate_legend(ax)
        ax.yaxis.set_major_locator(FixedLocator([[.2,.3,.4,.6],[.4,.6,1,2],[6,7,8,9,10]][col]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'));ax.yaxis.set_minor_locator(NullLocator())
        ax=axes[1,col]
        ax.axhline(1,color='.5',lw=.65,ls='-.',zorder=0);ax.axhline(2,color='.35',lw=.65,ls=':',zorder=0)
        ax.set(xlabel=r'Gradient index $t$',ylim=(-.08,[3.4,6.1,3.6][col]))
        ax.set_yticks([0,1,2,4,6] if col==1 else [0,1,2,3])
    axes[0,0].set_ylabel('Training loss');axes[1,0].set_ylabel('Scaled curvature')
    fig.legend(handles=[Line2D([],[],color='.2',label=r'$\eta\widehat\kappa_t$ (50-pair mean)'),
        Line2D([],[],color='.2',ls='--',marker='o',ms=3,mfc='white',label=r'$\eta\lambda_{\max}$'),
        Line2D([],[],color='.5',ls='-.',label='1: reversal'),Line2D([],[],color='.35',ls=':',label='2: quadratic loss threshold')],
        loc='lower center',bbox_to_anchor=(.52,.005),ncol=4,frameon=False,fontsize=7,handlelength=2,columnspacing=1)
    fig.subplots_adjust(left=.085,right=.985,top=.84,bottom=.19,hspace=.67,wspace=.29)
    save(fig,'gd_loss_curvature')

    fig,axes=setup(4,6.35)
    for col,task in enumerate(TASKS):
        for color,r in zip(COLORS,[r for r in result['trajectories'] if r['task']==task]):
            rid=r['run'];x=np.arange(1,3001)
            line(axes[0,col],np.arange(51,3001),smooth(a[f'{rid}_rho'][1:]),'rho_'+rid,
                 color=color,lw=.9,label=rf'$\eta={r["lr"]:g}$')
            line(axes[1,col],np.arange(50,3001),smooth(a[f'{rid}_norm']),'norm_'+rid,color=color,lw=.9)
            line(axes[2,col],x,a[f'{rid}_R'],'R_'+rid,color=color,lw=.95)
            line(axes[2,col],x,a[f'{rid}_Rperp'],'Rperp_'+rid,color=color,lw=.8,ls=':')
            even=a[f'{rid}_even']
            line(axes[3,col],even,a[f'{rid}_C0'],'C_'+rid,color=color,lw=.95)
            line(axes[3,col],even,a[f'{rid}_D0'],'D_'+rid,color=color,lw=.9,ls='--')
        rate_legend(axes[0,col])
        for row,ax in enumerate(axes[:,col]):
            ax.set(xlim=(0,3000),xticks=[0,1000,2000,3000])
            if row<3:ax.set_xticklabels([])
            else:ax.set_xlabel(r'Update $t$ (even prefixes for $C_t,D_t$)')
        axes[0,col].set(ylim=(-1.07,1.07),yticks=[-1,0,1]);axes[0,col].axhline(0,color='.5',lw=.6)
        axes[1,col].set(yscale='log');axes[1,col].yaxis.set_minor_locator(NullLocator())
        axes[2,col].set(yscale='log',ylim=(.015,1.2),yticks=[.02,.1,1])
        axes[2,col].yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'));axes[2,col].yaxis.set_minor_locator(NullLocator())
        axes[3,col].set(ylim=(-.03,1.06),yticks=[0,.5,1])
    for ax,label in zip(axes[:,0],[r'Cosine $\rho_t$',r'Norm $\|g_t\|$',r'Path ratio',r'Paired factors']):ax.set_ylabel(label)
    fig.legend(handles=[Line2D([],[],color='.2',label=r'$R_t$'),
        Line2D([],[],color='.2',ls=':',label=r'$R_t^{\perp}$'),
        Line2D([],[],color='.2',label=r'$C_t$: within-pair retention'),
        Line2D([],[],color='.2',ls='--',label=r'$D_t$: between-pair retention')],
        loc='lower center',bbox_to_anchor=(.53,.006),ncol=4,frameon=False,fontsize=7,handlelength=2,columnspacing=1.15)
    fig.subplots_adjust(left=.085,right=.985,top=.905,bottom=.115,hspace=.32,wspace=.30)
    save(fig,'gd_reversal_paired_displacement')

    fig,axes=setup(3,4.95)
    for col,task in enumerate(TASKS):
        values=a[f'lag_{task}_2256']
        line(axes[0,col],np.arange(1,257),values,'lag256_'+task,color=COLORS[0],lw=.65)
        line(axes[1,col],np.arange(1,33),values[:32],'lag32_'+task,color=COLORS[0],lw=.85,marker='o',ms=1.8)
        for row in [0,1]:
            axes[row,col].set(ylim=(-1.08,1.08),yticks=[-1,0,1],xlabel=r'Lag $k$')
            axes[row,col].axhline(0,color='.5',lw=.5)
        axes[0,col].set(xlim=(1,256),xticks=[1,64,128,192,256]);axes[1,col].set(xlim=(1,32),xticks=[1,8,16,24,32])
        selected=sorted([s for s in result['lag_summary'] if s['task']==task],key=lambda s:s['endpoint'])
        for field,color,marker,label in [('mean_odd_first32',COLORS[0],'o','Odd lags'),('mean_even_first32',COLORS[1],'s','Even lags')]:
            line(axes[2,col],[s['endpoint'] for s in selected],[s[field] for s in selected],field+'_'+task,
                 color=color,lw=.85,marker=marker,ms=3,label=label)
        axes[2,col].set(ylim=(-1.08,1.08),yticks=[-1,0,1],xticks=[756,2256,3256],xlabel=r'Endpoint state $s$')
        axes[2,col].axhline(0,color='.5',lw=.5)
    axes[0,0].set_ylabel(r'$\cos(g_t,g_{t-k})$');axes[1,0].set_ylabel(r'$\cos(g_t,g_{t-k})$')
    axes[2,0].set_ylabel('Mean cosine\nwithin lags 1--32')
    handles,labels=axes[2,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.53,.002),ncol=2,frameon=False,fontsize=8)
    fig.subplots_adjust(left=.085,right=.985,top=.90,bottom=.135,hspace=.88,wspace=.29)
    save(fig,'gd_lag_endpoints')

    fig,axes=setup(1,2.25)
    for col,(rid,task) in enumerate(zip(['122','120','126'],TASKS)):
        for offset,color in [(0,COLORS[0]),(1,COLORS[1])]:
            starts=a[f'{rid}_loss_pair_starts{offset}'];mask=starts>=2000
            line(axes[0,col],starts[mask]+2,a[f'{rid}_loss_pair_delta{offset}'][mask],f'loss_pairs_{rid}_{offset}',
                 color=color,lw=.45,alpha=.85,label=f'Offset {offset}')
        axes[0,col].axhline(0,color='.25',lw=.6)
        axes[0,col].set(xlabel='Second update of pair',xlim=(2000,3000),xticks=[2000,2500,3000])
        axes[0,col].ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
    axes[0,0].set_ylabel('Two-update loss change')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.53,.0),ncol=2,frameon=False,fontsize=8)
    fig.subplots_adjust(left=.09,right=.985,top=.76,bottom=.31,wspace=.44)
    save(fig,'gd_two_step_loss_changes')

    # All nine numerical endpoints, including the alternate pairing offset.
    lines=[r'% Generated by Analysis056/03_plot.py.',r'\begin{tabular}{@{}llrrrrrr@{}}',r'\toprule',
        r'Model & $\eta$ & $C_T$ & $D_T$ & $C_T^{\prime}$ & $D_T^{\prime}$ & $R_T^{\perp}$ & $Q_T$ \\',r'\midrule']
    for index,r in enumerate(result['trajectories']):
        if index in [3,6]:lines.append(r'\midrule')
        p0,p1=[e for e in result['endpoints'] if e['run']==r['run']]
        label={'mlp':'MLP','resnet':'ResNet','transformer':'Transformer'}[r['task']]
        vals=[p0['C'],p0['D'],p1['C'],p1['D'],p0['Rperp']]
        lines.append(' & '.join([label,f"{r['lr']:g}"]+[f'{v:.4f}' for v in vals]+[f"{p0['Q']:.2f}"])+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}','']
    table=DRAFT/'tables/analysis056_pairs.tex';table.write_text('\n'.join(lines),encoding='utf-8')
    np.savez_compressed(HERE/'plotted_arrays.npz',**plot_arrays)
    record=dict(status='awaiting_visual_review',generator_sha256=sha(__file__),
        results_sha256=sha(HERE/'results.json'),figures=figures,curves=curves,
        plotted_points=sum(c['points'] for c in curves),table_sha256=sha(table),
        plotted_arrays_sha256=sha(HERE/'plotted_arrays.npz'))
    (HERE/'figure_manifest.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(figures=len(figures),curves=len(curves),plotted_points=record['plotted_points'])))


if __name__=='__main__':main()
