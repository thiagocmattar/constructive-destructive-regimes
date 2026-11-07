"""Whole forward path and first-linear/tanh diagnostics, with static input explicit."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    r=json.loads((HERE/'all_layer_svd_results.json').read_text())
    assert json.loads((HERE/'all_layer_independent_verification.json').read_text())['status']=='passed'
    assert sha(HERE/'all_layer_svd_values.csv')==r['csv_sha256']
    with (HERE/'all_layer_svd_values.csv').open() as f:rows=list(csv.DictReader(f))
    with (HERE/'first_hidden_svd_values.csv').open() as f:previous=list(csv.DictReader(f))
    assert sha(HERE/'first_hidden_svd_values.csv')==r['sources'][(HERE/'first_hidden_svd_values.csv').relative_to(HERE.parents[1]).as_posix()]
    x=np.array([int(q['state']) for q in rows]);assert np.array_equal(x,np.arange(448,801))
    v=lambda key:np.array([float(q[key]) for q in rows])
    colors=dict(preactivation1='#a6761d',hidden1='#2166ac',preactivation2='#d6604d',hidden2='#762a83',output='#333333',parameter_step='#1b7837')
    names=dict(preactivation1=r'First linear $Z^{(1)}$',hidden1=r'First tanh $H^{(1)}$',preactivation2=r'Second linear $Z^{(2)}$',hidden2=r'Second tanh $H^{(2)}$',output=r'Output linear $F$')
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','pdf.fonttype':42,
        'font.size':11,'axes.labelsize':11,'axes.titlesize':12,'xtick.labelsize':9,'ytick.labelsize':9})
    line_count=heat_count=0

    def curve(ax,y,**kwargs):
        nonlocal line_count
        line,=ax.plot(x,y,**kwargs);np.testing.assert_array_equal(line.get_ydata(),y);line_count+=len(y)

    def style(ax,time=True):
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.16,lw=.5)
        if time:ax.set_xlim(448,800);ax.set_xticks([448,512,576,640,704,768,800])

    fig=plt.figure(figsize=(19,15.5))
    grid=fig.add_gridspec(5,5,left=.05,right=.97,bottom=.10,top=.865,
        height_ratios=[.75,1.1,1.1,1.1,2.5],hspace=.48,wspace=.34)
    ax=fig.add_subplot(grid[0,:]);curve(ax,v('loss'),color='#333333',lw=1.1)
    ax.set(title='Full training loss: all stages evaluated on the same samples and states',ylabel=r'$L_t$');style(ax);ax.tick_params(labelbottom=False)
    for c,name in enumerate(names):
        ax=fig.add_subplot(grid[1,c]);curve(ax,v(name+'_cosine'),color=colors[name],lw=1)
        ax.set(title=names[name],ylabel='Consecutive-change cosine',ylim=(-1.05,1.05));ax.axhline(0,color='gray',ls=':',lw=.6)
        style(ax);ax.tick_params(labelbottom=False)
        ax=fig.add_subplot(grid[2,c]);curve(ax,v(name+'_alpha_01')/5000**.5,color=colors[name],lw=.7)
        ax.axhline(0,color='gray',ls=':',lw=.6)
        ax.set(title=f'Leading mode: {100*r["streams"][name]["top1"]:.2f}% energy',ylabel=r'$\alpha_{t,1}/\sqrt{5000}$',xlabel='Completed updates, t');style(ax)
        ax=fig.add_subplot(grid[3,c]);f=np.array(r['streams'][name]['energy_fractions'])
        line,=ax.plot(np.arange(1,26),f.cumsum()[:25],color=colors[name],marker='.',ms=4,lw=1);np.testing.assert_array_equal(line.get_ydata(),f.cumsum()[:25])
        ax.axhline(.95,color='gray',ls=':',lw=.6)
        ax.set(title=f'Top 10 modes: {100*f[:10].sum():.2f}%',xlabel='Modes retained',ylabel='Cumulative change energy',ylim=(0,1.04),xlim=(1,25));ax.set_xticks([1,5,10,15,20,25]);style(ax,False)
        ax=fig.add_subplot(grid[4,c]);matrix=np.array([v(name+f'_alpha_{k:02d}')/5000**.5 for k in range(1,11)])
        limit=abs(matrix).max();im=ax.imshow(matrix,aspect='auto',interpolation='none',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),extent=[447.5,800.5,10.5,.5])
        np.testing.assert_array_equal(np.asarray(im.get_array()),matrix);heat_count+=matrix.size
        ax.set(title='Signed mode coefficients',xlabel='Completed updates, t',ylabel='Fixed SVD mode');ax.set_yticks(range(1,11));ax.set_xticks([448,512,640,768,800])
        fig.colorbar(im,ax=ax,fraction=.035,pad=.025).ax.tick_params(labelsize=8)
    fig.suptitle('CIFAR-10 MLP: the complete forward path of the oscillation',y=.977,fontsize=20)
    fig.text(.52,.946,r'Fixed input $X$ (5,000 samples x 3,072 input coordinates) $\longrightarrow Z^{(1)}\longrightarrow H^{(1)}\longrightarrow Z^{(2)}\longrightarrow H^{(2)}\longrightarrow F$',ha='center',fontsize=14)
    fig.text(.52,.913,r'INPUT CHECK: all 355 input matrices are exactly identical. $\Delta X=0$; input-change rank = 0; cosine and normalized SVD fractions are undefined.',ha='center',fontsize=12,bbox=dict(boxstyle='round,pad=.5',facecolor='#f2f2f2',edgecolor='#cccccc'))
    fig.text(.52,.063,'Each hidden/preactivation matrix has 5,000 sample rows and 200 activation columns; the output has 10 columns.',ha='center',fontsize=11)
    fig.text(.52,.041,r'Each fitted row is an outgoing change $X_{t+1}-X_t$. No sample averaging, centering, state subsampling or per-mode color normalization.',ha='center',fontsize=11)
    fig.text(.52,.019,'Fixed SVD bases are fitted independently; equal mode numbers do not identify the same spatial directions across stages.',ha='center',fontsize=11)
    p1=HERE/'figures/11-all-layer-svd-comparison.pdf';fig.savefig(p1);plt.close(fig)

    fig,axes=plt.subplots(6,1,figsize=(12,16),sharex=True,
        gridspec_kw=dict(left=.11,right=.96,bottom=.11,top=.905,hspace=.32,height_ratios=[.8,1.1,1,1,1,1]))
    curve(axes[0],v('loss'),color='#333333',lw=1.1);axes[0].set(title='(a) Full training loss',ylabel=r'$L_t$')
    for name,label,ls in [('parameter_step',r'Actual first-layer parameter steps $\delta\theta^{(1)}$','--'),('preactivation1',r'First linear changes $\Delta Z^{(1)}$','-'),('hidden1',r'First hidden changes $\Delta H^{(1)}$','-.'),('output',r'Output changes $\Delta F$',':')]:
        curve(axes[1],v(name+'_cosine'),label=label,color=colors[name],ls=ls,lw=1)
    axes[1].set(title='(b) Actual parameter-step and representation-change alignment',ylabel='Consecutive cosine',ylim=(-1.05,1.05));axes[1].legend(frameon=False,ncol=2,fontsize=9,loc='best')
    for name,label,ls in [('preactivation1',r'Before first tanh: $\Delta Z^{(1)}$','-'),('hidden1',r'After first tanh: $\Delta H^{(1)}$','--')]:
        curve(axes[2],v(name+'_norm')/5000**.5,label=label,color=colors[name],ls=ls,lw=1.1)
    axes[2].set(title='(c) First tanh changes the amplitude of the representation motion',ylabel='Per-example vector RMS');axes[2].legend(frameon=False,ncol=2,fontsize=9,loc='best')
    curve(axes[3],v('first_tanh_displacement_gain'),color=colors['hidden1'],label=r'First tanh: $\|\Delta H^{(1)}\|/\|\Delta Z^{(1)}\|$',lw=1.2)
    curve(axes[3],np.array([float(q['tanh_displacement_gain']) for q in previous]),color=colors['hidden2'],label=r'Second tanh: $\|\Delta H^{(2)}\|/\|\Delta Z^{(2)}\|$',lw=1.2)
    axes[3].set(title='(d) Measured finite-displacement gain of both tanh layers',ylabel='Norm ratio',ylim=(0,1.05));axes[3].axhline(1,color='gray',ls=':',lw=.7);axes[3].legend(frameon=False,ncol=2,fontsize=9,loc='best')
    energy=v('hidden1_norm')**2
    curve(axes[4],v('h_weights_dot')/energy,color=colors['preactivation1'],label=r'First-layer weight changes, $S\odot(X\Delta W_1^T)$',lw=1.2)
    curve(axes[4],v('h_bias_dot')/energy,color='#777777',label=r'First-layer bias changes, $S\odot\Delta b_1$',lw=1.2)
    axes[4].set(title='(e) Signed contributions to the first hidden change; the input matrix is fixed',ylabel='Signed projection share');axes[4].legend(frameon=False,ncol=1,fontsize=9,loc='best')
    curve(axes[5],v('secant_gt_half_coordinate_fraction'),color='#2166ac',label='Fraction of sample-neuron coordinates with secant > 0.5',lw=1.1)
    curve(axes[5],v('h1_change_energy_on_secant_gt_half'),color='#b2182b',label='Fraction of first-hidden change energy on those coordinates',lw=1.1)
    axes[5].set(title='(f) Which coordinates carry the motion through the first tanh?',ylabel='Fraction',ylim=(0,1.05),xlabel='Completed optimizer updates, t');axes[5].legend(frameon=False,ncol=1,fontsize=9,loc='best')
    for ax in axes:style(ax)
    fig.suptitle('CIFAR-10 MLP: fixed images, changing first-layer projection',y=.975,fontsize=18)
    fig.text(.535,.944,r'$\Delta Z^{(1)}_t=X\Delta W_{1,t}^{T}+\Delta b_{1,t}$; $\Delta H^{(1)}_t=S_t\odot\Delta Z^{(1)}_t$, with measured forward-roundoff retained.',ha='center',fontsize=12)
    fig.text(.535,.071,'Actual parameter increments are measured from successive weights and biases; they are not raw gradients or gradient differences.',ha='center',fontsize=10)
    fig.text(.535,.049,'The secant is the elementwise finite tanh slope. Threshold 0.5 summarizes transmission strength, not a binary neuron state.',ha='center',fontsize=10)
    fig.text(.535,.027,'These observations locate the earliest changing representation; they do not identify the unique cause of the optimizer oscillation.',ha='center',fontsize=10)
    p2=HERE/'figures/12-fixed-input-first-layer-transmission.pdf';fig.savefig(p2);plt.close(fig)
    audit=dict(status='numeric_checks_passed',time_series_values=line_count,heatmap_values=heat_count,spectrum_values=125,
        source_csv_sha256=sha(HERE/'all_layer_svd_values.csv'),generator_sha256=sha(Path(__file__)),
        pdfs={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in [p1,p2]})
    (HERE/'all_layer_figure_verification.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
