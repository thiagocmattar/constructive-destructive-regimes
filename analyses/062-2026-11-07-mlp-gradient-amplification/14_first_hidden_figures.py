"""Earlier-hidden oscillation and transmission figures from verified histories."""
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
    r=json.loads((HERE/'first_hidden_svd_results.json').read_text())
    assert json.loads((HERE/'first_hidden_independent_verification.json').read_text())['status']=='passed'
    assert sha(HERE/'first_hidden_svd_values.csv')==r['csv_sha256']
    with (HERE/'first_hidden_svd_values.csv').open() as f:rows=list(csv.DictReader(f))
    x=np.array([int(q['state']) for q in rows]);assert np.array_equal(x,np.arange(448,801))
    v=lambda key:np.array([float(q[key]) for q in rows])
    colors=dict(hidden1='#2166ac',preactivation2='#d6604d',hidden2='#762a83',output='#333333')
    names=dict(hidden1=r'First hidden $H^{(1)}$',preactivation2=r'Before second tanh $Z^{(2)}$',hidden2=r'Second hidden $H^{(2)}$',output=r'Outputs $F$')
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','pdf.fonttype':42,
                         'font.size':11,'axes.labelsize':11,'axes.titlesize':12,'xtick.labelsize':9,'ytick.labelsize':9})
    line_count=heat_count=0

    def curve(ax,y,**kwargs):
        nonlocal line_count
        line,=ax.plot(x,y,**kwargs);np.testing.assert_array_equal(line.get_ydata(),y);line_count+=len(y)

    def style(ax,time=True):
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.16,lw=.5)
        if time:ax.set_xlim(448,800);ax.set_xticks([448,512,576,640,704,768,800])

    fig=plt.figure(figsize=(16,15))
    grid=fig.add_gridspec(5,4,left=.06,right=.97,bottom=.105,top=.91,
        height_ratios=[.75,1.1,1.1,1.1,2.5],hspace=.48,wspace=.34)
    ax=fig.add_subplot(grid[0,:]);curve(ax,v('loss'),color='#333333',lw=1.1)
    ax.set(title='Full training loss: matched samples, states and trajectory',ylabel=r'$L_t$');style(ax);ax.tick_params(labelbottom=False)
    for c,name in enumerate(names):
        ax=fig.add_subplot(grid[1,c]);curve(ax,v(name+'_cosine'),color=colors[name],lw=1)
        ax.set(title=names[name],ylabel='Consecutive-change cosine',ylim=(-1.05,1.05));ax.axhline(0,color='gray',ls=':',lw=.6)
        style(ax);ax.tick_params(labelbottom=False)
        ax=fig.add_subplot(grid[2,c]);curve(ax,v(name+'_alpha_01')/5000**.5,color=colors[name],lw=.7)
        ax.axhline(0,color='gray',ls=':',lw=.6)
        ax.set(title=f'Leading mode: {100*r["streams"][name]["top1"]:.2f}% energy',ylabel=r'$\alpha_{t,1}/\sqrt{5000}$',xlabel='Completed updates, t')
        style(ax)
        ax=fig.add_subplot(grid[3,c]);fractions=np.array(r['streams'][name]['energy_fractions'])
        line,=ax.plot(np.arange(1,26),fractions.cumsum()[:25],color=colors[name],marker='.',ms=4,lw=1)
        np.testing.assert_array_equal(line.get_ydata(),fractions.cumsum()[:25])
        ax.axhline(.95,color='gray',ls=':',lw=.6)
        ax.set(title=f'Top 10: {100*fractions[:10].sum():.2f}%',xlabel='Number of modes retained',ylabel='Cumulative change energy',ylim=(0,1.04),xlim=(1,25))
        ax.set_xticks([1,5,10,15,20,25]);style(ax,False)
        ax=fig.add_subplot(grid[4,c]);matrix=np.array([v(name+f'_alpha_{k:02d}')/5000**.5 for k in range(1,11)])
        limit=abs(matrix).max();im=ax.imshow(matrix,aspect='auto',interpolation='none',cmap='RdBu_r',
            norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),extent=[447.5,800.5,10.5,.5])
        np.testing.assert_array_equal(np.asarray(im.get_array()),matrix);heat_count+=matrix.size
        ax.set(title='Signed mode coefficients',xlabel='Completed updates, t',ylabel='Fixed SVD mode')
        ax.set_yticks(range(1,11));ax.set_xticks([448,512,640,768,800])
        cb=fig.colorbar(im,ax=ax,fraction=.035,pad=.025);cb.ax.tick_params(labelsize=8)
    fig.suptitle('CIFAR-10 MLP: tracing output oscillation backward through both hidden layers',y=.975,fontsize=18)
    fig.text(.52,.946,r'$H^{(1)}$ (5,000 samples x 200 units) $\longrightarrow$ linear $Z^{(2)}$ (5,000 x 200) $\longrightarrow$ tanh $H^{(2)}$ (5,000 x 200) $\longrightarrow$ output $F$ (5,000 x 10)',ha='center',fontsize=12)
    fig.text(.52,.064,r'Each fitted row is $X_{t+1}-X_t$; each cosine compares outgoing and incoming changes around $t$. All examples are retained.',ha='center',fontsize=11)
    fig.text(.52,.042,'Fixed uncentered SVDs are fitted independently. Equal widths do not make hidden columns correspond between layers.',ha='center',fontsize=11)
    fig.text(.52,.020,'Signs are fixed within each basis; mode numbers and coefficient signs across spaces do not imply matched directions.',ha='center',fontsize=11)
    p1=HERE/'figures/09-first-hidden-svd-comparison.pdf';fig.savefig(p1);plt.close(fig)

    fig,axes=plt.subplots(6,1,figsize=(11.8,16),sharex=True,
        gridspec_kw=dict(left=.11,right=.96,bottom=.105,top=.915,hspace=.33,height_ratios=[.8,1.1,1,1,1,1]))
    curve(axes[0],v('loss'),color='#333333',lw=1.1);axes[0].set(title='(a) Full training loss',ylabel=r'$L_t$')
    for name,ls in [('hidden1','-'),('preactivation2','--'),('hidden2','-.'),('output',':')]:
        curve(axes[1],v(name+'_cosine'),label=names[name],color=colors[name],ls=ls,lw=1)
    axes[1].set(title='(b) Reversal geometry across the two hidden layers and outputs',ylabel='Consecutive cosine',ylim=(-1.05,1.05));axes[1].legend(frameon=False,ncol=2,fontsize=9,loc='upper left')
    for name,ls in [('hidden1','-'),('preactivation2','--'),('hidden2','-.')]:
        curve(axes[2],v(name+'_norm')/5000**.5,label=names[name],color=colors[name],ls=ls,lw=1)
    axes[2].set(title='(c) Displacement amplitude along the representation pathway',ylabel='Per-example vector RMS');axes[2].legend(frameon=False,ncol=3,fontsize=9,loc='upper right')
    curve(axes[3],v('linear_transmission_gain'),color=colors['preactivation2'],label=r'Existing linear layer: $\|\Delta H^{(1)}W_{2,t}^{T}\|/\|\Delta H^{(1)}\|$',lw=1)
    curve(axes[3],v('tanh_displacement_gain'),color=colors['hidden2'],label=r'Tanh: $\|\Delta H^{(2)}\|/\|\Delta Z^{(2)}\|$',lw=1)
    axes[3].axhline(1,color='gray',ls=':',lw=.7);axes[3].set(title='(d) Measured directional gain through the linear layer and tanh',ylabel='Norm ratio');axes[3].legend(frameon=False,ncol=1,fontsize=9,loc='best')
    energy=v('hidden2_norm')**2
    for name,label,color,ls in [('upstream','First-hidden change',colors['hidden1'],'-'),('weights','Second-layer weight change','#1b7837','--'),('interaction','Interaction',colors['preactivation2'],':'),('bias','Second-layer bias change','#777777','-.')]:
        curve(axes[4],v('h_'+name+'_dot')/energy,label=label,color=color,ls=ls,lw=1)
    axes[4].set(title='(e) Signed contributions to second-hidden motion, after tanh',ylabel='Signed projection share');axes[4].legend(frameon=False,ncol=2,fontsize=9,loc='best')
    for key,name,ls in [('h1_activation_rms','hidden1','-'),('z2_activation_rms','preactivation2','--'),('h2_activation_rms','hidden2','-.')]:
        curve(axes[5],v(key),label=names[name],color=colors[name],ls=ls,lw=1)
    axes[5].set(title='(f) Activation magnitude is distinct from the size of its stepwise changes',ylabel='Activation coordinate RMS',xlabel='Completed optimizer updates, t');axes[5].legend(frameon=False,ncol=3,fontsize=9,loc='best')
    for ax in axes:style(ax)
    fig.suptitle('CIFAR-10 MLP: how the first hidden oscillation reaches the second layer',y=.975,fontsize=17)
    fig.text(.535,.946,r'$Z^{(2)}_t=H^{(1)}_tW_{2,t}^{T}+b_{2,t},\quad H^{(2)}_t=\tanh(Z^{(2)}_t)$; rows are samples, columns are activations.',ha='center',fontsize=12)
    fig.text(.535,.067,'Panel (e) uses the observed elementwise tanh secant to transmit each additive preactivation contribution.',ha='center',fontsize=10)
    fig.text(.535,.046,'Signed projection shares can be negative; they sum to one including the retained forward-roundoff residual.',ha='center',fontsize=10)
    fig.text(.535,.025,'These are measured-trajectory identities, not interventions or an identification of the unique initiating cause.',ha='center',fontsize=10)
    p2=HERE/'figures/10-first-to-second-hidden-transmission.pdf';fig.savefig(p2);plt.close(fig)
    audit=dict(status='numeric_checks_passed',time_series_values=line_count,heatmap_values=heat_count,spectrum_values=100,
               source_csv_sha256=sha(HERE/'first_hidden_svd_values.csv'),generator_sha256=sha(Path(__file__)),
               pdfs={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in [p1,p2]})
    (HERE/'first_hidden_figure_verification.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
