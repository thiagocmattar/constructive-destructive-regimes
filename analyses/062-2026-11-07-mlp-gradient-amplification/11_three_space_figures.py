"""Comparative SVD and output-visible hidden-motion figures from verified tables."""
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
    r=json.loads((HERE/'three_space_svd_results.json').read_text())
    audit=json.loads((HERE/'three_space_svd_independent_verification.json').read_text())
    assert audit['status']=='passed'
    assert sha(HERE/'three_space_svd_values.csv')==r['csv_sha256']
    with (HERE/'three_space_svd_values.csv').open() as f:rows=list(csv.DictReader(f))
    x=np.array([int(q['state']) for q in rows]);assert np.array_equal(x,np.arange(448,801))
    value=lambda key:np.array([float(q[key]) for q in rows])
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','pdf.fonttype':42,
                         'font.size':11,'axes.labelsize':11,'axes.titlesize':12,'xtick.labelsize':9,'ytick.labelsize':9})
    colors=dict(gradient='#2166ac',output='#b2182b',hidden='#762a83',visible='#d6604d',null='#1b7837')
    line_values=0;heat_values=0

    def curve(ax,y,**kwargs):
        nonlocal line_values
        line,=ax.plot(x,y,**kwargs);np.testing.assert_array_equal(line.get_ydata(),y);line_values+=len(y)
        return line

    def decorate(ax,time_axis=True):
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.16,lw=.5)
        if time_axis:ax.set_xlim(448,800);ax.set_xticks([448,512,576,640,704,768,800])

    fig=plt.figure(figsize=(13,15))
    grid=fig.add_gridspec(5,3,left=.075,right=.96,bottom=.115,top=.91,
                          height_ratios=[.75,1.1,1.1,1.1,2.5],hspace=.47,wspace=.34)
    loss=fig.add_subplot(grid[0,:]);curve(loss,value('loss'),color='#333333',lw=1)
    loss.set(title='(a) Full training loss: the same trajectory for all three spaces',ylabel=r'$L_t$')
    decorate(loss);loss.tick_params(labelbottom=False)
    for c,(name,title,formula,scale) in enumerate([
        ('gradient','Raw gradients',r'$g_t$',1.),
        ('output','Output changes',r'$F_{t+1}-F_t$',5000**.5),
        ('hidden','Hidden changes',r'$H_{t+1}-H_t$',5000**.5)]):
        cosine=fig.add_subplot(grid[1,c]);curve(cosine,value(name+'_cosine'),color=colors[name],lw=1)
        cosine.set(title=f'{title}: {formula}',ylabel='Consecutive cosine',ylim=(-1.05,1.05));cosine.axhline(0,color='gray',lw=.6,ls=':')
        decorate(cosine);cosine.tick_params(labelbottom=False)
        alpha=fig.add_subplot(grid[2,c]);curve(alpha,value(name+'_alpha_01')/scale,color=colors[name],lw=.7)
        alpha.axhline(0,color='gray',lw=.6,ls=':')
        alpha.set(title=f'Leading mode: {100*r["streams"][name]["top1"]:.2f}% energy',
                  ylabel=r'$\alpha_{t,1}$' if name=='gradient' else r'$\alpha_{t,1}/\sqrt{5000}$')
        decorate(alpha);alpha.set_xlabel('Completed updates, t')
        spectrum=fig.add_subplot(grid[3,c]);fractions=np.array(r['streams'][name]['energy_fractions'])
        spectrum.plot(np.arange(1,26),fractions.cumsum()[:25],color=colors[name],marker='.',ms=4,lw=1)
        spectrum.axhline(.95,color='gray',ls=':',lw=.7)
        spectrum.set(title=f'Top 10 modes: {100*fractions[:10].sum():.2f}%',xlabel='Number of modes retained',ylabel='Cumulative energy',ylim=(0,1.04),xlim=(1,25))
        spectrum.set_xticks([1,5,10,15,20,25]);decorate(spectrum,False)
        heat=fig.add_subplot(grid[4,c]);matrix=np.array([value(name+f'_alpha_{k:02d}')/scale for k in range(1,11)])
        limit=abs(matrix).max();im=heat.imshow(matrix,aspect='auto',interpolation='none',cmap='RdBu_r',
            norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),extent=[447.5,800.5,10.5,.5])
        np.testing.assert_array_equal(np.asarray(im.get_array()),matrix);heat_values+=matrix.size
        heat.set_yticks(range(1,11));heat.set_xticks([448,512,640,768,800]);heat.set_xlabel('Completed updates, t')
        heat.set_ylabel('Fixed SVD mode');heat.set_title('Signed mode coefficients')
        cb=fig.colorbar(im,ax=heat,fraction=.035,pad=.025);cb.ax.tick_params(labelsize=8)
    fig.suptitle('CIFAR-10 MLP: raw-gradient, output-change and hidden-change SVDs',y=.975,fontsize=16)
    fig.text(.52,.947,'Same 353 states | All 5,000 examples | Fixed uncentered basis fitted independently in each space',ha='center',fontsize=11)
    fig.text(.52,.073,r'Gradient alignment: $\cos(g_t,g_{t-1})$. Output/hidden alignment: incoming versus outgoing displacement around $t$.',ha='center',fontsize=10)
    fig.text(.52,.050,'Gradient rows are raw gradients, NOT differences. Hidden rows use the second tanh, immediately before the linear head.',ha='center',fontsize=10)
    fig.text(.52,.028,'Mode signs are fixed within each space. Mode numbers and colors do not imply corresponding axes across spaces.',ha='center',fontsize=10)
    p1=HERE/'figures/07-three-space-svd-comparison.pdf';fig.savefig(p1);plt.close(fig)

    fig,axes=plt.subplots(5,1,figsize=(11.5,14),sharex=True,
        gridspec_kw=dict(left=.10,right=.96,bottom=.12,top=.91,hspace=.27,height_ratios=[.8,1.1,1.1,1.1,1.2]))
    curve(axes[0],value('loss'),color='#333333',lw=1.1);axes[0].set(title='(a) Full training loss',ylabel=r'$L_t$')
    for name,label,ls in [('output','Output changes',':'),('hidden','All hidden changes','-'),('visible','Visible hidden changes','--'),('null','Null hidden changes','-.')]:
        curve(axes[1],value(name+'_cosine'),label=label,color=colors[name],ls=ls,lw=1)
    axes[1].set(title='(b) Both hidden components reverse; the visible component tracks outputs most closely',ylabel='Consecutive cosine',ylim=(-1.05,1.05))
    axes[1].legend(frameon=False,ncol=2,fontsize=9,loc='upper left')
    for name,label,ls in [('hidden','All hidden changes','-'),('visible','Visible component','--'),('null','Null component','-.')]:
        curve(axes[2],value(name+'_norm')/5000**.5,label=label,color=colors[name],ls=ls,lw=1.1)
    axes[2].set(title='(c) Hidden displacement magnitude grows during the spikes',ylabel=r'Per-example RMS $\|\Delta h\|$')
    axes[2].legend(frameon=False,ncol=3,fontsize=9,loc='upper right')
    curve(axes[3],value('hidden_change_visible_share'),label='Share of hidden-change energy (fixed head)',color=colors['visible'],lw=1.2)
    curve(axes[3],value('hidden_change_instant_head_visible_share'),label='Share using instantaneous head',color='#555555',lw=.8,ls=':')
    curve(axes[3],value('hidden_activation_visible_share'),label='Share of hidden-activation energy',color='#2166ac',lw=1.2)
    axes[3].set(title='(d) Ten visible dimensions carry 68.1% of change energy, but only 4.5-4.9% of activation energy',ylabel='Visible energy fraction',ylim=(0,.85))
    axes[3].legend(frameon=False,fontsize=9,loc='center right')
    for key,label,color,ls in [('output_alpha_01','Observed output coefficient','#333333','-'),('alpha_hidden','Hidden-change contribution',colors['visible'],'--'),('alpha_head','Head-weight contribution','#2166ac','-'),('alpha_interaction','Interaction','#1b7837',':')]:
        curve(axes[4],value(key)/5000**.5,label=label,color=color,ls=ls,lw=1)
    axes[4].set(title='(e) Leading output oscillation is supplied mainly by hidden changes through the existing head',ylabel=r'Coefficient / $\sqrt{5000}$',xlabel='Completed optimizer updates, t')
    axes[4].legend(frameon=False,ncol=2,fontsize=9,loc='upper left');axes[4].axhline(0,color='gray',ls=':',lw=.6)
    for ax in axes:decorate(ax)
    fig.suptitle('CIFAR-10 MLP: where does hidden motion concentrate?',y=.975,fontsize=17)
    fig.text(.53,.948,'Fixed reference head at state 448: 10 visible dimensions and 190 null dimensions per example',ha='center',fontsize=11)
    fig.text(.53,.071,'Final hidden activation coordinate RMS stays within 0.3958-0.4009; stable magnitude does not mean stationary features.',ha='center',fontsize=10)
    fig.text(.53,.049,'The null component still carries 31.9% of change energy and also oscillates. Visibility is relative to the fixed reference head.',ha='center',fontsize=10)
    fig.text(.53,.027,'Panel (e) uses the output-space basis; bias and floating-point residuals are retained in the numerical decomposition.',ha='center',fontsize=10)
    p2=HERE/'figures/08-hidden-visible-nullspace.pdf';fig.savefig(p2);plt.close(fig)
    audit=dict(status='numeric_checks_passed',time_series_values=line_values,heatmap_values=heat_values,
               source_csv_sha256=sha(HERE/'three_space_svd_values.csv'),generator_sha256=sha(Path(__file__)),
               pdfs={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in [p1,p2]})
    (HERE/'three_space_figure_verification.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
