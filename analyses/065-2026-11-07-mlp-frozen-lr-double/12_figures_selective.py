"""Separate gate-method and paired-control PDFs; all earlier figures preserved."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

HERE=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
                    'font.size':10,'mathtext.fontset':'stix','pdf.fonttype':42})
COLORS={'control':'#2166ac','discard_q90':'#b8860b','cap_q90':'#a64d79',
        'neuron_q90':'#00876c','neuron_q75':'#00876c','neuron_q50':'#00876c'}
LABELS={'discard_q90':'Discard large coordinates','cap_q90':'Cap large coordinates',
        'neuron_q90':'Neuron cap: prefix q90','neuron_q75':'Neuron cap: prefix q75','neuron_q50':'Neuron cap: prefix q50'}


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def values(c):
    gate=c.get('gate',dict(retained_norm_fraction=[1.]*353,affected_fraction=[0.]*353))
    return [c['loss'],c['eta_kappa_hat'],c['gradient_rho'],c['raw_gradient_norm'],
            c['layers']['hidden1']['rho'],c['layers']['hidden1']['change_rms'],
            c['layers']['output']['rho'],c['layers']['output']['change_rms'],
            gate['retained_norm_fraction'],gate['affected_fraction']]


def main():
    source=HERE/'selective_comparison.json';r=json.loads(source.read_text());assert r['status']=='verified'
    assert r['generator_sha256']==sha(HERE/'11_reduce_selective.py')
    original={p.name:sha(p) for p in (HERE/'figures').glob('0[123]-*.pdf')}
    t=np.array(r['states']);curves=r['curves'];specs={q['arm']:q for q in r['plan']['arms']}
    allarms=['control',*specs];limits={}
    for i in [1,3,5,7,8]:
        data=np.concatenate([np.array(values(curves[a])[i]) for a in allarms])
        if i in [1,8]:limits[i]=(min(-.05,float(data.min())*1.1),max(1.05,float(data.max())*1.1))
        else:
            assert np.all(data>0);limits[i]=(float(data.min())/1.15,float(data.max())*1.15)
    rowlabels=['(a) Full training loss',r'(b) $\eta\widehat\kappa_t$ (symlog)',
        '(c) Raw-gradient\ncosine',r'(d) Full $\|g_t\|$ (log)',
        '(e) First-hidden\nchange cosine','(f) First-hidden\nRMS (log)',
        '(g) Output-change\ncosine','(h) Output RMS (log)',
        '(i) First-layer norm\napplied / proposed','(j) Fraction modified']
    outputs=[];checked=0

    def page(arms,title,paired):
        nonlocal checked
        fig,axes=plt.subplots(10,len(arms),figsize=(5.1*len(arms)+1.2,18),sharex=True,squeeze=False)
        loss_limits={}
        for col,arm in enumerate(arms):
            names=['control',arm]+(['uniform_'+arm] if paired else [])
            for name in names:
                color=COLORS.get(name,'#555555');style='--' if name.startswith('uniform_') else '-'
                for i,data in enumerate(values(curves[name])):
                    data=np.asarray(data,dtype=float);assert data.shape==t.shape and np.isfinite(data).all()
                    line,=axes[i,col].plot(t,data,color=color,ls=style,lw=1.05)
                    np.testing.assert_array_equal(line.get_ydata(),data);checked+=len(t)
            maximum=max(max(curves[n]['loss']) for n in names)
            axes[0,col].set_ylim(0,maximum*1.09);loss_limits[arm]=list(axes[0,col].get_ylim())
            for i in [2,4,6]:axes[i,col].set_ylim(-1.05,1.05);axes[i,col].axhline(0,color='.5',lw=.6)
            for i in [3,5,7]:axes[i,col].set_yscale('log')
            axes[1,col].set_yscale('symlog',linthresh=1);axes[1,col].axhline(1,color='.5',lw=.6,ls=':')
            axes[8,col].set_yscale('symlog',linthresh=1);axes[8,col].axhline(1,color='.5',lw=.6,ls=':')
            axes[9,col].set_ylim(-.03,1.03)
            for i,lim in limits.items():axes[i,col].set_ylim(*lim)
            for i,ax in enumerate(axes[:,col]):
                ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.2,lw=.6);ax.set_xlim(448,800)
                if col and i:ax.tick_params(which='both',labelleft=False)
            threshold=r['thresholds']['coordinate_q90'] if specs[arm]['kind']!='neuron' else r['thresholds']['neuron'][specs[arm]['threshold_key']]
            axes[0,col].set_title(specs[arm]['label']+': '+LABELS[arm]+f'\nFixed threshold {threshold:.5g}',fontsize=11,pad=9)
            axes[-1,col].set_xlabel('Completed optimizer updates $t$')
            if col:
                for i in range(10):np.testing.assert_array_equal(axes[i,0].lines[0].get_ydata(),axes[i,col].lines[0].get_ydata())
        for ax,label in zip(axes[:,0],rowlabels):ax.set_ylabel(label)
        fig.suptitle(title,fontsize=16,y=.985)
        fig.text(.5,.965,'All gradients and Adam moments retained; first-layer updates transformed from 448; all learning rates stay 0.001.',ha='center',fontsize=10)
        legend=[Line2D([],[],color=COLORS['control'],label='Unchanged Adam')]
        if paired and all(COLORS[a]=='#00876c' for a in arms):
            legend.append(Line2D([],[],color='#00876c',label='Selective policy (column color)'))
        else:
            legend.extend(Line2D([],[],color=COLORS[a],label=LABELS[a]) for a in arms)
        if paired:legend.append(Line2D([],[],color='#555555',ls='--',label='Uniform control: matched first-layer norm'))
        fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.54,.954),ncol=len(legend),frameon=False)
        fig.text(.5,.022,'Loss: independent linear axes. Other rows use common scales. All 5,000 samples; outgoing changes; no smoothing.'
            '\nRow (i): first-layer norm after modification / native proposal; a uniform control may exceed 1. Row (j): coordinate or neuron fraction modified.'
            '\nThresholds use only untreated updates 416-447. Norm controls follow the selective arm at the same update; only first-layer norms are matched.',
            ha='center',va='center',fontsize=9)
        fig.subplots_adjust(left=.075 if len(arms)==3 else .105,right=.985,bottom=.064,top=.912,hspace=.22,wspace=.20)
        return fig,dict(arms=arms,paired=paired,loss_limits=loss_limits)

    filenames=['04-selective-update-methods-cifar10-mlp.pdf','05-selective-update-norm-controls-cifar10-mlp.pdf']
    plans=[[(list(LABELS)[:3],'MLP: three selective-update methods',False)],
           [(['neuron_q90','neuron_q75','neuron_q50'],'MLP: neuron displacement caps and matched-norm controls',True),
            (['discard_q90','cap_q90'],'MLP: coordinate gates and matched-norm controls',True)]]
    for filename,pages in zip(filenames,plans):
        target=HERE/'figures'/filename;records=[]
        with PdfPages(target) as pdf:
            for args in pages:
                fig,record=page(*args);pdf.savefig(fig);plt.close(fig);records.append(record)
        outputs.append(dict(file=filename,pdf_sha256=sha(target),pages=records))
    assert original=={p.name:sha(p) for p in (HERE/'figures').glob('0[123]-*.pdf')}
    audit=dict(status='numeric_checks_passed',source_sha256=sha(source),generator_sha256=sha(Path(__file__)),
        outputs=outputs,plotted_values_checked=checked,identical_baselines=True,independent_loss_axes=True,
        original_figures_unchanged=original)
    (HERE/'selective_figure_verification.json').write_text(json.dumps(audit,indent=2)+'\n',newline='\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
