"""New three-page LR-location comparison; preserve all earlier figures."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

HERE=Path(__file__).resolve().parent
PAGES=[('First layer only','first_half','first_quarter'),
       ('All layers: global LR controls','global_half','global_quarter'),
       ('Later layers only: location controls','later_half','later_quarter')]
COLORS=['#2166ac','#a64d79','#b8860b']
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
                    'font.size':10,'mathtext.fontset':'stix','pdf.fonttype':42})


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def values(c):
    return [c['loss'],c['eta_kappa_hat'],c['gradient_rho'],c['raw_gradient_norm'],c['active_gradient_norm'],
            c['layers']['hidden1']['rho'],c['layers']['hidden1']['change_rms'],
            c['layers']['output']['rho'],c['layers']['output']['change_rms'],c['first_lr']]


def main():
    source=HERE/'layer_lr_comparison.json';record=json.loads(source.read_text());assert record['status']=='verified'
    assert record['generator_sha256']==sha(HERE/'07_reduce_layer_lr.py')
    previous={p.name:sha(p) for p in (HERE/'figures').glob('0[12]-*.pdf')}
    t=np.array(record['states']);assert np.array_equal(t,np.arange(448,801))
    arms=['control',*[arm for page in PAGES for arm in page[1:]]];curves=record['curves']
    specs={q['arm']:q for q in record['plan']['arms']}
    labels=['(a) Full training loss',r'(b) $\eta\widehat\kappa_t$ (symlog)',
            '(c) Raw-gradient cosine',r'(d) Full $\|g_t\|$ (log)',r'(e) Later-layer $\|g_t\|$ (log)',
            '(f) First-hidden change cosine','(g) First-hidden RMS (log)',
            '(h) Output-change cosine','(i) Output RMS (log)','(j) Applied learning rate']
    limits={}
    for i in [1,3,4,6,8]:
        a=np.concatenate([np.asarray(values(curves[arm])[i]) for arm in arms])
        limits[i]=(min(-.1,float(a.min())*1.1),max(1.1,float(a.max())*1.1)) if i==1 else (float(a.min())/1.15,float(a.max())*1.15)
    target=HERE/'figures/03-selective-layer-lr-cifar10-mlp.pdf';checked=0;pages=[]
    with PdfPages(target) as pdf:
        for page,(title,left,right) in enumerate(PAGES,1):
            fig,axes=plt.subplots(10,2,figsize=(12,19),sharex=True)
            for i in range(1,10):axes[i,1].sharey(axes[i,0]);axes[i,1].tick_params(labelleft=False)
            handles={};loss_ranges={}
            for j,arm in enumerate([left,right]):
                spec=specs[arm]
                for name,color in [('control',COLORS[0]),(arm,COLORS[j+1])]:
                    c=curves[name]
                    for i,ys in enumerate(values(c)):
                        ys=np.asarray(ys,dtype=float);assert ys.shape==t.shape and np.isfinite(ys).all()
                        line,=axes[i,j].plot(t,ys,color=color,lw=1.1)
                        np.testing.assert_array_equal(line.get_ydata(),ys);checked+=len(ys)
                        if i==0:handles[name]=line
                    for i,key in [(2,'active_gradient_rho'),(9,'later_lr')]:
                        line,=axes[i,j].plot(t,c[key],color=color,lw=1,ls='--')
                        np.testing.assert_array_equal(line.get_ydata(),c[key]);checked+=len(t)
                maximum=max(max(curves[name]['loss']) for name in ['control',arm])
                axes[0,j].set_ylim(0,1.08*maximum);loss_ranges[arm]=list(axes[0,j].get_ylim())
                for i in [2,5,7]:axes[i,j].set_ylim(-1.05,1.05);axes[i,j].axhline(0,color='.5',lw=.6)
                for i in [3,4,6,8]:axes[i,j].set_yscale('log')
                axes[1,j].set_yscale('symlog',linthresh=1);axes[1,j].axhline(1,color='.5',lw=.6,ls=':')
                for i,lim in limits.items():axes[i,j].set_ylim(*lim)
                axes[9,j].set_ylim(.00015,.0011);axes[9,j].set_yticks([.00025,.0005,.001],['0.00025','0.0005','0.001'])
                for ax in axes[:,j]:
                    ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.2,lw=.6);ax.set_xlim(448,800)
                multiplier='0.5' if j==0 else '0.25'
                axes[0,j].set_title(f'I{spec["run_id"]-243}: {multiplier}'+r'$\times$ LR'+
                    f'\nFirst layer {spec["first_lr"]:g}; later layers {spec["later_lr"]:g}',fontsize=12,pad=9)
                axes[-1,j].set_xlabel('Completed optimizer updates $t$')
            for i in range(10):
                axes[i,0].set_ylabel(labels[i]);np.testing.assert_array_equal(axes[i,0].lines[0].get_ydata(),axes[i,1].lines[0].get_ydata())
            assert not axes[0,0].get_shared_y_axes().joined(axes[0,0],axes[0,1])
            fig.suptitle('MLP: '+title,fontsize=16,y=.986)
            fig.text(.5,.968,'All layers trainable; LR changes at 448; same warm Adam state and all 5,000 training examples.',ha='center',fontsize=10)
            fig.legend([handles[n] for n in ['control',left,right]],['Unchanged Adam (LR 0.001)',r'0.5$\times$ intervention',r'0.25$\times$ intervention'],
                       loc='upper center',bbox_to_anchor=(.55,.958),ncol=3,frameon=False)
            fig.text(.5,.024,'Loss: independent linear axes. Other rows share scales across all pages. No smoothing.'
                '\nRow (c): solid = full raw gradient; dashed = later-layer gradient. Row (j): solid = first-layer LR; dashed = later-layer LR.'
                '\nFirst-hidden and output metrics retain all sample coordinates. Changes are outgoing. '+r'$\eta\widehat\kappa_t$ is a projected-gradient diagnostic under Adam.',
                ha='center',va='center',fontsize=9)
            fig.text(.975,.012,f'{page}/3',ha='right',fontsize=9)
            fig.subplots_adjust(left=.105,right=.98,bottom=.064,top=.912,hspace=.22,wspace=.21)
            pdf.savefig(fig);plt.close(fig);pages.append(dict(page=page,arms=[left,right],loss_limits=loss_ranges))
    assert previous=={p.name:sha(p) for p in (HERE/'figures').glob('0[12]-*.pdf')}
    audit=dict(status='numeric_checks_passed',pages=pages,source_sha256=sha(source),generator_sha256=sha(Path(__file__)),
               pdf_sha256=sha(target),plotted_values_checked=checked,identical_baselines=True,
               independent_loss_axes=True,other_scales_shared_across_pages=True,original_figures_unchanged=previous)
    (HERE/'figure03_verification.json').write_text(json.dumps(audit,indent=2)+'\n',newline='\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
