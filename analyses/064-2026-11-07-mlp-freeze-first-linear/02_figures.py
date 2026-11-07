"""Matched Adam/frozen-first-layer comparison: full follow-up and architectural trace."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

HERE=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
    'font.size':10,'mathtext.fontset':'stix','pdf.fonttype':42})
COLORS={'control':'#2166ac','freeze':'#d95f02'}
LABELS={'control':'Continue Adam','freeze':'Freeze linear 1 (weights + bias)'}


def style(ax):
    ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.2,lw=.6)


def main():
    source=HERE/'comparison.json';r=json.loads(source.read_text());assert r['status']=='verified'
    (HERE/'figures').mkdir(exist_ok=True);target=HERE/'figures/01-freeze-first-linear-cifar10-mlp.pdf'
    t=np.array(r['states'])
    with PdfPages(target) as pdf:
        fig,axes=plt.subplots(7,2,figsize=(12,15),sharex='col')
        for j,(a,b,title) in enumerate([(448,800,'Full follow-up: 448-800'),(450,500,'Event window: 450-500')]):
            mask=(t>=a)&(t<=b);tt=t[mask]
            for arm in ['control','freeze']:
                c=r['curves'][arm]
                values=[c['loss'],c['eta_kappa_hat'],c['gradient_rho'],c['raw_gradient_norm'],c['active_gradient_norm'],
                        c['layers']['output']['rho'],c['layers']['output']['change_rms']]
                for i,y in enumerate(values):
                    axes[i,j].plot(tt,np.array(y,dtype=float)[mask],color=COLORS[arm],label=LABELS[arm],lw=1.2)
            for i in [2,5]:
                axes[i,j].axhline(0,color='.4',lw=.6);axes[i,j].set_ylim(-1.05,1.05)
            for i in [3,4,6]:axes[i,j].set_ylim(bottom=0)
            axes[1,j].axhline(1,color='.5',ls='--',lw=.7)
            for ax in axes[:,j]:
                style(ax);ax.set_xlim(a,b)
                if j==0:ax.axvspan(450,500,color='.5',alpha=.09,lw=0)
                else:ax.axvline(474,color='.45',ls=':',lw=.8)
            axes[0,j].set_title(title,fontsize=13,pad=12)
            axes[-1,j].set_xlabel('Completed optimizer updates $t$')
        for ax,label in zip(axes[:,0],['(a) Full training loss',r'(b) $\eta\widehat{\kappa}_t$',
                                     '(c) Raw-gradient cosine',r'(d) Full raw-gradient norm $\|g_t\|$',
                                     '(e) Updated-layer gradient norm','(f) Output-change cosine','(g) Output displacement RMS']):
            ax.set_ylabel(label)
        axes[0,1].legend(frameon=False,fontsize=9,loc='upper right')
        fig.suptitle('Freeze the first linear layer; keep Adam and the remaining learning rates unchanged',fontsize=15,y=.985)
        fig.text(.5,.963,'Fork at state 448; all 5,000 fixed training examples; LR = 0.001. No sample averaging or smoothing.',ha='center',fontsize=10)
        fig.text(.5,.020,r'$\eta\widehat{\kappa}_t=1-\langle g_t,g_{t-1}\rangle/\|g_{t-1}\|^2$: projected-gradient diagnostic under Adam, not a curvature threshold.'+'\n'
                 'Full gradients include all 656,810 parameters; row (e) uses only the 42,210 still updated. Output changes are outgoing.',ha='center',fontsize=10)
        fig.subplots_adjust(left=.12,right=.98,bottom=.070,top=.93,hspace=.22,wspace=.21)
        pdf.savefig(fig);plt.close(fig)

        names=['preactivation1','hidden1','preactivation2','hidden2','output']
        rowlabels=['Linear 1 ($Z_1$)','Tanh 1 ($H_1$)','Linear 2 ($Z_2$)','Tanh 2 ($H_2$)','Linear 3 ($F$)']
        tt=np.array(r['primary_states']);control=r['event']['control'];freeze=r['event']['freeze']
        fig,axes=plt.subplots(5,3,figsize=(13,11),sharex='col')
        for i,(name,label) in enumerate(zip(names,rowlabels)):
            for arm in ['control','freeze']:
                curve=control['curves'][name] if arm=='control' else freeze[name]
                yy=np.array([np.nan if x is None else x for x in curve['rho']])
                axes[i,0].plot(tt,yy,color=COLORS[arm],lw=1.15,label=LABELS[arm])
                axes[i,1].plot(tt,curve['change_rms'],color=COLORS[arm],lw=1.15)
                if curve['spectrum'] is not None:
                    cumulative=np.cumsum(curve['spectrum'])
                    axes[i,2].plot(np.arange(1,16),cumulative[:15],color=COLORS[arm],lw=1.15)
            axes[i,0].set_ylim(-1.05,1.05);axes[i,0].axhline(0,color='.5',lw=.6)
            axes[i,0].set_ylabel(label,rotation=0,ha='right',va='center',labelpad=14)
            axes[i,1].set_ylim(bottom=0)
            axes[i,2].set_ylim(0,1.03);axes[i,2].set_xlim(1,15);axes[i,2].axhline(.95,color='.5',ls=':',lw=.6)
            for ax in axes[i,:2]:ax.set_xlim(450,500);ax.axvline(474,color='.6',ls=':',lw=.7)
            if i<2:
                axes[i,0].text(.5,.80,'Frozen: zero motion\ncosine undefined',transform=axes[i,0].transAxes,ha='center',color=COLORS['freeze'],fontsize=9)
                axes[i,2].text(.5,.25,'Frozen: zero energy\nnormalized spectrum undefined',transform=axes[i,2].transAxes,ha='center',color=COLORS['freeze'],fontsize=9)
            for ax in axes[i]:style(ax)
        for ax,label in zip(axes[0],['Consecutive representation-change cosine','Displacement coordinate RMS','Cumulative singular energy']):
            ax.set_title(label,fontsize=11,pad=10)
        axes[-1,0].set_xlabel('Outgoing update $t$');axes[-1,1].set_xlabel('Outgoing update $t$');axes[-1,2].set_xlabel('Number of fixed SVD modes')
        handles,labels=axes[2,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.57,.948),ncol=2,frameon=False)
        fig.suptitle('Does the downstream network keep oscillating when the first representation is fixed?',fontsize=15,y=.985)
        fig.text(.5,.027,'Each stream has its own uncentered SVD fitted on 450-500, retaining all sample-feature coordinates.\n'
                 'Concentrated singular energy remains after damping; it does not by itself diagnose oscillation or a stationary local axis.',ha='center',fontsize=10)
        fig.subplots_adjust(left=.17,right=.98,bottom=.085,top=.89,hspace=.25,wspace=.30)
        pdf.savefig(fig);plt.close(fig)
    audit=dict(status='numeric_checks_passed',pages=2,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        pdf_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        full_window=[448,800],event_window=[450,500],zero_streams_rendered_as_undefined=True,
        overview_rows=['loss','eta_kappa_hat','raw_gradient_rho','raw_gradient_norm','active_gradient_norm','output_rho','output_rms'])
    (HERE/'figure_verification.json').write_text(json.dumps(audit,indent=2)+'\n',newline='\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
