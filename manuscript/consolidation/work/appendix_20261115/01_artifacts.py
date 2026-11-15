"""Compact reviewer appendix from frozen matched-intervention reductions."""
from pathlib import Path
import csv, hashlib, json, importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];D=BASE/'draft'
DATA=BASE/'evidence/05-adam/c-intervention/data';OUT=D/'figures'
spec=importlib.util.spec_from_file_location('style',BASE/'work/intervention_manuscript_20261114/01_prepare.py')
style=importlib.util.module_from_spec(spec);spec.loader.exec_module(style)
CASES=style.CASES;ARMS=style.ARMS
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
arrsha=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
CURVES=np.load(DATA/'matched-curves.npz')
v=lambda c,a,m:CURVES[f'{c}__{a}__{m}']
FAMILIES=[('LR reduction',['lr_0p75','lr_0p5','lr_0p1']),
 (r'$\beta_2$ reduction',['beta2_0p95','beta2_0p9','beta2_0p5']),
 ('Gradient clipping',['clip_0p5','clip_1p0']),('Displacement cap',['cap_q25','cap_q50','cap_q75'])]
SHORT={'control':'Adam','lr_0p75':r'$0.75\eta$','lr_0p5':r'$0.5\eta$','lr_0p1':r'$0.1\eta$',
 'beta2_0p95':'0.95','beta2_0p9':'0.9','beta2_0p5':'0.5','clip_0p5':'0.5','clip_1p0':'1.0',
 'cap_q25':'q25','cap_q50':'q50','cap_q75':'q75'}
RECORD={'sources':{p.name:sha(p) for p in [DATA/'matched-curves.npz',DATA/'matched-results.json',DATA/'matched-costs.csv']},
        'lines':[],'maps':[],'figures':{},'last32':[]}

def line(ax,c,a,metric,x,y,**kw):
    artist,=ax.plot(x,y,**kw)
    np.testing.assert_array_equal(artist.get_xdata(),x);np.testing.assert_array_equal(artist.get_ydata(),y)
    RECORD['lines'].append(dict(case=c,arm=a,metric=metric,n=len(y),x_sha256=arrsha(x),y_sha256=arrsha(y)))

def save(fig,name):
    fig.savefig(OUT/name,metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(HERE/'scratch'/name.replace('.pdf','.png'),dpi=175)
    plt.close(fig);RECORD['figures'][name]=sha(OUT/name)

def main():
    (HERE/'scratch').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','font.size':8,
     'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
     'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
     'pdf.fonttype':42,'path.simplify':False})
    fig,axes=plt.subplots(4,3,figsize=(7.16,4.55),sharex='col')
    for col,c in enumerate(CASES):
        x=v(c,'control','state')[1:]
        all_y=np.concatenate([v(c,a,'actual_norm') for a in ARMS]);assert np.all(all_y>0)
        low=all_y.min()/1.1;high=all_y.max()*(all_y.max()/low)**.20
        for row,(family,arms) in enumerate(FAMILIES):
            ax=axes[row,col]
            for a in ['control']+arms:
                line(ax,c,a,'actual_norm',x,v(c,a,'actual_norm'),color=style.COLORS[a],lw=.8,label=SHORT[a])
            ax.set_yscale('log');ax.set_ylim(low,high);ax.set_xlim(x[0]-1,x[-1]);ax.yaxis.set_minor_locator(NullLocator())
            ax.legend(loc='upper right',frameon=False,ncol=4,fontsize=6,handlelength=1,columnspacing=.5,handletextpad=.3)
            ax.tick_params(length=2,pad=2);ax.set_xticks([int(x[0]-1),int((x[0]-1+x[-1])/2),int(x[-1])])
            if col==0:ax.set_ylabel(family+'\n'+r'$\|u_{t-1}\|$',fontsize=7.5)
            if row==0:ax.set_title(style.TITLES[c],pad=7)
            if row==3:ax.set_xlabel('Training step $t$')
    fig.subplots_adjust(left=.105,right=.99,top=.92,bottom=.09,hspace=.25,wspace=.35)
    save(fig,'a01-matched-applied-updates.pdf')

    fig,axes=plt.subplots(2,3,figsize=(7.16,2.65),sharex='col')
    for col,c in enumerate(CASES):
        x=v(c,'control','state')[1:]
        for row,(metric,arms) in enumerate([('clip_factor',FAMILIES[2][1]),('cap_active',FAMILIES[3][1])]):
            ax=axes[row,col]
            for a in arms:line(ax,c,a,metric,x,v(c,a,metric),color=style.COLORS[a],lw=.7,label=SHORT[a])
            ax.set_ylim(-.02,1.3);ax.set_yticks([0,.5,1]);ax.set_xlim(x[0]-1,x[-1]);ax.tick_params(length=2,pad=2)
            ax.set_xticks([int(x[0]-1),int((x[0]-1+x[-1])/2),int(x[-1])])
            ax.legend(loc='upper right',frameon=False,ncol=3,handlelength=1.3,columnspacing=.6)
            if row==0:ax.set_title(style.TITLES[c],pad=7)
            else:ax.set_xlabel('Training step $t$')
    axes[0,0].set_ylabel('Gradient factor');axes[1,0].set_ylabel('Fraction capped')
    fig.subplots_adjust(left=.105,right=.99,top=.87,bottom=.16,hspace=.22,wspace=.35)
    save(fig,'a02-matched-controller-activity.pdf')

    # One plate: only the selected treatment's heatmaps; Adam is a loss reference.
    selected=['beta2_0p5','lr_0p1','clip_1p0']
    titles=[r'$\beta_2=0.5$',r'LR $\times0.1$',r'Clip $1.0$']
    den={c:v(c,'control','stage_change_rms')[:,1:].max(axis=1) for c in CASES}
    relative={(c,a):v(c,a,'stage_change_rms')[:,1:]/den[c][:,None] for c in CASES for a in selected}
    vmax=max(1.,max(float(z.max()) for z in relative.values()))
    RECORD['selected_heatmaps']=dict(arms=selected,rms_color_maximum=vmax,normalization='same per-site unchanged-Adam post-fork maximum for every intervention')
    fig,axes=plt.subplots(9,3,figsize=(7.16,7.45),sharex='col',gridspec_kw={'height_ratios':[.85,1.1,1.1]*3})
    for col,c in enumerate(CASES):
        state=v(c,'control','state');x=state[1:];labels=[s.replace('Stage','Group') for s in style.STAGE_LABELS[c]]
        losses=np.concatenate([v(c,a,'loss') for a in selected+['control']]);lo,hi=losses.min()/1.12,losses.max()*1.15
        for group,(a,title) in enumerate(zip(selected,titles)):
            ax=axes[3*group,col]
            for arm in ['control',a]:line(ax,c,arm,'loss',state,v(c,arm,'loss'),color=style.COLORS[arm],lw=.75)
            ax.set_yscale('log');ax.set_ylim(lo,hi);ax.yaxis.set_minor_locator(NullLocator())
            if col==0:ax.set_ylabel(title+'\nLoss',fontsize=7)
            for offset,metric,z in [(1,'stage_rho',v(c,a,'stage_rho')[:,1:]),(2,'relative_rms',relative[c,a])]:
                ax=axes[3*group+offset,col]
                im=ax.imshow(z,aspect='auto',interpolation='none',extent=[x[0]-.5,x[-1]+.5,len(labels)-.5,-.5],
                    cmap='RdBu_r' if offset==1 else 'viridis',vmin=-1 if offset==1 else 0,vmax=1 if offset==1 else vmax)
                np.testing.assert_array_equal(im.get_array().data,z)
                ax.set_yticks(range(len(labels)),labels=labels,fontsize=6.5);ax.tick_params(axis='y',length=0,pad=2)
                if col==0:ax.set_ylabel('Cosine' if offset==1 else 'Rel. RMS',fontsize=7)
                RECORD['maps'].append(dict(case=c,arm=a,metric=metric,shape=list(z.shape),sha256=arrsha(z),denominator=den[c].tolist()))
                if offset==1:cosine=im
                else:rms=im
        for ax in axes[:,col]:
            ax.set_xlim(state[0],state[-1]);ax.set_xticks([int(state[0]),int((state[0]+state[-1])/2),int(state[-1])]);ax.tick_params(axis='x',length=2,pad=2)
        axes[0,col].set_title(style.TITLES[c],pad=7);axes[-1,col].set_xlabel('Training step $t$')
    fig.subplots_adjust(left=.13,right=.90,top=.96,bottom=.065,hspace=.21,wspace=.55)
    cax=fig.add_axes([.926,.56,.012,.30]);fig.colorbar(cosine,cax=cax,ticks=[-1,0,1]);cax.set_title('Cosine',fontsize=7,pad=5)
    cax=fig.add_axes([.926,.14,.012,.30]);fig.colorbar(rms,cax=cax);cax.set_title('Rel. RMS',fontsize=7,pad=5)
    save(fig,'a03-selected-intervention-motion.pdf')

    report=json.loads((DATA/'matched-results.json').read_text());rows={(r['case'],r['arm']):r for r in report['summary']}
    labels=['Adam',r'LR $\times.75$',r'LR $\times.5$',r'LR $\times.1$',r'$\beta_2=.95$',r'$\beta_2=.9$',r'$\beta_2=.5$','Clip .5','Clip 1','Cap q25','Cap q50','Cap q75']
    tex=[r'\begin{table}[!t]',r'\caption{Final-32-step mean training loss, multiplied by $10^3$. Every matched setting is retained; bold marks the full-precision minimum in each column.}',r'\label{tab:app-late}',r'\centering\small',r'\setlength{\tabcolsep}{6pt}',r'\begin{tabular}{lrrr}',r'\toprule',r'Intervention & MLP & ResNet & Transformer \\',r'\midrule']
    for a,label in zip(ARMS,labels):
        cells=[]
        for c in CASES:
            value=float(v(c,a,'loss')[-32:].mean());np.testing.assert_allclose(value,rows[c,a]['last32_loss'],rtol=2e-12)
            token=f'{1000*value:.1f}';cells.append(r'\textbf{'+token+'}' if value==min(rows[c,b]['last32_loss'] for b in ARMS) else token)
            RECORD['last32'].append(dict(case=c,arm=a,value=value,display=token))
        tex.append(' & '.join([label]+cells)+r' \\')
    tex +=[r'\bottomrule',r'\end{tabular}',r'\end{table}','']
    (D/'tables/appendix-last32.tex').write_text('\n'.join(tex),encoding='utf8',newline='\n')
    costs=list(csv.DictReader((DATA/'matched-costs.csv').open()))
    tex=[r'\begin{table}[!t]',r'\caption{Recorded q50 component times (ms). Reference evaluation includes the forward pass used to collect cap inputs. These are instrumented component costs, not matched time-to-loss benchmarks.}',r'\label{tab:app-cost}',r'\centering\small',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lrrr}',r'\toprule',r'Case / GPU & Gradient & Reference & Cap \\',r'\midrule']
    for c,name in zip(CASES,['MLP / GB10','ResNet / RTX 4090','Transformer / RTX 5090']):
        r=next(x for x in costs if x['case']==c and x['arm']=='cap_q50')
        nums=[1000*float(r[k])/int(r[n]) for k,n in [('gradient_seconds','full_gradient_passes'),('probe_seconds','full_gradient_passes'),('controller_seconds','updates')]]
        tex.append(' & '.join([name]+[f'{x:.1f}' for x in nums])+r' \\')
    tex +=[r'\bottomrule',r'\end{tabular}',r'\end{table}','']
    (D/'tables/appendix-costs.tex').write_text('\n'.join(tex),encoding='utf8',newline='\n')
    RECORD['generator_sha256']=sha(__file__)
    (HERE/'manifest.json').write_text(json.dumps(RECORD,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps({'figures':len(RECORD['figures']),'lines':len(RECORD['lines']),'heatmaps':len(RECORD['maps']),'last32_cells':len(RECORD['last32'])}))

if __name__=='__main__':main()
