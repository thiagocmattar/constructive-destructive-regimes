"""Absolute intervention comparison and q25 motion from frozen H051 arrays."""
from pathlib import Path
import hashlib, json, importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, NullLocator

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1];DRAFT=BASE/'draft';DATA=BASE/'evidence/05-adam/c-intervention/data'
spec=importlib.util.spec_from_file_location('previous_presentation',BASE/'work/intervention_manuscript_20261114/01_prepare.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
array_sha=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def main():
    scratch=HERE/'scratch';scratch.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','font.size':8,
        'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
        'legend.fontsize':8,'axes.linewidth':.6,'pdf.fonttype':42,'axes.spines.top':False,
        'axes.spines.right':False,'path.simplify':False})
    report=json.loads((DATA/'matched-results.json').read_text())
    rows={(r['case'],r['arm']):r for r in report['summary']}
    curves=np.load(DATA/'matched-curves.npz')
    v=lambda c,a,m:curves[f'{c}__{a}__{m}']
    manifest={'sources':{p.name:sha(p) for p in [DATA/'matched-results.json',DATA/'matched-curves.npz']},
              'points':[],'maps':[],'lines':[],'table':[],'motion':{},'figures':{}}
    # A small linear interval retains exact zero; all other y values are log-scaled.
    fig,axes=plt.subplots(1,3,figsize=(7.16,3.1))
    thresholds={'mlp':.001,'resnet':.01,'transformer':.01}
    for ax,case in zip(axes,old.CASES):
        values=[rows[case,a] for a in old.ARMS]
        xs=[r['mean_loss'] for r in values];ys=[r['positive_variation'] for r in values]
        span=max(xs)-min(xs)
        ax.set_xlim(min(xs)-.18*span,max(xs)+.22*span)
        ax.set_yscale('symlog',linthresh=thresholds[case],linscale=.65)
        ax.set_ylim(-thresholds[case]*.22,max(ys)*2.5)
        ax.axhline(rows[case,'control']['positive_variation'],color='.7',lw=.55)
        ax.axvline(rows[case,'control']['mean_loss'],color='.7',lw=.55)
        labels=[]
        for arm in old.ARMS:
            row=rows[case,arm];x,y=row['mean_loss'],row['positive_variation']
            ax.scatter(x,y,s=23,c=old.COLORS[arm],marker='x' if arm=='control' else old.MARKERS[row['family']],zorder=4)
            # Labels use offsets only; the underlying measured points are untouched.
            label='Adam' if arm=='control' else old.POINT_LABELS[arm]
            labels.append(ax.annotate(label,(x,y),xytext=(4,5),textcoords='offset points',fontsize=7,
                color='.25' if arm=='control' else old.LABEL_COLORS[arm],
                bbox=dict(fc='white',ec='none',alpha=.85,pad=.1),
                arrowprops=dict(arrowstyle='-',lw=.4,color='.5',shrinkA=1,shrinkB=3)))
            manifest['points'].append(dict(case=case,arm=arm,mean_loss=x,positive_variation=y))
        ax.set_title(old.TITLES[case],pad=7)
        ax.set_xlabel(r'Mean training loss $\bar L$')
        ax.tick_params(length=2.5,pad=2)
        ax.yaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax._comparison_labels=labels
    axes[0].set_ylabel(r'Upward loss variation $\mathcal{P}$')
    handles=[Line2D([],[],marker=old.MARKERS[f],color=old.COLORS[a],ls='',label=l,markersize=4)
       for f,a,l in [('lr','lr_0p5','LR reduction'),('beta2','beta2_0p9',r'$\beta_2$ reduction'),
                    ('clipping','clip_1p0','Gradient clipping'),('cap','cap_q50','Displacement cap')]]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.005),ncol=4,frameon=False,columnspacing=1.2)
    fig.subplots_adjust(left=.085,right=.99,bottom=.22,top=.89,wspace=.35)
    # Deterministic label placement: greedy screen-space candidates around each point.
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for ax,case in zip(axes,old.CASES):
        occupied=[]
        for label in ax._comparison_labels:
            best=None
            candidates=[(4,5),(4,-9),(-22,6),(-30,-9),(5,17),(-25,19),(5,-22),(-32,-23),
                        (18,28),(-32,32),(18,-35),(-45,-36),(25,5),(-48,5)]
            for dx,dy in candidates:
                label.set_position((dx,dy));box=label.get_bbox_patch()
                label.update_bbox_position_size(renderer)
                b=box.get_window_extent(renderer).expanded(1.07,1.12)
                overlap=sum(max(0,min(b.x1,o.x1)-max(b.x0,o.x0))*max(0,min(b.y1,o.y1)-max(b.y0,o.y0)) for o in occupied)
                outside=max(ax.bbox.x0-b.x0,0)+max(b.x1-ax.bbox.x1,0)+max(ax.bbox.y0-b.y0,0)+max(b.y1-ax.bbox.y1,0)
                score=overlap+100*outside+.015*(dx*dx+dy*dy)
                if best is None or score<best[0]:best=(score,(dx,dy),b)
            label.set_position(best[1]);occupied.append(best[2])
        overrides={'mlp':{'control':(7,-15)},'resnet':{'control':(7,12),'cap_q50':(-29,-25)},'transformer':{}}
        for arm,offset in overrides[case].items():
            ax._comparison_labels[old.ARMS.index(arm)].set_position(offset)
    name='24-main-intervention-tradeoff.pdf'
    fig.savefig(DRAFT/'figures'/name,metadata={'CreationDate':None});fig.savefig(scratch/'tradeoff.png',dpi=190);plt.close(fig)
    manifest['figures'][name]=sha(DRAFT/'figures'/name)

    fig=plt.figure(figsize=(7.16,5.0));axes=np.empty((5,3),object)
    bottoms=[.775,.585,.425,.265,.105];heights=[.16,.14,.14,.14,.14]
    den={c:v(c,'control','stage_change_rms')[:,1:].max(axis=1) for c in old.CASES}
    rel={(c,a):v(c,a,'stage_change_rms')[:,1:]/den[c][:,None] for c in old.CASES for a in ['control','cap_q25']}
    maximum=max(1.,max(float(x.max()) for x in rel.values()))
    for col,c in enumerate(old.CASES):
        state=v(c,'control','state');x=state[1:]
        labels=[l.replace('Stage','Group') for l in old.STAGE_LABELS[c]]
        for row,y in enumerate(bottoms):
            ax=axes[row,col]=fig.add_axes([.13+col*.282,y,.21,heights[row]])
            ax.set_xlim(state[0],state[-1]);ax.tick_params(length=2,pad=2)
            ax.set_xticks([int(state[0]),int((state[0]+state[-1])/2),int(state[-1])])
            if row<4:ax.tick_params(axis='x',labelbottom=False)
        for a in ['control','cap_q25']:
            y=v(c,a,'loss');line,=axes[0,col].plot(state,y,color=old.COLORS[a],lw=.8,ls='-' if a=='control' else (0,(4,1.5)))
            np.testing.assert_array_equal(line.get_ydata(),y)
            manifest['lines'].append(dict(case=c,arm=a,loss_sha256=array_sha(y),state_sha256=array_sha(state)))
        axes[0,col].set_yscale('log');axes[0,col].set_title(old.TITLES[c],fontsize=9,pad=7)
        axes[0,col].yaxis.set_minor_locator(NullLocator())
        mats=[(a,'cosine',v(c,a,'stage_rho')[:,1:]) for a in ['control','cap_q25']]
        mats +=[(a,'relative_rms',rel[c,a]) for a in ['control','cap_q25']]
        for row,(a,metric,z) in enumerate(mats,1):
            im=axes[row,col].imshow(z,aspect='auto',interpolation='none',extent=[x[0]-.5,x[-1]+.5,len(labels)-.5,-.5],
                cmap='RdBu_r' if row<3 else 'viridis',vmin=-1 if row<3 else 0,vmax=1 if row<3 else maximum)
            np.testing.assert_array_equal(im.get_array().data,z)
            axes[row,col].set_yticks(range(len(labels)),labels=labels);axes[row,col].tick_params(axis='y',length=0,labelsize=7)
            manifest['maps'].append(dict(case=c,arm=a,metric=metric,shape=list(z.shape),sha256=array_sha(z),denominator=den[c].tolist()))
            if row<3:cosine_image=im
            else:rms_image=im
        axes[-1,col].set_xlabel('Training step $t$',labelpad=3)
        base=v(c,'control','stage_change_rms')[-1,1:];cap=v(c,'cap_q25','stage_change_rms')[-1,1:]
        manifest['motion'][c]=dict(max_output_rms_reduction_percent=100*(1-cap.max()/base.max()),
            squared_movement_change_percent=100*((cap@cap)/(base@base)-1),
            baseline_negative_cosine_percent=100*np.mean(v(c,'control','stage_rho')[-1,1:]<0),
            q25_negative_cosine_percent=100*np.mean(v(c,'cap_q25','stage_rho')[-1,1:]<0))
    for row,label in enumerate(['Training loss','Adam: cosine','q25: cosine','Adam: rel. RMS','q25: rel. RMS']):
        axes[row,0].set_ylabel(label,labelpad=5,fontsize=8)
    fig.legend(handles=[Line2D([],[],color=old.COLORS['control'],label='Unchanged Adam'),
        Line2D([],[],color=old.COLORS['cap_q25'],ls=(0,(4,1.5)),label='Cap q25')],
        loc='center',bbox_to_anchor=(.53,.745),ncol=2,frameon=False,fontsize=8)
    fig.colorbar(cosine_image,cax=fig.add_axes([.925,.425,.012,.30]),ticks=[-1,0,1])
    fig.colorbar(rms_image,cax=fig.add_axes([.925,.105,.012,.30]))
    name='25-main-q25-functional-motion.pdf'
    fig.savefig(DRAFT/'figures'/name,metadata={'CreationDate':None});fig.savefig(scratch/'motion.png',dpi=190);plt.close(fig)
    manifest['figures'][name]=sha(DRAFT/'figures'/name)
    # Milliloss units permit one decimal without collapsing small nonzero losses.
    metrics=['mean_loss','positive_variation','final_loss']
    names=['Adam','LR reduction','LR reduction','LR reduction',r'$\beta_2$ reduction',r'$\beta_2$ reduction',r'$\beta_2$ reduction','Gradient clipping','Gradient clipping','Displacement cap','Displacement cap','Displacement cap']
    values=['---',r'$0.75\eta$',r'$0.5\eta$',r'$0.1\eta$','0.95','0.9','0.5','0.5','1','q25','q50','q75']
    tex=[r'\begin{table*}[!t]',r'\caption{Absolute matched training outcomes. All numeric outcomes are multiplied by $10^3$; lower is better. Bold marks each column minimum at full precision. Windows are 512--800 (MLP), 256--576 (ResNet), and 544--928 (Transformer), with one initialization per model.}',r'\label{tab:vc-outcomes}',r'\centering\fontsize{8.5}{10.5}\selectfont',r'\setlength{\tabcolsep}{4pt}',r'\renewcommand{\arraystretch}{1.13}',r'\begin{tabular}{llrrrrrrrrr}',r'\toprule',r' & & \multicolumn{3}{c}{CIFAR-10 / MLP} & \multicolumn{3}{c}{CIFAR-10 / ResNet-32} & \multicolumn{3}{c}{WikiText-2 / Transformer} \\',r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}',r'Intervention & Parameter value & $\bar L$ & $\mathcal P$ & $L_{t_f}$ & $\bar L$ & $\mathcal P$ & $L_{t_f}$ & $\bar L$ & $\mathcal P$ & $L_{t_f}$ \\',r'\midrule']
    for i,a in enumerate(old.ARMS):
        if i in [1,4,7,9]:tex.append(r'\addlinespace[3pt]')
        cells=[];plain=[]
        for c in old.CASES:
            for m in metrics:
                value=rows[c,a][m]
                # Independent reduction from complete loss states.
                loss=v(c,a,'loss');check={'mean_loss':loss[1:].mean(),'positive_variation':np.maximum(np.diff(loss),0).sum(),'final_loss':loss[-1]}[m]
                np.testing.assert_allclose(value,check,rtol=2e-12,atol=2e-12)
                val=f'{1000*value:.1f}';plain.append(val)
                cells.append(r'\textbf{'+val+'}' if value==min(rows[c,b][m] for b in old.ARMS) else val)
        tex.append(' & '.join([names[i],values[i]]+cells)+r' \\')
        manifest['table'].append(dict(arm=a,display_values=plain))
    tex +=[r'\bottomrule',r'\end{tabular}',r'\end{table*}','']
    (DRAFT/'tables/matched-main-outcomes.tex').write_text('\n'.join(tex),encoding='utf8',newline='\n')
    for a in ['control','cap_q25']:
        rise=v('mlp',a,'mse_delta_loss')>1e-8
        linear=v('mlp',a,'mse_linear');quadratic=v('mlp',a,'mse_quadratic')
        corrective=rise & (linear<0)
        assert np.all(quadratic[corrective]>-linear[corrective])
        np.testing.assert_allclose(v('mlp',a,'mse_delta_loss'),linear+quadratic,atol=1e-15)
        manifest.setdefault('mse_rises',{})[a]=dict(total=int(rise.sum()),corrective=int(corrective.sum()))
    (HERE/'presentation-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'motion':manifest['motion'],'mse_rises':manifest['mse_rises']}))

if __name__=='__main__':main()
