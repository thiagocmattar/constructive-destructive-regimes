"""New baseline-referenced H048 PDFs; preserve all earlier publication figures."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

HERE=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':10,'mathtext.fontset':'stix','pdf.fonttype':42})
BLUE='#2166ac';GREEN='#00876c';GOLD='#b8860b';RED='#b2182b';PURPLE='#8856a7';GRAY='#555555'
EARLY={'mlp':'2','resnet':'conv1','transformer':'transformer_encoder.layers.0.self_attn'}
TITLES={'mlp':'MLP / CIFAR-10','resnet':'ResNet-32 / CIFAR-10','transformer':'Transformer / WikiText-2'}
LABELS={'discard_entry':'Discard large coordinates (q90)','cap_entry':'Cap large coordinates (q90)',
 'functional_entry':'Local displacement cap (q75)','functional_trunk':'Distributed displacement cap (q75)','functional_trunk_q50':'Distributed displacement cap (q50)',
 'functional_entry_q50':'First-layer displacement cap (q50)','global_half':'Uniform half increments',
 'block_shock':'Block reversal + norm surge','global_shock':'Global reversal + norm surge','block_secant':'Block secant damping',
 'embedding_x2':'Embedding LR 2x','embedding_x4':'Embedding LR 4x','embedding_x4_functional_trunk_q50':'Embedding LR 4x + distributed cap',
 'axis_remove':'Remove reversing component','axis_half':'Halve reversing component','axis_cap':'Cap reversing component',
 'bounded_axis_remove':'Remove + native norm bound','bounded_axis_half':'Halve + native norm bound','bounded_axis_cap':'Cap + native norm bound',
 'output_budget_q50':'Output movement budget (q50)','representation_budget_q50':'All tracked movements (q50)','output_budget_q75':'Output movement budget (q75)',
 'output_budget_slow':'Output budget + slow recovery','representation_budget_slow':'All tracked budgets + slow recovery',
 'probe_loss_guard':'Probe-loss backtracking'}


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def load():
    workers=[];sources={}
    for filename in ['transfer_comparison.json']+['transfer_comparison_run'+str(n)+'.json' for n in range(272,286)]:
        p=HERE/filename;r=json.loads(p.read_text());assert r['status']=='verified';assert r['generator_sha256']==sha(HERE/'17_reduce_transfer.py')
        sources[filename]=sha(p);workers+=r['workers']
    return workers,sources


def find(workers,run,arm):return next(w for w in workers if w['run_id']==run and arm in w['curves'])


def values(w,name,embedding=False,projection=False,probe_guard=False):
    c=w['curves'][name];r=c['representations'];early='1' if projection else EARLY[w['case']]
    if embedding:
        return [c['loss'],c['eta_kappa_hat'],c['rho'],c['raw_gradient_norm'],r['encoder']['rho'],r[early]['rho'],r['output']['rho'],
            r['encoder']['change_rms'],r[early]['change_rms'],r['output']['change_rms'],c['retained_norm_ratio'],c['embedding_lr']]
    return [c['loss'],c['eta_kappa_hat'],c['rho'],c['raw_gradient_norm'],r[early]['rho'],r[early]['change_rms'],
            r['output']['rho'],r['output']['change_rms'],c['retained_norm_ratio'],c['mean_target_fraction_modified']]+([c['probe_loss']] if probe_guard else [])


def page(columns,title,subtitle,embedding=False,confirmation=False,logloss=False,projection=False,probe_guard=False):
    n=12 if embedding else 11 if probe_guard else 10;fig,axes=plt.subplots(n,len(columns),figsize=(5.1*len(columns)+1.1,20 if embedding or probe_guard else 18),sharex='col',squeeze=False)
    labels=['Full training loss',r'$\eta\widehat\kappa_t$ (symlog)','Raw-gradient cosine',r'Full $\|g_t\|$ (log)']
    labels+=['Embedding-change cosine','Attention-change cosine','Output-change cosine','Embedding RMS (log)','Attention RMS (log)','Output RMS (log)','Update norm / proposal','Embedding LR'] if embedding else ['Early change cosine','Early RMS (log)','Output-change cosine','Output RMS (log)','Update norm / proposal','Mean fraction modified']
    if probe_guard:labels+=['Fixed-probe loss']
    checks=0;meta=[]
    for col,spec in enumerate(columns):
        w=spec['worker'];t=np.array(w['states']);mask=t<=spec.get('end',int(t[-1]));t=t[mask]
        series=[(w,'control','Unchanged Adam',BLUE,'-')]+spec['curves']
        for source,name,label,color,style in series:
            np.testing.assert_array_equal(np.array(source['states'])[mask],t)
            # Supplementary paired controls must have the identical untreated trajectory.
            if source is not w:np.testing.assert_array_equal(np.asarray(source['curves']['control']['loss'])[mask],np.asarray(w['curves']['control']['loss'])[mask])
            for i,y in enumerate(values(source,name,embedding,projection,probe_guard)):
                y=np.array(y,dtype=float)[mask]
                if i not in ([2,4,5,6] if embedding else [2,4,6]):assert np.isfinite(y).all()
                else:assert not np.isinf(y).any()  # Undefined zero-denominator cosines remain gaps.
                line,=axes[i,col].plot(t,y,color=color,ls=style,lw=1.05,label=label)
                np.testing.assert_array_equal(line.get_ydata(),y);checks+=len(t)
        axes[0,col].set_title(spec['title'],fontsize=11,pad=9)
        axes[0,col].legend(loc='best',frameon=False,fontsize=8.5)
        if logloss:axes[0,col].set_yscale('log',nonpositive='mask')
        else:axes[0,col].set_ylim(bottom=0)
        axes[1,col].set_yscale('symlog',linthresh=1);axes[1,col].axhline(1,color='.6',lw=.6,ls=':')
        cosine_rows=[2,4,5,6] if embedding else [2,4,6]
        log_rows=[3,7,8,9] if embedding else [3,5,7]
        for i in cosine_rows:axes[i,col].set_ylim(-1.05,1.05);axes[i,col].axhline(0,color='.6',lw=.6)
        for i in log_rows:axes[i,col].set_yscale('log',nonpositive='mask')
        ratio=10 if embedding else 8;axes[ratio,col].axhline(1,color='.6',lw=.6,ls=':')
        if not embedding:axes[9,col].set_ylim(-.03,1.03)
        for i,ax in enumerate(axes[:,col]):
            ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.2,lw=.6);ax.set_xlim(t[0],t[-1])
            if confirmation:ax.axvspan(800,t[-1],color='#f0e1bc',alpha=.3,zorder=-10);ax.axvline(800,color='.4',ls=':',lw=.8)
        axes[-1,col].set_xlabel('Completed optimizer updates $t$')
        meta.append(dict(title=spec['title'],run=w['run_id'],arms=[name for _,name,_,_,_ in series],window=[int(t[0]),int(t[-1])]))
    for i,label in enumerate(labels):axes[i,0].set_ylabel(f'({chr(97+i)}) '+label)
    fig.suptitle(title,fontsize=16,y=.987);fig.text(.5,.965,subtitle,ha='center',fontsize=10)
    note='Full objectives; fixed diagnostic probes; outgoing representation changes; no smoothing. Each column uses its own unchanged baseline.'
    if embedding:note+='\nEmbedding and decoder are untied. Base LR 0.003; only embedding LR is multiplied. Update-norm ratio is relative to that native proposal.\nColumns 1/2 use RTX 4090; column 3 uses RTX 5090. Their control trajectories differ; each effect is compared within its own worker.'
    elif projection:note+='\nEarly is the first affine projection. Only its negative component along the preceding actual affine change is modified.'
    else:note+='\nEarly: MLP first tanh, ResNet conv1 output, Transformer first attention output. Loss and magnitude scales are independent by column.'
    if confirmation:note+='\nShaded region 801-1152 was unseen at initial policy selection. The scalar-per-layer norm control is retrospective, not an online policy.'
    elif projection:note+='\nLast row is 1 when that component is modified. This uses one preceding direction, not a future-fitted SVD basis; all other increments stay native.'
    elif not embedding:note+='\nThe modified-fraction row averages targeted coordinate/neuron/block fractions; this is not a whole-model parameter fraction.'
    if probe_guard:note+='\nProbe-loss acceptance covers all MLP examples, one ResNet normalization group, or 64 Transformer sequences; full-objective behavior is measured separately.'
    note+='\nUndefined cosines and zero values on logarithmic axes appear as gaps.'
    fig.text(.5,.024,note,ha='center',va='center',fontsize=9)
    fig.subplots_adjust(left=.084 if len(columns)==3 else .12,right=.985,bottom=.065,top=.925,hspace=.23,wspace=.24)
    return fig,dict(columns=meta,plotted_values_checked=checks)


def depth_page(columns,title,subtitle):
    stages={'mlp':[('1','Linear 1'),('2','Tanh 1'),('3','Linear 2'),('4','Tanh 2'),('output','Output')],
        'resnet':[(n,label) for n,label in [('conv1','Conv 1'),('layer1','Stage 1'),('layer2','Stage 2'),('layer3','Stage 3'),('avgpool','Pooling'),('output','Output')]],
        'transformer':[('encoder','Embedding'),('transformer_encoder.layers.0.self_attn','Attention 1'),('transformer_encoder.layers.0','Block 1'),('transformer_encoder.layers.1.self_attn','Attention 2'),('transformer_encoder.layers.1','Block 2'),('output','Output')]}
    prepared=[];maximum=1.
    for spec in columns:
        w=spec['worker'];t=np.asarray(w['states']);mask=t<=spec.get('end',int(t[-1]));names=stages[w['case']]
        baseline=w['curves']['control'];treatment=w['curves']['functional_trunk_q50']
        rms=[np.asarray([c['representations'][n]['change_rms'] for n,_ in names],dtype=float)[:,mask] for c in [baseline,treatment]]
        denominator=rms[0].max(axis=1);assert np.all(denominator>0)
        relative=[v/denominator[:,None] for v in rms];maximum=max(maximum,max(float(v.max()) for v in relative))
        cosine=[np.asarray([c['representations'][n]['rho'] for n,_ in names],dtype=float)[:,mask] for c in [baseline,treatment]]
        prepared.append(dict(worker=w,t=t[mask],mask=mask,names=names,cosine=cosine,relative=relative,denominator=denominator))
    fig,axes=plt.subplots(5,3,figsize=(16.4,13),sharex='col',gridspec_kw={'height_ratios':[1.2,1.8,1.8,1.8,1.8]})
    checks=0;records=[]
    for col,data in enumerate(prepared):
        w=data['worker'];t=data['t'];names=data['names'];extent=[t[0]-.5,t[-1]+.5,len(names)-.5,-.5]
        for name,label,color in [('control','Unchanged Adam',BLUE),('functional_trunk_q50','Distributed cap (q50)',GREEN)]:
            values=np.asarray(w['curves'][name]['loss'])[data['mask']];line,=axes[0,col].plot(t,values,label=label,color=color,lw=1.1)
            np.testing.assert_array_equal(line.get_ydata(),values);checks+=len(t)
        axes[0,col].set_title(TITLES[w['case']],fontsize=12);axes[0,col].set_ylim(bottom=0);axes[0,col].legend(frameon=False,fontsize=9)
        for row,values in enumerate(data['cosine']+data['relative'],start=1):
            im=axes[row,col].imshow(values,aspect='auto',interpolation='none',extent=extent,cmap='RdBu_r' if row<3 else 'viridis',vmin=-1 if row<3 else 0,vmax=1 if row<3 else maximum)
            np.testing.assert_array_equal(np.asarray(im.get_array().data),values);checks+=values.size
            axes[row,col].set_yticks(range(len(names)),[label for _,label in names],fontsize=9)
            if row<3:cosine_image=im
            else:rms_image=im
        for ax in axes[:,col]:ax.set_xlim(t[0],t[-1]);ax.spines[['top','right']].set_visible(False)
        axes[-1,col].set_xlabel('Completed optimizer updates $t$')
        records.append(dict(run=w['run_id'],case=w['case'],arm='functional_trunk_q50',window=[int(t[0]),int(t[-1])],stages=[n for n,_ in names],baseline_stage_rms_maxima=data['denominator'].tolist()))
    for row,label in enumerate(['(a) Full training loss','(b) Baseline change cosine','(c) Cap change cosine','(d) Baseline relative RMS','(e) Cap relative RMS']):axes[row,0].set_ylabel(label,labelpad=13)
    fig.suptitle(title,fontsize=16,y=.985);fig.text(.5,.952,subtitle,ha='center',fontsize=10)
    fig.subplots_adjust(left=.10,right=.90,bottom=.13,top=.91,hspace=.27,wspace=.35)
    fig.colorbar(cosine_image,cax=fig.add_axes([.93,.48,.012,.30]),label='Consecutive-change cosine')
    fig.colorbar(rms_image,cax=fig.add_axes([.93,.14,.012,.30]),label='RMS / baseline stage maximum')
    fig.text(.5,.035,'Captured stages preserve all probe sample/token/spatial coordinates. No smoothing or new SVD decomposition.\nBoth RMS maps use each stage\'s baseline maximum in the displayed window; treatment amplitudes are not normalized away.\nSame-worker unchanged Adam in every column. These maps show intervention effects on motion, not a unique source of instability.',ha='center',fontsize=9)
    return fig,dict(kind='captured_stage_heatmaps',columns=records,relative_rms_color_maximum=maximum,plotted_values_checked=checks)


def main():
    workers,sources=load();original={p.name:sha(p) for p in (HERE/'figures').glob('0[12345]-*.pdf')};outputs=[]
    def column(run,arm,title=None,end=None,extras=None):
        w=find(workers,run,arm);curves=[(w,arm,LABELS.get(arm,arm),GREEN,'-')]+(extras or [])
        return dict(worker=w,title=title or LABELS[arm],curves=curves,**({} if end is None else dict(end=end)))
    plans=[]
    selective=[]
    for case,run in [('mlp',269),('resnet',270),('transformer',271)]:
        selective.append(dict(columns=[column(run,a) for a in ['discard_entry','cap_entry','functional_entry']],title=TITLES[case]+': selective entry updates',
            subtitle='Thresholds fixed from eight preceding untreated updates; native gradients and Adam moments retained.'))
    plans.append(('06-selective-transfer-diagnostics.pdf',selective))
    matched=find(workers,275,'matched_trunk_q50')
    distributed=[dict(columns=[column(run,'functional_trunk_q50',TITLES[case],800 if case=='mlp' else None) for case,run in [('mlp',272),('resnet',273),('transformer',274)]],
        title='Distributed local-displacement caps across architectures',subtitle='Same prefix-q50 rule; each architecture calibrates its own affine modules from eight untreated updates.'),
        dict(render=depth_page,columns=[column(run,'functional_trunk_q50',end=800 if case=='mlp' else None) for case,run in [('mlp',272),('resnet',273),('transformer',274)]],
        title='Distributed caps: motion across captured representation stages',subtitle='The same q50 local rule; compare reversal patterns and amplitudes with unchanged Adam.'),
        dict(columns=[column(272,'functional_trunk_q50',extras=[(matched,'matched_trunk_q50','Matched per-layer norms',GRAY,'--')]),column(272,'functional_entry_q50'),column(272,'global_half')],
        title='MLP: later-window confirmation and matched norms',subtitle='Continue the selected policies through 1152; unchanged Adam remains the baseline in every column.',confirmation=True)]
    seeds=[]
    for seed,run,slow_run in [(0,272,279),(1,282,282),(2,283,283)]:
        w=find(workers,run,'functional_trunk_q50');slow=find(workers,slow_run,'output_budget_slow')
        seeds.append(column(run,'functional_trunk_q50','MLP initialization '+str(seed),extras=[(slow,'output_budget_slow','Output budget + slow recovery',RED,'-'),(w,'global_half','Uniform half increments',GRAY,'--')]))
    distributed.append(dict(columns=seeds,title='MLP: replication across three initializations',subtitle='Same rules and calibration length; initialization 0 selected the candidates, initializations 1/2 received no policy tuning.'))
    distributed.append(dict(columns=[column(run,'output_budget_slow',TITLES[case]) for case,run in [('mlp',279),('resnet',280),('transformer',281)]],title='Slow recovery of an output-movement budget',subtitle='Transfer the same q50 rule and 1% recovery limit; calibration thresholds differ by architecture.'))
    distributed.append(dict(columns=[column(run,'probe_loss_guard',TITLES[case]) for case,run in [('mlp',284),('resnet',285),('transformer',281)]],title='A loss-aware comparator: probe-loss backtracking',subtitle='Accept the first strictly decreasing dyadic candidate on the fixed probe; otherwise take a zero increment. Adam moments/clocks advance.',probe_guard=True))
    plans.append(('07-distributed-displacement-and-controls.pdf',distributed))
    other=[dict(columns=[column(269,a) for a in ['block_shock','global_shock','block_secant']],title='MLP: transient diagnostic-triggered damping fails',
        subtitle='Loss uses log axes to retain the untreated trajectory beside the much larger intervention excursions.',logloss=True),
        dict(columns=[column(276,a,extras=[(find(workers,277,'bounded_'+a),'bounded_'+a,'With native parameter-norm bound',RED,'--')]) for a in ['axis_remove','axis_half','axis_cap']],title='MLP: past-axis damping, with and without a parameter-norm bound',
        subtitle='Loss uses log axes. Fewer reversals and better endpoints can coexist with much larger transient loss.',logloss=True,projection=True),
        dict(columns=[column(278,a,extras=([(find(workers,279,b),b,'With slow recovery',RED,'--')] if b else [])) for a,b in [('output_budget_q50','output_budget_slow'),('representation_budget_q50','representation_budget_slow'),('output_budget_q75',None)]],title='MLP: movement budgets and the timing of release',
        subtitle='Backtrack the complete Adam increment until actual fixed-probe displacement RMS meets its prefix budget.',logloss=True),
        dict(columns=[column(270,a) for a in ['block_shock','global_half','functional_trunk']],title='ResNet: diagnostic damping, uniform scaling, and distributed caps',
        subtitle='Compare a blockwise reversal/norm rule, half increments, and prefix-q75 caps across all 31 convolutions.',logloss=True),
        dict(columns=[column(271,'embedding_x2'),column(271,'embedding_x4'),column(274,'embedding_x4_functional_trunk_q50',extras=[(find(workers,274,'functional_trunk_q50'),'functional_trunk_q50','Distributed cap only',GRAY,'--')])],title='Transformer: larger embedding learning rates',
        subtitle='All supervised tokens retained; fixed 64-sequence probe. The combined intervention also caps both QKV and feed-forward affines.',embedding=True)]
    plans.append(('08-adaptive-and-embedding-tests.pdf',other))
    for filename,pages in plans:
        target=HERE/'figures'/filename;records=[]
        with PdfPages(target) as pdf:
            for spec in pages:
                spec=dict(spec);render=spec.pop('render',page)
                fig,record=render(**spec);pdf.savefig(fig);plt.close(fig);records.append(record)
        outputs.append(dict(file=filename,sha256=sha(target),pages=records))
    assert original=={p.name:sha(p) for p in (HERE/'figures').glob('0[12345]-*.pdf')}
    (HERE/'transfer_figure_audit.json').write_text(json.dumps(dict(generator_sha256=sha(Path(__file__)),sources=sources,outputs=outputs,earlier_figures_preserved=original),indent=2)+'\n')
    print(json.dumps(dict(files=[q['file'] for q in outputs],pages=sum(len(q['pages']) for q in outputs))))


if __name__=='__main__':main()
