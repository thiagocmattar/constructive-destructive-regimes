"""Figures and report for the exploratory Adam state map."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
COLORS = {.5:'#0072B2', 1.:'#777777', 2.:'#D55E00'}
MARKERS = {.5:'o', 1.:'s', 2.:'^'}
SHORT = {
    'raw_cosine':'Raw gradient cosine', 'projected_change':'Projected gradient change',
    'input_cosine':'Input / prior moment cosine', 'incoming_ratio':'Incoming / retained magnitude',
    'moment_cosine':'Moment cosine', 'adaptive_cosine':'Adaptive direction cosine',
    'update_cosine':'Actual update cosine', 'descent_cosine':'Gradient / descent cosine',
    'log_raw_norm':'Log gradient norm', 'log_moment_norm':'Log moment norm',
    'log_adaptive_gain':'Log adaptive / moment norm', 'log_step_norm':'Log step norm',
    'first_order_decrease':'First-order sampled decrease', 'clipping_fraction':'Clipping fraction',
    'update_retention':'Actual displacement / path length', 'raw_retention':'Raw accumulation ratio',
    'moment_retention':'Moment accumulation ratio', 'adaptive_retention':'Adaptive accumulation ratio',
    'raw_growth':'Raw norm growth', 'moment_growth':'Moment norm growth',
    'probe_gain':'Fixed-probe loss benefit', 'log_previous_moment_norm':'Log prior moment norm'}


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    data=json.loads((HERE/'figure_data.json').read_text())['rows']
    result=json.loads((HERE/'results.json').read_text())
    features=json.loads((HERE/'features.json').read_text())
    experiments={(e['cutoff'],e['target']):e for e in result['experiments']}
    e=experiments[16,'gain_final']; initial=experiments[0,'gain_final']
    figdir=HERE/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':8,'axes.titlesize':9,'axes.labelsize':8,
        'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    metadata={'Creator':'Analysis031 / 02_plot.py','Subject':'Exploratory Adam state map; three seed clusters'}
    paths=[]
    def output(fig,stem):
        path=figdir/(stem+'.pdf');fig.savefig(path,metadata=metadata);plt.close(fig);paths.append(path)

    fig,axes=plt.subplots(1,2,figsize=(7.1,3.25),layout='constrained',sharex=True,sharey=True)
    norm=TwoSlopeNorm(vmin=-.06,vcenter=0,vmax=.06)
    for j,size in enumerate(['14M','70M']):
        ax=axes[j]
        for mult in [.5,1.,2.]:
            rows=[r for r in data if r['size']==size and r['multiplier']==mult]
            sc=ax.scatter([r['state16']['input_cosine'] for r in rows],
                [r['state16']['incoming_ratio'] for r in rows],
                c=[r['gain_final'] for r in rows],norm=norm,cmap='RdBu',
                marker=MARKERS[mult],s=55,edgecolors='.25',linewidths=.4,zorder=3)
        ax.axvline(0,color='.75',lw=.7);ax.grid(alpha=.15)
        ax.set_title(f'({chr(97+j)}) Pythia {size}',loc='left',pad=8)
        ax.set_xlabel('Incoming gradient / prior moment cosine')
    axes[0].set_ylabel('Incoming / retained moment magnitude')
    cb=fig.colorbar(sc,ax=axes,shrink=.85,pad=.025)
    cb.set_label('Final validation benefit vs. held LR\n(positive = lower loss)')
    fig.legend(handles=[Line2D([],[],marker=MARKERS[m],color='.4',ls='',label=f'{m:g}x LR') for m in [.5,1.,2.]],
               loc='outside upper center',ncol=3,frameon=False)
    output(fig,'optimizer_state_map')

    fig,axes=plt.subplots(4,2,figsize=(7.1,7.8),layout='constrained',sharex=True,sharey='row')
    metrics=[('probe_gain','Fixed-probe loss benefit\n(held minus candidate)'),
             ('input_cosine','Input / prior moment cosine\n(candidate minus held)'),
             ('incoming_ratio','Incoming / retained magnitude\n(candidate minus held)'),
             ('update_retention','Displacement / path length\n(candidate minus held)')]
    for j,size in enumerate(['14M','70M']):
        axes[0,j].set_title(f'({chr(97+j)}) Pythia {size}',loc='left',pad=8)
        for i,(key,label) in enumerate(metrics):
            ax=axes[i,j];ax.axhline(0,color='.5',lw=.7);ax.axvspan(1,16,color='.9',zorder=-2)
            for mult in [.5,2.]:
                rows=[r for r in data if r['size']==size and r['multiplier']==mult]
                ys=[]
                for row in rows:
                    if key=='probe_gain':
                        x=sorted(map(int,row[key]));y=[row[key][str(t)] for t in x]
                    else:x=row['series'][key]['x'];y=row['series'][key]['y']
                    ys.append(y);ax.plot(x,y,color=COLORS[mult],alpha=.25,lw=.8)
                ax.plot(x,np.mean(ys,axis=0),color=COLORS[mult],lw=1.5,
                        marker='o' if key=='probe_gain' else None,ms=3)
            ax.grid(alpha=.12);ax.set_xlim(0,128)
            if j==0:ax.set_ylabel(label)
            if i==3:ax.set_xlabel('Updates after fork');ax.set_xticks([0,16,32,64,96,128])
    fig.legend(handles=[Line2D([],[],color=COLORS[m],lw=1.5,label=f'{m:g}x LR') for m in [.5,2.]],
               loc='outside upper center',ncol=2,frameon=False)
    output(fig,'paired_optimizer_response')

    keys=list(features['post_labels']);prekeys=list(features['pre_labels'])
    scores=[e['candidates'][k] for k in keys]+[initial['candidates'][k] for k in prekeys]
    labels=[SHORT[k] for k in keys]+['Initial: '+SHORT[k].lower() for k in prekeys]
    arr=np.array([s['fold_mae_improvements_percent']+[s['mae_improvement_percent']] for s in scores])
    fig,ax=plt.subplots(figsize=(7.1,7.5),layout='constrained')
    im=ax.imshow(arr,aspect='auto',cmap='RdBu',vmin=-60,vmax=60)
    ax.set_xticks(range(4),['Seed 1234','Seed 1235','Seed 1236','Pooled'])
    ax.xaxis.tick_top();ax.tick_params(axis='x',length=0,pad=7)
    ax.set_yticks(range(len(labels)),labels);ax.tick_params(axis='y',length=0)
    ax.axhline(len(keys)-.5,color='white',lw=3)
    for (i,j),v in np.ndenumerate(arr):
        ax.text(j,i,f'{v:+.0f}',ha='center',va='center',fontsize=7,color='white' if abs(v)>36 else '#222222')
    cb=fig.colorbar(im,ax=ax,pad=.025,shrink=.7,extend='both')
    cb.set_label('Prediction MAE reduction (%)\npositive = better than model/LR baseline')
    output(fig,'predictive_value_all_features')

    candidate=e['candidates']['update_retention']; baseline=e['baseline']
    actual=np.array(candidate['target'])-baseline['predictions']
    predicted=np.array(candidate['predictions'])-baseline['predictions']
    rows=[r for r in features['rows'] if r['cutoff']==16]
    fig,axes=plt.subplots(1,3,figsize=(7.1,2.8),layout='constrained')
    ax=axes[0]
    for mult in [.5,2.]:
        for size,marker in [('14M','o'),('70M','^')]:
            ix=[i for i,r in enumerate(rows) if r['multiplier']==mult and r['size']==size]
            ax.scatter(actual[ix],predicted[ix],c=COLORS[mult],marker=marker,s=25)
    low=min(actual.min(),predicted.min())-.002;high=max(actual.max(),predicted.max())+.002
    ax.plot([low,high],[low,high],color='.6',lw=.7)
    ax.axhline(0,color='.85',lw=.5);ax.axvline(0,color='.85',lw=.5)
    ax.set_xlabel('Observed benefit residual');ax.set_ylabel('Predicted benefit residual')
    ax.set_title('(a) Prediction beyond model / LR',loc='left',pad=9)
    ax=axes[1]
    for name,v,color in [('Model / LR',baseline,'#777777'),('+ Actual retention',candidate,'#0072B2')]:
        ax.plot([1,2,3],[f['mae'] for f in v['folds']],marker='o',ms=3,color=color,label=name)
    ax.set_xticks([1,2,3],['1234','1235','1236']);ax.set_xlabel('Held-out seed');ax.set_ylabel('Prediction MAE')
    ax.set_title('(b) Each seed held out',loc='left',pad=9);ax.legend(frameon=False,fontsize=7,loc='upper right')
    ax.set_ylim(.003,.0105)
    ax=axes[2]
    for k,label,color in [('update_retention','Actual','#0072B2'),('adaptive_retention','Adaptive','#D55E00'),('moment_retention','Moment','#009E73')]:
        ax.plot([8,16,32,64],[experiments[t,'gain_final']['candidates'][k]['mae_improvement_percent'] for t in [8,16,32,64]],
            marker='o',ms=3,color=color,label=label)
    ax.axhline(0,color='.5',lw=.7);ax.set_xticks([8,16,32,64]);ax.set_xlabel('Last observed update')
    ax.set_ylabel('MAE reduction (%)');ax.set_title('(c) Timing sensitivity',loc='left',pad=9)
    ax.legend(frameon=False,fontsize=7)
    for ax in axes:ax.grid(alpha=.12)
    output(fig,'retention_prediction_audit')

    lines=['# Adam state map: exploratory results','',
        'The recorded state contains a candidate signal for final paired loss benefit: early adaptive-direction and actual-update accumulation ratios. It is not a validated scheduling rule. The effect is sensitive to the observation window and target, and no tested pre-intervention summary improves primary prediction.','',
        '## Target and comparison','',
        'Benefit = held-LR validation loss at128 minus candidate validation loss at128. Positive is better. Both models, all three seeds, and both altered rates are retained:12 paired outcomes, six starting states, three seed clusters. The baseline predicts the same model/multiplier cell mean from the other two seeds. Each additional signal gets one slope, fitted after training-only cell centering. The test seed contributes neither cell means nor fitted coefficients.','',
        'The early response uses updates2-16, except accumulation ratios which use the full prefix1-16. Features are candidate-minus-held values. Initial features use the common pre-update gradient and stored moment, multiplied by log2(LR multiplier). These are predictions after observing both candidate and held responses, not a single-trajectory controller.','',
        '## Main comparison','',
        '| Predictor beyond model/LR | MAE | RMSE | MAE reduction | Per-seed reductions |',
        '| --- | ---: | ---: | ---: | --- |',
        f"| Model/LR only | {baseline['mae']:.6f} | {baseline['rmse']:.6f} | 0% | reference |"]
    for k in keys:
        v=e['candidates'][k]
        folds=', '.join(f'{x:+.1f}%' for x in v['fold_mae_improvements_percent'])
        lines.append(f"| {SHORT[k]} | {v['mae']:.6f} | {v['rmse']:.6f} | {v['mae_improvement_percent']:+.1f}% | {folds} |")
    lines+=['','All candidate features are shown; this is an outcome-known exploratory scan. Adaptive and actual retention are near-duplicate signals, not independent replications. Raw and moment accumulation ratios were added after the first scan as specificity controls.','',
        'For actual retention, all three fitted slopes are negative. Conditional on model and LR multiplier, a more negative retention difference predicts greater final benefit in these data. This is not evidence that cancellation should be maximized: the altered LR and evolving trajectory jointly affect geometry and loss.','',
        '| Subgroup | Model/LR MAE | + Actual retention MAE |',
        '| --- | ---: | ---: |']
    for size in ['14M','70M']:
        for mult in [.5,2.]:
            ix=[i for i,row in enumerate(rows) if row['size']==size and row['multiplier']==mult]
            base_err=np.mean(abs(np.array(baseline['predictions'])[ix]-np.array(candidate['target'])[ix]))
            cand_err=np.mean(abs(np.array(candidate['predictions'])[ix]-np.array(candidate['target'])[ix]))
            lines.append(f'| {size}, {mult:g}x | {base_err:.6f} | {cand_err:.6f} |')
    lines+=['','These subgroup errors reveal whether the pooled result is concentrated in a particular intervention; three outcomes per subgroup cannot establish a subgroup-specific prediction law.','',
        '## Initial-state prediction','',
        '| Common-state feature (action interaction) | MAE | MAE reduction |','| --- | ---: | ---: |']
    for k in prekeys:
        v=initial['candidates'][k];lines.append(f"| {SHORT[k]} | {v['mae']:.6f} | {v['mae_improvement_percent']:+.1f}% |")
    lines+=['','## Timing and target sensitivity','',
        '| Observation cutoff | Actual retention | Adaptive retention | Moment retention |',
        '| ---: | ---: | ---: | ---: |']
    for t in [8,16,32,64]:
        cs=experiments[t,'gain_final']['candidates']
        lines.append('| '+str(t)+' | '+' | '.join(f"{cs[k]['mae_improvement_percent']:+.1f}%" for k in ['update_retention','adaptive_retention','moment_retention'])+' |')
    lines+=['','Entries are pooled MAE reductions versus the same model/LR baseline. The main retention result does not persist across these windows. A separate gradient/descent-angle candidate improves final-benefit MAE at32, illustrating the multiplicity of this search rather than an independently confirmed discovery.','',
        'To separate future improvement from an advantage already realized at16, the secondary target is benefit128 minus benefit16. It is validation-based; the predictor uses the distinct fixed training probe. The target itself contains the early loss term, so predictability by early loss need not identify a mechanism.','',
        '| Target | Model/LR MAE | + Early probe | + Actual retention | + Both |','| --- | ---: | ---: | ---: | ---: |']
    for target,label in [('gain_final','Final benefit'),('gain_after16','Additional benefit16-128')]:
        ex=experiments[16,target]
        lines.append(f"| {label} | {ex['baseline']['mae']:.6f} | {ex['candidates']['probe_gain']['mae']:.6f} | {ex['candidates']['update_retention']['mae']:.6f} | {ex['conditional_on_probe']['update_retention']['mae']:.6f} |")
    lines+=['','The two-feature combination helps pooled future-benefit prediction relative to the probe comparator, but that conditional improvement is not consistent in every held-seed fold. Thus the standalone final-benefit result does not establish a robust predictor of subsequent recovery. All fold outcomes, coefficients, and candidate combinations are in results.json.','',
        '## What the state maps show','',
        'The incoming-gradient/prior-moment cosine and their relative contribution magnitudes separate the LR interventions. Color encodes final benefit, not a stability class. This separation alone is largely an action effect: it cannot establish incremental predictive value. Temporal panels show all three seed traces and their mean, with the first16 updates shaded. Fixed-probe loss is measured only at0/16/32/64/128; connecting lines are guides, not dense evaluations.','',
        'All doubled-rate branches are worse than holding LR on validation at16, yet better at128. The state maps describe this adverse early response and later recovery; the shared best action prevents evidence of state-dependent LR selection.','',
        '## Limits and next scientific use','',
        'Only three seed clusters and one fork stage are available; folds overlap in training data and their scores are not independent replications. A single-slope fit has four training states, and feature/window selection uses the same known cohort. No p-values, population confidence intervals, curvature interpretation, or large-model extrapolation are claimed. The scalar map does not observe the full second-moment spectrum.','',
        'The defensible next hypothesis is that early adaptive-update accumulation contains information about the magnitude of the eventual LR response, beyond action and size. Freeze the16-update feature and target on a new cohort spanning several training stages and both beneficial and harmful LR increases. A policy claim also needs a practical information budget and equal-compute comparison.','',
        '## Artifacts','',
        '- [Optimizer state map](figures/optimizer_state_map.pdf)',
        '- [Paired temporal responses](figures/paired_optimizer_response.pdf)',
        '- [All primary feature scores](figures/predictive_value_all_features.pdf)',
        '- [Retention prediction and sensitivity](figures/retention_prediction_audit.pdf)',
        '- [Figure sources, reductions, captions, and caveats](observations/001-state-map.md)',
        '', 'Verification and visual review are recorded separately in verification.json and figure_checks.json. No new training or rental.']
    report='\n'.join(lines)+'\n'
    for old,new in [('at128','at 128'),('at16','at 16'),('at32','at 32'),('updates2-16','updates 2-16'),
        ('prefix1-16','prefix 1-16'),('retained:12','retained: 12'),('ratio16','ratio 16'),
        ('benefit128','benefit at 128'),('benefit16','benefit at 16'),('from16','from 16'),
        ('to128','to 128'),('benefit16-128','benefit 16-128'),('first16','first 16'),
        ('at0/16/32/64/128','at 0/16/32/64/128'),('the16-update','the 16-update')]:
        report=report.replace(old,new)
    (HERE/'REPORT.md').write_text(report,encoding='utf-8')
    manifest=dict(generator_sha256=sha(__file__),figures=[dict(path=p.relative_to(HERE).as_posix(),sha256=sha(p)) for p in paths],
                  counts=dict(state_map_points=18,prediction_score_cells=int(arr.size),held_seed_prediction_points=12),visual_review='pending')
    (HERE/'figure_checks.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
