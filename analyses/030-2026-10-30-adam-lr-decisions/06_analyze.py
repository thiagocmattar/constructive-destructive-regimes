"""Prespecified H024 paired effects, seed-held-out predictions and vector figures."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
COLORS={.5:'#0072B2',1.:'#009E73',2.:'#D55E00'}

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def ridge_predict(x,y,train,test,alpha=10.):
    mean=x[train].mean(0);sd=x[train].std(0);sd[sd==0]=1
    a=(x[train]-mean)/sd;b=(x[test]-mean)/sd
    center=y[train].mean()
    beta=np.linalg.solve(a.T@a+alpha*np.eye(a.shape[1]),a.T@(y[train]-center))
    return b@beta+center

def main():
    cohort=json.loads((HERE/'cohort.json').read_text())
    data=[];sources=[]
    for row in cohort:
        output=ROOT/'runs'/row['run']/'artifacts/attempt-01/full-pass'
        receipt=ROOT/'runs'/row['run']/'prelaunch/attempt-01-retrieval.json'
        assert json.loads(receipt.read_text())['scientific_verification']['verified']
        events=output/'events.jsonl';summary=json.loads((output/'summary.json').read_text())
        updates=[json.loads(l) for l in events.read_text().splitlines() if json.loads(l)['event']=='update']
        c=json.loads((output/'resolved_config.json').read_text())
        early=[r for r in updates if 2<=r['local_update']<=16]
        probe={p['local_update']:p['loss'] for p in summary['probes']}
        val={p['local_update']:p['loss'] for p in summary['validations']}
        raw_norm=np.array([r['gradient_norm'] for r in early])
        features=[probe[0],probe[16]-probe[0],np.log(updates[0]['learning_rate']),np.log(c['model']['parameter_count']),
            np.mean([r['adjacent_cosine'] for r in early]),
            np.mean([r['momentum']['adjacent_cosine'] for r in early]),
            np.mean([r['adaptive']['adjacent_cosine'] for r in early]),
            np.mean([r['incoming_previous_moment_cosine'] for r in early]),
            np.log(raw_norm[-4:].mean()/raw_norm[:4].mean())]
        data.append(dict(**row,features=[float(v) for v in features],validation=val,probe=probe,updates=updates,final_loss=val[128],later_loss_change=val[128]-val[16]))
        sources.append(dict(run=row['run'],events_sha256=sha(events),summary_sha256=sha(output/'summary.json'),inventory_sha256=sha(output/'inventory.json')))
    # Matched-state invariance: the first measured gradient precedes the intervention.
    for block in sorted({r['block'] for r in data}):
        rows=[r for r in data if r['block']==block]
        for r in rows[1:]:
            assert np.isclose(r['validation'][0],rows[0]['validation'][0],rtol=1e-5,atol=1e-4)
            assert np.isclose(r['probe'][0],rows[0]['probe'][0],rtol=1e-5,atol=1e-4)
            for key in ['loss','gradient_norm','adjacent_cosine']:
                assert np.isclose(r['updates'][0][key],rows[0]['updates'][0][key],rtol=1e-5,atol=1e-7),(block,key)
            assert [u['batch_indices_sha256'] for u in r['updates']]==[u['batch_indices_sha256'] for u in rows[0]['updates']]
    x=np.array([r['features'] for r in data]);y=np.array([r['final_loss'] for r in data]);seeds=np.array([r['seed'] for r in data])
    # A state-independent tuning baseline must compete with diagnostic models.
    # Its multiplier is chosen using only the training seeds of each fold.
    fixed=np.full(len(data),np.nan)
    for seed in sorted(set(seeds)):
        train=np.flatnonzero(seeds!=seed);test=np.flatnonzero(seeds==seed)
        scores={mult:np.mean([y[i] for i in train if data[i]['multiplier']==mult]) for mult in [.5,1.,2.]}
        for i in test:fixed[i]=scores[data[i]['multiplier']]
    predictions={'Fixed LR':fixed}
    for label,count in [('Loss baseline',4),('Plus raw cosine',5),('Joint diagnostics',9)]:
        pred=np.full(len(data),np.nan)
        for seed in sorted(set(seeds)):
            train=np.flatnonzero(seeds!=seed);test=np.flatnonzero(seeds==seed)
            pred[test]=ridge_predict(x[:,:count],y,train,test)
        predictions[label]=pred
    effects=[];decisions=[]
    blocks=sorted({r['block'] for r in data})
    for block in blocks:
        indices=[i for i,r in enumerate(data) if r['block']==block]
        hold=next(i for i in indices if data[i]['multiplier']==1)
        best=min(indices,key=lambda i:y[i]);d=dict(block=block,best_multiplier=data[best]['multiplier'],held_regret=float(y[hold]-y[best]),choices={})
        for label,pred in predictions.items():
            chosen=min(indices,key=lambda i:pred[i])
            d['choices'][label]=dict(multiplier=data[chosen]['multiplier'],regret=float(y[chosen]-y[best]))
        decisions.append(d)
        for i in indices:
            effects.append(dict(block=block,size=data[i]['size'],seed=data[i]['seed'],multiplier=data[i]['multiplier'],final_loss=float(y[i]),loss_difference_from_held=float(y[i]-y[hold]),early_raw_cosine=float(x[i,4]),early_moment_cosine=float(x[i,5]),early_adaptive_cosine=float(x[i,6]),early_input_moment_cosine=float(x[i,7])))
    figdir=HERE/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(7.1,4.4),layout='constrained')
    for si,size in enumerate(['14M','70M']):
        for sj,seed in enumerate([1234,1235,1236]):
            ax=axes[si,sj]
            for r in data:
                if r['size']!=size or r['seed']!=seed:continue
                ax.plot([0,16,128],[r['validation'][k] for k in [0,16,128]],color=COLORS[r['multiplier']],marker='o',ms=3,lw=1.3,label=f"{r['multiplier']:g} × LR")
            ax.set_title(f'({chr(97+3*si+sj)}) Pythia{size}, seed {seed}',loc='left',pad=7)
            ax.set_xlabel('Updates after fork');ax.set_ylabel('Validation loss');ax.grid(alpha=.15)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=3,frameon=False)
    fig.savefig(figdir/'matched_lr_validation.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(7.1,2.5),layout='constrained')
    for j,(key,label) in enumerate([('early_raw_cosine','Raw gradients'),('early_moment_cosine','Momentum'),('early_adaptive_cosine','Adaptive direction')]):
        ax=axes[j]
        for mult in [.5,2.]:
            rows=[r for r in effects if r['multiplier']==mult]
            ax.scatter([r[key] for r in rows],[r['loss_difference_from_held'] for r in rows],s=25,color=COLORS[mult],label=f'{mult:g} × LR')
        ax.axhline(0,color='.5',lw=.7);ax.set_title(f'({chr(97+j)}) {label}',loc='left',pad=7)
        ax.set_xlabel('Mean cosine, updates 2–16');ax.set_ylabel('Final loss minus held LR');ax.grid(alpha=.15)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='outside upper center',ncol=2,frameon=False)
    fig.savefig(figdir/'early_geometry_later_loss.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.1,2.8),layout='constrained')
    methods=['Hold LR',*predictions];palette=['#777777','#CC79A7','#0072B2','#D55E00','#009E73'];positions=np.arange(len(decisions))
    for k,(method,color) in enumerate(zip(methods,palette)):
        values=[d['held_regret'] if method=='Hold LR' else d['choices'][method]['regret'] for d in decisions]
        ax.bar(positions+(k-2)*.16,values,width=.15,color=color,label=method)
    ax.set_xticks(positions,[d['block'].replace('-','\n') for d in decisions]);ax.set_ylabel('Loss above best observed branch');ax.grid(axis='y',alpha=.15)
    fig.legend(loc='outside upper center',ncol=5,frameon=False)
    fig.savefig(figdir/'held_seed_decision_regret.pdf');plt.close(fig)
    result=dict(status='completed pilot; exploratory prediction',generator_sha256=sha(Path(__file__)),sources=sources,effects=effects,decisions=decisions,feature_contract='README.md; no outcome-driven tuning',features=x.tolist(),feature_names=['initial_probe_loss','probe_loss_change_0_16','log_lr','log_parameters','raw_cosine','moment_cosine','adaptive_cosine','input_previous_moment_cosine','log_raw_norm_change'],ridge_penalty=10,predictions={k:v.tolist() for k,v in predictions.items()},run_order=[r['run'] for r in data],best_multiplier_varies=len({d['best_multiplier'] for d in decisions})>1,seed_clusters=3)
    (HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    lines=['# Matched Adam learning-rate pilot','', 'All18 branches completed and passed local artifact/scientific acceptance. Three seed clusters, two sizes and one fork stage support a pilot, not a validated scheduler.','', '| State | Lower LR minus held loss | Higher LR minus held loss | Best multiplier |','| --- | ---: | ---: | ---: |']
    for d in decisions:
        rs={r['multiplier']:r for r in effects if r['block']==d['block']}
        lines.append(f"| {d['block']} | {rs[.5]['loss_difference_from_held']:+.6f} | {rs[2.]['loss_difference_from_held']:+.6f} | {d['best_multiplier']:g} |")
    lines+=['','Negative differences mean the LR intervention improved final full-validation loss at matched tokens. Shrink factors were matched; this is not native fixed-decay schedule tuning.','',f"Best LR multiplier varies across blocks: {result['best_multiplier_varies']}. A common winning multiplier does not demonstrate state-dependent decision value.",'','Exploratory leave-one-seed-out mean regret (six correlated size/state blocks):']
    for method in methods:
        values=[d['held_regret'] if method=='Hold LR' else d['choices'][method]['regret'] for d in decisions]
        lines.append(f'- {method}: {np.mean(values):.6f}.')
    lines+=['','No significance test or controller-validation claim is made. All predictions, including failures to select the best arm, are retained in results.json. Figures require visual review before publication.']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    observations=HERE/'observations';observations.mkdir(exist_ok=True)
    observation='''# H024 matched LR figures

Sources: all18 original events and summary files, individually hash verified and
listed in results.json. Generator: ../06_analyze.py; execution and input contracts
are in ../README.md. GPU/scientific acceptance precedes these reductions.

## Matched LR validation

File: ../figures/matched_lr_validation.pdf. Caption: Full-validation loss at the
common fork and16/128 subsequent updates for three constant LR multipliers,
holding per-update AdamW shrinkage constant. Columns show seeds, rows show sizes.
Reduction: original token-weighted evaluations on all500 validation documents,
693,329 targets including the packed tail. No smoothing or omitted evaluations.
Uncertainty: show each of the three seed clusters; curves are not confidence bands.
Caveat: one training stage and short continuations; no billion-parameter claim.

## Early geometry and later loss

File: ../figures/early_geometry_later_loss.pdf. Caption: Mean adjacent cosine over
local updates2--16 versus final loss difference from the held-LR branch, separately
for raw gradients, moments and adaptive directions. Negative Y means the rate
change improves final validation loss. Each panel has12 altered-rate arms.
Reduction: arithmetic mean of original pointwise cosines; matched validation
difference at128. Update1 is excluded because its gradient precedes intervention.
Uncertainty: six matched states with three seed clusters; points are not independent
replications across sizes/arms. Caveat: associations do not establish mediation or
a useful decision threshold; momentum/adaptive cosines are not actual displacement.

## Exploratory decision regret

File: ../figures/held_seed_decision_regret.pdf. Caption: Achieved loss above the
best observed arm per starting state when holding LR or selecting from fixed ridge
models using early loss, additional raw cosine, or joint optimizer diagnostics.
Reduction: fixed penalty10, training-fold standardization, leave-one-seed-out;
both sizes and all arms of each held seed are excluded during fitting. No tuning.
Coverage:18 arms, six states, three folds. Uncertainty: per-state outcomes shown,
no significance intervals. Caveat: the observed best arm is a finite-comparison
reference, not an oracle beyond these rates; this pilot does not validate a scheduler.
The fixed-LR comparator chooses its multiplier only on each fold's training seeds;
it tests whether diagnostics add value beyond ordinary state-independent LR tuning.
If every state favors the same rate, conditional decision value is unestablished.

Visual review: pending. Automated text-bound checks and preview generation are
recorded separately in figure_checks.json; no figure is yet approved for the paper.
'''
    (observations/'001-matched-lr-decisions.md').write_text(observation,encoding='utf-8')
    print(json.dumps(dict(completed=18,best_multiplier_varies=result['best_multiplier_varies'],decisions=decisions),indent=2))

if __name__=='__main__':main()
