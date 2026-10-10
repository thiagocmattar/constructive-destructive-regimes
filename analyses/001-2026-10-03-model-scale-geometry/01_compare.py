"""Reduce the verified scale cohort and generate publication figures."""
from pathlib import Path
import csv
import hashlib
import json
import math
import platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = [
 ('14M','002-2026-10-03-pythia14m-adamw-adaptive-geometry','attempt-02'),
 ('31M','003-2026-10-03-pythia31m-adamw-scale','attempt-01'),
 ('70M','004-2026-10-03-pythia70m-adamw-scale','attempt-01'),
]
COLORS = ['#2874A6','#D35400','#138D75']
STYLES = ['-','--','-.']

def sha(p):
    return hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def load_cohort():
    cohort = []
    reference = None
    for label, folder, attempt in SOURCES:
        run = ROOT/'runs'/folder
        base = run/'artifacts'/attempt/'full-pass'
        cfg, ident, env, summary = map(read,[base/'resolved_config.json',base/'input_identity.json',
                                             base/'environment.json',base/'summary.json'])
        assert read(base/'manifest.json')['status'] == 'completed'
        assert read(base/'verification.json')['verified']
        receipt_name = 'attempt2-retrieval.json' if label == '14M' else 'attempt1-retrieval.json'
        assert read(run/'prelaunch'/receipt_name)['verified']
        rows = [e for line in (base/'events.jsonl').read_text().splitlines()
                    if (e:=json.loads(line))['event'] == 'update']
        assert len(rows)==712 and summary['all_training_tokens_covered']
        assert rows[-1]['sequences']==311 and all(r['sequences']==1024 for r in rows[:-1])
        matched = {'dataset':cfg['data']['dataset'],'revision':cfg['data']['revision'],
            'tokenizer_revision':cfg['data']['tokenizer_revision'],
            'train_sha256':cfg['data']['train_sha256'],'validation_sha256':cfg['data']['validation_sha256'],
            'sequence_length':cfg['data']['sequence_length'],'seeds':cfg['seeds'],
            'training':cfg['training'],'measurements':cfg['measurements'],
            'validation':cfg['validation'],'checkpoints':cfg['checkpoints'],
            'order_sha256':ident['order_sha256']}
        if reference is None: reference = matched
        else: assert matched==reference, 'Cohort protocols are not matched'
        assert [r['learning_rate'] for r in rows] == [r['learning_rate'] for r in (cohort[0]['rows'] if cohort else rows)]
        record = {'label':label,'run_id':cfg['run_id'],'run_folder':folder,'attempt':attempt,
            'parameters':env['model']['parameter_count'],'architecture':env['model']['architecture'],
            'model_revision':cfg['model']['revision'],'initial_parameter_sha256':env['initial_parameter_sha256'],
            'rows':rows,'summary':summary,
            'sources':{name:{'path':(base/name).relative_to(ROOT).as_posix(),'sha256':sha(base/name)}
               for name in ['events.jsonl','summary.json','resolved_config.json','input_identity.json',
                            'environment.json','verification.json']}}
        cohort.append(record)
    return cohort, reference

def stats(rows, adaptive=False):
    x = [r['adaptive'] if adaptive else r for r in rows]
    cos = np.array([r['adjacent_cosine'] for r in x if r['adjacent_cosine'] is not None])
    norm = 'direction_norm' if adaptive else 'gradient_norm'
    b = 'cumulative_direction_norm' if adaptive else 'cumulative_gradient_norm'
    minimum = min(zip(rows,x), key=lambda pair: pair[1]['cumulative_ratio'])
    return {'median_adjacent_cosine':float(np.median(cos)), 'mean_adjacent_cosine':float(np.mean(cos)),
        'min_adjacent_cosine':float(cos.min()),'negative_cosines':int((cos<0).sum()),
        'defined_cosines':len(cos),'negative_fraction':float((cos<0).mean()),
        'A':x[-1]['cumulative_norm_sum'],'B':x[-1][b],'R':x[-1]['cumulative_ratio'],
        'median_norm':float(np.median([r[norm] for r in x])),
        'minimum_R':minimum[1]['cumulative_ratio'],'minimum_R_update':minimum[0]['update']}

def rolling(y, width=31):
    a = np.asarray(y, dtype=float)
    # Centered descriptive smoothing; finite observations only near the first undefined cosine.
    return np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else np.nan
        for i in range(len(a)) for v in [a[max(0,i-width//2):min(len(a),i+width//2+1)]]])

def save(fig, stem):
    fig.savefig(HERE/'figures'/f'{stem}.pdf', bbox_inches='tight')
    fig.savefig(HERE/'scratch'/f'{stem}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

def plots(cohort, aggregate):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig, axes = plt.subplots(3,2,figsize=(12,10),sharex=True,layout='constrained')
    for j, adaptive in enumerate([False,True]):
        axes[0,j].set_title('Raw gradient g (before clipping)' if not adaptive else 'Adaptive Adam direction d (before LR/decay)')
        for record,color,style in zip(cohort,COLORS,STYLES):
            rows = record['rows'][:-1]
            x = np.arange(1,len(rows)+1)
            v = [r['adaptive'] if adaptive else r for r in rows]
            cos = [r['adjacent_cosine'] if r['adjacent_cosine'] is not None else np.nan for r in v]
            axes[0,j].plot(x,cos,color=color,alpha=.13,lw=.55)
            axes[0,j].plot(x,rolling(cos),color=color,ls=style,lw=1.6,label=record['label'])
            axes[1,j].plot(x,[r['cumulative_ratio'] for r in v],color=color,ls=style,lw=1.7,label=record['label'])
            nk='direction_norm' if adaptive else 'gradient_norm'
            axes[2,j].plot(x,[r[nk] for r in v],color=color,ls=style,lw=.9,label=record['label'])
        axes[0,j].axhline(0,color='#777777',lw=.7,ls=':')
        axes[0,j].set_ylabel('Adjacent cosine\nfaint: each step; bold: 31-step median')
        lower = min(r['adaptive']['adjacent_cosine'] for c in cohort for r in c['rows'][1:-1])-.03 if adaptive else -1.01
        axes[0,j].set_ylim(lower,1.01)
        axes[1,j].set_ylabel('R = B / A')
        axes[1,j].set_ylim(0,1.025)
        axes[2,j].set_ylabel('Vector L2 norm (log scale)')
        axes[2,j].set_yscale('log')
        axes[2,j].set_xlabel('Optimizer update (matched data order)')
        axes[0,j].legend(loc='lower right',frameon=False)
        for ax in axes[:,j]: ax.grid(alpha=.18)
    fig.suptitle('Training geometry across Pythia sizes\nOne seed per size; updates 1–711; final partial batch omitted from display',fontsize=13)
    save(fig,'geometry_by_scale')

    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,layout='constrained')
    for j,adaptive in enumerate([False,True]):
        for record,color,style in zip(cohort,COLORS,STYLES):
            v=[r['adaptive'] if adaptive else r for r in record['rows'][:-1]]
            x=np.arange(1,len(v)+1)
            axes[0,j].plot(x,[r['cumulative_norm_sum'] for r in v],color=color,ls=style,label=record['label'])
            bk='cumulative_direction_norm' if adaptive else 'cumulative_gradient_norm'
            axes[1,j].plot(x,[r[bk] for r in v],color=color,ls=style,label=record['label'])
        axes[0,j].set_title('Raw gradients' if not adaptive else 'Adaptive Adam directions')
        axes[0,j].set_ylabel('A = cumulative sum of norms')
        axes[1,j].set_ylabel('B = norm of cumulative vector sum')
        axes[1,j].set_xlabel('Optimizer update')
        axes[0,j].legend(frameon=False)
        for ax in axes[:,j]: ax.grid(alpha=.18)
    fig.suptitle('Cumulative A and B versus model scale (updates 1–711)',fontsize=13)
    save(fig,'cumulative_by_scale')

    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    xx=np.array([r['parameters']/1e6 for r in cohort])
    fields=[('median_adjacent_cosine','Median adjacent cosine',False),('negative_fraction','Fraction of negative cosines',False),
        ('R','Final B / A',False),('A','Final A (sum of norms)',True),
        ('B','Final B (norm of sum)',True),('median_norm','Median vector norm',True)]
    for ax,(key,title,log) in zip(axes.flat,fields):
        for stream,color,marker in [('raw','#2874A6','o'),('adaptive','#C0392B','s')]:
            yy=[a['all_updates'][stream][key] for a in aggregate]
            ax.plot(xx,yy,marker=marker,color=color,label=stream,lw=1.1)
        ax.set_title(title)
        ax.set_xlabel('Actual parameter count (millions)')
        ax.set_xticks(xx,labels=[f'{p:.2f}' for p in xx])
        if log: ax.set_yscale('log')
        ax.grid(alpha=.18)
    axes[0,0].legend(frameon=False)
    fig.suptitle('Three sizes, one seed each: descriptive scale comparison\nFull-pass statistics (712 updates); connecting lines are guides, not fitted laws',fontsize=13)
    save(fig,'scale_summary')

    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    for record,color,style in zip(cohort,COLORS,STYLES):
        rows=record['rows'][:-1]
        ax.plot(np.arange(1,712),[r['adaptive_to_raw_norm_ratio'] for r in rows],color=color,ls=style,label=record['label'],lw=.9)
    ax.set_yscale('log')
    ax.set(xlabel='Optimizer update',ylabel='||d|| / ||g|| (log scale)',
        title='Adaptive/raw norm comparison by scale\nBefore learning-rate scaling and weight decay; updates 1–711')
    ax.legend(frameon=False); ax.grid(alpha=.18)
    save(fig,'norm_comparison_by_scale')

def main():
    for name in ['figures','scratch','observations']:(HERE/name).mkdir(exist_ok=True)
    cohort,matched=load_cohort()
    aggregate=[]
    reduced=[]
    for r in cohort:
        all_stats={'raw':stats(r['rows']), 'adaptive':stats(r['rows'],True),
            'median_adaptive_to_raw_norm_ratio':float(np.median([e['adaptive_to_raw_norm_ratio'] for e in r['rows']]))}
        display_stats={'raw':stats(r['rows'][:-1]),'adaptive':stats(r['rows'][:-1],True)}
        quarters=[]
        for first,last in [(1,178),(179,356),(357,534),(535,712)]:
            q=r['rows'][first-1:last]
            quarters.append({'first':first,'last':last,'raw':stats(q),'adaptive':stats(q,True)})
        aggregate.append({'label':r['label'],'parameters':r['parameters'],'run_id':r['run_id'],
            'all_updates':all_stats,'displayed_updates':display_stats,'quarters':quarters,
            'initial_validation_loss':r['summary']['initial_validation']['loss'],
            'final_validation_loss':r['summary']['final_validation']['loss'],
            'overflow_retries':sum(e['overflow_retries'] for e in r['rows'])})
        for row in r['rows']:
            d=row['adaptive']
            reduced.append({'model':r['label'],'parameters':r['parameters'],'update':row['update'],
                'sequences':row['sequences'],'loss':row['loss'],'raw_cosine':row['adjacent_cosine'],
                'adaptive_cosine':d['adjacent_cosine'],'raw_norm':row['gradient_norm'],'adaptive_norm':d['direction_norm'],
                'raw_A':row['cumulative_norm_sum'],'raw_B':row['cumulative_gradient_norm'],'raw_R':row['cumulative_ratio'],
                'adaptive_A':d['cumulative_norm_sum'],'adaptive_B':d['cumulative_direction_norm'],'adaptive_R':d['cumulative_ratio'],
                'adaptive_to_raw_norm_ratio':row['adaptive_to_raw_norm_ratio']})
    with (HERE/'figure_data.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(reduced[0]));w.writeheader();w.writerows(reduced)
    output={'schema_version':1,'script_sha256':sha(Path(__file__)),
        'source_lineage':[{k:v for k,v in r.items() if k not in ['rows','summary']} for r in cohort],
        'matched_contract':matched,'aggregate':aggregate,
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__},
        'figure_coverage':{'evolution_updates':[1,711],'full_statistics_updates':[1,712],
                           'smoothing':'31-update centered median for adjacent cosine only; faint unsmoothed values'}}
    (HERE/'results.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    plots(cohort,aggregate)
    print(json.dumps(aggregate,indent=2))

if __name__=='__main__':main()
