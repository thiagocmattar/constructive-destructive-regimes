"""Verified nine-run comparison; seed trajectories are the replicate units."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PILOT=ROOT/'analyses/001-2026-10-03-model-scale-geometry/01_compare.py'
spec=importlib.util.spec_from_file_location('pilot_reduction',PILOT)
pilot=importlib.util.module_from_spec(spec); spec.loader.exec_module(pilot)
SIZES=('14M','31M','70M')
SEEDS=(1234,1235,1236)
COLORS=('#2874A6','#D35400','#138D75')
METRICS=('median_adjacent_cosine','R','A','B','median_norm','negative_fraction')

def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,obj): path.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')

def load():
    sources=[(size,1234,folder,attempt) for size,folder,attempt in pilot.SOURCES]
    sources += [(r['size'],r['seed'],r['run'],'attempt-01')
                for r in read(ROOT/'cache/runpod/seed-cohort.json')]
    records=[]; protocol=None; orders={}; arch_by_size={}
    for size,seed,folder,attempt in sources:
        run=ROOT/'runs'/folder; base=run/'artifacts'/attempt/'full-pass'
        cfg,ident,env,summary=map(read,[base/'resolved_config.json',base/'input_identity.json',
            base/'environment.json',base/'summary.json'])
        assert read(base/'manifest.json')['status']=='completed'
        assert read(base/'verification.json')['verified']
        receipt=run/'prelaunch'/('attempt2-retrieval.json' if folder[:3]=='002' else 'attempt1-retrieval.json')
        assert read(receipt)['verified']
        assert cfg['seeds']=={'model':seed,'data_order':seed}
        rows=[e for line in (base/'events.jsonl').read_text().splitlines()
              if (e:=json.loads(line))['event']=='update']
        assert len(rows)==712 and [r['update'] for r in rows]==list(range(1,713))
        assert summary['all_training_tokens_covered'] and rows[-1]['cumulative_input_tokens']==1491711416
        assert rows[-1]['sequences']==311 and all(r['sequences']==1024 for r in rows[:-1])
        matched={k:cfg[k] for k in ('training','measurements','validation','checkpoints')}
        matched['data']={k:cfg['data'][k] for k in ('dataset','revision','tokenizer_revision',
            'sequence_length','train_sha256','validation_sha256','coverage','loss_reduction')}
        if protocol is None: protocol=matched
        else: assert matched==protocol,'Scientific protocols differ'
        if seed in orders: assert ident['order_sha256']==orders[seed]
        else: orders[seed]=ident['order_sha256']
        model=env['model']
        if size in arch_by_size: assert model==arch_by_size[size]
        else: arch_by_size[size]=model
        assert [r['learning_rate'] for r in rows]==[r['learning_rate'] for r in (records[0]['rows'] if records else rows)]
        records.append({'size':size,'seed':seed,'run_id':folder[:3],'folder':folder,'attempt':attempt,
            'parameters':model['parameter_count'],'architecture':model['architecture'],
            'gpu':env['gpu'],'initial_parameter_sha256':env['initial_parameter_sha256'],
            'order_sha256':ident['order_sha256'],'rows':rows,'summary':summary,
            'sources':{name:{'path':(base/name).relative_to(ROOT).as_posix(),'sha256':sha(base/name)}
            for name in ('events.jsonl','summary.json','resolved_config.json','input_identity.json',
                         'environment.json','verification.json')},
            'acceptance':{'path':receipt.relative_to(ROOT).as_posix(),'sha256':sha(receipt)}})
    assert len(orders)==3 and len(set(orders.values()))==3
    assert {(r['size'],r['seed']) for r in records}=={(m,s) for m in SIZES for s in SEEDS}
    return sorted(records,key=lambda r:(r['seed'],SIZES.index(r['size']))),protocol

def summarize(records):
    runs=[]
    for r in records:
        stats={stream:pilot.stats(r['rows'],stream=='adaptive') for stream in ('raw','adaptive')}
        shown={stream:pilot.stats(r['rows'][:-1],stream=='adaptive') for stream in ('raw','adaptive')}
        quarters=[{'first':start+1,'last':start+178,**{stream:pilot.stats(r['rows'][start:start+178],
            stream=='adaptive') for stream in ('raw','adaptive')}} for start in (0,178,356,534)]
        runs.append({'size':r['size'],'seed':r['seed'],'run_id':r['run_id'],'parameters':r['parameters'],
            'all_updates':stats,'displayed_updates':shown,'quarters':quarters,
            'median_adaptive_to_raw_norm_ratio':float(np.median([e['adaptive_to_raw_norm_ratio'] for e in r['rows']])),
            'initial_validation_loss':r['summary']['initial_validation']['loss'],
            'final_validation_loss':r['summary']['final_validation']['loss'],
            'overflow_retries':sum(e['overflow_retries'] for e in r['rows'])})
    groups=[]
    for size in SIZES:
        group=[r for r in runs if r['size']==size]
        for stream in ('raw','adaptive'):
            for key in METRICS:
                values=[r['all_updates'][stream][key] for r in group]
                groups.append({'size':size,'stream':stream,'metric':key,'n_seeds':3,
                    'seeds':list(SEEDS),'values':values,'mean':float(np.mean(values)),
                    'sample_sd':float(np.std(values,ddof=1)),'min':float(min(values)),'max':float(max(values))})
    paired=[]
    for stream in ('raw','adaptive'):
        for key in METRICS:
            expected='increase' if stream=='adaptive' and key in ('A','B','median_norm') else 'decrease'
            if key in ('negative_fraction',): expected='increase' if stream=='raw' else 'no_negative_pairs'
            if key=='median_norm' and stream=='raw': expected='no_monotone_pilot_expectation'
            for seed in SEEDS:
                block=[next(r for r in runs if r['seed']==seed and r['size']==m) for m in SIZES]
                values=[r['all_updates'][stream][key] for r in block]
                delta=np.diff(values)
                strict=bool(np.all(delta<0)) if expected=='decrease' else bool(np.all(delta>0)) if expected=='increase' else None
                paired.append({'seed':seed,'stream':stream,'metric':key,'values':values,
                    'difference_31M_minus_14M':float(delta[0]),'difference_70M_minus_31M':float(delta[1]),
                    'pilot_expectation':expected,'strict_pilot_ordering':strict})
    return runs,groups,paired

def save(fig,stem):
    fig.savefig(HERE/'figures'/f'{stem}.pdf',bbox_inches='tight')
    fig.savefig(HERE/'scratch'/f'{stem}.png',dpi=150,bbox_inches='tight'); plt.close(fig)

def plots(records,runs,groups):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(3,2,figsize=(12,10),sharex=True,layout='constrained')
    x=np.arange(1,712)
    for j,stream in enumerate(('raw','adaptive')):
        axes[0,j].set_title('Raw gradient before clipping' if j==0 else 'Adaptive Adam direction before LR/decay')
        for size,color in zip(SIZES,COLORS):
            block=[r for r in records if r['size']==size]
            series=[[],[],[]]
            for r in block:
                vectors=[e if j==0 else e['adaptive'] for e in r['rows'][:-1]]
                cos=[v['adjacent_cosine'] if v['adjacent_cosine'] is not None else np.nan for v in vectors]
                smoothed=pilot.rolling(cos)
                smoothed[0]=np.nan  # Keep the first adjacent cosine undefined in the display.
                series[0].append(smoothed)
                series[1].append([v['cumulative_ratio'] for v in vectors])
                series[2].append([v['gradient_norm' if j==0 else 'direction_norm'] for v in vectors])
            for ax,arr in zip(axes[:,j],series):
                z=np.array(arr); ax.fill_between(x,np.min(z,axis=0),np.max(z,axis=0),color=color,alpha=.16)
                for trajectory in z: ax.plot(x,trajectory,color=color,lw=.5,alpha=.25)
                ax.plot(x,np.mean(z,axis=0),color=color,lw=1.4,label=size)
                ax.grid(alpha=.18)
        axes[0,j].set_ylim((-1.01,1.01) if j==0 else (.75,1.01))
        axes[0,j].set_ylabel('Adjacent cosine\n31-step median in each seed')
        axes[0,j].legend(frameon=False,loc='lower right')
        axes[1,j].set(ylabel='B / A',ylim=(0,1.02))
        axes[2,j].set(xlabel='Optimizer update',ylabel='Vector norm (log scale)',yscale='log')
    fig.suptitle('Geometry across model sizes: three seeds per size\nLines: seed mean; shading: seed range (not a confidence interval); updates 1-711',fontsize=13)
    save(fig,'geometry_across_seeds')

    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    titles=('Median adjacent cosine','Final B / A','Final A','Final B','Median vector norm','Fraction of negative cosines')
    keys=('median_adjacent_cosine','R','A','B','median_norm','negative_fraction')
    for ax,key,title in zip(axes.flat,keys,titles):
        for stream,color,marker in (('raw','#2874A6','o'),('adaptive','#C0392B','s')):
            xx=np.array([next(r['parameters'] for r in runs if r['size']==s)/1e6 for s in SIZES])
            group=[next(g for g in groups if g['size']==s and g['stream']==stream and g['metric']==key) for s in SIZES]
            ax.errorbar(xx,[g['mean'] for g in group],yerr=[g['sample_sd'] for g in group],color=color,marker=marker,
                        capsize=4,lw=1,label=stream+' mean ± sample SD')
            for seed in SEEDS:
                yy=[next(r['all_updates'][stream][key] for r in runs if r['size']==s and r['seed']==seed) for s in SIZES]
                ax.plot(xx,yy,color=color,alpha=.3,lw=.6,marker=marker,ms=4)
        if key in ('A','B','median_norm'): ax.set_yscale('log')
        ax.set(title=title,xlabel='Actual parameters (millions)'); ax.set_xticks(xx,labels=[f'{v:.2f}' for v in xx]); ax.grid(alpha=.18)
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle('Full-pass seed replication (712 updates): n=3 per size\nFaint connected points: paired seeds; lines are guides, not fitted laws',fontsize=13)
    save(fig,'scale_summary_across_seeds')

    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,layout='constrained')
    for j in range(2):
        for size,color in zip(SIZES,COLORS):
            for i,key in enumerate(('cumulative_norm_sum','cumulative_gradient_norm' if j==0 else 'cumulative_direction_norm')):
                z=np.array([[((e if j==0 else e['adaptive'])[key]) for e in r['rows'][:-1]] for r in records if r['size']==size])
                axes[i,j].fill_between(x,z.min(0),z.max(0),color=color,alpha=.16)
                axes[i,j].plot(x,z.mean(0),color=color,label=size,lw=1.4);axes[i,j].grid(alpha=.18)
        axes[0,j].set(title='Raw gradients' if j==0 else 'Adaptive Adam directions',ylabel='A: cumulative sum of norms')
        axes[1,j].set(xlabel='Optimizer update',ylabel='B: norm of cumulative vector sum')
        axes[0,j].legend(frameon=False)
    fig.suptitle('Cumulative geometry: seed mean and seed range; updates 1-711',fontsize=13)
    save(fig,'cumulative_across_seeds')

    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    for size,color in zip(SIZES,COLORS):
        z=np.array([[e['adaptive_to_raw_norm_ratio'] for e in r['rows'][:-1]] for r in records if r['size']==size])
        ax.fill_between(x,z.min(0),z.max(0),color=color,alpha=.16);ax.plot(x,z.mean(0),color=color,label=size)
    ax.set(xlabel='Optimizer update',ylabel='||d|| / ||g||',yscale='log',title='Adaptive/raw norm ratio: mean and range across three seeds\nDirection before LR/decay; updates 1-711')
    ax.grid(alpha=.18); ax.legend(frameon=False); save(fig,'norm_ratio_across_seeds')

def main():
    for name in ('figures','scratch','observations'): (HERE/name).mkdir(exist_ok=True)
    records,protocol=load(); runs,groups,paired=summarize(records)
    result={'schema_version':1,'script_sha256':sha(Path(__file__)),
        'pilot_reduction_script_sha256':sha(PILOT),'matched_contract':protocol,
        'source_lineage':[{k:v for k,v in r.items() if k not in ('rows','summary')} for r in records],
        'runs':runs,'across_seeds':groups,'paired_size_comparisons':paired,
        'replicate_unit':'One complete training trajectory; three seed blocks, not individual steps.',
        'quarter_reduction':'Cosine/norm summaries use events in each quarter; A/B/R retain full cumulative history through that quarter endpoint. Cosines are assigned by their later update.',
        'figure_coverage':{'evolution_updates':'1-711','statistics_updates':'1-712',
            'cosine_smoothing':'31-step centered median per seed before seed mean/range',
            'envelope':'range across three seeds, not confidence interval'},
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__}}
    write(HERE/'results.json',result)
    with (HERE/'figure_data.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['model','seed','parameters','update','sequences','loss','raw_cosine','adaptive_cosine',
            'raw_norm','adaptive_norm','raw_A','raw_B','raw_R','adaptive_A','adaptive_B','adaptive_R','adaptive_to_raw_norm_ratio']
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for r in records:
            for e in r['rows']:
                d=e['adaptive'];writer.writerow(dict(zip(fields,[r['size'],r['seed'],r['parameters'],e['update'],e['sequences'],e['loss'],
                    e['adjacent_cosine'],d['adjacent_cosine'],e['gradient_norm'],d['direction_norm'],e['cumulative_norm_sum'],
                    e['cumulative_gradient_norm'],e['cumulative_ratio'],d['cumulative_norm_sum'],d['cumulative_direction_norm'],
                    d['cumulative_ratio'],e['adaptive_to_raw_norm_ratio']])))
    plots(records,runs,groups)
    write(HERE/'generation.json',{'results_sha256':sha(HERE/'results.json'),'figure_data_sha256':sha(HERE/'figure_data.csv'),
        'figures':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted((HERE/'figures').glob('*.pdf'))}})
    print(json.dumps({'runs':len(runs),'step_rows':sum(len(r['rows']) for r in records),'figures':4}))

if __name__=='__main__': main()
