"""Reconstruct Pythia diagnostics from frozen events; create four vector figures."""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WIDTH = 25
SOURCES = {}
RECORDS = []
CURVES = []
ARRAYS = {}
FIGURES = {}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def source(relative, expected=None, jsonl=False):
    p = ROOT / relative
    digest = sha(p)
    if expected is not None:
        assert digest == expected, relative
    SOURCES[p.relative_to(ROOT).as_posix()] = digest
    text = p.read_text(encoding='utf-8')
    return [json.loads(s) for s in text.splitlines()] if jsonl else json.loads(text)


def reduce_record(cohort, rid, size, seed, condition, events, validation, **metadata):
    assert [e['update'] for e in events] == list(range(1, 713))
    assert all(e['sequences'] == 1024 for e in events[:-1]) and events[-1]['sequences'] == 311
    a = {k: np.array([e[field] for e in events], dtype=float) for k, field in
         [('loss', 'loss'), ('lr', 'learning_rate'), ('tokens', 'supervised_tokens'),
          ('g_norm', 'gradient_norm'), ('g_cos', 'adjacent_cosine'), ('g_R', 'cumulative_ratio')]}
    a['clip'] = np.array([e['gradient_clip_returned_norm'] > 1 for e in events], dtype=float)
    for stream, prefix, norm in [('adaptive', 'd', 'direction_norm'), ('momentum', 'm', 'momentum_norm')]:
        if stream in events[0]:
            for suffix, key in [('norm', norm), ('cos', 'adjacent_cosine'), ('R', 'cumulative_ratio')]:
                a[prefix + '_' + suffix] = np.array([e[stream][key] for e in events], dtype=float)
    a['q'] = np.r_[np.nan, 1-a['g_cos'][1:]*a['g_norm'][1:]/a['g_norm'][:-1]]
    a['q_norm_correction'] = a['q'] - (1-a['g_cos'])
    if 'm_norm' in a:
        previous, current = a['m_norm'][:-1], a['m_norm'][1:]
        dot = previous*current*a['m_cos'][1:]
        input_squared = (current**2 + .9**2*previous**2 - 2*.9*dot)/.1**2
        assert np.all(input_squared > 0)
        reconstructed = np.sqrt(input_squared)
        a['input_norm'] = np.r_[np.nan, reconstructed]
        a['input_m_cos'] = np.r_[np.nan, (dot-.9*previous**2)/(.1*previous*reconstructed)]
        a['input_memory_ratio'] = np.r_[np.nan, .1*reconstructed/(.9*previous)]
        assert np.nanmax(np.abs(a['input_m_cos'])) <= 1+1e-8
        clip_return = np.array([e['gradient_clip_returned_norm'] for e in events])
        expected_input = a['g_norm']*np.minimum(1, 1/(clip_return+1e-6))
        a['input_norm_relative_error'] = (a['input_norm']-expected_input)/expected_input
        assert np.nanmax(np.abs(a['input_norm_relative_error'])) < 5e-6
    r = dict(cohort=cohort, run=rid, size=size, seed=seed, condition=condition,
             validation=validation, arrays=a, **metadata)
    RECORDS.append(r)
    return r


def load_records():
    evidence = source('analyses/028-2026-10-30-adam-diagnostics/evidence.json')
    sweep = source('analyses/009-2026-10-09-intermediate-lr/complete-seven-lr/comparison_data.json',
                   'a02a1ba7ace7a7011063767e6e7fb1c25548be0a12be06b6dad6826e5e7b2739')
    lr_records = []
    for rid, old in sorted(sweep['runs'].items(), key=lambda kv: kv[1]['lr']):
        relative = next(p for p in evidence['event_sources'] if
                        p.startswith('runs/'+rid+'-') or '/'+rid+'/events.jsonl' in p)
        events = source(relative, old['sources']['events.jsonl'], True)
        updates = [e for e in events if e['event'] == 'update']
        assert updates == old['events']
        val = [dict(update=712, loss=e['loss']) for e in events if e['event'] == 'final_validation']
        lr_records.append(reduce_record('lr', rid, '14M', 1234, old['lr'], updates, val,
                                        hardware=old['hardware'], event_source=relative))
    scale = source('analyses/002-2026-10-04-scale-seed-replication/results.json',
                   '4dcc899b5572c97e40345ba871af8445632ba1c2d83568ecdd36760094136a79')
    scale_records = []
    parent_updates = {}
    for old in scale['source_lineage']:
        src = old['sources']['events.jsonl']
        events = source(src['path'], src['sha256'], True)
        updates = [e for e in events if e['event'] == 'update']
        parent_updates[old['folder']] = updates
        val = [dict(update=712, loss=e['loss']) for e in events if e['event'] == 'final_validation']
        scale_records.append(reduce_record('size', old['run_id'], old['size'], old['seed'], old['size'],
                                           updates, val, hardware=old['gpu'], event_source=src['path']))
    schedules = source('analyses/004-2026-10-04-lr-schedule-intervention/results.json',
                       'b15537ab31b1a97b80e3fee5ef98392830e7ab5da36da2160c97fe6fea93f682')
    schedule_records = []
    for old in schedules['records']:
        original = next(s for s in schedules['source'] if s['run'] == old['run'])
        relative = 'runs/'+old['run']+'/artifacts/attempt-01/full-pass/events.jsonl'
        events = source(relative, original['events_sha256'], True)
        updates = parent_updates[old['parent_run']][:356]+[e for e in events if e['event'] == 'update']
        val = [dict(update=e['update'], loss=e['loss']) for e in events if e['event'] == 'validation']
        assert [v['update'] for v in val] == [356,445,534,623,712]
        schedule_records.append(reduce_record('schedule', old['run'][:3], old['size'], 1235,
                                              old['variant'], updates, val, hardware='RTX 5090',
                                              event_source=relative, parent_run=old['parent_run'][:3]))
    return lr_records, schedule_records, scale_records


def trailing(y, weights=None):
    out = np.full(len(y), np.nan)
    for end in range(WIDTH-1, len(y)):
        chunk = y[end-WIDTH+1:end+1]
        if np.isfinite(chunk).all():
            out[end] = np.average(chunk, weights=None if weights is None else weights[end-WIDTH+1:end+1])
    return out


def values(r, metric, smooth=False, first=1, last=711):
    if metric == 'validation':
        return np.array([v['update'] for v in r['validation']]), np.array([v['loss'] for v in r['validation']])
    y = r['arrays'][metric][first-1:last]
    if smooth:
        y = trailing(y, r['arrays']['tokens'][first-1:last] if metric == 'loss' else None)
    return np.arange(first, last+1), y


def keep(fig, records, metric, smooth, first, last, aggregation, x, y):
    key = f'curve_{len(CURVES):04d}'
    CURVES.append(dict(key=key, figure=fig, runs=[r['run'] for r in records], cohort=records[0]['cohort'],
                       metric=metric, smooth=smooth, first=first, last=last, aggregation=aggregation))
    ARRAYS[key+'_x'], ARRAYS[key+'_y'] = np.array(x), np.array(y)


def draw(ax, name, r, metric, color, smooth=True, first=1, last=711, faint=True, marker=None):
    x, y = values(r, metric, smooth, first, last)
    if smooth and faint:
        raw_x, raw_y = values(r, metric, False, first, last)
        ax.plot(raw_x, raw_y, color=color, lw=.45, alpha=.12)
        keep(name, [r], metric, False, first, last, 'individual', raw_x, raw_y)
    ax.plot(x, y, color=color, lw=1.55, marker=marker, ms=4)
    keep(name, [r], metric, smooth, first, last, 'individual', x, y)


def style(ax, ylabel, title=None, xlim=(0,720), log=False, zero=False, one=False):
    if title:
        ax.set_title(title, loc='left', pad=8)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.grid(axis='y', alpha=.2, lw=.5)
    ax.spines[['top', 'right']].set_visible(False)
    if log:
        ax.set_yscale('log')
    if zero:
        ax.axhline(0, color='#777777', ls=':', lw=.75)
    if one:
        ax.axhline(1, color='#777777', ls=':', lw=.75)


def save(fig, name):
    for ax in fig.axes:
        for line in ax.lines:
            x, y = np.asarray(line.get_xdata(),dtype=float), np.asarray(line.get_ydata(),dtype=float)
            if len(x) <= 2:  # Algebraic reference lines use axes-fraction X coordinates.
                continue
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.any():
                assert x[valid].min() >= ax.get_xlim()[0]-1e-8 and x[valid].max() <= ax.get_xlim()[1]+1e-8
                assert y[valid].min() >= ax.get_ylim()[0]-1e-8 and y[valid].max() <= ax.get_ylim()[1]+1e-8, (name,ax.get_ylabel())
    p = HERE/'figures'/(name+'.pdf')
    fig.savefig(p, metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    FIGURES[name] = dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p), data_bounds_within_axes=True)


def lr_legend(fig, records):
    handles = [Line2D([], [], color=r['color'], lw=2, label=rf"$\eta={r['condition']:.3g}$") for r in records]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.51,1), ncol=4, frameon=False,
               columnspacing=2.1, handlelength=2.4, title='Pythia-14M: peak learning rate',title_fontsize=12.5)


def plot_lr(records):
    palette = plt.get_cmap('plasma_r')(np.linspace(.22,.94,len(records)))
    for r, color in zip(records,palette):
        r['color'] = color
    name = 'pythia_lr_response'
    fig, axes = plt.subplots(2,3,figsize=(11.8,7.4), sharex=True)
    panels = [('loss','Training loss (nats)','(a) Training loss',True,False,False),
              ('g_cos',r'$\rho_t$','(b) Raw-gradient alignment',False,True,False),
              ('q','Projected gradient change',r'(c) SGD $\eta\widehat{\kappa}$ expression',False,False,True),
              ('g_norm',r'$\|g_t\|$','(d) Raw-gradient norm',True,False,True),
              ('m_norm',r'$\|m_t\|$','(e) First-moment norm',True,False,False),
              ('input_m_cos',r'$\cos(g_t,m_{t-1})$','(f) Input vs. prior moment',False,True,False)]
    for ax,(metric,ylabel,title,log,zero,one) in zip(axes.flat,panels):
        for r in records:
            draw(ax,name,r,metric,r['color'])
        style(ax,ylabel,title,log=log,zero=zero,one=one)
        if metric in ['g_cos','input_m_cos']:
            ax.set_ylim(-1.05,1.05)
    for ax in axes[-1]:ax.set_xlabel('Optimizer update')
    lr_legend(fig,records)
    fig.subplots_adjust(left=.075,right=.985,bottom=.08,top=.85,wspace=.35,hspace=.38)
    save(fig,name)

    name = 'pythia_optimizer_streams'
    fig, axes = plt.subplots(2,3,figsize=(11.8,6.8), sharex=True)
    for col,(prefix,label) in enumerate([('g','Raw gradients'),('m','First moments'),('d','Adaptive directions')]):
        for r in records:
            draw(axes[0,col],name,r,prefix+'_cos',r['color'])
            draw(axes[1,col],name,r,prefix+'_R',r['color'],smooth=False)
        style(axes[0,col],'Consecutive cosine',f'({"abc"[col]}) {label}',zero=True)
        style(axes[1,col],'Stream accumulation ratio',f'({"def"[col]}) {label}')
        axes[0,col].set_ylim(-1.05 if col==0 else 0,1.03)
        axes[1,col].set_ylim(0,1.03);axes[1,col].set_xlabel('Optimizer update')
    lr_legend(fig,records)
    fig.subplots_adjust(left=.075,right=.985,bottom=.08,top=.82,wspace=.35,hspace=.38)
    save(fig,name)


def plot_schedules(records):
    name = 'pythia_schedule_response'
    fig, axes = plt.subplots(5,2,figsize=(9,12),sharex=True,sharey='row')
    colors = dict(original='#0072B2',hold='#D55E00',accelerated='#009E73')
    labels = dict(original='Original decay',hold='Held rate',accelerated='Faster decay')
    panels = [('lr',r'Learning rate $\eta_t$',False),('g_cos',r'Raw cosine $\rho_t$',True),
              ('q',r'$\eta\widehat{\kappa}$ expression',True),('d_cos','Adaptive cosine',True),
              ('validation','Validation loss (nats)',False)]
    for col,size in enumerate(['14M','70M']):
        group = [r for r in records if r['size']==size]
        for row,(metric,ylabel,smooth) in enumerate(panels):
            ax=axes[row,col]
            for r in group:
                draw(ax,name,r,metric,colors[r['condition']],smooth,first=358 if smooth else 357,
                     marker='o' if metric=='validation' else None)
            style(ax,ylabel if col==0 else None,xlim=(350,720),log=metric=='lr',
                  zero=metric=='g_cos',one=metric=='q')
            if row==0:ax.set_title(f'({"ab"[col]}) Pythia-{size}',loc='left',pad=8)
            if metric=='validation':ax.set_yscale('log')
        axes[-1,col].set_xlabel('Optimizer update')
    fig.legend(handles=[Line2D([],[],color=colors[k],lw=2,label=labels[k]) for k in colors],
               loc='upper center',bbox_to_anchor=(.52,.995),ncol=3,frameon=False)
    fig.subplots_adjust(left=.12,right=.985,bottom=.055,top=.94,wspace=.18,hspace=.2)
    save(fig,name)


def plot_sizes(records):
    name = 'pythia_scale_response'
    fig, axes = plt.subplots(3,2,figsize=(10.4,9.3),sharex=True)
    colors = {'14M':'#0072B2','31M':'#D55E00','70M':'#009E73'}
    panels=[('loss','Training loss (nats)','(a) Training loss',True),
            ('g_cos',r'$\rho_t$','(b) Raw-gradient alignment',True),
            ('q','Projected gradient change',r'(c) SGD $\eta\widehat{\kappa}$ expression',True),
            ('d_cos','Consecutive cosine','(d) Adaptive-direction alignment',True),
            ('g_R','Stream accumulation ratio','(e) Raw-gradient retention',False),
            ('d_R','Stream accumulation ratio','(f) Adaptive-direction retention',False)]
    for ax,(metric,ylabel,title,smooth) in zip(axes.flat,panels):
        for size,color in colors.items():
            rs=sorted([r for r in records if r['size']==size],key=lambda r:r['seed'])
            ys=[]
            for r in rs:
                x,y=values(r,metric,smooth);ys.append(y)
                ax.plot(x,y,color=color,lw=.55,alpha=.25)
                keep(name,[r],metric,smooth,1,711,'individual',x,y)
            y=np.array(ys)
            mean,lo,hi=y.mean(axis=0),y.min(axis=0),y.max(axis=0)
            ax.fill_between(x,lo,hi,color=color,alpha=.14,lw=0)
            ax.plot(x,mean,color=color,lw=1.8)
            for key,arr in [('mean',mean),('min',lo),('max',hi)]:keep(name,rs,metric,smooth,1,711,key,x,arr)
        style(ax,ylabel,title,log=metric=='loss',zero=metric=='g_cos',one=metric=='q')
        if metric=='g_cos':ax.set_ylim(-.55,1.05)
        if metric=='d_cos':ax.set_ylim(.9,1.005)
        if metric.endswith('_R'):ax.set_ylim(0,1.03)
    for ax in axes[-1]:ax.set_xlabel('Optimizer update')
    fig.legend(handles=[Line2D([],[],color=c,lw=2,label='Pythia-'+s) for s,c in colors.items()],
               loc='upper center',bbox_to_anchor=(.53,.995),ncol=3,frameon=False)
    fig.subplots_adjust(left=.09,right=.985,bottom=.07,top=.93,wspace=.25,hspace=.32)
    save(fig,name)


def summarize(r):
    a=r['arrays'];out={k:v for k,v in r.items() if k not in ['arrays','color']}
    windows={'full':(2,712),'full_batches':(2,711),'late':(612,711)}
    if r['cohort']=='schedule':windows['affected']=(358,712)
    out['windows']={}
    for label,(first,last) in windows.items():
        region=slice(first-1,last)
        out['windows'][label]={k:float(np.median(v[region])) for k,v in a.items()
                              if k in ['q','g_cos','m_cos','d_cos','g_norm','m_norm','input_m_cos','input_memory_ratio']}
    out['q_correction_abs_p95_full_batches']=float(np.nanpercentile(abs(a['q_norm_correction'][:711]),95))
    out['clipped_updates']=int(a['clip'].sum())
    out['raw_negative_pairs']=int((a['g_cos'][1:]<0).sum())
    out['terminal_R']={p:float(a[p+'_R'][-1]) for p in ['g','m','d'] if p+'_R' in a}
    if 'input_norm_relative_error' in a:
        out['max_input_norm_relative_error']=float(np.nanmax(abs(a['input_norm_relative_error'])))
        out['negative_momentum_pairs']=int((a['m_cos'][1:]<0).sum())
    return out


def main():
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
                         'mathtext.fontset':'stix','font.size':12.4,'axes.titlesize':12.5,
                         'legend.fontsize':12,'pdf.fonttype':42,'savefig.facecolor':'white'})
    lr,schedule,size=load_records()
    plot_lr(lr);plot_schedules(schedule);plot_sizes(size)
    summaries=[summarize(r) for r in RECORDS]
    data=HERE/'plot_arrays.npz';np.savez_compressed(data,**ARRAYS)
    result=dict(script_sha256=sha(Path(__file__)),sources=SOURCES,window=WIDTH,
                records=summaries,curves=CURVES,figures=FIGURES,plot_arrays_sha256=sha(data),
                observed_update_rows=len(RECORDS)*712,defined_projected_changes=len(RECORDS)*711,
                reconstructed_input_pairs=7*711)
    (HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    for name in FIGURES:
        cmd=['pdftoppm','-singlefile','-scale-to','1600','-png',str(HERE/'figures'/(name+'.pdf')),str(HERE/'scratch'/name)]
        completed=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        assert completed.returncode==0,completed.stderr
    print(json.dumps(dict(figures=len(FIGURES),records=len(RECORDS),curves=len(CURVES),
                         reconstruction_max_relative_error=max(r.get('max_input_norm_relative_error',0) for r in summaries))))


if __name__=='__main__':main()
