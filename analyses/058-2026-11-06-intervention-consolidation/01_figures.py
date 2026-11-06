"""Publication views of frozen interventions; no training or source mutation."""
from pathlib import Path
import csv
import hashlib
import json
import shutil

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, LogLocator, NullFormatter, FuncFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG = HERE / 'figures'
SCRATCH = ROOT / 'tmp/pdfs/analysis058'
SOURCES, DATA, DEFINITIONS, CURVES, VALUES, OUTPUTS = {}, {}, {}, [], [], {}
GRAY, BLUE, GREEN, ORANGE = '#777777', '#0072B2', '#009E73', '#D55E00'
METRICS = ['loss', 'eta_kappa_hat', 'rho', 'raw_gradient_norm', 'applied_lr']
LABELS = dict(loss='Training loss', eta_kappa_hat=r'$\eta\widehat\kappa_t$',
              rho=r'$\rho_t$', raw_gradient_norm=r'$\|g_t\|$', applied_lr='Applied / base LR')


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def source(p):
    p = Path(p)
    if not p.is_absolute():
        p = ROOT / p
    rel = p.relative_to(ROOT).as_posix()
    SOURCES[rel] = dict(sha256=sha(p), bytes=p.stat().st_size)
    return p


def read(p):
    return json.loads(source(p).read_text(encoding='utf-8'))


def rows(p):
    return [json.loads(s) for s in source(p).read_text(encoding='utf-8').splitlines() if s.strip()]


def analysis(n):
    return next((ROOT / 'analyses').glob(f'{n:03d}-*'))


def worker(n, host='spark'):
    return next((ROOT / 'runs').glob(f'{n:03d}-*')) / f'artifacts/{host}/payload/worker'


def add(key, prefix, branch, base, start, end, clipped=False):
    raw = rows(prefix / 'prefix/boundaries.jsonl')[:-1] + rows(branch / 'boundaries.jsonl')
    assert [r['state'] for r in raw] == list(range(end + 1)), key
    data = {r['state']: dict(r) for r in raw}
    for r in data.values():
        if r.get('applied_lr') is not None:
            r['applied_lr'] /= base
    if clipped:
        for r in rows(branch / 'metrics.jsonl'):
            data[r['state']]['optimizer_input_norm'] = r['optimizer_input_norm']
    DATA[key] = data
    DEFINITIONS[key] = dict(prefix=(prefix/'prefix/boundaries.jsonl').relative_to(ROOT).as_posix(),
        branch=(branch/'boundaries.jsonl').relative_to(ROOT).as_posix(),
        metrics=(branch/'metrics.jsonl').relative_to(ROOT).as_posix() if clipped else None,
        base_lr=base, fork=start, end=end, kind='full_objective')


def load():
    # Authenticate unchanged inputs against the owning analysis manifests.
    for n in [44, 45, 46, 47, 50, 52]:
        result = read(analysis(n) / 'results.json')
        for item in result.get('source_files', result.get('sources', [])):
            assert sha(source(item['path'])) == item['sha256'], item['path']
    mlp, res, tr = worker(223), worker(223, 'duqoihms8w1gle'), worker(223, '159vjccj7w3996')
    for case, p, base, start, end in [('mlp', mlp, .001, 512, 800), ('resnet', res, .003, 256, 576), ('transformer', tr, .003, 544, 928)]:
        add(case+'.control', p, p/'control', base, start, end)
        add(case+'.half', p, p/'fixed_half', base, start, end)
    for key, n, arm in [('cosinehalf',223,'cosine32_half'), ('cosine75',226,'cosine32_three_quarters'),
                        ('half_return',224,'fixed_half_return075'), ('cosine_return',224,'cosine32_half_return075'),
                        ('cosinerise',225,'cosine32_half_rise70'), ('twostage',227,'cosine32_075_then_05'),
                        ('clip1',228,'global_clip1'), ('clip05',229,'global_clip05')]:
        add('mlp.'+key, mlp, worker(n)/arm, .001, 512, 800, key.startswith('clip'))
    # The ResNet clipping comparison must use its own fresh Spark control.
    w = worker(231)
    assert read(w/'manifest.json')['baseline'] == 'fresh_spark'
    inv = read(w.parent/'transfer_inventory.json')
    for item in inv['files']:
        if item['path'].endswith(('boundaries.jsonl', 'metrics.jsonl')):
            assert sha(source(w.parent/item['path'])) == item['sha256']
    add('resnet.spark_control', res, w/'control', .003, 256, 576)
    add('resnet.clip1', res, w/'clipping', .003, 256, 576, True)
    p4 = analysis(4)
    old = read(p4/'results.json')
    hashes = {r['run']: r for r in old['source']}
    for entry in read(p4/'cohort.json'):
        run = ROOT/'runs'/entry['run']/'artifacts/attempt-01/full-pass'
        parent = ROOT/'runs'/entry['parent_run']/'artifacts/attempt-01/full-pass/events.jsonl'
        assert sha(source(run/'events.jsonl')) == hashes[entry['run']]['events_sha256']
        assert sha(source(parent)) == hashes[entry['run']]['parent_events_sha256']
        before = [r for r in rows(parent) if r['event']=='update'][:356]
        after = [r for r in rows(run/'events.jsonl') if r['event']=='update']
        combined = before + after
        assert [r['update'] for r in combined] == list(range(1,713))
        cfg = read(run/'resolved_config.json')
        base = before[-1]['learning_rate']
        key = 'pythia'+entry['size']+'.'+entry['variant']
        DATA[key] = {}
        for i, r in enumerate(combined):
            prev = combined[i-1]['gradient_norm'] if i else None
            q = 1-r['adjacent_cosine']*r['gradient_norm']/prev if prev and r['adjacent_cosine'] is not None else None
            DATA[key][r['update']] = dict(loss=r['loss'], rho=r['adjacent_cosine'],
                raw_gradient_norm=r['gradient_norm'], eta_kappa_hat=q,
                applied_lr=r['learning_rate']/base, supervised_tokens=r['supervised_tokens'])
        DEFINITIONS[key] = dict(kind='stochastic_adamw', prefix=parent.relative_to(ROOT).as_posix(),
            branch=(run/'events.jsonl').relative_to(ROOT).as_posix(), base_lr=base,
            fork=356, end=712, loss='sampled token-mean training loss',
            peak_lr=cfg['training']['peak_learning_rate'])


def draw_curve(ax, name, col, key, metric, label, color, begin, end, ls='-', alpha=1., lw=.75):
    points = [(x, r[metric]) for x,r in sorted(DATA[key].items())
              if begin<=x<=end and r.get(metric) is not None]
    if metric=='applied_lr' and DEFINITIONS[key]['kind']=='full_objective':
        points = [(x,y) for x,y in points if x<end]
    x, y = map(np.asarray, zip(*points))
    xx, yy = x, y
    close_interval = metric=='applied_lr' and DEFINITIONS[key]['kind']=='full_objective'
    if close_interval:
        xx, yy = np.append(x,end), np.append(y,y[-1])
    line, = ax.plot(xx, yy, color=color, lw=lw, ls=ls, alpha=alpha, label=label,
                    drawstyle='steps-post' if metric=='applied_lr' else 'default')
    np.testing.assert_array_equal(line.get_ydata(), yy)
    index = len(CURVES)
    CURVES.append(dict(figure=name, column=col, series=key, metric=metric, points=len(x),
        x_sha256=hashlib.sha256(np.asarray(x,dtype='<f8').tobytes()).hexdigest(),
        y_sha256=hashlib.sha256(np.asarray(y,dtype='<f8').tobytes()).hexdigest(),
        terminal_interval_closed=close_interval, label=label, linestyle=ls))
    VALUES.extend(dict(curve=index, state=int(a), value=float(b)) for a,b in zip(x,y))


def panel(title, series, begin=448, end=800, fork=512, marks=(), clip=False):
    return dict(title=title, series=series, begin=begin, end=end, fork=fork, marks=marks, clip=clip)


def log_axis(ax):
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1,2,5), numticks=5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p: f'{v:g}'))
    ax.yaxis.set_minor_formatter(NullFormatter())


def save(fig, name):
    fig.canvas.draw()
    box = fig.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans)
    assert box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.width+1 and box.y1<=fig.bbox.height+1, (name,box)
    dest = FIG/(name+'.pdf')
    fig.savefig(dest, metadata={'CreationDate':None, 'ModDate':None})
    plt.close(fig)
    with fitz.open(dest) as doc:
        assert len(doc)==1 and not doc[0].get_images()
        for word in doc[0].get_text('words'):
            assert doc[0].rect.contains(fitz.Rect(word[:4])), (name,word)
        for font in doc[0].get_fonts(full=True):
            assert font[2]!='Type3' and doc.extract_font(font[0])[3]
        doc[0].get_pixmap(matrix=fitz.Matrix(2,2)).save(SCRATCH/(name+'.png'))
    OUTPUTS[name] = dict(path=dest.relative_to(ROOT).as_posix(), sha256=sha(dest))


def plot(name, panels, metrics=METRICS, subtitle=None):
    nr,nc = len(metrics),len(panels)
    fig,axs = plt.subplots(nr,nc,figsize=(7.16,6.35 if nr==5 else 5.45),sharex='col',squeeze=False)
    top = (.82 if nr==4 else .865) if subtitle else .89
    fig.subplots_adjust(left=.09,right=.986,bottom=.075,top=top,hspace=.18,wspace=.30)
    for col,p in enumerate(panels):
        for row,metric in enumerate(metrics):
            ax=axs[row,col]
            for key,label,color in p['series']:
                draw_curve(ax,name,col,key,metric,label,color,p['begin'],p['end'])
                if metric=='raw_gradient_norm' and p['clip'] and '.clip' in key:
                    draw_curve(ax,name,col,key,'optimizer_input_norm',label+' input',color,p['begin'],p['end'],ls='--',lw=.85)
            ax.axvspan(p['begin'],p['fork'],color='.92',zorder=-5,lw=0)
            ax.axvline(p['fork'],color='.45',lw=.6)
            for m in p['marks']:
                ax.axvline(m,color='.55',lw=.55,ls=':')
            ax.set_xlim(p['begin'],p['end'])
            ax.grid(alpha=.15,lw=.45)
            ax.tick_params(length=2.3,width=.6,pad=2)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4,integer=True))
            ax.margins(y=.10)
            if metric=='loss': log_axis(ax)
            if metric=='rho':
                ax.set_ylim(-1.08,1.08);ax.set_yticks([-1,0,1]);ax.axhline(0,color='.7',lw=.4)
            if metric=='applied_lr':
                if name.startswith('06'):
                    ymax=max(r['applied_lr'] for key,_,_ in p['series'] for r in DATA[key].values())
                    ax.set_ylim(0,1.07*ymax);ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
                else:
                    ax.set_ylim(.40,1.07);ax.set_yticks([.5,.75,1])
            if col==0:ax.set_ylabel(LABELS[metric])
            if row==0:
                ax.set_title(f'({chr(97+col)}) '+p['title'],pad=32,fontsize=9)
                handles=[Line2D([],[],color=c,lw=1,label=l) for _,l,c in p['series']]
                ax.legend(handles=handles,loc='lower left',bbox_to_anchor=(-.015,1.015),frameon=False,
                          fontsize=7.2,handlelength=1.8,labelspacing=.12,borderaxespad=0)
            if row==nr-1:
                ax.set_xlabel('Optimizer update' if name.startswith('06') else r'Parameter state $s$')
            if metric=='raw_gradient_norm' and p['clip']:
                ax.text(.98,.97,'Dashed: after clipping',ha='right',va='top',transform=ax.transAxes,fontsize=6.8)
    if subtitle:fig.text(.54,.978,subtitle,ha='center',va='top',fontsize=9.5)
    save(fig,name)


def main():
    FIG.mkdir(exist_ok=True); SCRATCH.mkdir(parents=True,exist_ok=True)
    load()
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix',
        'font.size':8.5,'axes.labelsize':9,'axes.titlesize':9,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'pdf.fonttype':42,
        'path.simplify':False})
    overview=source(analysis(55)/'figures/adam_full_trajectories_high.pdf')
    original=read(analysis(55)/'plot_manifest.json')
    assert sha(overview)==original['outputs']['figures/adam_full_trajectories_high.pdf']
    name='00-high-lr-overview-cifar10-mlp-resnet-wikitext2-transformer'
    shutil.copy2(overview,FIG/(name+'.pdf'))
    OUTPUTS[name]=dict(path=(FIG/(name+'.pdf')).relative_to(ROOT).as_posix(),sha256=sha(overview),
                       copied_from=overview.relative_to(ROOT).as_posix())
    with fitz.open(overview) as doc:doc[0].get_pixmap(matrix=fitz.Matrix(2,2)).save(SCRATCH/(name+'.png'))
    plot('01-persistent-half-lr-cifar10-mlp-resnet-wikitext2-transformer',[
        panel(title,[(case+'.control','Unchanged',GRAY),(case+'.half','Immediate half LR',BLUE)],begin,end,start)
        for case,title,begin,start,end in [('mlp','CIFAR-10 / MLP',448,512,800),
        ('resnet','CIFAR-10 / ResNet-32',192,256,576),('transformer','WikiText-2 / Transformer',480,544,928)]])
    plot('02-lr-floors-and-abrupt-return-cifar10-mlp',[
        panel('Cosine floor 0.75x',[('mlp.control','Unchanged',GRAY),('mlp.cosine75','Cosine32 to 0.75x',BLUE)]),
        panel('Cosine floor 0.5x',[('mlp.control','Unchanged',GRAY),('mlp.cosinehalf','Cosine32 to 0.5x',GREEN)]),
        panel('Immediate half, then return',[('mlp.half','Held at 0.5x',GRAY),('mlp.half_return','Return to 0.75x',ORANGE)],marks=(650,))
    ],metrics=['loss','eta_kappa_hat','raw_gradient_norm','applied_lr'],subtitle='CIFAR-10 / MLP')
    plot('03-lr-return-styles-cifar10-mlp',[
        panel(title,[(key,label,GRAY),('mlp.cosinerise','Cosine70 return',GREEN)],marks=(650,720))
        for title,key,label in [('Versus unchanged','mlp.control','Unchanged'),
            ('Versus held half','mlp.cosinehalf','Held at 0.5x'),
            ('Versus abrupt return','mlp.cosine_return','Abrupt return to 0.75x')]
    ],subtitle='CIFAR-10 / MLP: cosine32 to half, then return at state 650')
    plot('04-two-stage-decay-cifar10-mlp',[
        panel(title,[(key,label,GRAY),('mlp.twostage','Two-stage cosine decay',GREEN)],marks=(650,682))
        for title,key,label in [('Versus unchanged','mlp.control','Unchanged'),
                               ('Versus held 0.75x','mlp.cosine75','Cosine32 to 0.75x')]
    ],subtitle='CIFAR-10 / MLP: 1x to 0.75x, then 0.75x to 0.5x')
    plot('05-gradient-clipping-cifar10-mlp-resnet',[
        panel('CIFAR-10 / MLP',[('mlp.control','Unclipped',GRAY),('mlp.clip1','Cap 1.0',BLUE),('mlp.clip05','Cap 0.5',ORANGE)],clip=True),
        panel('CIFAR-10 / ResNet-32',[('resnet.spark_control','Unclipped',GRAY),('resnet.clip1','Cap 1.0',BLUE)],192,576,256,clip=True)
    ])
    plot('06-lr-schedules-minipile-pythia14m-70m',[
        panel('MiniPile / Pythia-'+size,[(f'pythia{size}.'+v,l,c) for v,l,c in
              [('original','Original decay',GRAY),('hold','Held midpoint LR',BLUE),('accelerated','Accelerated decay',GREEN)]],
              1,712,356) for size in ['14M','70M']
    ],subtitle='Sampled training diagnostics; base LR = midpoint LR')
    with (HERE/'plotted_values.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['curve','state','value'],lineterminator='\n');w.writeheader();w.writerows(VALUES)
    manifest=dict(sources=SOURCES,series=DEFINITIONS,curves=CURVES,outputs=OUTPUTS,
                  generator_sha256=sha(__file__),plotted_values_sha256=sha(HERE/'plotted_values.csv'),
                  points=len(VALUES),smoothing=None,figure00='Exact preserved Analysis055 PDF; its own scale/event contract applies')
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(f'{len(OUTPUTS)} figures; {len(CURVES)} curves; {len(VALUES)} plotted source points')


if __name__=='__main__':main()
