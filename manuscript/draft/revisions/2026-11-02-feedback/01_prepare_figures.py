"""Rearrange verified evidence for the author's 28 PDF comments; no training."""
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator, FuncFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DRAFT = HERE.parents[1]
A24 = ROOT / 'analyses/024-2026-10-17-eos-gd-lr-calibration'
A27 = ROOT / 'analyses/027-2026-10-30-minibatch-noise'
A35 = ROOT / 'analyses/035-2026-11-02-eos-backward-anchors'
A37 = ROOT / 'analyses/037-2026-11-02-sgd-batch-extension'
TASKS = ['mlp', 'resnet', 'transformer']
TITLES = ['CIFAR-10 / MLP', 'CIFAR-10 / ResNet-32', 'WikiText-2 / Transformer']
COLORS = ['#0072B2', '#D55E00', '#009E73']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def arrsha(a):
    return hashlib.sha256(np.asarray(a, dtype='<f8').tobytes()).hexdigest()


def main():
    manifest = dict(generator=Path(__file__).relative_to(ROOT).as_posix(),
                    generator_sha256=sha(__file__), sources={}, figures=[], curves=[],
                    tables=[], visual_review='pending')
    def source(path):
        manifest['sources'][path.relative_to(ROOT).as_posix()] = sha(path)
        return path

    m24 = read(source(A24 / 'single_rows_figure_manifest.json'))
    arrays_path = source(ROOT / m24['arrays'])
    assert sha(arrays_path) == m24['arrays_sha256'] and m24['status'] == 'passed'
    arrays = np.load(arrays_path)
    results = read(source(A24 / 'results.json'))
    assert results['status'] == 'passed'
    runs = sorted(results['runs'], key=lambda r: (TASKS.index(r['task']), r['lr']))
    audit = read(source(A24 / 'section_iv_review_checks.json'))
    assert audit['status'] == 'passed'
    assert audit['sources']['analyses/024-2026-10-17-eos-gd-lr-calibration/results.json'] == sha(A24/'results.json')
    for r in audit['runs']:
        assert sha(ROOT/r['source']) == r['source_sha256']
    plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
                         'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .6, 'pdf.fonttype': 42, 'path.simplify': False})

    def curve(ax, x, y, tag, **kwargs):
        assert len(x) == len(y) and np.isfinite(y).all()
        line, = ax.plot(x, y, **kwargs)
        np.testing.assert_array_equal(line.get_xdata(), x)
        np.testing.assert_array_equal(line.get_ydata(), y)
        manifest['curves'].append(dict(tag=tag, points=len(x), x_sha256=arrsha(x), y_sha256=arrsha(y)))
        return line

    def style(ax, letter=None, label_y=1.025):
        if letter is not None:
            ax.text(0, label_y, f'({letter})', transform=ax.transAxes, va='bottom')
        ax.tick_params(length=3, width=.6)
        ax.grid(axis='y', alpha=.15, lw=.5)

    def save(fig, analysis, name):
        target = analysis/'figures'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, metadata={'CreationDate': None, 'ModDate': None})
        plt.close(fig)
        folder = DRAFT/'figures'/f'analysis{analysis.name[:3]}'
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(target, folder/name)
        manifest['figures'].append(dict(source=target.relative_to(ROOT).as_posix(),
            target=(folder/name).relative_to(ROOT).as_posix(), sha256=sha(target)))

    specs = [
        ('gd_loss_projected_change.pdf', [('loss_log', 'Training loss', 'log'),
          ('eta_kappa_hat_smoothed', r'$\eta\widehat\kappa_t$', 'linear')], 3.45),
        ('gd_reversal_norm_retention.pdf', [('rho_smoothed', r'$\rho_t$', 'linear'),
          ('gradient_norms_smoothed', r'$\|g_t\|$', 'log'),
          ('normalized_displacement', r'$R_t$', 'linear')], 4.85),
    ]
    for name, rows, height in specs:
        fig, axes = plt.subplots(len(rows), 3, figsize=(7.16, height), sharex=True)
        for col, task in enumerate(TASKS):
            subset = [r for r in runs if r['task'] == task]
            # The panel letter shares the title line; the legend has its own
            # compact line below it, without competing with a separate letter.
            axes[0, col].set_title(f'({chr(97+col)})  {TITLES[col]}', pad=15)
            for row, (metric, ylabel, scale) in enumerate(rows):
                ax = axes[row, col]
                for color, r in zip(COLORS, subset):
                    tag = f'{metric}_run{r["run_id"]}'
                    x, y = arrays[tag+'_x'], arrays[tag+'_y']
                    curve(ax, x, y, name+':'+tag, color=color, lw=.85,
                          label=rf'$\eta={r["lr"]:g}$')
                ax.set(xlim=(0, 3000), xticks=[0, 1000, 2000, 3000], yscale=scale)
                if metric == 'loss_log':
                    ax.yaxis.set_major_locator(FixedLocator([[.2,.3,.4,.6],[.4,.6,1,2],[6,7,8,9,10]][col]))
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:g}'))
                    ax.yaxis.set_minor_locator(NullLocator())
                if metric == 'eta_kappa_hat_smoothed':
                    ax.axhline(2, color='.35', lw=.7, ls='--', zorder=0)
                    ax.set(ylim=(-.08, 2.18), yticks=[0, 1, 2])
                if metric == 'rho_smoothed':
                    ax.axhline(0, color='.5', lw=.6, zorder=0)
                    ax.set(ylim=(-1.08, 1.08), yticks=[-1, 0, 1])
                if metric == 'normalized_displacement':
                    ax.set(ylim=(0, 1.04), yticks=[0, .5, 1])
                style(ax, chr(97+row*3+col) if row else None)
                if col == 0:
                    ax.set_ylabel(ylabel)
                if row == len(rows)-1:
                    ax.set_xlabel('Update index')
            axes[0,col].legend(loc='lower center', bbox_to_anchor=(.5, 1.015), borderaxespad=0,
                               ncol=3, fontsize=7, frameon=False, columnspacing=.7,
                               handlelength=1.1, handletextpad=.25)
        fig.subplots_adjust(left=.085, right=.985, top=1-32/(72*height),
                            bottom=.13 if len(rows)==2 else .10, hspace=.32, wspace=.29)
        save(fig, A24, name)

    v27 = read(source(A27/'verification.json'))
    assert v27['status'] == 'passed'
    assert sha(source(A27/'results.json')) == v27['results_sha256']
    reduced = read(source(A27/'reduction.json'))
    assert reduced['source_results_sha256'] == v27['results_sha256']
    assert sha(A27/'reduction.json') == read(source(A27/'scientific_summary.json'))['source_reduction_sha256']
    extension = read(source(A37/'results.json'))
    assert extension['status']=='passed'
    assert extension['verified_forks']==extension['expected_forks']==348
    coverage = read(source(A37/'cohort_acceptance.json'))
    assert coverage['status']=='passed' and coverage['results_sha256']==sha(A37/'results.json')
    batch_sets = {'mlp':[32,128,512,2048,5000], 'resnet':[1,5,20,50],
                  'transformer':[8,32,128,512,59680]}
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.15), sharey=True)
    sgd_values = []
    field = 'mean_rho'
    for col, task in enumerate(TASKS):
        batches=batch_sets[task]
        x=np.array(batches)*(1000 if task=='resnet' else 1)
        ax=axes[col]
        ax.set_title(f'({chr(97+col)})  {TITLES[col]}', pad=6)
        for checkpoint,color,marker in zip([0,2000],COLORS[:2],['o','s']):
            if task=='mlp':
                blocks=np.array([[next(r[field] for r in reduced['moving'] if
                    (r['seed'],r['checkpoint'],r['batch'])==(seed,checkpoint,b))
                    for b in batches] for seed in range(3)])
            else:
                blocks=np.array([next(r['seed_means'][field] for r in extension['cells'] if
                    (r['task'],r['state'],r['batch'])==(task,checkpoint,b)) for b in batches]).T
            assert blocks.shape==(3,len(batches))
            mean,low,high=blocks.mean(0),blocks.min(0),blocks.max(0)
            ax.fill_between(x,low,high,color=color,alpha=.14,lw=0)
            curve(ax,x,mean,f'sgd:{task}:{checkpoint}:{field}',color=color,marker=marker,
                  ms=3.2,lw=1,label='At initialization' if checkpoint==0 else 'After 2,000 GD updates')
            ax.plot(x[-1],mean[-1],marker=marker,ms=4,color=color,mfc='white',lw=0)
            sgd_values.append(dict(task=task,checkpoint=checkpoint,field=field,batches=batches,
                plotted_batches=x.tolist(),initialization_means=blocks.tolist(),mean=mean.tolist(),
                minimum=low.tolist(),maximum=high.tolist()))
        ax.set(xscale='log',xlim=(x[0]/1.2,x[-1]*(2 if task=='transformer' else 1.4)),
               ylim=(-1.08,1.08),yticks=[-1,0,1])
        ax.xaxis.set_major_locator(FixedLocator(x))
        labels=[f'{b:,}' for b in x];labels[-1]+='\n(full)'
        ax.xaxis.set_major_formatter(FixedFormatter(labels))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis='x',labelsize=7.5)
        ax.axhline(0,color='.5',lw=.6,zorder=0)
        ax.set_xlabel(r'Batch size $b$ ('+('sequences' if task=='transformer' else 'images')+')')
        if col==0: ax.set_ylabel(r'Mean cosine $\rho_t$')
        style(ax)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.53,1.005),
               ncol=2,frameon=False,fontsize=9,columnspacing=2,handlelength=2)
    fig.subplots_adjust(left=.085,right=.985,top=.76,bottom=.28,wspace=.18)
    save(fig,A37,'sgd_batch_size_compact.pdf')
    manifest['sgd_values']=sgd_values

    profiles = read(source(A35/'results.json'))['profiles']
    assert read(source(A35/'verification.json'))['status']=='passed'
    fig,axes=plt.subplots(2,3,figsize=(7.16,3.30),sharey=True)
    summaries=[]
    for col,task in enumerate(TASKS):
        p=next(p for p in profiles if (p['job']['optimizer'],p['job']['task'],p['job']['start'])==('gd',task,2000))
        x=np.array([r['lag'] for r in p['rows']]); y=np.array([r['cosine'] for r in p['rows']])
        assert p['endpoint']==2256 and np.array_equal(x,np.arange(1,257))
        for row,limit in enumerate([256,32]):
            ax=axes[row,col]
            curve(ax,x[:limit],y[:limit],('backward:' if row==0 else 'backward_zoom:')+task,
                  color=COLORS[0],lw=.55 if row==0 else .7,
                  marker=None if row==0 else 'o',ms=1.8)
            ax.set(xlim=(1,limit),ylim=(-1.05,1.05),
                   xticks=[1,64,128,192,256] if row==0 else [1,8,16,24,32],
                   yticks=[-1,0,1],xlabel=r'Backward lag $k$'+(' (full range)' if row==0 else ' (zoom)'))
            ax.axhline(0,color='.5',lw=.6,zorder=0)
            style(ax,chr(97+row*3+col) if row else None)
            if col==0: ax.set_ylabel(r'$\cos(g_t,g_{t-k})$')
        axes[0,col].set_title(f'({chr(97+col)})  {TITLES[col]}',pad=6)
    fig.subplots_adjust(left=.085,right=.985,top=.90,bottom=.13,wspace=.2,hspace=.62)
    save(fig,A35,'gd_backward_compact.pdf')
    for task in TASKS:
        for start in [500,2000,3000]:
            p=next(p for p in profiles if (p['job']['optimizer'],p['job']['task'],p['job']['start'])==('gd',task,start))
            y=np.array([r['cosine'] for r in p['rows']])
            summaries.append(dict(task=task,state=p['endpoint'],rho1=float(y[0]),
                                  min_last64=float(y[-64:].min()),max_last64=float(y[-64:].max())))
    manifest['backward_all_endpoints']=summaries

    labels={'mlp':'MLP','resnet':'ResNet-32','transformer':'Transformer'}
    lines=[r'% Generated by revisions/2026-11-02-feedback/01_prepare_figures.py.',
           r'\begin{tabular}{@{}lllrrrrr@{}}',r'\toprule',
           r'Dataset & Model & $\eta$ & $\mathcal L(\theta_T)$ & $\overline\rho_{\rm late}$ & $R_T$ & $\overline{\eta\widehat\kappa}_{\rm late}$ & $\|\Delta_T\|$ \\',r'\midrule']
    for i,r in enumerate(runs):
        if i==3: lines.append(r'\cmidrule(l){2-8}')
        if i==6: lines.append(r'\midrule')
        a=next(a for a in audit['runs'] if a['run_id']==r['run_id'])
        q=r['late_1000_q_mean']; qtext=f'{q:.4f}'+(r'$^{*}$' if 0<abs(q)<.00005 else '')
        dataset=r'\multirow{6}{*}{CIFAR-10}' if i==0 else (r'\multirow{3}{*}{WikiText-2}' if i==6 else '')
        model=rf'\multirow{{3}}{{*}}{{{labels[r["task"]]}}}' if i%3==0 else ''
        cells=[dataset,model,f'{r["lr"]:g}',
               f'{r["final_loss"]:.4f}',f'{r["late_1000_cosine_mean"]:.4f}',f'{r["final_B_over_A"]:.4f}',qtext,
               f'{a["final_gradient_sum_displacement"]:.4f}']
        lines.append(' & '.join(cells)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}','']
    table=DRAFT/'tables/analysis024_endpoints.tex';table.write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    manifest['tables'].append(dict(path=table.relative_to(ROOT).as_posix(),sha256=sha(table),rows=9))
    (HERE/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(figures=len(manifest['figures']),curves=len(manifest['curves']),
                          plotted_points=sum(c['points'] for c in manifest['curves']))))


if __name__ == '__main__':
    main()
