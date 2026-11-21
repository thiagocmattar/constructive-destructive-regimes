"""Revise Figures 5, 7, and 8 using unchanged retained measurements."""
from pathlib import Path
import hashlib
import importlib.util
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, FixedLocator, NullLocator, MaxNLocator
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
ROOT = BASE.parents[1]
DRAFT = BASE / 'draft'
SCRATCH = HERE / 'scratch'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest = {'sources': {}, 'figures': {}, 'loss_overlays': [], 'tradeoff_points': [], 'array_checks': []}


def source(path):
    manifest['sources'][path.relative_to(ROOT).as_posix()] = sha(path)
    return path


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, source(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v = load(BASE / 'work/figure_revision_20261114/01_figures.py', 'retained_style')
o = load(BASE / 'work/intervention_manuscript_20261114/01_prepare.py', 'retained_interventions')
plt.rcParams.update({'font.family':'Times New Roman', 'mathtext.fontset':'stix', 'font.size':8,
    'axes.titlesize':9, 'axes.labelsize':8, 'xtick.labelsize':7, 'ytick.labelsize':7,
    'legend.fontsize':7.5, 'axes.linewidth':.6, 'pdf.fonttype':42, 'ps.fonttype':42,
    'axes.spines.top':False, 'axes.spines.right':False, 'path.simplify':False})
v.XLABEL = r'Training step $t$'


def save(fig, name):
    fig.canvas.draw()
    box = fig.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans)
    assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.width+1 and box.y1 <= fig.bbox.height+1, (name, box)
    path = DRAFT / 'figures' / name
    fig.savefig(path, metadata={'CreationDate':None, 'ModDate':None})
    fig.savefig(SCRATCH / (path.stem+'.png'), dpi=200)
    manifest['figures'][name] = sha(path)
    plt.close(fig)


# Figure 5: notation only. Compare every numerical series with the retained export.
def overview_save(fig, name):
    for artist in fig.findobj(match=Text):
        label = re.sub(r'(?<![A-Za-z])s(?![A-Za-z])', 't', artist.get_text())
        label = label.replace(r'$\eta\widehat\kappa_t$', 'Projected\nresponse')
        artist.set_text(label)
    save(fig, name)


v.save = overview_save
v.adam_overview()
reference = np.load(source(BASE / 'work/figure_revision_20261114/plotted_arrays.npz'))
for key, value in v.PLOTTED.items():
    np.testing.assert_array_equal(value, reference[key], err_msg=key)
manifest['array_checks'].append({'figure':5, 'unchanged_arrays':len(v.PLOTTED)})
v.PLOTTED.clear()

# Figure 7: layerwise concentration and temporal overlap, then output reversal and loss.
data = v.trace_data()
resnet = v.read(ROOT / 'analyses/063-2026-11-07-architecture-instability-tracing/resnet_390_450_results.json')
motion_path = source(ROOT / 'analyses/063-2026-11-07-architecture-instability-tracing/resnet_390_450_motion.json')
motion = json.loads(motion_path.read_text())
np.testing.assert_array_equal(resnet['completed_steps'], motion['completed_steps'])
data['resnet']['result'] = resnet
data['resnet']['x'] = np.array(resnet['completed_steps'])
data['resnet']['loss'] = np.array(motion['loss'])
# Match the GD panel height in PDF points. Leave room between rows for x-axis labels.
panel_height = 64.19
row_gap, top_margin, bottom_margin = 37, 23, 48
figure_height = 2 * panel_height + row_gap + top_margin + bottom_margin
fig, axes = plt.subplots(2, 3, figsize=(7.16, figure_height / 72))
loss_axes = []
for col, case in enumerate(v.CASES):
    d = data[case]; r = d['result']; x = d['x']
    records = {s['name']:s for s in r['stages']}; out = d['names'][-1]
    y = np.arange(len(d['names']))
    for key,label,color,marker in [('top1',r'First direction $E_1$',v.COLORS[1],'o'),('top10',r'First ten $E_{10}$',v.COLORS[0],'s'),('temporal_top10_overlap',r'Temporal overlap $O_{10}$',v.COLORS[2],'^')]:
        values = np.array([records[n][key] for n in d['names']])
        v.curve(axes[0,col],values,y,f'spectral-stages:{case}:{key}',color=color,marker=marker,ms=3,lw=.7,label=label)
    axes[0,col].set(xlim=(0,1.03),xticks=[0,.5,1],ylim=(len(y)-.5,-.5),xlabel='Movement fraction / overlap')
    axes[0,col].set_yticks(y,[label.replace('Stage','Group') for label in d['labels']],fontsize=7)
    axes[0,col].set_title(f'({chr(97+col)}) {v.TITLES[col]}',pad=8)
    v.style(axes[0,col])
    alpha = np.array(r['curves'][out]['alpha'])/np.sqrt(records[out]['coordinates'])
    v.curve(axes[1,col],x,alpha,f'output-coefficient:{case}',color=v.COLORS[1],lw=.9)
    axes[1,col].axhline(0,color='.7',lw=.5)
    axes[1,col].set(xlim=(x[0],x[-1]),xlabel=v.XLABEL)
    axes[1,col].set_xticks({'mlp':[451,475,501],'resnet':[390,420,450],'transformer':[553,577,601]}[case])
    v.style(axes[1,col]); v.letter(axes[1,col],chr(100+col))
    axes[1,col].tick_params(axis='y',colors=v.COLORS[1])
    twin = axes[1,col].twinx(); loss_axes.append(twin)
    twin.spines['right'].set_visible(True)
    loss = np.asarray(d['loss'])
    v.curve(twin,x,loss,f'loss-overlay:{case}',color='.3',ls='--',lw=.9)
    peak = int(np.argmax(loss))
    twin.plot(x[peak],loss[peak],'o',color='.3',ms=3)
    twin.yaxis.set_major_locator(MaxNLocator(nbins=3))
    twin.yaxis.set_major_formatter(FuncFormatter(lambda value,_:f'{value:g}'))
    twin.tick_params(axis='y',colors='.3',length=2,pad=2)
    twin.set_xlim(x[0],x[-1])
    manifest['loss_overlays'].append({'case':case,'completed_window':[int(x[0]),int(x[-1])],
        'points':len(x),'peak_step':int(x[peak]),'peak_loss':float(loss[peak]),
        'loss_sha256':v.arrsha(loss),'clock_sha256':v.arrsha(x), 'scope':'full training objective; no smoothing'})
axes[1,0].set_ylabel(r'Output $\alpha_{t,1}/\sqrt{P}$', color=v.COLORS[1])
loss_axes[-1].set_ylabel('Training loss',color='.3',labelpad=4)
handles = axes[0,0].get_legend_handles_labels()[0]
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5,.005), ncol=3,
           frameon=False, columnspacing=1.4, handlelength=2)
fig.subplots_adjust(left=.12, right=.92, top=1-top_margin/figure_height,
                    bottom=bottom_margin/figure_height, hspace=row_gap/panel_height, wspace=.85)
save(fig,'01-three-model-temporal-structure.pdf')
reference = np.load(source(BASE / 'work/resnet_temporal_revision_20261114/plotted_arrays.npz'))
retained = {key for key in reference.files if not key.startswith('spectrum:')}
for key in retained:
    np.testing.assert_array_equal(v.PLOTTED[key],reference[key],err_msg=key)
assert set(v.PLOTTED) == retained | {f'loss-overlay:{c}:{xy}' for c in v.CASES for xy in ['x','y']}
manifest['array_checks'].append({'figure':7,'unchanged_arrays':len(retained),'added_loss_arrays':6})

# Figure 8: every point retains its original metrics and axes; highlight caps/control.
report = json.loads(source(BASE / 'evidence/05-adam/c-intervention/data/matched-results.json').read_text())
curves = np.load(source(BASE / 'evidence/05-adam/c-intervention/data/matched-curves.npz'))
rows = {(r['case'],r['arm']):r for r in report['summary']}
fig, axes = plt.subplots(1,3,figsize=(7.16,3.1))
offsets = {'mlp':{'control':(5,7),'cap_q25':(6,3),'cap_q50':(6,-3),'cap_q75':(6,-6)},
    'resnet':{'control':(5,10),'cap_q25':(8,0),'cap_q50':(8,-2),'cap_q75':(8,-6)},
    'transformer':{'control':(-34,-12),'cap_q25':(6,5),'cap_q50':(6,-3),'cap_q75':(6,-6)}}
for ax,case in zip(axes,o.CASES):
    threshold = .001 if case=='mlp' else .01
    xs = [rows[case,a]['final_loss'] for a in o.ARMS]
    ys = [rows[case,a]['positive_variation'] for a in o.ARMS]
    ax.set_xscale('log'); ax.set_xlim(min(xs)*.82,max(xs)*1.22)
    ax.set_yscale('symlog',linthresh=threshold,linscale=.65)
    ax.set_ylim(-threshold*.22,max(ys)*2.5)
    ax.axhline(rows[case,'control']['positive_variation'],color='.75',lw=.55)
    ax.axvline(rows[case,'control']['final_loss'],color='.75',lw=.55)
    for arm in o.ARMS:
        r=rows[case,arm]; x=r['final_loss']; y=r['positive_variation']
        loss=curves[f'{case}__{arm}__loss']
        np.testing.assert_allclose([x,y],[loss[-1],np.maximum(np.diff(loss),0).sum()],rtol=2e-12,atol=2e-12)
        cap=arm.startswith('cap_'); control=arm=='control'
        color=o.COLORS[arm] if cap else ('.15' if control else '.65')
        marker='x' if control else o.MARKERS[r['family']]
        artist=ax.scatter(x,y,s=30 if cap or control else 21,c=color,marker=marker,zorder=5 if cap or control else 3)
        np.testing.assert_array_equal(artist.get_offsets().data,[[x,y]])
        if cap or control:
            ax.annotate('Adam' if control else o.POINT_LABELS[arm],(x,y),xytext=offsets[case][arm],textcoords='offset points',
                fontsize=7.5,color=color,bbox=dict(fc='white',ec='none',alpha=.85,pad=.15))
        manifest['tradeoff_points'].append(dict(case=case,arm=arm,final_loss=x,positive_variation=y))
    ax.set_title(o.TITLES[case],pad=7); ax.set_xlabel(r'Final training loss $L_{t_f}$')
    ax.tick_params(length=2.5,pad=2); ax.yaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_locator(FixedLocator({'mlp':[.01,.02,.04],'resnet':[.005,.02,.1],'transformer':[.5,.7,1]}[case]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}')); ax.xaxis.set_minor_locator(NullLocator())
axes[0].set_ylabel(r'Upward loss variation $\mathcal{P}$')
handles=[Line2D([],[],marker=o.MARKERS[f],color='.65',ls='',label=label,markersize=4)
    for f,label in [('lr','LR reduction'),('beta2',r'$\beta_2$ reduction'),('clipping','Gradient clipping')]]
handles.append(Line2D([],[],marker='D',color=o.COLORS['cap_q50'],ls='',label='Displacement cap',markersize=4))
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.005),ncol=4,frameon=False,columnspacing=1.2)
fig.subplots_adjust(left=.085,right=.99,bottom=.22,top=.89,wspace=.35)
save(fig,'24-main-intervention-tradeoff.pdf')
prior = json.loads(source(BASE / 'work/author_feedback2_20261115/presentation-manifest.json').read_text())
assert manifest['tradeoff_points']==prior['points']
manifest['array_checks'].append({'figure':8,'unchanged_points':len(prior['points'])})
manifest['sources'].update(v.AUDIT['sources'])
manifest['generator_sha256']=sha(__file__)
(HERE/'figure-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'figures':list(manifest['figures']),'checks':manifest['array_checks'],'overlays':manifest['loss_overlays']},indent=2))
