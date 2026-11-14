"""Update the temporal companion with a newly fitted ResNet 390--450 window."""
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OLD = HERE.parent / 'figure_revision_20261114'
NAME = '01-three-model-temporal-structure.pdf'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value): path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def main():
    spec = importlib.util.spec_from_file_location('figure_style', OLD / '01_figures.py')
    v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
    v.OUT = HERE / 'figures'; v.OUT.mkdir(exist_ok=True)
    if not (HERE / 'preservation.json').exists():
        evidence = ROOT / 'manuscript/consolidation/evidence'
        keep = list((evidence / '05-adam/c-intervention').rglob('*'))
        keep += [p for p in evidence.rglob('*.pdf') if p.name != NAME]
        plan = (ROOT / 'manuscript/consolidation/PLAN.md').read_bytes()
        section = plan[plan.index(b'### C. Intervention'):plan.index(b'## VI.')]
        write(HERE / 'preservation.json', dict(files={p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(keep)) if p.is_file()}, plan_intervention_sha256=hashlib.sha256(section).hexdigest()))
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','font.size':9,
        'axes.titlesize':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'pdf.fonttype':42,'path.simplify':False})
    data = v.trace_data()
    result = v.read(ROOT / 'analyses/063-2026-11-07-architecture-instability-tracing/resnet_390_450_results.json')
    assert result['status'] == 'verified'
    data['resnet']['result'] = result
    data['resnet']['x'] = np.array(result['completed_steps'])
    fig, axes = plt.subplots(3, 3, figsize=(7.16, 5.65), gridspec_kw={'height_ratios':[1,1.25,1]})
    for col, case in enumerate(v.CASES):
        d=data[case]; r=d['result']; x=d['x']; records={s['name']:s for s in r['stages']}; out=d['names'][-1]
        for name,label,color,ls in [(out,'Output changes',v.COLORS[1],'-'),('raw_gradient','Raw gradients',v.COLORS[0],'--')]:
            values=np.asarray(r['curves'][name]['spectrum']); assert abs(values.sum()-1)<1e-9
            v.curve(axes[0,col],np.arange(1,len(values)+1),np.cumsum(values),f'spectrum:{case}:{name}',color=color,ls=ls,lw=1,label=label)
        axes[0,col].axhline(.95,color='.5',ls=':',lw=.6)
        axes[0,col].set(xlim=(1,len(x)),ylim=(0,1.03),yticks=[0,.5,1],xlabel='Number of modes')
        axes[0,col].set_title(f'({chr(97+col)}) {v.TITLES[col]}',pad=8); v.style(axes[0,col])
        y=np.arange(len(d['names']))
        for key,label,color,marker in [('top1','Top-1 energy',v.COLORS[1],'o'),('top10','Top-10 energy',v.COLORS[0],'s'),('temporal_top10_overlap','Output overlap',v.COLORS[2],'^')]:
            values=np.array([records[n][key] for n in d['names']])
            v.curve(axes[1,col],values,y,f'spectral-stages:{case}:{key}',color=color,marker=marker,ms=3,lw=.7,label=label)
        axes[1,col].set(xlim=(0,1.03),xticks=[0,.5,1],ylim=(len(y)-.5,-.5),xlabel='Energy fraction / overlap')
        axes[1,col].set_yticks(y,d['labels'],fontsize=7.5); v.style(axes[1,col]); v.letter(axes[1,col],chr(100+col))
        alpha=np.array(r['curves'][out]['alpha'])/np.sqrt(records[out]['coordinates'])
        v.curve(axes[2,col],x,alpha,f'output-coefficient:{case}',color=v.COLORS[1],lw=.9)
        axes[2,col].axhline(0,color='.5',lw=.6); axes[2,col].set(xlim=(x[0],x[-1]),xlabel=v.XLABEL)
        if case == 'resnet': axes[2,col].set_xticks([390,405,420,450])
        v.style(axes[2,col]); v.letter(axes[2,col],chr(103+col))
        axes[2,col].ticklabel_format(axis='y',style='sci',scilimits=(-2,2),useMathText=True)
    axes[0,0].set_ylabel('Cumulative energy'); axes[2,0].set_ylabel(r'Output $\alpha_1/\sqrt{P}$')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.53,1.01),ncol=2,frameon=False,fontsize=7.5)
    fig.legend(*axes[1,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.53,.002),ncol=3,frameon=False,fontsize=7.5)
    fig.subplots_adjust(left=.13,right=.97,top=.91,bottom=.13,hspace=.70,wspace=.55)
    v.save(fig,NAME)
    np.savez_compressed(HERE / 'plotted_arrays.npz', **v.PLOTTED)
    write(HERE / 'figure_manifest.json',dict(status='numerical_checks_passed',generator_sha256=sha(__file__),
        style_generator_sha256=sha(OLD / '01_figures.py'),sources=v.AUDIT['sources'],curves=v.AUDIT['curves'],figures=v.AUDIT['figures'],
        completed_windows={c:[int(data[c]['x'][0]),int(data[c]['x'][-1])] for c in v.CASES},
        plotted_arrays_sha256=sha(HERE / 'plotted_arrays.npz'),scope='Only the ResNet temporal column changes; all its spectra, overlap and coefficients are refitted for completed steps390--450.'))
    print(json.dumps(dict(figure=NAME,completed_steps=[390,450])))


if __name__ == '__main__': main()
