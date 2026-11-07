"""Loss, alignment overlay and output-coordinate heatmap on a shared time axis."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
RAW=ROOT/'runs/239-2026-11-07-mlp-output-change-alignment/artifacts/spark/payload/worker'
CLASSES=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main(representation,example_index):
    torch.set_num_threads(4)
    audit=json.loads((HERE/'control_coherence_verification.json').read_text())
    assert audit['status']=='passed'
    paths=[RAW/'outputs.pt',RAW/'boundaries.jsonl',HERE/'control_coherence_448_800.csv']
    wanted=[audit['output_alignment']['sources']['outputs.pt'],audit['output_alignment']['sources']['boundaries.jsonl'],audit['csv']['sha256']]
    for p,h in zip(paths,wanted):assert sha(p)==h
    saved=torch.load(paths[0],weights_only=False)
    assert saved['states']==list(range(447,802)) and tuple(saved['outputs'].shape)==(355,5000,10)
    states=np.arange(448,801);outputs=saved['outputs'][1:-1].numpy()
    boundaries={r['state']:r for r in map(json.loads,paths[1].read_text().splitlines())}
    loss=np.array([boundaries[int(t)]['loss'] for t in states])
    with paths[2].open() as f:rows=list(csv.DictReader(f))
    assert [int(r['state']) for r in rows]==states.tolist()
    gradient=np.array([float(r['rho']) for r in rows]);alignment=np.array([float(r['output_change_cosine']) for r in rows])
    if representation=='all':
        matrix=outputs.reshape(353,50000).T
        reduction='None: row r=10*i+c is output c of fixed example i (zero-based).'
        subtitle='All 50,000 output coordinates; fixed example-major row order'
        ylabel='Output coordinate r = 10i + c (example i, output c)'
    elif representation=='mean':
        matrix=outputs.astype(np.float64).mean(axis=1).T
        reduction='Arithmetic mean across all5000 fixed examples, separately per output coordinate and state.'
        subtitle='10 output dimensions, averaged across all 5,000 examples'
        ylabel='Output dimension'
    else:
        assert 0<=example_index<5000
        matrix=outputs[:,example_index,:].T
        reduction=f'One fixed example, zero-based index {example_index}; all10 raw outputs.'
        subtitle=f'Fixed example {example_index} (label: {CLASSES[int(saved["labels"][example_index])]}); all 10 outputs'
        ylabel='Output dimension'
    assert matrix.shape[1]==353 and np.isfinite(matrix).all()
    values_path=None
    if representation=='mean':
        values_path=HERE/'mean_output_heatmap_values.csv'
        with values_path.open('w',encoding='utf-8',newline='') as f:
            writer=csv.writer(f,lineterminator='\n');writer.writerow(['state']+[f'output_{i}' for i in range(10)])
            writer.writerows([[int(t),*matrix[:,j].tolist()] for j,t in enumerate(states)])
    # No centering, per-row scaling, normalization, clipping or output differencing.
    vmin=float(matrix.min());vmax=float(matrix.max())
    diverging=vmin<0<vmax
    norm=TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax) if diverging else Normalize(vmin=vmin,vmax=vmax)
    cmap='RdBu_r' if diverging else 'viridis'
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','pdf.fonttype':42,
        'font.size':11,'axes.labelsize':12,'axes.titlesize':12,'xtick.labelsize':10,'ytick.labelsize':10})
    fig=plt.figure(figsize=(11.5,15.5))
    grid=fig.add_gridspec(3,2,width_ratios=[1,.025],height_ratios=[1,1.2,5.2],
        left=.12,right=.93,bottom=.09,top=.925,hspace=.17,wspace=.045)
    ax_loss=fig.add_subplot(grid[0,0]);ax_align=fig.add_subplot(grid[1,0],sharex=ax_loss)
    ax_heat=fig.add_subplot(grid[2,0],sharex=ax_loss);cax=fig.add_subplot(grid[2,1])
    fig.suptitle('CIFAR-10 MLP: loss, temporal alignment and network outputs',fontsize=16,y=.98)
    fig.text(.515,.955,'Unchanged Adam control | States 448-800 | '+subtitle,ha='center',fontsize=10)
    line,=ax_loss.plot(states,loss,color='#2166ac',lw=1.2)
    np.testing.assert_array_equal(line.get_ydata(),loss)
    ax_loss.set_title('(a) Full training loss',loc='left');ax_loss.set_ylabel(r'$L_t$')
    for values,label,color,ls in [(gradient,r'Gradient $\rho_t$','#2166ac','-'),
                                 (alignment,r'Output changes $\rho_t^{(\Delta F)}$','#b2182b','--')]:
        line,=ax_align.plot(states,values,label=label,color=color,ls=ls,lw=1.15)
        np.testing.assert_array_equal(line.get_ydata(),values)
    ax_align.set_title('(b) Temporal alignment overlay',loc='left');ax_align.set_ylabel('Cosine alignment')
    ax_align.set_ylim(-1.06,1.06);ax_align.axhline(0,color='gray',ls=':',lw=.6)
    ax_align.legend(frameon=False,loc='upper left',fontsize=10)
    for ax in [ax_loss,ax_align]:
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15,lw=.6);ax.tick_params(labelbottom=False)
    ax_heat.set_title('(c) Mean raw output across the 5,000 examples' if representation=='mean' else '(c) Raw network outputs',loc='left')
    nrows=matrix.shape[0]
    # interpolation='none' embeds the complete source raster in the vector PDF.
    heat=ax_heat.imshow(matrix,cmap=cmap,norm=norm,aspect='auto',interpolation='none',
        extent=[447.5,800.5,nrows-.5,-.5])
    np.testing.assert_array_equal(np.asarray(heat.get_array()),matrix)
    if representation=='all':ax_heat.set_yticks([0,9999,19999,29999,39999,49999],['0','9,999','19,999','29,999','39,999','49,999'])
    else:ax_heat.set_yticks(range(10),[f'{i}: {name}' for i,name in enumerate(CLASSES)])
    ax_heat.set_ylabel(ylabel);ax_heat.set_xlabel('Completed optimizer updates, t')
    ax_heat.set_xlim(447.5,800.5);ax_heat.set_xticks([448,512,576,640,704,768,800])
    bar=fig.colorbar(heat,cax=cax);bar.set_label('Mean raw output value' if representation=='mean' else 'Raw output value')
    bar.solids.set_rasterized(False)
    bar.solids.set_edgecolor('face')
    fig.text(.515,.05,r'$\rho_t=\cos(g_t,g_{t-1}),\quad \rho_t^{(\Delta F)}=\cos(F_{t+1}-F_t,F_t-F_{t-1})$',ha='center',fontsize=11)
    footer=('All coordinates are embedded; individual rows require zooming. No averaging, sorting or value clipping.'
            if representation=='all' else reduction+' No value clipping or row normalization.')
    fig.text(.515,.025,footer,ha='center',fontsize=9)
    target=HERE/'figures/05-network-output-heatmap.pdf';fig.savefig(target);plt.close(fig)
    result=dict(status='numeric_checks_passed',representation=representation,example_index=example_index if representation=='example' else None,
        state_range=[448,800],columns=353,rows=nrows,heatmap_values=int(matrix.size),line_values=1059,
        reduction=reduction,row_order='example-major, output index within each example' if representation=='all' else 'output class index0..9',
        heatmap_quantity='Mean raw final linear-layer output across examples' if representation=='mean' else 'Raw final linear-layer output; not probability, residual or output change',
        heatmap_range=[float(matrix.min()),float(matrix.max())],color_normalization=dict(kind='TwoSlopeNorm' if diverging else 'Normalize',center=0 if diverging else None,vmin=vmin,vmax=vmax,clipped_values=0),
        rendering='Full-resolution heatmap raster embedded with interpolation=none; vector axes and lines. Screen previews may resample rows.',
        formulas=dict(gradient='cos(g_t,g_(t-1))',output_change='cos(F_(t+1)-F_t,F_t-F_(t-1))'),
        sources={p.relative_to(ROOT).as_posix():sha(p) for p in paths},generator_sha256=sha(Path(__file__)),
        values_csv=dict(path=values_path.name,sha256=sha(values_path)) if values_path else None,
        pdf=dict(path=target.relative_to(HERE).as_posix(),bytes=target.stat().st_size,sha256=sha(target)))
    (HERE/'output_heatmap_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--representation',choices=['all','mean','example'],default='mean')
    parser.add_argument('--example-index',type=int,default=0)
    args=parser.parse_args();main(args.representation,args.example_index)
