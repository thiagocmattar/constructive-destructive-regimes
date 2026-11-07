"""Same-state temporal overlay and scatter of gradient/output-change alignment."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    source=HERE/'control_coherence_448_800.csv'
    audit=json.loads((HERE/'control_coherence_verification.json').read_text())
    assert audit['status']=='passed' and sha(source)==audit['csv']['sha256']
    with source.open() as f:records=list(csv.DictReader(f))
    states=np.array([int(r['state']) for r in records])
    gradient=np.array([float(r['rho']) for r in records])
    output=np.array([float(r['output_change_cosine']) for r in records])
    assert np.array_equal(states,np.arange(448,801))
    assert np.isfinite(gradient).all() and np.isfinite(output).all()
    refs=[ROOT/'runs/238-2026-11-07-mlp-intrabatch-coherence/artifacts/spark/payload/worker/boundaries.jsonl',
          ROOT/'runs/239-2026-11-07-mlp-output-change-alignment/artifacts/spark/payload/worker/output_alignment.jsonl']
    expected_hashes=[audit['sources']['boundaries.jsonl'],audit['output_alignment']['sources']['output_alignment.jsonl']]
    for path,expected_hash,values,key in zip(refs,expected_hashes,[gradient,output],['rho','output_change_cosine']):
        assert sha(path)==expected_hash
        by_state={r['state']:r for r in map(json.loads,path.read_text().splitlines())}
        np.testing.assert_allclose(values,[by_state[int(t)][key] for t in states],rtol=1e-12,atol=1e-14)
    correlation=float(np.corrcoef(gradient,output)[0,1])
    mae=float(np.mean(np.abs(gradient-output)))
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','pdf.fonttype':42,
        'font.size':11,'axes.titlesize':12,'axes.labelsize':12,'xtick.labelsize':10,'ytick.labelsize':10})
    fig=plt.figure(figsize=(13,5.4))
    grid=fig.add_gridspec(1,2,width_ratios=[1.8,1],left=.065,right=.95,bottom=.23,top=.81,wspace=.25)
    time_ax=fig.add_subplot(grid[0]);scatter_ax=fig.add_subplot(grid[1])
    fig.suptitle('CIFAR-10 MLP: gradient alignment and output-change alignment',fontsize=16,y=.96)
    fig.text(.5,.895,'Unchanged Adam control | 353 paired states, 448-800 | Same t, no lag shift or smoothing',ha='center',fontsize=11)
    traces=[]
    for values,label,color,ls in [(gradient,r'Gradient $\rho_t$','#2166ac','-'),
                                 (output,r'Output changes $\rho_t^{(\Delta F)}$','#b2182b','--')]:
        line,=time_ax.plot(states,values,label=label,color=color,ls=ls,lw=1.15)
        np.testing.assert_array_equal(line.get_xdata(),states);np.testing.assert_array_equal(line.get_ydata(),values)
        traces.append(line)
    time_ax.set_title('(a) Temporal overlay',loc='left');time_ax.set_xlabel('Completed optimizer updates, t')
    time_ax.set_ylabel('Cosine alignment');time_ax.set_xlim(448,800);time_ax.set_ylim(-1.06,1.06)
    time_ax.set_xticks([448,512,576,640,704,768,800]);time_ax.axhline(0,color='gray',lw=.7,ls=':')
    time_ax.legend(loc='upper left',frameon=False,fontsize=10)
    points=scatter_ax.scatter(gradient,output,c=states,cmap='viridis',vmin=448,vmax=800,s=24,alpha=.8,edgecolors='none')
    np.testing.assert_array_equal(np.asarray(points.get_offsets()),np.column_stack([gradient,output]))
    scatter_ax.plot([-1,1],[-1,1],color='gray',ls='--',lw=.8,zorder=0)
    scatter_ax.set_title('(b) Same-state comparison',loc='left')
    scatter_ax.set_xlabel(r'Gradient alignment $\rho_t$');scatter_ax.set_ylabel(r'Output-change alignment $\rho_t^{(\Delta F)}$')
    scatter_ax.set_xlim(-1.06,1.06);scatter_ax.set_ylim(-1.06,1.06);scatter_ax.set_aspect('equal',adjustable='box')
    scatter_ax.text(.05,.95,f'Pearson r = {correlation:.3f}\nDashed line: equality',transform=scatter_ax.transAxes,va='top',fontsize=10)
    bar=fig.colorbar(points,ax=scatter_ax,fraction=.045,pad=.035);bar.set_label('State t',fontsize=10)
    bar.solids.set_rasterized(False)
    bar.set_ticks([448,512,576,640,704,768,800]);bar.ax.tick_params(labelsize=9)
    for ax in [time_ax,scatter_ax]:
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15,lw=.6)
    fig.text(.5,.115,r'$\rho_t=\cos(g_t,g_{t-1}),\qquad \rho_t^{(\Delta F)}=\cos(F_{t+1}-F_t,\,F_t-F_{t-1})$',ha='center',fontsize=12)
    fig.text(.5,.055,'F concatenates all 10 raw outputs for all 5,000 examples. Correlation is descriptive for this dependent time series.',ha='center',fontsize=10)
    target=HERE/'figures/04-gradient-output-alignment-comparison.pdf'
    fig.savefig(target);plt.close(fig)
    result=dict(status='numeric_checks_passed',states=353,state_range=[448,800],pairing='same state t; no lag shift',
        temporal_values=706,scatter_points=353,plotted_scalar_values=1412,pearson_correlation=correlation,
        mean_absolute_alignment_difference=mae,statistical_inference='Descriptive only; dependent states from one seed, no IID test or confidence interval.',
        sources={source.name:sha(source),**{p.relative_to(ROOT).as_posix():sha(p) for p in refs}},
        formulas=dict(gradient='cos(g_t,g_(t-1))',output_change='cos(F_(t+1)-F_t,F_t-F_(t-1))'),
        generator_sha256=sha(Path(__file__)),pdf=dict(path=target.relative_to(HERE).as_posix(),bytes=target.stat().st_size,sha256=sha(target)))
    (HERE/'alignment_comparison_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
