"""Loss, alignment overlay and fixed example-preserving SVD coefficients."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
RAW=ROOT/'runs/239-2026-11-07-mlp-output-change-alignment/artifacts/spark/payload/worker'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
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
    # D[t] is the outgoing output displacement at state t, preserving all examples.
    displacement=(saved['outputs'][2:].double()-saved['outputs'][1:-1].double()).flatten(1)
    assert tuple(displacement.shape)==(353,50000)
    gram=displacement@displacement.T
    eigenvalues,left=torch.linalg.eigh(gram)
    eigenvalues=eigenvalues.flip(0);left=left.flip(1)
    assert float(eigenvalues.min())>-1e-10*float(eigenvalues.max())
    eigenvalues=eigenvalues.clamp_min(0);singular=eigenvalues[:10].sqrt()
    patterns=(left[:,:10].T@displacement)/singular[:,None]
    coefficients=left[:,:10]*singular[None,:]
    # A fixed deterministic sign per pattern, not a sign choice made per state.
    for k in range(10):
        if patterns[k,int(patterns[k].abs().argmax())]<0:
            patterns[k].neg_();coefficients[:,k].neg_()
    fractions=eigenvalues/eigenvalues.sum()
    retained=float(fractions[:10].sum())
    matrix=(coefficients/5000**.5).T.numpy()
    # Check basis, independent direct projection, energy and reconstruction.
    orthogonality_error=float((patterns@patterns.T-torch.eye(10,dtype=torch.float64)).abs().max())
    direct=displacement@patterns.T
    projection_error=float((direct-coefficients).abs().max())
    residual_energy=0.
    for start in range(0,353,32):
        residual=displacement[start:start+32]-coefficients[start:start+32]@patterns
        residual_energy+=float(residual.square().sum())
    total_energy=float(displacement.square().sum())
    assert orthogonality_error<1e-10 and projection_error<1e-10
    assert abs(residual_energy/total_energy-(1-retained))<1e-12
    assert abs(float(gram.trace())-total_energy)<1e-8
    # All source coordinates enter the fit. No averaging, centering or row scaling.
    artifacts=HERE/'artifacts/svd-output-changes';artifacts.mkdir(parents=True,exist_ok=True)
    pattern_path=artifacts/'patterns.pt'
    torch.save(dict(states=states.tolist(),patterns=patterns.reshape(10,5000,10),coefficients=coefficients,
        singular_values=eigenvalues.sqrt(),energy_fractions=fractions,
        labels=saved['labels'],source_sha256=sha(paths[0]),
        sign_rule='largest-absolute pattern coordinate positive'),pattern_path)
    values_path=HERE/'svd_output_heatmap_values.csv'
    with values_path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['state']+[f'mode_{i+1:02d}_coefficient_over_sqrtN' for i in range(10)])
        writer.writerows([[int(t),*matrix[:,j].tolist()] for j,t in enumerate(states)])
    limit=float(np.max(np.abs(matrix)));vmin=-limit;vmax=limit
    norm=TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax);cmap='RdBu_r'
    subtitle=f'Fixed SVD basis; all 5,000 examples retained; top 10 patterns capture {100*retained:.2f}% of change energy'
    ylabel='SVD pattern (share of full-window change energy)'
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','pdf.fonttype':42,
        'font.size':11,'axes.labelsize':12,'axes.titlesize':12,'xtick.labelsize':10,'ytick.labelsize':10})
    fig=plt.figure(figsize=(11.5,15.5))
    grid=fig.add_gridspec(3,2,width_ratios=[1,.025],height_ratios=[1,1.2,5.2],
        left=.17,right=.93,bottom=.105,top=.925,hspace=.17,wspace=.045)
    ax_loss=fig.add_subplot(grid[0,0]);ax_align=fig.add_subplot(grid[1,0],sharex=ax_loss)
    ax_heat=fig.add_subplot(grid[2,0],sharex=ax_loss);cax=fig.add_subplot(grid[2,1])
    fig.suptitle('CIFAR-10 MLP: loss, temporal alignment and SVD output-change patterns',fontsize=16,y=.98)
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
    ax_heat.set_title('(c) Signed coefficients of fixed SVD output-change patterns',loc='left')
    nrows=matrix.shape[0]
    # interpolation='none' embeds the complete source raster in the vector PDF.
    heat=ax_heat.imshow(matrix,cmap=cmap,norm=norm,aspect='auto',interpolation='none',
        extent=[447.5,800.5,nrows-.5,-.5])
    np.testing.assert_array_equal(np.asarray(heat.get_array()),matrix)
    ax_heat.set_yticks(range(10),[f'{i+1} ({100*float(fractions[i]):.2f}%)' for i in range(10)])
    ax_heat.set_ylabel(ylabel);ax_heat.set_xlabel('Completed optimizer updates, t')
    ax_heat.set_xlim(447.5,800.5);ax_heat.set_xticks([448,512,576,640,704,768,800])
    bar=fig.colorbar(heat,cax=cax);bar.set_label(r'Signed mode coefficient $a_{t,k}/\sqrt{5000}$')
    bar.solids.set_rasterized(False)
    bar.solids.set_edgecolor('face')
    fig.text(.515,.05,r'$\rho_t=\cos(g_t,g_{t-1}),\quad \rho_t^{(\Delta F)}=\cos(F_{t+1}-F_t,F_t-F_{t-1})$',ha='center',fontsize=11)
    fig.text(.515,.035,r'$D_t=F_{t+1}-F_t\approx\sum_{k=1}^{10}a_{t,k}v_k;\quad v_k\in\mathbb{R}^{50000}$ is fixed across updates.',ha='center',fontsize=10)
    fig.text(.515,.017,'No example averaging. Signs share one fixed orientation per pattern; colors use one symmetric scale.',ha='center',fontsize=9)
    target=HERE/'figures/05-network-output-heatmap.pdf';fig.savefig(target);plt.close(fig)
    result=dict(status='numeric_checks_passed',representation='fixed_full_example_output_change_svd',
        state_range=[448,800],source_output_states=[448,801],columns=353,rows=10,heatmap_values=3530,line_values=1059,
        matrix_shape=[353,50000],centered=False,example_averaging=False,
        reduction='D_t=flatten(F_(t+1)-F_t); fixed uncentered SVD over all353 updates. Plot a_(t,k)/sqrt(5000), top10 modes.',
        sign_rule='Largest-absolute coordinate in each fixed50000-dimensional pattern is positive.',
        row_order='Descending full-window squared singular value',energy_fractions=fractions.tolist(),
        retained_energy_fraction=retained,discarded_energy_fraction=1-retained,
        orthogonality_max_absolute_error=orthogonality_error,projection_max_absolute_error=projection_error,
        reconstruction_relative_squared_error=residual_energy/total_energy,
        heatmap_range=[float(matrix.min()),float(matrix.max())],
        color_normalization=dict(kind='TwoSlopeNorm',center=0,vmin=vmin,vmax=vmax,clipped_values=0),
        rendering='Complete10x353 coefficient raster embedded with interpolation=none; vector axes and lines.',
        formulas=dict(gradient='cos(g_t,g_(t-1))',output_change='cos(F_(t+1)-F_t,F_t-F_(t-1))'),
        sources={p.relative_to(ROOT).as_posix():sha(p) for p in paths},generator_sha256=sha(Path(__file__)),
        values_csv=dict(path=values_path.name,sha256=sha(values_path)),
        patterns=dict(path=pattern_path.relative_to(HERE).as_posix(),bytes=pattern_path.stat().st_size,sha256=sha(pattern_path)),
        pdf=dict(path=target.relative_to(HERE).as_posix(),bytes=target.stat().st_size,sha256=sha(target)))
    (HERE/'output_heatmap_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
