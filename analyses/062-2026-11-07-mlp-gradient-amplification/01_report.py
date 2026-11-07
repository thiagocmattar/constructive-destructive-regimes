"""Reduce complete retained diagnostics and produce two vector research figures."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
RAW=ROOT/'runs/237-2026-11-07-mlp-gradient-amplification/artifacts/spark/payload/worker'
EVENTS=[(465,474,485),(492,493,508),(502,503,505),(512,522,529)]
PLOTS=[]


def rows(name): return [json.loads(x) for x in (RAW/name).read_text().splitlines()]
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,value): p.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')
def line(ax,x,y,label,**kwargs):
    x=np.asarray(x);y=np.asarray(y)
    assert x.shape==y.shape and np.isfinite(y).all()
    PLOTS.append(dict(panel=ax.get_title(),label=label,x=x.tolist(),y=y.tolist()))
    ax.plot(x,y,label=label,**kwargs)
def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(alpha=.16,linewidth=.6);ax.tick_params(labelsize=8)
def legend(ax,**kwargs): ax.legend(frameon=False,fontsize=8,**kwargs)


def main():
    torch.set_num_threads(4)
    b={r['state']:r for r in rows('backward.jsonl')}
    a={r['state']:r for r in rows('adam.jsonl')}
    d={r['state']:r for r in rows('directional.jsonl')}
    vectors=torch.load(RAW/'output_vectors.pt',weights_only=False)
    states=np.array(vectors['states']);ss=states[:-1]
    residual=vectors['residuals'].double().numpy()
    flat=residual.reshape(len(states),-1);dr=flat[1:]-flat[:-1]
    exact_linear=np.sum(flat[:-1]*dr,axis=1)/5000
    exact_quadratic=.5*np.sum(dr**2,axis=1)/5000
    cosine=np.sum(dr[1:]*dr[:-1],axis=1)/(np.linalg.norm(dr[1:],axis=1)*np.linalg.norm(dr[:-1],axis=1))
    perloss=.5*np.sum(residual**2,axis=2)
    events=[];projections={}
    for before,peak,recovery in EVENTS:
        i,j,k=[int(np.where(states==s)[0][0]) for s in [before,peak,recovery]]
        delta=flat[j]-flat[i];unit=delta/np.linalg.norm(delta)
        projections[peak]=(flat-flat[i])@unit/5000**.5
        losses=perloss[j]-perloss[i];positive=np.maximum(losses,0)
        p,q=b[before],b[peak]
        backward_gains={}
        for name,num,den in [('head','dh2','residual'),('tanh2','dz2','dh2'),('W2','dh1','dz2'),('tanh1','dz1','dh1'),('W1','dx','dz1')]:
            backward_gains[name]=[b[s]['backward'][num]['norm']/b[s]['backward'][den]['norm'] for s in [before,peak,recovery]]
        neuron_concentration={}
        for layer in ['dz1','dz2']:
            energy=np.array(q['backward'][layer]['neuron_squared_energy'])
            neuron_concentration[layer]=dict(top1_fraction=float(energy.max()/energy.sum()),top10pct_fraction=float(np.sort(energy)[-20:].sum()/energy.sum()))
        events.append(dict(before=before,peak=peak,recovery=recovery,
            losses=[b[s]['native_loss'] for s in [before,peak,recovery]],
            gradient_norm=[b[s]['gradient_norm'] for s in [before,peak,recovery]],
            unit_gradient_rms=[b[s]['unit_gradient_mean_squared_norm']**.5 for s in [before,peak,recovery]],
            coherence=[b[s]['gradient_coherence'] for s in [before,peak,recovery]],
            gradient_norm_ratio=q['gradient_norm']/p['gradient_norm'],
            unit_gradient_rms_ratio=(q['unit_gradient_mean_squared_norm']/p['unit_gradient_mean_squared_norm'])**.5,
            coherence_ratio=q['gradient_coherence']/p['gradient_coherence'],
            parameter_jacobian_rms_ratio=(q['jacobian']['parameter_jacobian_trace']['mean']/p['jacobian']['parameter_jacobian_trace']['mean'])**.5,
            input_jacobian_rms_ratio=(q['jacobian']['input_jacobian_trace']['mean']/p['jacobian']['input_jacobian_trace']['mean'])**.5,
            fraction_examples_loss_increased=float(np.mean(losses>0)),
            top1pct_positive_loss_increase_share=float(np.sort(positive)[-50:].sum()/positive.sum()),
            incoming_previous_output_change_cosine=float(cosine[j-2]),
            incoming_outgoing_output_change_cosine=float(cosine[j-1]),
            incoming_step=d[peak-1],exact_incoming_output_loss_identity=dict(
                residual_dot_output_change=float(exact_linear[j-1]),output_change_energy=float(exact_quadratic[j-1]),
                actual_loss_change=float(perloss[j].mean()-perloss[j-1].mean())),
            backward_gains=backward_gains,peak_neuron_energy_concentration=neuron_concentration))
    actual=np.array([d[s]['actual_loss_change'] for s in ss])
    linear=np.array([d[s]['linear'] for s in ss]);ggn=np.array([d[s]['gauss_newton_quadratic'] for s in ss])
    second=np.array([d[s]['residual_curvature_quadratic'] for s in ss])
    assert np.allclose(exact_linear+exact_quadratic,np.diff(perloss.mean(axis=1)),rtol=1e-10,atol=1e-14)
    result=dict(status='completed',events=events,coverage=dict(states=[448,535],examples=5000,updates=87,seeds=[0]),
        all_updates_initially_downhill=bool(np.all(linear<0)),loss_increasing_updates=int(np.sum(actual>0)),
        linear_network_prediction_rmse=float(np.mean((linear+ggn-actual)**2)**.5),
        second_order_prediction_rmse=float(np.mean((linear+ggn+second-actual)**2)**.5),
        loss_change_rms=float(np.mean(actual**2)**.5),
        linear_network_prediction_sign_accuracy=float(np.mean(np.sign(linear+ggn)==np.sign(actual))),
        minimum_denominator_ratio=min(q['minimum_denominator_ratio'] for q in a.values()),
        minimum_absolute_denominator=min(g['denominator']['min'] for q in a.values() for g in q['groups'].values()),
        maximum_epsilon_near_fraction=max(g['epsilon_near_fraction'] for q in a.values() for g in q['groups'].values()),
        old_denominator_step_norm_ratio_range=[min(q['cf_old_denominator_norm']/q['actual_step_norm'] for q in a.values()),max(q['cf_old_denominator_norm']/q['actual_step_norm'] for q in a.values())],
        output_increment_cosine_range=[float(cosine.min()),float(np.median(cosine)),float(cosine.max())],
        cross_jacobian=json.loads((HERE/'cross_jacobian_probe.json').read_text()),
        sources={name:sha(RAW/name) for name in ['backward.jsonl','adam.jsonl','directional.jsonl','output_vectors.pt','architecture_audit.json','qualification.json']},
        generator_sha256=sha(Path(__file__)))
    dump(HERE/'results.json',result)
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','pdf.fonttype':42,
        'font.size':10,'axes.titlesize':11,'axes.labelsize':10,'lines.linewidth':1.35})
    colors=['#2166ac','#b2182b','#16856b','#8c6bb1']
    fig,axs=plt.subplots(4,2,figsize=(11,11.5),layout='constrained')
    fig.suptitle('MLP Adam control: coherent gradient spikes without a normalization collapse',fontsize=14)
    ax=axs[0,0];ax.set_title('(a) Full training objective')
    line(ax,states,[b[s]['native_loss'] for s in states],'Loss',color=colors[0]);ax.set_ylabel('Half squared error')
    ax=axs[0,1];ax.set_title('(b) Batch gradient grows faster than individual gradients')
    line(ax,states,[b[s]['gradient_norm']/b[465]['gradient_norm'] for s in states],r'$\|g\|$',color=colors[1])
    line(ax,states,[(b[s]['unit_gradient_mean_squared_norm']/b[465]['unit_gradient_mean_squared_norm'])**.5 for s in states],r'$(E_i\|g_i\|^2)^{1/2}$',color=colors[0]);ax.set_ylabel('Relative to state 465');legend(ax)
    ax=axs[1,0];ax.set_title('(c) Less cancellation between example gradients')
    line(ax,states,[100*b[s]['gradient_coherence'] for s in states],r'$\|E_i g_i\|^2/E_i\|g_i\|^2$',color=colors[1]);ax.set_ylabel('Gradient coherence (%)');legend(ax)
    ax=axs[1,1];ax.set_title('(d) Average output sensitivity changes gradually')
    for key,label,color in [('parameter_jacobian_trace','Parameters',colors[0]),('input_jacobian_trace','Inputs',colors[2])]:
        line(ax,states,[(b[s]['jacobian'][key]['mean']/b[465]['jacobian'][key]['mean'])**.5 for s in states],label,color=color)
    ax.set_ylabel('Jacobian RMS / state 465');legend(ax)
    ax=axs[2,0];ax.set_title('(e) Unit-loss backpropagation strengthens')
    for key,label,color in [('residual',r'$r$',colors[0]),('dz2',r'$\partial\ell_i/\partial z_2$',colors[2]),('dz1',r'$\partial\ell_i/\partial z_1$',colors[1])]:
        line(ax,states,[b[s]['backward'][key]['norm']/b[465]['backward'][key]['norm'] for s in states],label,color=color)
    ax.set_ylabel('Vector norm / state 465');legend(ax)
    ax=axs[2,1];ax.set_title('(f) Actual Adam increments become larger')
    line(ax,ss,[a[s]['actual_step_norm'] for s in ss],r'$\|\theta_{s+1}-\theta_s\|$',color=colors[1]);ax.set_ylabel('Parameter increment norm');legend(ax)
    ax=axs[3,0];ax.set_title('(g) Adam denominators do not collapse')
    line(ax,ss,[100*(a[s]['minimum_denominator_ratio']-1) for s in ss],'Minimum coordinate change',color=colors[0])
    line(ax,ss,[100*a[s]['groups']['1.weight']['step_weighted_denominator_relative_change'] for s in ss],'W1: update-weighted RMS change',color=colors[1]);ax.set_ylabel('One-step denominator change (%)');legend(ax,loc='upper left')
    ax=axs[3,1];ax.set_title('(h) Successive prediction changes reverse direction')
    line(ax,states[1:-1],cosine,r'$\cos(\Delta f_{s-1},\Delta f_s)$',color=colors[2]);ax.axhline(0,color='gray',lw=.7);ax.set_ylabel('Cosine');ax.set_ylim(-1.05,.3)
    for ax in axs.flat:
        for peak in [474,493,503,522]:ax.axvline(peak,color='gray',ls=':',lw=.7,alpha=.65)
        ax.set_xlim(448,535);ax.set_xlabel('Completed updates / outgoing update state s');style(ax)
    figures=HERE/'figures';figures.mkdir(exist_ok=True)
    fig.savefig(figures/'01-gradient-amplification.pdf');plt.close(fig)
    fig,axs=plt.subplots(4,2,figsize=(11,11),layout='constrained')
    fig.suptitle('Growing oscillations along existing sensitive directions',fontsize=15)
    for col,(before,peak,recovery) in enumerate([EVENTS[0],EVENTS[3]]):
        mask=(states>=before)&(states<=recovery);smask=(ss>=before)&(ss<recovery)
        ax=axs[0,col];ax.set_title(f'Peak {peak}: signed prediction displacement')
        line(ax,states[mask],projections[peak][mask],'Projection onto before-to-peak displacement',color=colors[0],marker='o',markersize=3)
        ax.axhline(0,color='gray',lw=.7);ax.set_ylabel('Projection / sqrt(5000)')
        ax=axs[1,col];ax.set_title('Measured loss change and local output prediction')
        line(ax,ss[smask],actual[smask],'Actual',color='black')
        line(ax,ss[smask],(linear+ggn)[smask],r'$g^T\Delta\theta+\frac{1}{2} E\|J\Delta\theta\|^2$',color=colors[1],ls='--');ax.axhline(0,color='gray',lw=.7);ax.set_ylabel('Next-step loss change');legend(ax)
        ax=axs[2,col];ax.set_title('Two large opposing terms determine a small loss change')
        line(ax,ss[smask],-exact_linear[smask],r'Residual correction $-E[r^T\Delta f]$',color=colors[0])
        line(ax,ss[smask],exact_quadratic[smask],r'Output-change cost $\frac{1}{2} E\|\Delta f\|^2$',color=colors[1]);ax.set_ylabel('Exact loss-change terms');legend(ax)
        for ax in axs[:3,col]:
            ax.axvline(peak,color='gray',ls=':',lw=.8);ax.set_xlim(before,recovery)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8,integer=True))
            ax.set_xlabel('Completed updates / outgoing update state s');style(ax)
        ax=axs[3,col];ax.set_title('Cross-check: freeze Jacobian, change direction')
        e=next(e for e in result['cross_jacobian']['events'] if e['peak']==peak)
        for index,jstate in enumerate([before,peak-1]):
            y=[next(c['directional_gain'] for c in e['crosses'] if c['jacobian_state']==jstate and c['direction_state']==s) for s in [before,peak-1]]
            ax.bar(np.arange(2)+(index-.5)*.28,y,width=.28,color=colors[index],label=f'Jacobian at {jstate}')
            PLOTS.append(dict(panel=ax.get_title()+f' peak {peak}',label=f'Jacobian {jstate}',x=[0,1],y=y))
        ax.set_xticks([0,1],[f'Adam direction {before}',f'Adam direction {peak-1}']);ax.set_ylabel(r'$\sqrt{E\|J\Delta\theta\|^2}/\|\Delta\theta\|$');ax.set_ylim(0,6.8);legend(ax);style(ax)
    fig.savefig(figures/'02-oscillation-and-direction-probes.pdf');plt.close(fig)
    dump(HERE/'plot_data.json',PLOTS)
    dump(HERE/'figure_verification.json',dict(status='numeric_checks_passed',series=len(PLOTS),plotted_values=sum(len(p['y']) for p in PLOTS),
        independent_source_verification='verification.json',pdfs={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(figures.glob('*.pdf'))}))
    print(json.dumps(dict(events=events,summary={k:v for k,v in result.items() if k not in ['events','cross_jacobian','sources']}),indent=2))


if __name__=='__main__':main()
