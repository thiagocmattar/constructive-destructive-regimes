"""Five requested rows, complete unchanged Adam control states448..800."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
RUN=ROOT/'runs/238-2026-11-07-mlp-intrabatch-coherence'
RAW=RUN/'artifacts/spark/payload/worker'
BASE=ROOT/'runs/223-2026-11-04-multispike-interventions/artifacts/spark/payload/worker'
METRICS=['loss_change','eta_kappa_hat','rho','raw_gradient_norm','intrabatch_coherence']


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rows(path):return [json.loads(x) for x in path.read_text().splitlines()]
def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')
def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def main():
    torch.set_num_threads(4)
    inputs=json.loads((RUN/'input_manifest.json').read_text())['files']
    for item in inputs:
        path=ROOT/item.get('local_path',item['path'])
        assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
    files=json.loads((RAW.parent/'transfer_inventory.json').read_text())['files']
    for item in files:
        path=RAW.parent/item['path']
        assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
    reference=rows(BASE/'prefix/boundaries.jsonl')[:-1]+rows(BASE/'control/boundaries.jsonl')
    boundaries=rows(RAW/'boundaries.jsonl');assert len(boundaries)==801
    for t,row in enumerate(boundaries):
        assert row['state']==t
        assert row['loss']==reference[t]['loss'] and row['raw_gradient_norm']==reference[t]['raw_gradient_norm']
        if t:
            for key in ['rho','eta_kappa_hat']:assert abs(row[key]-reference[t][key])<1e-12
    end=torch.load(RAW/'end.pt',weights_only=False);original=torch.load(BASE/'control/end.pt',weights_only=False)
    for key in ['model','optimizer','previous_gradient','current_gradient','rng']:equal(end[key],original[key])
    measured=rows(RAW/'coherence.jsonl');states=[q['state'] for q in measured]
    assert states==list(range(448,801))
    saved=torch.load(RAW/'example_energies.pt',weights_only=False)
    assert saved['states']==states and saved['examples']==5000 and tuple(saved['energies'].shape)==(353,5000)
    energies=saved['energies'].numpy();assert np.isfinite(energies).all() and (energies>=0).all()
    norm_sums=np.sqrt(energies).sum(axis=1)
    old={q['state']:q for q in rows(ROOT/'runs/237-2026-11-07-mlp-gradient-amplification/artifacts/spark/payload/worker/backward.jsonl')}
    records=[];max_c_error=0.;overlap_error=0.
    for i,q in enumerate(measured):
        t=q['state'];p=reference[t-1];r=reference[t]
        # Independent NumPy reductions from all5000 per-example energies.
        c=5000*r['raw_gradient_norm']/norm_sums[i]
        expected=dict(state=t,loss_change=r['loss']-p['loss'],
            eta_kappa_hat=1-r['rho']*r['raw_gradient_norm']/p['raw_gradient_norm'],
            rho=r['rho'],raw_gradient_norm=r['raw_gradient_norm'],intrabatch_coherence=c)
        for key in METRICS:assert np.isclose(q[key],expected[key],rtol=1e-12,atol=1e-14),(t,key)
        max_c_error=max(max_c_error,abs(c-q['intrabatch_coherence']))
        assert 0<=c<=1+1e-6
        if t in old:
            error=abs(float(energies[i].mean())/old[t]['unit_gradient_mean_squared_norm']-1)
            assert error<1e-12;overlap_error=max(overlap_error,error)
        records.append(expected)
    values=HERE/'control_coherence_448_800.csv'
    with values.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['state']+METRICS,lineterminator='\n');writer.writeheader();writer.writerows(records)
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','pdf.fonttype':42,
        'font.size':11,'axes.labelsize':13,'axes.titlesize':11,'xtick.labelsize':10,'ytick.labelsize':10})
    fig,axes=plt.subplots(5,1,figsize=(9,10.5),sharex=True)
    fig.subplots_adjust(left=.15,right=.98,bottom=.105,top=.93,hspace=.25)
    fig.suptitle('CIFAR-10 MLP: unchanged Adam control',fontsize=16,y=.975)
    fig.text(.565,.945,'Full 5,000-example objective; every state from 448 to 800',ha='center',fontsize=10)
    labels=[r'$L_t-L_{t-1}$',r'$\eta\widehat{\kappa}_t$',r'$\rho_t$',r'$\|g_t\|$',r'$C_t$']
    titles=['(a) Loss change','(b) Projected gradient change','(c) Consecutive raw-gradient cosine',
            '(d) Full-batch raw-gradient norm','(e) Intra-batch gradient coherence']
    colors=['#2166ac','#c26921','#2166ac','#2166ac','#16856b']
    for ax,key,label,title,color in zip(axes,METRICS,labels,titles,colors):
        y=np.array([q[key] for q in records]);assert np.isfinite(y).all()
        trace,=ax.plot(states,y,color=color,lw=1.05)
        np.testing.assert_array_equal(trace.get_xdata(),states);np.testing.assert_array_equal(trace.get_ydata(),y)
        ax.set_ylabel(label);ax.set_title(title,loc='left',pad=5)
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.18,linewidth=.6)
        ax.set_xlim(448,800)
    axes[0].axhline(0,color='gray',ls='--',lw=.65)
    axes[1].axhline(1,color='gray',ls='--',lw=.65)
    axes[2].axhline(0,color='gray',ls='--',lw=.65);axes[2].set_ylim(-1.06,1.06)
    axes[3].set_ylim(bottom=0);axes[4].set_ylim(bottom=0)
    axes[-1].set_xticks([448,512,576,640,704,768,800]);axes[-1].set_xlabel('Completed optimizer updates, t')
    fig.text(.565,.035,r'$C_t=\|\sum_i g_{i,t}\|\,/\,\sum_i\|g_{i,t}\|$',ha='center',fontsize=12)
    fig.text(.565,.012,'Raw gradients include all weights and biases. No smoothing or subsampling.',ha='center',fontsize=9)
    target=HERE/'figures/03-control-coherence-448-800.pdf';fig.savefig(target);plt.close(fig)
    result=dict(status='numeric_checks_passed',run_id='238',state_range=[448,800],states=353,examples_per_state=5000,
        plotted_values=353*5,metrics=METRICS,exact_reference_pairs=801,adjacent_reference_pairs=800,
        full_final_state_exact=800,remote_exact_states=[512,800],input_hashes=len(inputs),artifact_hashes=len(files),
        independently_reduced_example_energies=int(energies.size),maximum_C_absolute_difference=max_c_error,
        previous_energy_coherence_overlap_states=len(old),overlap_max_relative_energy_difference=overlap_error,
        C_definition='norm(sum_i g_i) / sum_i norm(g_i); not squared-energy coherence',
        loss_change_definition='L_t-L_(t-1), transition ending at state t',
        eta_kappa_hat_definition='1-rho_t*norm(g_t)/norm(g_(t-1)); projected gradient change under Adam',
        ranges={key:[min(q[key] for q in records),max(q[key] for q in records)] for key in METRICS},
        sources={name:sha(RAW/name) for name in ['coherence.jsonl','boundaries.jsonl','example_energies.pt']},
        csv=dict(path=values.name,sha256=sha(values)),pdf=dict(path=target.relative_to(HERE).as_posix(),bytes=target.stat().st_size,sha256=sha(target)),
        generator_sha256=sha(Path(__file__)))
    dump(HERE/'control_coherence_verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
