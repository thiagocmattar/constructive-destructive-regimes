"""Reduce complete verified H023 evidence; conditional uncertainty and vector PDF figures."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullLocator, StrMethodFormatter
import numpy as np

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent
sys.path.insert(0,str(ROOT))
from src.cifar_eos.core import file_hash
OUT=ROOT/'runs/186-2026-10-30-mlp-minibatch-noise/artifacts/attempt-01/production'
BATCHES=[32,128,512,2048,5000];CHECKPOINTS=[0,1000,2000]
COLORS={32:'#0072B2',128:'#D55E00',512:'#009E73',2048:'#CC79A7',5000:'#333333'}
EXPECTED_SHA={}


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')


def read_rows(path):
    content=path.read_bytes()
    assert hashlib.sha256(content).hexdigest()==EXPECTED_SHA[path.relative_to(OUT).as_posix()]
    return [json.loads(s) for s in content.decode().splitlines() if s.strip()]


def ci(values,rng):
    values=np.asarray(values,dtype=float)
    if len(values)==1:return [float(values[0])]*2
    estimates=values[rng.integers(len(values),size=(4000,len(values)))].mean(1)
    return np.quantile(estimates,[.025,.975]).tolist()


def smoothed(values,window=20):
    result=np.full(len(values),np.nan)
    # Gradient 1 has no preceding pair; gradient 21 ends the first twenty-pair window.
    result[window:]=np.convolve(values[1:],np.ones(window)/window,mode='valid')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--reduce-only',action='store_true');args=p.parse_args()
    verification=json.loads((HERE/'verification.json').read_text());assert verification['status']=='passed'
    assert file_hash(HERE/'results.json')==verification['results_sha256']
    assert file_hash(OUT/'inventory.json')==verification['production_inventory_sha256']
    EXPECTED_SHA.update({r['path']:r['sha256'] for r in json.loads((OUT/'inventory.json').read_text())['files']})
    data=json.loads((HERE/'results.json').read_text());rng=np.random.default_rng(20261030)
    frozen=[];moving=[]
    for seed in range(3):
        for cp in CHECKPOINTS:
            for b in BATCHES:
                records=[r for r in data['frozen'] if (r['seed'],r['checkpoint'],r['batch'])==(seed,cp,b)]
                cell=dict(seed=seed,checkpoint=cp,batch=b,signal_energy=records[0]['signal_energy'],
                          predicted_variance=records[0]['predicted_variance'],predicted_rho=records[0]['predicted_rho'])
                for field in ('mean_rho','mean_noise_energy','mean_noise_projection','norm_cv','norm_product_cv','R'):
                    values=[r[field] for r in records]
                    cell[field]=float(np.mean(values));cell[field+'_conditional_ci95']=ci(values,rng)
                cell['rho_prediction_error']=cell['mean_rho']-cell['predicted_rho']
                frozen.append(cell)
                records=[r for r in data['moving'] if (r['seed'],r['checkpoint'],r['batch'])==(seed,cp,b)]
                cell=dict(seed=seed,checkpoint=cp,batch=b,repetitions=len(records))
                for field in ('mean_rho','mean_full_rho','R','actual_R','loss_change','final_loss','absolute_displacement','mean_q',
                              'mean_noise_projection','mean_previous_noise_energy'):
                    values=[r[field] for r in records]
                    cell[field]=float(np.mean(values));cell[field+'_conditional_ci95']=ci(values,rng)
                residuals=[r['mean_noise_projection']+r['mean_previous_noise_energy'] for r in records]
                cell['mean_noise_identity_residual']=float(np.mean(residuals))
                cell['mean_noise_identity_residual_conditional_ci95']=ci(residuals,rng)
                moving.append(cell)
    contrasts=[]
    for seed in range(3):
        for cp in CHECKPOINTS:
            full=next(c for c in moving if (c['seed'],c['checkpoint'],c['batch'])==(seed,cp,5000))
            for b in BATCHES[:-1]:
                c=next(c for c in moving if (c['seed'],c['checkpoint'],c['batch'])==(seed,cp,b))
                contrast=dict(seed=seed,checkpoint=cp,batch=b,
                    rho_difference=c['mean_rho']-full['mean_rho'],R_difference=c['R']-full['R'],
                    final_loss_difference=c['final_loss']-full['final_loss'],
                    sampled_minus_full_cosine=c['mean_rho']-c['mean_full_rho'])
                records=[r for r in data['moving'] if (r['seed'],r['checkpoint'],r['batch'])==(seed,cp,b)]
                for name,field in [('rho_difference','mean_rho'),('R_difference','R'),('final_loss_difference','final_loss')]:
                    contrast[name+'_conditional_ci95']=ci([r[field]-full[field] for r in records],rng)
                contrast['sampled_minus_full_cosine_conditional_ci95']=ci([r['mean_rho']-r['mean_full_rho'] for r in records],rng)
                contrasts.append(contrast)
    reduction=dict(frozen=frozen,moving=moving,contrasts_to_full_batch=contrasts,
        numpy=np.__version__,bootstrap_resamples=4000,bootstrap_seed=20261030,
        maximum_abs_frozen_cosine_approximation_error=max(abs(c['rho_prediction_error']) for c in frozen),
        uncertainty='Bootstrap sampling repetitions within fixed state; ranges of three initialization means are descriptive.',
        source_results_sha256=file_hash(HERE/'results.json'))
    write_json(HERE/'reduction.json',reduction)
    if args.reduce_only:
        print(json.dumps({'frozen_cells':len(frozen),'moving_cells':len(moving),'contrasts':len(contrasts),
                          'maximum_abs_frozen_cosine_approximation_error':reduction['maximum_abs_frozen_cosine_approximation_error']}));return
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','font.size':9,
        'axes.titlesize':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'pdf.fonttype':42})
    figures=HERE/'figures';figures.mkdir(exist_ok=True)
    manifest={'source_results_sha256':file_hash(HERE/'results.json'),'script_sha256':file_hash(__file__),
              'production_inventory_sha256':file_hash(OUT/'inventory.json'),
              'python':sys.version.split()[0],'numpy':np.__version__,'matplotlib':matplotlib.__version__,
              'arrays':[],'pdfs':{},'visual_review':'pending'}
    def save(fig,name):
        path=figures/(name+'.pdf');fig.savefig(path,metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
        manifest['pdfs'][path.name]=file_hash(path)
    def panel(ax,label):
        ax.text(0,1.04,label,transform=ax.transAxes,ha='left',va='bottom');ax.grid(alpha=.15,lw=.5)
    fig,axes=plt.subplots(1,3,figsize=(7.16,2.65))
    markers={0:'o',1000:'s',2000:'^'}
    for c in frozen:
        scale=c['signal_energy']+c['predicted_variance']
        pairs=[(c['predicted_variance']/scale,c['mean_noise_energy']/scale,'mean_noise_energy'),
               (-c['predicted_variance']/scale,c['mean_noise_projection']/scale,'mean_noise_projection'),
               (c['predicted_rho'],c['mean_rho'],'mean_rho')]
        for j,(x,y,field) in enumerate(pairs):
            bounds=np.array(c[field+'_conditional_ci95'])/(scale if j<2 else 1)
            err=np.maximum(0,[[y-bounds[0]],[bounds[1]-y]])
            axes[j].errorbar(x,y,yerr=err,color=COLORS[c['batch']],marker=markers[c['checkpoint']],
                            ms=3,alpha=.8,lw=.6,linestyle='none',capsize=1)
            manifest['arrays'].append(dict(figure='frozen_calibration',panel=j,seed=c['seed'],checkpoint=c['checkpoint'],batch=c['batch'],x=x,y=y,ci=bounds.tolist()))
    titles=['Noise variance','Noise projection','Cosine approximation']
    for j,ax in enumerate(axes):
        lo,hi=(-1.05,.05) if j==1 else (-.05,1.05)
        ax.plot([lo,hi],[lo,hi],color='.45',lw=.8,zorder=0)
        ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Predicted',title=titles[j]);panel(ax,f'({chr(97+j)})')
        ax.set_ylabel('Measured'+(' / (S + V)' if j<2 else ' cosine'))
        if j<2:ax.set_xlabel('Predicted / (S + V)')
    handles=[Line2D([],[],color=COLORS[b],lw=1.5,label=f'b = {b:,}') for b in BATCHES]
    handles += [Line2D([],[],color='.3',marker=markers[c],ls='none',label=f'State {c:,}') for c in CHECKPOINTS]
    fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,fontsize=8)
    fig.subplots_adjust(left=.075,right=.99,top=.82,bottom=.34,wspace=.42)
    save(fig,'frozen_calibration')
    histories={}
    for cp in CHECKPOINTS:
        for b in BATCHES:
            for seed in range(3):
                histories[cp,b,seed]=[read_rows(OUT/f'moving-seed{seed}-state{cp}-b{b}-rep{rep:02d}/metrics.jsonl') for rep in range(1 if b==5000 else 8)]
    def trace(cp,b,field,smooth=False,include_endpoint=False):
        blocks=[]
        for seed in range(3):
            all_values=[]
            for rep,records in enumerate(histories[cp,b,seed]):
                values=np.array([np.nan if r.get(field) is None else r[field] for r in records])
                if include_endpoint:
                    assert field=='full_loss'
                    endpoint=next(r['final_loss'] for r in data['moving'] if (r['seed'],r['checkpoint'],r['batch'],r['replicate'])==(seed,cp,b,rep))
                    values=np.append(values,endpoint)
                all_values.append(smoothed(values) if smooth else values)
            blocks.append(np.mean(all_values,axis=0))
        return np.array(blocks)
    for raw in (False,True):
        name='sgd_geometry_raw' if raw else 'sgd_geometry'
        fig,axes=plt.subplots(3,3,figsize=(7.16,5.3),sharex=True)
        for col,cp in enumerate(CHECKPOINTS):
            axes[0,col].set_title(f'Fork at GD update {cp:,}',pad=17)
            for row,(field,label) in enumerate([('cosine',r'Sampled $\rho_k$'),('rho_full',r'Full-gradient $\rho_k$'),('B_over_A',r'Normalized displacement $R_K$')]):
                ax=axes[row,col]
                for b in BATCHES:
                    values=trace(cp,b,field,smooth=(not raw and row<2));center=values.mean(0);x=np.arange(1,201)
                    ax.fill_between(x,values.min(0),values.max(0),color=COLORS[b],alpha=.10,lw=0)
                    line,=ax.plot(x,center,color=COLORS[b],lw=1.1)
                    assert np.array_equal(line.get_ydata(),center,equal_nan=True)
                    manifest['arrays'].append(dict(figure=name,panel=[row,col],batch=b,
                        y_sha256=hashlib.sha256(center.tobytes()).hexdigest(),points=int(np.isfinite(center).sum())))
                ax.set_ylim((-.04,1.04) if row==2 else (-1.04,1.04));panel(ax,f'({chr(97+3*row+col)})')
                if col==0:ax.set_ylabel(label)
                if row==2:ax.set_xlabel('Updates since fork')
        fig.legend(handles=handles[:5],loc='lower center',ncol=5,frameon=False,fontsize=8)
        fig.subplots_adjust(left=.09,right=.99,top=.9,bottom=.11,hspace=.37,wspace=.27)
        save(fig,name)
    fig,axes=plt.subplots(1,3,figsize=(7.16,2.45))
    for col,cp in enumerate(CHECKPOINTS):
        ax=axes[col]
        for b in BATCHES:
            values=trace(cp,b,'full_loss',include_endpoint=True);center=values.mean(0);x=np.arange(201)
            line,=ax.plot(x,center,color=COLORS[b],lw=1.1)
            assert np.array_equal(line.get_ydata(),center,equal_nan=True)
            manifest['arrays'].append(dict(figure='sgd_full_loss',panel=col,batch=b,
                y_sha256=hashlib.sha256(center.tobytes()).hexdigest(),points=201))
            ax.fill_between(x,values.min(0),values.max(0),color=COLORS[b],alpha=.10,lw=0)
        ax.set(title=f'Fork at GD update {cp:,}',xlabel='Updates since fork',yscale='log');panel(ax,f'({chr(97+col)})')
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4,min_n_ticks=3))
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.3g}'))
        ax.yaxis.set_minor_locator(NullLocator())
        if col==0:ax.set_ylabel('Training loss (log scale)')
    fig.legend(handles=handles[:5],loc='lower center',ncol=5,frameon=False,fontsize=8)
    fig.subplots_adjust(left=.09,right=.99,top=.78,bottom=.28,wspace=.35)
    save(fig,'sgd_full_loss')
    fig,axes=plt.subplots(2,3,figsize=(7.16,4.1))
    residuals=[]
    for col,cp in enumerate(CHECKPOINTS):
        axes[0,col].set_title(f'Fork at GD update {cp:,}',pad=15)
        for b in BATCHES:
            q=trace(cp,b,'q',True);corrected=trace(cp,b,'corrected',True)
            for row,val in enumerate([q,corrected]):
                center=val.mean(0);line,=axes[row,col].plot(np.arange(1,201),center,color=COLORS[b],lw=1.1)
                assert np.array_equal(line.get_ydata(),center,equal_nan=True)
                manifest['arrays'].append(dict(figure='sgd_proxy_validation',panel=[row,col],batch=b,
                    y_sha256=hashlib.sha256(center.tobytes()).hexdigest(),points=int(np.isfinite(center).sum())))
                axes[row,col].fill_between(np.arange(1,201),val.min(0),val.max(0),color=COLORS[b],alpha=.10,lw=0)
            values=[];corrected_values=[]
            for seed in range(3):
                seed_values=[];seed_corrected=[]
                for records in histories[cp,b,seed]:
                    probes=[r for r in records if r['taylor_residual'] is not None]
                    seed_values.append([r['previous_hvp_eta_kappa'] for r in probes])
                    seed_corrected.append([r['corrected'] for r in probes])
                    residuals.append(dict(seed=seed,checkpoint=cp,batch=b,
                        comparisons=[dict(update=r['update'],q=r['q'],epsilon=r['epsilon'],corrected=r['corrected'],
                            hvp=r['previous_hvp_eta_kappa'],residual=r['taylor_residual'],
                            raw_proxy_error=r['q']-r['previous_hvp_eta_kappa']) for r in probes],
                        absolute_errors=[abs(r['taylor_residual']) for r in probes]))
                values.append(np.mean(seed_values,axis=0))
                corrected_values.append(np.mean(seed_corrected,axis=0))
            center=np.mean(values,axis=0)
            line,=axes[1,col].plot([2,52,102,152],center,marker='o',ms=5,mfc='none',color=COLORS[b],ls='none')
            assert np.array_equal(line.get_ydata(),center)
            manifest['arrays'].append(dict(figure='sgd_proxy_validation',panel=['hvp',col],batch=b,
                y_sha256=hashlib.sha256(center.tobytes()).hexdigest(),points=4))
            center=np.mean(corrected_values,axis=0)
            line,=axes[1,col].plot([2,52,102,152],center,marker='.',ms=3,color=COLORS[b],ls='none')
            assert np.array_equal(line.get_ydata(),center)
            manifest['arrays'].append(dict(figure='sgd_proxy_validation',panel=['corrected_raw',col],batch=b,
                y_sha256=hashlib.sha256(center.tobytes()).hexdigest(),points=4))
        for row in range(2):
            axes[row,col].set_xlim(-5,205);axes[row,col].set_xticks([0,50,100,150,200])
            panel(axes[row,col],f'({chr(97+3*row+col)})')
            if row==1:axes[row,col].set_xlabel('Updates since fork')
        if col==0:
            axes[0,col].set_ylabel(r'Gradient proxy $q_k$')
            axes[1,col].set_ylabel(r'Corrected $q_k+\varepsilon_k$')
    extra=Line2D([],[],color='.3',marker='o',ms=5,mfc='none',ls='none',label='Directional HVP (raw)')
    corrected_marker=Line2D([],[],color='.3',marker='.',ms=3,ls='none',label='Corrected (raw)')
    fig.legend(handles=handles[:5]+[extra,corrected_marker],loc='lower center',ncol=4,frameon=False,fontsize=8)
    fig.subplots_adjust(left=.085,right=.99,top=.87,bottom=.2,hspace=.4,wspace=.3)
    save(fig,'sgd_proxy_validation')
    write_json(HERE/'hvp_residuals.json',residuals)
    write_json(HERE/'figure_manifest.json',manifest)
    print(json.dumps({'figures':list(manifest['pdfs']),'visual_review':'pending'}))


if __name__=='__main__':main()
