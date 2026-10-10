"""Signed lag profiles and exact calibration from frozen raw-gradient sketches."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
COLORS={'14M':'#2166ac','31M':'#d95f02','70M':'#1b9e77'}
STYLES={'14M':'-','31M':'--','70M':'-.'}
WINDOWS=['early','middle','late']
def sha(p):
 with Path(p).open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()

def profiles(gram,norms):
    n=len(norms)
    denominator=norms[:,None]*norms[None,:]
    cosine=np.divide(gram,denominator[None],out=np.full_like(gram,np.nan),where=denominator[None]>0)
    cs=[]; weights=[]; counts=[]
    for k in range(1,n):
        c=np.diagonal(cosine,offset=k,axis1=1,axis2=2)
        cs.append(np.nanmean(c,axis=1)); counts.append(int(np.isfinite(c[0]).sum()))
        weights.append(np.diagonal(gram,offset=k,axis1=1,axis2=2).sum(axis=1))
    return cosine,np.array(cs).T,np.array(weights).T,counts

def common_origins(cosine,max_lag):
    origins=np.arange(cosine.shape[1]-max_lag)
    curves=[]; counts=[]
    for k in range(1,max_lag+1):
        selected=cosine[:,origins,origins+k]
        curves.append(np.nanmean(selected,axis=1)); counts.append(int(np.isfinite(selected[0]).sum()))
    return np.array(curves).T,counts

def main():
    results=[]; curve_rows=[]; sources=[]
    gram_windows={}
    for row in json.loads((HERE/'cohort.json').read_text()):
        run=ROOT/'runs'/row['run']; attempt=json.loads((run/'prelaunch/lease.json').read_text())['attempt']
        base=run/'artifacts'/attempt/'full-pass'
        receipt=json.loads((run/'prelaunch'/f'{attempt}-retrieval.json').read_text())
        assert receipt['verified']
        cfg=json.loads((base/'resolved_config.json').read_text())
        events=[json.loads(x) for x in (base/'events.jsonl').read_text().splitlines()]
        updates=[e for e in events if e['event']=='update']; assert len(updates)==712
        norms=np.array([e['gradient_norm'] for e in updates])
        history=np.load(base/'temporal/raw_sketches.npy',mmap_mode='r')
        exact_pairs=[json.loads(x) for x in (base/'temporal/exact_pairs.jsonl').read_text().splitlines()]
        exact_windows={r['window']:r for r in map(json.loads,(base/'temporal/exact_windows.jsonl').read_text().splitlines())}
        for pair in exact_pairs:
            left=pair['source_update']-1; right=pair['target_update']-1
            assert right-left==pair['lag']
            assert pair['source_norm']==norms[left] and pair['target_norm']==norms[right]
            recomputed=np.sum(np.array(history[left],dtype=np.float64)*np.array(history[right],dtype=np.float64),axis=1)
            np.testing.assert_allclose(recomputed,pair['projected_dots'],rtol=1e-12,atol=1e-12)
            if pair['exact_cosine'] is not None:
                np.testing.assert_allclose(recomputed/(norms[left]*norms[right]),pair['projected_cosines'],rtol=1e-12,atol=1e-12)
        source={'run':row['run'],'size':row['size'],'receipt_sha256':sha(run/'prelaunch'/f'{attempt}-retrieval.json'),
            'sources':{f:sha(base/f) for f in ['events.jsonl','resolved_config.json','input_identity.json','summary.json','environment.json','temporal/raw_sketches.npy',
                'temporal/exact_pairs.jsonl','temporal/exact_windows.jsonl','temporal/sketch_metadata.json']}}
        sources.append(source)
        reference=ROOT/'runs'/row['reference']/'artifacts/attempt-01/full-pass'
        source['reference_run']=row['reference']
        source['reference_sources']={f:sha(reference/f) for f in ['events.jsonl','summary.json','input_identity.json','environment.json','resolved_config.json']}
        ref_events=[json.loads(x) for x in (reference/'events.jsonl').read_text().splitlines()]
        ref=[e for e in ref_events if e['event']=='update']
        assert json.loads((base/'input_identity.json').read_text())['order_sha256']==json.loads((reference/'input_identity.json').read_text())['order_sha256']
        assert all(e['input_tokens']==f['input_tokens'] and e['learning_rate']==f['learning_rate'] for e,f in zip(updates,ref))
        r={'run':row['run'],'size':row['size'],'seed':1235,
            'gpu':json.loads((base/'environment.json').read_text())['gpu'],
            'reference_gpu':json.loads((reference/'environment.json').read_text())['gpu'],
            'validation_loss':json.loads((base/'summary.json').read_text())['final_validation']['loss'],
            'reference_validation_loss':json.loads((reference/'summary.json').read_text())['final_validation']['loss'],
            'reference_cosine_rmse':float(np.sqrt(np.mean([(e['adjacent_cosine']-f['adjacent_cosine'])**2 for e,f in zip(updates[1:],ref[1:])]))),
            'windows':{}}
        energy=float(np.sum(norms**2)); b2=updates[-1]['cumulative_gradient_norm']**2
        i1=float(np.sum(np.array([e['adjacent_cosine'] or 0. for e in updates[1:]])*norms[:-1]*norms[1:]))
        total=(b2-energy)/2
        r['whole_pass']={'updates':712,'energy':energy,'cumulative_squared_norm':b2,
            'lag_one_interaction':i1,'all_pair_interaction':total,'beyond_lag_one_interaction':total-i1,
            'lag_one_truncation_relative_squared_sum_error':abs(b2-(energy+2*i1))/b2 if b2 else None,
            'signed_lag_one_share':i1/total if total else None}
        for window,(first,last) in cfg['temporal_sketch']['windows'].items():
            z=np.array(history[first-1:last],dtype=np.float64).transpose(1,0,2)
            n=norms[first-1:last]; gram=z@z.transpose(0,2,1)
            cosine,cs,weighted,counts=profiles(gram,n)
            fixed,fixed_counts=common_origins(cosine,64)
            gram_windows[row['size'],window]=cosine.mean(axis=0)
            mean=cs.mean(axis=0); lower=cs.min(axis=0); upper=cs.max(axis=0)
            off_diagonal=~np.eye(len(n),dtype=bool)
            shuffled=np.nanmean(cosine[:,off_diagonal],axis=1)
            exact=exact_windows[window]
            assert np.isclose(np.sum(n*n),exact['energy'],rtol=1e-12,atol=1e-12)
            assert np.isclose(exact['energy']+2*exact['all_pair_interaction'],exact['cumulative_squared_norm'],rtol=1e-12,atol=1e-12)
            total=weighted.sum(axis=1)
            eligible=[p for p in exact_pairs if first<=p['source_update'] and p['target_update']<=last and p['exact_cosine'] is not None]
            errors=np.array([np.mean(p['projected_cosines'])-p['exact_cosine'] for p in eligible])
            calibration_by_lag={}
            for lag in sorted(set(p['lag'] for p in eligible)):
                selected=[p for p in eligible if p['lag']==lag]
                error=np.array([np.mean(p['projected_cosines'])-p['exact_cosine'] for p in selected])
                calibration_by_lag[str(lag)]={'pairs':len(selected),'rmse':float(np.sqrt(np.mean(error**2))),
                    'mean_error':float(error.mean()),'max_abs_error':float(np.max(np.abs(error)))}
            exact_adj=np.array([e['adjacent_cosine'] for e in updates[first:last]])
            norms_adj=n[:-1]*n[1:]
            assert np.isclose(np.sum(exact_adj*norms_adj),exact['lag_one_interaction'],rtol=1e-10,atol=1e-9)
            window_report={'first_update':first,'last_update':last,'updates':len(n),
                'valid_pairs_by_lag':counts,'cosine_by_lag':mean.tolist(),
                'projection_min':lower.tolist(),'projection_max':upper.tolist(),
                'profiles_by_projection':cs.tolist(),'weighted_interaction_by_lag':weighted.mean(axis=0).tolist(),
                'fixed_origin_profiles_by_projection':fixed.tolist(),'fixed_origin_pair_counts':fixed_counts,
                'fixed_origin_exact_lag_one_mean':float(np.mean([e['adjacent_cosine'] for e in updates[first:first+64]])),
                'shuffled_order_expectation':float(shuffled.mean()),'shuffled_by_projection':shuffled.tolist(),
                'exact_lag_one_mean_cosine':float(exact_adj.mean()),
                'exact_lag_one_negative_frequency':float(np.mean(exact_adj<0)),
                'long_lag_mean_32_64':float(mean[31:64].mean()),
                'odd_even_contrast_1_16':float(mean[0:16:2].mean()-mean[1:16:2].mean()),
                'projection_sensitive_signs_1_64':int(np.sum((lower[:64]<=0)&(upper[:64]>=0))),
                'exact_lag_one_share_of_all_interaction':float(exact['lag_one_interaction']/exact['all_pair_interaction']) if exact['all_pair_interaction'] else None,
                'lag_one_truncation_relative_squared_sum_error':float(abs(exact['cumulative_squared_norm']-(exact['energy']+2*exact['lag_one_interaction']))/exact['cumulative_squared_norm']) if exact['cumulative_squared_norm'] else None,
                'exact_window':exact,'projected_total_interaction':float(total.mean()),
                'projected_totals_by_map':total.tolist(),
                'interaction_relative_error':float((total.mean()-exact['all_pair_interaction'])/abs(exact['all_pair_interaction'])) if exact['all_pair_interaction'] else None,
                'exact_calibration_pairs':len(eligible), 'exact_calibration_lags':sorted(set(p['lag'] for p in eligible)),
                'cosine_calibration_rmse':float(np.sqrt(np.mean(errors**2))),
                'cosine_calibration_max_abs':float(np.max(np.abs(errors))),
                'calibration_by_lag':calibration_by_lag,
                'late_tail_exclusion_cosine':profiles(gram[:,:-1,:-1],n[:-1])[1].mean(axis=0).tolist() if last==712 else None}
            if last==712:
                window_report['late_tail_exclusion_max_cosine_change_1_64']=float(np.max(np.abs(np.array(window_report['late_tail_exclusion_cosine'])[:64]-mean[:64])))
            r['windows'][window]=window_report
            for k in range(1,len(n)):
                curve_rows.append({'size':row['size'],'window':window,'lag':k,'pairs':counts[k-1],
                    'cosine':mean[k-1],'projection_min':lower[k-1],'projection_max':upper[k-1],
                    'shuffled_expectation':shuffled.mean(),'mean_dot':weighted.mean(axis=0)[k-1]/counts[k-1],
                    'interaction':weighted.mean(axis=0)[k-1]})
        results.append(r)
    (HERE/'figures').mkdir(exist_ok=True); (HERE/'scratch').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    comparable=np.concatenate([np.array(w[key])[:,:64].ravel() for r in results for w in r['windows'].values()
                              for key in ['profiles_by_projection','fixed_origin_profiles_by_projection']])
    raw_limits=(.1*np.floor(min(float(np.nanmin(comparable)),0)*10)-.03,
                .1*np.ceil(max(float(np.nanmax(comparable)),0)*10)+.03)
    for kind,name in [('raw','lag_profiles'),('fixed','lag_profiles_fixed_origins'),('excess','lag_profiles_above_shuffled'),('weighted','cumulative_lag_interaction')]:
        fig,axs=plt.subplots(1,3,figsize=(12,3.7),sharey=True)
        for ax,window in zip(axs,WINDOWS):
            for r in results:
                w=r['windows'][window]; size=r['size']; x=np.arange(1,65)
                if kind=='weighted':
                    y=2*np.cumsum(w['weighted_interaction_by_lag'])[:64]/w['exact_window']['energy']
                    ax.plot(x,y,label=size,color=COLORS[size],ls=STYLES[size],lw=1.7)
                else:
                    curves=np.array(w['fixed_origin_profiles_by_projection'] if kind=='fixed' else w['profiles_by_projection'])[:,:64]
                    if kind=='excess':curves-=np.array(w['shuffled_by_projection'])[:,None]
                    y=curves.mean(axis=0)
                    ax.plot(x,y,label=size,color=COLORS[size],ls=STYLES[size],lw=1.6)
                    ax.fill_between(x,curves.min(axis=0),curves.max(axis=0),color=COLORS[size],alpha=.13)
                    if kind in ['raw','fixed']:
                        ax.scatter([1],[w['fixed_origin_exact_lag_one_mean'] if kind=='fixed' else w['exact_lag_one_mean_cosine']],s=24,color=COLORS[size],marker='o',zorder=5)
                ax.axhline(0,color='0.45',lw=.8)
            w=results[0]['windows'][window]
            ax.set_title(window.capitalize()+f" training (updates {w['first_update']}–{w['last_update']})")
            ax.set_xlabel('Lag (optimizer updates)'); ax.set_xlim(.5,64.5)
            if kind in ['raw','fixed']:ax.set_ylim(*raw_limits)
            ax.set_xticks([1,8,16,32,48,64]); ax.grid(axis='y',alpha=.18)
        ylabel={'raw':'Mean raw-gradient cosine','fixed':'Mean raw-gradient cosine','excess':'Cosine minus unordered-window reference',
                'weighted':'Cumulative interaction / gradient energy'}[kind]
        axs[0].set_ylabel(ylabel); axs[-1].legend(frameon=False)
        note='Seed1235; all within-window pairs; bands show range across four fixed projections.' if kind!='weighted' else 'Seed1235; 2 × cumulative lag interaction / sum of squared raw-gradient norms; four-map mean.'
        if kind=='fixed':note='Seed1235; same first 64 source updates at every lag; bands: range across four fixed projections.'
        fig.text(.5,.025,note,ha='center',fontsize=8)
        fig.subplots_adjust(left=.075,right=.99,bottom=.19,top=.86,wspace=.1)
        fig.savefig(HERE/'figures'/f'{name}.pdf'); fig.savefig(HERE/'scratch'/f'{name}.png',dpi=160); plt.close(fig)
    fig,axs=plt.subplots(3,3,figsize=(10,9),sharex=True,sharey=True)
    for i,size in enumerate(COLORS):
        for j,window in enumerate(WINDOWS):
            ax=axs[i,j]; im=ax.imshow(gram_windows[size,window],vmin=-1,vmax=1,cmap='RdBu_r',origin='lower',extent=[.5,128.5,.5,128.5],interpolation='nearest',rasterized=True)
            ax.set_xticks([1,32,64,96,128]); ax.set_yticks([1,32,64,96,128])
            if i==0:ax.set_title(window.capitalize())
            if j==0:ax.set_ylabel(size+'\nUpdate within window')
            if i==2:ax.set_xlabel('Update within window')
    fig.subplots_adjust(right=.88,wspace=.12,hspace=.15,bottom=.1,top=.93)
    cax=fig.add_axes([.91,.18,.018,.65]); fig.colorbar(im,cax=cax,label='Projected raw-gradient cosine')
    fig.text(.5,.025,'Seed1235; four-map mean; exact original gradient norms; diagonal includes projection error.',ha='center',fontsize=8)
    fig.savefig(HERE/'figures/temporal_gram.pdf',dpi=160); fig.savefig(HERE/'scratch/temporal_gram.png',dpi=130); plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(12,3.7),sharey=True)
    for ax,window in zip(axs,WINDOWS):
        for r in results:
            w=r['windows'][window]; size=r['size']; x=np.arange(1,128)
            # Weighted curves are reconstructed per map from the stored Gram.
            row=next(row for row in json.loads((HERE/'cohort.json').read_text()) if row['size']==size)
            run=ROOT/'runs'/row['run']; attempt=json.loads((run/'prelaunch/lease.json').read_text())['attempt']
            base=run/'artifacts'/attempt/'full-pass'
            history=np.load(base/'temporal/raw_sketches.npy',mmap_mode='r')
            z=np.array(history[w['first_update']-1:w['last_update']],dtype=np.float64).transpose(1,0,2)
            gram=z@z.transpose(0,2,1)
            weighted=np.array([np.diagonal(gram,offset=k,axis1=1,axis2=2).sum(axis=1) for k in x]).T
            curves=2*np.cumsum(weighted,axis=1)/w['exact_window']['energy']
            ax.plot(x,curves.mean(axis=0),color=COLORS[size],ls=STYLES[size],label=size,lw=1.6)
            ax.fill_between(x,curves.min(axis=0),curves.max(axis=0),color=COLORS[size],alpha=.13)
            ax.scatter([127],[2*w['exact_window']['all_pair_interaction']/w['exact_window']['energy']],
                color=COLORS[size],marker='D',s=23,zorder=5)
        ax.axhline(0,color='0.45',lw=.8); ax.set_xlim(.5,129)
        ax.set_title(window.capitalize()); ax.set_xlabel('Maximum included lag (updates)')
        ax.set_xticks([1,32,64,96,127]); ax.grid(axis='y',alpha=.18)
    axs[0].set_ylabel('Cumulative interaction / gradient energy'); axs[-1].legend(frameon=False)
    fig.text(.5,.025,'2 × cumulative lag interaction / energy; diamonds: exact all-pair endpoint; 128 − k pairs at lag k.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.075,right=.99,bottom=.19,top=.86,wspace=.1)
    fig.savefig(HERE/'figures/cumulative_lag_interaction_all_lags.pdf')
    fig.savefig(HERE/'scratch/cumulative_lag_interaction_all_lags.png',dpi=160); plt.close(fig)
    with (HERE/'curve_data.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=curve_rows[0].keys()); writer.writeheader(); writer.writerows(curve_rows)
    payload={'definition':'mean uncentered signed cosine of raw gradients at lag k; both endpoints in window',
        'coverage':'712 updates per model; final partial update retained; sensitivity excluding it included',
        'uncertainty':'range across four fixed map realizations only; one training seed',
        'results':results,'sources':sources}
    (HERE/'results.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print(json.dumps([{'size':r['size'],'windows':{k:{x:v[x] for x in ['exact_lag_one_mean_cosine','long_lag_mean_32_64','odd_even_contrast_1_16','cosine_calibration_rmse','interaction_relative_error']} for k,v in r['windows'].items()}} for r in results],indent=2))
if __name__=='__main__':main()
