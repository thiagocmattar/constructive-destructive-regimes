"""Independent FP64 verification, anchored-profile reduction and publication PDF."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent
RUN=ROOT/'runs/205-2026-11-01-eos-exact-lag10'
sys.path.insert(0,str(ROOT))
from src.eos_tasks.task import model_for
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42})


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def verify_window(condition,result,folder,parameter_names):
    gradients=np.load(folder/'gradients.npy',mmap_mode='r')
    assert gradients.dtype==np.float32 and gradients.shape==(11,sum(v for _,v in parameter_names))
    squares=np.zeros(11,dtype=np.float64);dots=np.zeros(11,dtype=np.float64)
    adjacent=np.zeros(10,dtype=np.float64)
    # Chunked NumPy reductions independently check the CUDA/Torch recorder.
    for start in range(0,gradients.shape[1],65536):
        chunk=np.array(gradients[:,start:start+65536],dtype=np.float64)
        assert np.isfinite(chunk).all()
        squares+=np.einsum('ij,ij->i',chunk,chunk)
        dots+=chunk@chunk[0]
        adjacent+=np.einsum('ij,ij->i',chunk[:-1],chunk[1:])
    norms=np.sqrt(squares);denominator=norms[0]*norms
    cosines=np.divide(dots,denominator,out=np.full(11,np.nan),where=denominator>0)
    adjacent=np.divide(adjacent,norms[:-1]*norms[1:],out=np.full(10,np.nan),where=norms[:-1]*norms[1:]>0)
    max_cosine_error=max_norm_error=0.
    for offset,row in enumerate(result['rows']):
        assert row['state']==result['anchor']+offset
        max_norm_error=max(max_norm_error,abs(row['norm']-norms[offset]))
        if row['anchor_cosine'] is None:assert not np.isfinite(cosines[offset])
        else:max_cosine_error=max(max_cosine_error,abs(row['anchor_cosine']-cosines[offset]))
        if offset and row['adjacent_cosine'] is not None:
            assert abs(row['adjacent_cosine']-adjacent[offset-1])<1e-10
    assert max_cosine_error<1e-10 and max_norm_error<1e-9*max(1.,float(norms.max()))
    start=torch.load(folder/'start.pt',map_location='cpu',weights_only=False)
    end=torch.load(folder/'end.pt',map_location='cpu',weights_only=False)
    assert start['step']==result['anchor'] and end['step']==result['anchor']+10
    assert not start['optimizer']['state'] and not end['optimizer']['state']
    assert start['optimizer']['param_groups'][0]['lr']==condition['lr']
    if not condition['reconstructed']:
        parent=next(p for p in condition['parents'] if p['state']==result['anchor'])
        original=torch.load(ROOT/parent['path'],map_location='cpu',weights_only=False)
        assert set(start['model'])==set(original['model'])
        assert all(torch.equal(start['model'][k],v) for k,v in original['model'].items())
        del original
    displacement_sq=residual_sq=parameter_sq=0.
    offset=0
    for name,count in parameter_names:
        initial=start['model'][name].numpy().reshape(-1).astype(np.float64)
        actual=end['model'][name].numpy().reshape(-1).astype(np.float64)-initial
        expected=-condition['lr']*np.sum(gradients[:10,offset:offset+count],axis=0,dtype=np.float64)
        displacement_sq+=float(actual@actual);residual_sq+=float((actual-expected)@(actual-expected));parameter_sq+=float(initial@initial)
        offset+=count
    rounding_bound=32*np.finfo(np.float32).eps*10*(np.sqrt(parameter_sq)+condition['lr']*float(norms[:10].sum()))
    assert np.sqrt(residual_sq)<rounding_bound
    return dict(gradient_vectors=11,profile_values=10,max_cosine_error=max_cosine_error,max_norm_error=max_norm_error,
        saved_start_exact=None if condition['reconstructed'] else True,
        actual_displacement_norm=float(np.sqrt(displacement_sq)),gd_rounding_residual_norm=float(np.sqrt(residual_sq)),
        conservative_fp32_rounding_bound=float(rounding_bound),cosines=[None if not np.isfinite(v) else float(v) for v in cosines[1:]])


def figure(profiles,path,include_reconstructions=False):
    colors=['#2674b8','#e1812c','#359746'];tasks=['mlp','resnet','transformer'];anchors=[500,2000,3000]
    names=['CIFAR-10 / MLP','CIFAR-10 / ResNet-32','WikiText-2 / Transformer']
    fig,axes=plt.subplots(3,3,figsize=(10.8,7.4),sharex=True,sharey=True)
    artists=[]
    for row,anchor in enumerate(anchors):
        for col,task in enumerate(tasks):
            ax=axes[row,col];group=sorted([p for p in profiles if p['task']==task and p['anchor']==anchor],key=lambda p:p['lr'])
            for rank,p in enumerate(group):
                if not p['primary_eligible'] and not include_reconstructions:continue
                y=np.array([np.nan if v is None else v for v in p['cosines']])
                line,=ax.plot(range(1,11),y,color=colors[rank],marker='o',markersize=3,linewidth=1.5,
                    linestyle='-' if p['primary_eligible'] else '--',label=f"LR {p['lr']:g}"+(' (replay unqualified)' if not p['primary_eligible'] else ''))
                assert np.array_equal(line.get_xdata(),np.arange(1,11)) and np.allclose(line.get_ydata(),y,equal_nan=True,rtol=0,atol=0)
                artists.append(dict(parent=p['parent_run_id'],anchor=anchor,values=10,primary_eligible=p['primary_eligible']))
            ax.axhline(0,color='.55',linewidth=.7)
            ax.set_ylim(-1.07,1.07);ax.set_xlim(.7,10.3);ax.set_xticks([1,2,4,6,8,10]);ax.grid(alpha=.16)
            ax.set_title(f'{names[col]}\nStates {anchor}–{anchor+10}',fontsize=9)
            ax.legend(fontsize=6.8,loc='best',framealpha=.85)
            if col==0:ax.set_ylabel('Anchored gradient cosine')
            if row==2:ax.set_xlabel('Lag (full-GD updates)')
    fig.tight_layout(pad=1.0)
    fig.savefig(path,metadata={'Title':'Exact anchored gradient cosine through lag10','Creator':'Analysis032 / 02_analyze.py'})
    plt.close(fig)
    return artists


def main():
    cfg=json.loads((RUN/'config.json').read_text())
    shared=json.loads((ROOT/cfg['original_inputs']).read_text())
    profiles=[];checks=[];source_files=[]
    for condition in cfg['conditions']:
        folder=RUN/f'artifacts/attempt-01/worker{condition["worker"]}'/condition['parent_run_id']
        result=json.loads((folder/'result.json').read_text())
        replay=json.loads((folder/'replay_checks.json').read_text())
        references=json.loads((ROOT/condition['reference']).read_text())
        refs={r['state']:r for r in references['rows']}
        tol=cfg['qualification_tolerances']
        for q in replay:
            ref=refs[q['state']]
            expected=dict(loss=abs(q['loss_error'])<=tol['loss_absolute']+tol['loss_relative']*abs(ref['loss']),
                norm=abs(q['norm_error'])<=tol['norm_absolute']+tol['norm_relative']*abs(ref['norm']))
            if q['cosine_error'] is not None:expected['adjacent_cosine']=abs(q['cosine_error'])<=tol['cosine_absolute']
            assert expected==q['checks'] and all(expected.values())==q['passed']
        model=model_for(condition['task'],ROOT/cfg['task_cache'],shared['wikitext']['vocabulary_size'])
        names=[(name,p.numel()) for name,p in model.named_parameters()];del model
        for w in result['windows']:
            target=folder/f'state-{w["anchor"]:06d}'
            verification=verify_window(condition,w,target,names)
            eligible_checks=replay[:w['preceding_replay_checks']]+w['qualification'] if condition['reconstructed'] else w['qualification']
            qualified=bool(eligible_checks) and all(q['passed'] for q in eligible_checks)
            assert qualified==w['original_trajectory_qualified']
            assert w['primary_eligible']==(qualified if condition['reconstructed'] else True)
            checks.append(dict(parent=condition['parent_run_id'],anchor=w['anchor'],**verification))
            source_files.append(dict(path=(target/'result.json').relative_to(ROOT).as_posix(),sha256=sha(target/'result.json'),
                vectors_sha256=sha(target/'gradients.npy')))
            profiles.append(dict(parent_run_id=condition['parent_run_id'],task=condition['task'],lr=condition['lr'],anchor=w['anchor'],
                primary_eligible=w['primary_eligible'],parent_mode=w['parent_mode'],original_trajectory_qualified=w['original_trajectory_qualified'],
                cosines=verification['cosines'],first_loss=w['rows'][0]['loss'],last_loss=w['rows'][-1]['loss'],
                first_norm=w['rows'][0]['norm'],last_norm=w['rows'][-1]['norm'],
                historical_checked_records=len(eligible_checks),
                first_failed_historical_check=next((q for q in eligible_checks if not q['passed']),None),
                qualification_failures=[q for q in w['qualification'] if not q['passed']],
                fixed_repeat=w['fixed_repeat'],seconds=w['seconds']))
    assert len(profiles)==27
    save(HERE/'results.json',dict(profiles=profiles,sources=source_files,script_sha256=sha(Path(__file__))))
    save(HERE/'verification.json',dict(status='passed',windows=len(checks),gradient_vectors=11*len(checks),profile_values=10*len(checks),checks=checks,
        maximum_cosine_error=max(c['max_cosine_error'] for c in checks),independent_method='Chunked NumPy FP64 reductions of complete stored FP32 vectors; source-checkpoint equality; SGD displacement with explicit FP32 bound'))
    with (HERE/'lag_profiles.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['parent_run_id','task','lr','anchor','lag','source_state','target_state','cosine','primary_eligible','parent_mode','original_trajectory_qualified']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for p in profiles:
            for k,value in enumerate(p['cosines'],1):
                writer.writerow({**{f:p[f] for f in fields if f in p},'lag':k,'source_state':p['anchor'],'target_state':p['anchor']+k,'cosine':value})
    figures=HERE/'figures';figures.mkdir(exist_ok=True)
    primary=figures/'exact_gradient_lag10.pdf'
    artists=figure(profiles,primary)
    assets=[dict(path=primary.relative_to(HERE).as_posix(),sha256=sha(primary),curves=artists)]
    if any(not p['primary_eligible'] for p in profiles):
        exploratory=figures/'all_profiles_with_reconstruction_flags.pdf'
        exploratory_artists=figure(profiles,exploratory,True)
        assets.append(dict(path=exploratory.relative_to(HERE).as_posix(),sha256=sha(exploratory),curves=exploratory_artists))
    save(HERE/'figure_manifest.json',dict(figures=assets,source_sha256=sha(HERE/'results.json'),generator_sha256=sha(Path(__file__)),
        aggregation='none: one anchored pair per lag; primary excludes unqualified historical reconstruction',uncertainty='one initialization; no error bars; deterministic conditional profiles; numerical tolerance in verification.json',
        primary_curves=len(artists),primary_values=10*len(artists)))
    print(json.dumps(dict(windows=len(profiles),primary_windows=len(artists),verified_values=10*len(profiles),max_cosine_error=max(c['max_cosine_error'] for c in checks))))


if __name__=='__main__':main()
