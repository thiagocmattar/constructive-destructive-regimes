"""Independent reductions from retained activation moments and maxima."""
from pathlib import Path
import csv,hashlib,json,math
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
CFG=json.loads((ROOT/'runs/296-2026-11-15-checkpoint-activation-audit/config.json').read_text())
CASES=['mlp','resnet','transformer'];ARMS=CFG['arms']
FORK={'mlp':512,'resnet':256,'transformer':544};END={'mlp':800,'resnet':576,'transformer':928}
def load(case,arm,state,scope):
    if state==FORK[case]:arm='control'
    stem=HERE/'raw'/case/f'{arm}-{state:04d}-{scope}'
    return json.loads(stem.with_suffix('.json').read_text()),np.load(stem.with_suffix('.npz'))
def primary(s):return s['kind'] in ['tanh','relu']
def label(case,site):
    if case=='mlp':return {'1':'Linear 1','2':'Tanh 1','3':'Linear 2','4':'Tanh 2','5':'Output'}[site]
    if case=='resnet':return site.replace('layer','G').replace('.relu#',' / ReLU ').replace('avgpool','Pool').replace('fc','Output')
    return site.replace('transformer_encoder.layers.0.','Block 1 / ').replace('transformer_encoder.layers.1.','Block 2 / ').replace('linear2','FF ReLU').replace('linear1','FF pre').replace('norm1','Attn residual').replace('norm2','Block output').replace('pos_encoder','Position added')
def csvout(name,rows):
    with (HERE/name).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def main():
    result=[];thresholds=[];checked=0;coverage=[];sources={};authenticated=0
    for c in CASES:
        manifest=json.loads((HERE/'raw'/c/'manifest.json').read_text());assert manifest['status']=='completed' and not manifest['smoke']
        run=next((ROOT/'runs').glob(str(CFG['source_runs'][c])+'-*'))
        prior={}
        for inventory in run.glob('*_immutable_prestage.json'):
            for item in json.loads(inventory.read_text())['inventory']:
                prior[item['path']]=item['sha256']
        for file,digest in manifest['sources'].items():
            if Path(file).suffix=='.pt' and '/artifacts/' in file:
                key='/'.join(Path(file).parts[-2:])
                if prior:
                    assert prior[key]==digest,(key,digest,prior[key]);authenticated+=1
        assert len(manifest['records'])==18 and sum('alias' in r for r in manifest['records'])==2
        base,_=load(c,'control',FORK[c],'full_training');scale={s['site']:s['rms'] for s in base['sites']}
        for r in manifest['records']:
            if 'alias' in r:continue
            for scope,info in r['scopes'].items():
                d,a=load(c,r['arm'],r['state'],scope);sources[c+'/'+info['stem']]=hashlib.sha256((HERE/'raw'/c/(info['stem']+'.npz')).read_bytes()).hexdigest()
                coverage.append(dict(case=c,arm=r['arm'],state=r['state'],scope=scope,loss=d['loss'],loss_units=d['loss_units'],historical_relative_error=info['historical_relative_error']))
                for s in d['sites']:
                    prefix=s['array_prefix']+'_';mx=a[prefix+'maximum'];sq=a[prefix+'square'];n=s['n_rows'];width=s['units'];rms=np.sqrt(sq/n);energy=sq.sum();shares=sq/energy if energy else np.zeros_like(sq)
                    np.testing.assert_allclose(rms,a[prefix+'rms'],rtol=1e-14);np.testing.assert_allclose(shares,a[prefix+'share'],rtol=1e-14)
                    assert a[prefix+'threshold_counts'].sum()==n*width and a[prefix+'hist'].sum()==n*width
                    assert np.all(mx>=rms-1e-12) and np.all(a[prefix+'zero']<=n)
                    dead=float(np.mean(mx==0));zero=float(a[prefix+'zero'].sum()/(n*width));assert dead==s['dead_zero_fraction'] and zero==s['zero_entry_fraction']
                    eff=1/np.square(shares).sum()/width if energy else None;assert eff is None or math.isclose(eff,s['effective_unit_fraction'],rel_tol=1e-12)
                    assert s['threshold_entry_fractions']==(a[prefix+'threshold_counts'].cumsum()[:-1]/(n*width)).tolist()
                    row=dict(case=c,arm=r['arm'],state=r['state'],scope=scope,site=s['site'],label=label(c,s['site']),kind=s['kind'],primary=primary(s),rows=n,units=width,rms=s['rms'],fork_rms=scale[s['site']],dead_zero_fraction=dead,zero_entry_fraction=zero,effective_unit_fraction=eff,top10pct_energy=s['top10pct_energy'],dead_relative_001=float(np.mean(mx<=.01*scale[s['site']])),weak_relative_001=float(np.mean(rms<=.01*scale[s['site']])),saturated_entry_fraction=s['saturated_entry_fraction'])
                    result.append(row);checked+=1
                    for typ,grid in [('absolute',CFG['absolute_thresholds']),('relative',CFG['relative_thresholds']+[3,10,30,100])]:
                        previous=0
                        for t in grid:
                            threshold=t*(scale[s['site']] if typ=='relative' else 1.)
                            frac=float(np.mean(mx<=threshold));assert frac>=previous;previous=frac
                            thresholds.append(dict(case=c,arm=r['arm'],state=r['state'],scope=scope,site=s['site'],kind=s['kind'],scale=typ,threshold=t,absolute_threshold=threshold,inactive_fraction=frac,weak_rms_fraction=float(np.mean(rms<=threshold))))
                if scope=='full_training':assert d['loss_units']=={'mlp':5000,'resnet':50000,'transformer':2088600}[c]
        # Reference near-zero claims must not exceed full-data inactive-unit counts in reverse.
        for row in [v for v in result if v['case']==c and v['scope']=='full_training']:
            probe=next(v for v in result if all(v[k]==row[k] for k in ['case','arm','state','site']) and v['scope']=='reference')
            assert row['dead_zero_fraction']<=probe['dead_zero_fraction']
    temporal=[]
    for c in CASES:
        states=[FORK[c],FORK[c]+1,FORK[c]+32,FORK[c]+33,END[c]-1,END[c]]
        base,_=load(c,'control',FORK[c],'full_training')
        for arm in ARMS:
            for s in base['sites']:
                maxima=[]
                for state in states:
                    d,a=load(c,arm,state,'reference');ss=next(v for v in d['sites'] if v['site']==s['site'])
                    maxima.append(a[ss['array_prefix']+'_maximum'])
                maxima=np.stack(maxima)
                for threshold in [0,.01]:
                    mask=maxima<=threshold*s['rms']
                    temporal.append(dict(case=c,arm=arm,site=s['site'],kind=s['kind'],relative_threshold=threshold,states=len(states),always_inactive_fraction=float(mask.all(0).mean()),ever_inactive_fraction=float(mask.any(0).mean()),inactive_identity_changes=int(np.count_nonzero(mask[1:]!=mask[:-1]))))
    csvout('temporal-inactivity.csv',temporal)
    csvout('layer-summary.csv',result);csvout('threshold-sweep.csv',thresholds);csvout('coverage.csv',coverage)
    structure=json.loads((HERE/'resnet-structure.json').read_text());assert structure['status']=='passed' and len(structure['rows'])==90
    for row in structure['rows']:
        d,a=load('resnet',row['arm'],row['state'],'full_training');s=next(v for v in d['sites'] if v['site']==row['site'])
        assert np.flatnonzero(a[s['array_prefix']+'_maximum']==0).tolist()==row['inactive_indices']
    out=dict(status='passed',site_scope_observations=checked,threshold_rows=len(thresholds),full_training_passes=sum(r['scope']=='full_training' for r in coverage),reference_states=sum(r['scope']=='reference' for r in coverage),historical_checkpoint_hashes_matched=authenticated,resnet_structural_audits=90,maximum_loss_relative_error=max(r['historical_relative_error'] for r in coverage),sources=sources)
    (HERE/'verification.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8',newline='\n')
    print({k:v for k,v in out.items() if k!='sources'})
if __name__=='__main__':main()
