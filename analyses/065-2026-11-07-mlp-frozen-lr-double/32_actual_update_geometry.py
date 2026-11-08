"""Audit block contributions to parameter-update geometry at retained anchors."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
torch.set_num_threads(4);threadpool_limits(4)


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def array(tensor):return tensor.numpy().astype(np.float64)


def main():
    sources={};records=[];table=[]
    for number in [272,273,274]:
        path=HERE/f'transfer_comparison_run{number}.json';data=json.loads(path.read_text())
        assert data['status']=='verified' and data['generator_sha256']==sha(HERE/'17_reduce_transfer.py')
        sources[path.name]=sha(path);assert len(data['workers'])==1;w=data['workers'][0]
        groups=w['layout']['groups'];parts={'target_affines':set(w['layout']['trunk'])}
        if w['case']=='transformer':parts.update(embedding={'encoder'},decoder={'decoder'})
        used=set().union(*parts.values());assert sum(map(len,parts.values()))==len(used)
        parts['remaining']=set(groups)-used;assert set().union(*parts.values())==set(groups)
        for arm in ['control','functional_trunk_q50']:
            curve=w['curves'][arm]
            for t in [w['states'][0]+32,w['states'][-1]]:
                path=ROOT/w['worker']/arm/f'audit_{t}.pt';a=torch.load(path,map_location='cpu',weights_only=False)
                before=array(a['before_parameters']);u=array(a['actual_parameters'])-before
                previous=before-array(a['previous_parameters'][-1]);offset=0;rebuilt={}
                for name,shape in a['parameter_names']:
                    owner=name.rsplit('.',1)[0];group=rebuilt.setdefault(owner,dict(start=offset,end=offset,names=[]))
                    assert group['end']==offset;offset+=int(np.prod(shape));group['end']=offset;group['names'].append(name)
                assert rebuilt==groups and offset==len(u)==len(previous)
                energy=float(u@u);old_energy=float(previous@previous);product=float(u@previous)
                cosine=product/np.sqrt(energy*old_energy) if energy*old_energy else None
                index=w['states'].index(t);logged=curve['actual_update_rho'][index]
                if cosine is None:assert logged is None
                else:np.testing.assert_allclose(cosine,logged,rtol=2e-9,atol=2e-11)
                block_records=[]
                for label,names in parts.items():
                    current=old=inner=0.;count=0
                    for name in sorted(names):
                        group=groups[name];sl=slice(group['start'],group['end']);x=u[sl];y=previous[sl]
                        current+=float(x@x);old+=float(y@y);inner+=float(x@y);count+=len(x)
                    block_records.append(dict(part=label,groups=sorted(names),parameters=count,parameter_fraction=count/len(u),
                        current_energy=current,previous_energy=old,inner_product=inner,
                        current_energy_fraction=current/energy if energy else None,
                        previous_energy_fraction=old/old_energy if old_energy else None,
                        consecutive_cosine=inner/np.sqrt(current*old) if current*old else None,
                        signed_share_of_global_inner_product=inner/product if product else None))
                for key,total in [('current_energy',energy),('previous_energy',old_energy),('inner_product',product)]:
                    np.testing.assert_allclose(sum(r[key] for r in block_records),total,rtol=2e-12,atol=2e-14)
                assert sum(r['parameters'] for r in block_records)==len(u)
                row=dict(run=number,case=w['case'],seed=w['seed'],worker=w['worker'],arm=arm,state=t,
                    raw_gradient_cosine=curve['rho'][index],actual_update_cosine=cosine,
                    output_change_cosine=curve['representations']['output']['rho'][index],
                    current_energy=energy,previous_energy=old_energy,inner_product=product,
                    audit_file=path.relative_to(ROOT).as_posix(),audit_sha256=sha(path),parts=block_records)
                records.append(row)
                for part in block_records:
                    table.append({**{k:row[k] for k in ['run','case','seed','arm','state','actual_update_cosine','output_change_cosine']},
                        **{k:v for k,v in part.items() if k!='groups'}})
    result=dict(status='verified',generator_sha256=sha(Path(__file__)),sources=sources,records=records,
        scope='Secondary analysis of fork+32 and terminal retained anchors, using actual outgoing increments and their actual predecessors. Target affines identify the intervention scope in both arms; the control is unchanged. These are parameter-space contributions, not a causal attribution of output motion.',
        denominator_note='Energy fractions sum to one. Signed inner-product shares can be negative or exceed one. Zero denominators are undefined.')
    (HERE/'actual_update_geometry.json').write_text(json.dumps(result,indent=2)+'\n')
    with (HERE/'actual_update_geometry.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    print(json.dumps(dict(status='verified',snapshots=len(records),part_rows=len(table))))


if __name__=='__main__':main()
