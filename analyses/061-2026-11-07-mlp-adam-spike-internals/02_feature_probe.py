"""Post hoc, full-example feature geometry on retained Run236 snapshots."""
import hashlib
import json
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
RUN=ROOT/'runs/236-2026-11-07-mlp-adam-spike-internals'
OUT=RUN/'artifacts/attempt-01/worker'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(state,kind):
    p=OUT/f'snapshots/{kind}{state}.pt'
    SOURCES[p.relative_to(ROOT).as_posix()]=sha(p)
    assert SOURCES[p.relative_to(ROOT).as_posix()]==EXPECTED[p.relative_to(OUT.parent).as_posix()]
    return torch.load(p,map_location='cuda:0',weights_only=False)


def main():
    global SOURCES,EXPECTED
    SOURCES={};EXPECTED={f['path']:f['sha256'] for f in json.loads((OUT.parent/'transfer_inventory.json').read_text())['files']}
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    records=[]
    for event in json.loads((RUN/'config.json').read_text())['events']:
        a=load(event['before'],'state')['model'];b=load(event['peak'],'state')['model'];c=load(event['recovery'],'state')['model']
        aa=load(event['before'],'activations')['arrays'];bb=load(event['peak'],'activations')['arrays'];cc=load(event['recovery'],'activations')['arrays']
        record=dict(event=event['name'],before=event['before'],peak=event['peak'],recovery=event['recovery'],layers={})
        for index in [1,2]:
            h0,h1,h2=[z[f'h{index}'].double() for z in [aa,bb,cc]]
            z0,z1=[z[f'z{index}'].double() for z in [aa,bb]]
            dh=h1-h0;energy=dh.square();den=float(energy.sum())
            # Native FP32 threshold matches the original observer's predicate.
            saturated=aa[f'h{index}'].abs()>.95
            per_unit=energy.sum(0);slope=1-h0.square()
            recovery=h2-h1
            record['layers'][f'h{index}']=dict(
                pre_saturated_fraction=float(saturated.double().mean()),
                change_energy_from_pre_saturated=float(energy[saturated].sum())/den,
                change_energy_from_pre_unsaturated=float(energy[~saturated].sum())/den,
                tanh_vector_change_ratio=float(dh.norm()/(z1-z0).norm()),
                tangent_prediction_relative_error=float((dh-slope*(z1-z0)).norm()/dh.norm()),
                top10pct_units_change_energy=float(per_unit.topk(20).values.sum()/per_unit.sum()),
                unit_change_energy=per_unit.tolist(),
                rise_recovery_cosine=float((dh*recovery).sum()/(dh.norm()*recovery.norm())),
                recovery_to_baseline_change_ratio=float((h2-h0).norm()/dh.norm()))
        w=a['5.weight'].double();hdelta=bb['h2'].double()-aa['h2'].double()
        u,s,v=torch.linalg.svd(w,full_matrices=False)
        projected=hdelta@v.T;output_change=hdelta@w.T
        record['readout']=dict(spectral_norm=float(s[0]),rank=int(torch.linalg.matrix_rank(w)),
            feature_change_rowspace_energy=float(projected.square().sum()/hdelta.square().sum()),
            isotropic_rowspace_fraction=len(s)/w.shape[1],
            actual_vector_gain=float(output_change.norm()/hdelta.norm()),
            baseline_feature_rowspace_energy=float(((aa['h2'].double()@v.T).square().sum()/aa['h2'].double().square().sum())),
            singular_values=s.tolist(),rowspace_mode_change_energy=projected.square().sum(0).tolist())
        assert record['readout']['actual_vector_gain']<=record['readout']['spectral_norm']+1e-12
        delta=bb['output'].double()-aa['output'].double()
        values,vectors=torch.linalg.eigh(delta.T@delta)
        axis=vectors[:,-1]
        if axis[axis.abs().argmax()]<0:axis=-axis
        score=delta@axis
        labels=load(event['before'],'activations')['labels']
        record['output_mode']=dict(energy_fraction=float(values[-1]/values.sum()),class_direction=axis.tolist(),
            score_by_label={str(label):dict(mean=float(score[labels==label].mean()),rms=float(score[labels==label].square().mean().sqrt())) for label in range(10)})
        record['parameter_norms']={key:[float(q[key].double().norm()) for q in [a,b,c]] for key in a}
        records.append(record)
    result=dict(status='completed',examples=5000,events=records,source_sha256=SOURCES,generator_sha256=sha(Path(__file__)),
        additional_training_updates=0,selection='Post hoc probe after Run236; no new model fitting.')
    target=HERE/'feature_probe.json';target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(path=str(target),bytes=target.stat().st_size,sha256=sha(target))))


if __name__=='__main__':main()
