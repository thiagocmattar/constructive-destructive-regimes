"""Independent CPU checks for post hoc Spark feature and magnitude probes."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
RUN=ROOT/'runs/236-2026-11-07-mlp-adam-spike-internals'
OUT=RUN/'artifacts/spark/payload/worker'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(RUN))
from src.cifar_eos.core import make_model,file_hash
import diagnostics as d
torch.set_num_threads(4)
def load(state,kind):return torch.load(OUT/f'snapshots/{kind}{state}.pt',map_location='cpu',weights_only=False)
def near(a,b):np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-12)


def main():
    for stem,script in [('feature_probe','02_feature_probe.py'),('norm_direction_probe','03_norm_direction_probe.py')]:
        p=HERE/(stem+'.json');record=json.loads(p.read_text());receipt=json.loads((HERE/(stem+'_retrieval.json')).read_text())
        assert receipt['verified'] and p.stat().st_size==receipt['bytes'] and file_hash(p)==receipt['sha256']
        assert record['generator_sha256']==file_hash(HERE/script)
        for source,expected in record['source_sha256'].items():
            local=ROOT/source.replace('/artifacts/attempt-01/worker/','/artifacts/spark/payload/worker/')
            assert file_hash(local)==expected
    feature=json.loads((HERE/'feature_probe.json').read_text())
    for event in feature['events']:
        aa,bb,cc=[load(event[k],'activations')['arrays'] for k in ['before','peak','recovery']]
        model=load(event['before'],'state')['model']
        for j in [1,2]:
            h0,h1,h2=[q[f'h{j}'].double() for q in [aa,bb,cc]]
            dh=h1-h0;dz=bb[f'z{j}'].double()-aa[f'z{j}'].double();energy=dh.square();saturated=aa[f'h{j}'].abs()>.95
            r=event['layers'][f'h{j}']
            near(float(energy[~saturated].sum()/energy.sum()),r['change_energy_from_pre_unsaturated'])
            near(float(dh.norm()/dz.norm()),r['tanh_vector_change_ratio'])
            near(energy.sum(0).tolist(),r['unit_change_energy'])
            near(float(((h2-h1)*dh).sum()/((h2-h1).norm()*dh.norm())),r['rise_recovery_cosine'])
        delta=bb['h2'].double()-aa['h2'].double();w=model['5.weight'].double()
        _,sv,v=torch.linalg.svd(w,full_matrices=False)
        near(float((delta@v.T).square().sum()/delta.square().sum()),event['readout']['feature_change_rowspace_energy'])
        near(float((delta@w.T).norm()/delta.norm()),event['readout']['actual_vector_gain'])
        near(float(sv[0]),event['readout']['spectral_norm'])
    data=torch.load(ROOT/'cache/cifar10-eos/data.pt',weights_only=True)
    model=make_model();max_error=0.
    norm=json.loads((HERE/'norm_direction_probe.json').read_text())
    for event in norm['events']:
        a,b=[load(event[k],'state')['model'] for k in ['before','peak']]
        def scaled(p,q,weights_only):
            return {k:(v*(q[k].double().norm()/v.double().norm()).to(v.dtype) if not weights_only or k.endswith('weight') else v) for k,v in p.items()}
        variants=dict(before=a,peak=b,peak_restore_weight_norms=scaled(b,a,True),peak_restore_all_norms=scaled(b,a,False),
            before_grow_weight_norms=scaled(a,b,True),before_grow_all_norms=scaled(a,b,False),
            peak_old_head={k:a[k] if k.startswith('5.') else b[k] for k in a},
            before_new_head={k:b[k] if k.startswith('5.') else a[k] for k in a})
        for name,params in variants.items():
            model.load_state_dict(params)
            value=d.native_loss(model,data['x_train'],data['y_train'],1000)
            error=abs(value-event['losses'][name]);max_error=max(max_error,error)
            assert error<2e-7,(name,error)
    result=dict(status='passed',feature_events=4,norm_variants_verified=32,examples=5000,
        cpu_spark_max_loss_difference=max_error,verifier_sha256=file_hash(Path(__file__)))
    (HERE/'probe_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
