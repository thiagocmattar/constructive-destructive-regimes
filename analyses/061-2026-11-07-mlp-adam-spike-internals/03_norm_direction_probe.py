"""Static magnitude-only versus norm-restored peak networks on Spark."""
import importlib.util
import json
from pathlib import Path
import sys
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
RUN=ROOT/'runs/236-2026-11-07-mlp-adam-spike-internals'
sys.path.insert(0,str(RUN));sys.path.insert(0,str(ROOT))
import diagnostics as d
from src.cifar_eos.core import make_model,file_hash
from src.eos_tasks.task import configure


def main():
    configure();torch.use_deterministic_algorithms(True)
    inputs=json.loads((RUN/'input_manifest.json').read_text())['files']
    for item in inputs:
        p=ROOT/item['path'];assert p.stat().st_size==item['bytes'] and file_hash(p)==item['sha256']
    worker=RUN/'artifacts/attempt-01/worker'
    expected={f['path']:f['sha256'] for f in json.loads((worker.parent/'transfer_inventory.json').read_text())['files']}
    data=torch.load(ROOT/'cache/cifar10-eos/data.pt',map_location='cuda:0',weights_only=True)
    x,y=data['x_train'],data['y_train'];model=make_model().cuda()
    records=[];sources={}
    def checkpoint(state):
        path=worker/f'snapshots/state{state}.pt'
        sources[path.relative_to(ROOT).as_posix()]=file_hash(path)
        assert file_hash(path)==expected[path.relative_to(worker.parent).as_posix()]
        return torch.load(path,map_location='cuda:0',weights_only=False)['model']
    for event in json.loads((RUN/'config.json').read_text())['events']:
        a,b=checkpoint(event['before']),checkpoint(event['peak'])
        def rescale(params,target,weights_only):
            return {k:(v*(target[k].double().norm()/v.double().norm()).to(v.dtype) if not weights_only or k.endswith('weight') else v) for k,v in params.items()}
        variants=dict(before=a,peak=b,
            peak_restore_weight_norms=rescale(b,a,True),peak_restore_all_norms=rescale(b,a,False),
            before_grow_weight_norms=rescale(a,b,True),before_grow_all_norms=rescale(a,b,False),
            peak_old_head={k:(a[k] if k.startswith('5.') else b[k]) for k in a},
            before_new_head={k:(b[k] if k.startswith('5.') else a[k]) for k in a})
        losses={}
        for name,params in variants.items():
            model.load_state_dict(params);losses[name]=d.native_loss(model,x,y,1000)
        restored=variants['peak_restore_all_norms'];grown=variants['before_grow_all_norms']
        norm_error=max(abs(float(restored[k].double().norm()/a[k].double().norm()-1)) for k in a)
        direction_error=max(abs(d.cosine(a[k],grown[k])-1) for k in a)
        assert norm_error<2e-7 and direction_error<1e-12
        records.append(dict(event=event['name'],before=event['before'],peak=event['peak'],losses=losses,
            restored_norm_max_relative_error=norm_error,grown_direction_max_cosine_error=direction_error))
    result=dict(status='completed',events=records,source_sha256=sources,generator_sha256=file_hash(Path(__file__)),
        diagnostics_sha256=file_hash(RUN/'diagnostics.py'),input_manifest_sha256=file_hash(RUN/'input_manifest.json'),
        examples=5000,additional_training_updates=0,
        interpretation='Static norm-preserving/magnitude-only parameter recombinations; no retraining or unique causal attribution.')
    target=HERE/'norm_direction_probe.json';target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(path=str(target),bytes=target.stat().st_size,sha256=file_hash(target))))


if __name__=='__main__':main()
