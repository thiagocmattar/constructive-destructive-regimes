"""Authenticated matched replay and dense fixed-probe histories, owned Spark."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.eos_tasks.task import configure,model_for
from src.eos_tasks.production import ProductionTask
from src.eos_tasks import matched_interventions as m
r=m.r; policy=m.policy
CFG=json.loads(Path(__file__).with_name('config.json').read_text())

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(q,p)
def rows(p):return [json.loads(t) for t in Path(p).read_text().splitlines() if t.strip()]
def tensor_error(a,b):
    a=a.detach().cpu().double();b=b.detach().cpu().double()
    return dict(exact=bool(torch.equal(a,b)),relative=float((a-b).norm()/b.norm()) if float(b.norm()) else None,max_absolute=float((a-b).abs().max()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=CFG['source_runs'],required=True)
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    configure();torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    begin=time.monotonic();out=a.out/a.case;out.mkdir(parents=True,exist_ok=False)
    run=next((a.source_root/'runs').glob(str(CFG['source_runs'][a.case])+'-*'))
    cfg=json.loads((run/'config.json').read_text());n=1 if a.smoke else CFG['updates']
    cache=a.data_root/'cache/eos-multitask'
    datafile=a.data_root/('cache/cifar10-eos/data.pt' if a.case=='mlp' else f'cache/eos-multitask/{a.case}_data.pt')
    expected={x['path']:x for x in json.loads((run/'input_manifest.json').read_text())['files']}
    sources={}
    for f in [datafile,cache/'inputs.json',cache/'reference/resnet_cifar.py',cache/'reference/model.py']:
        key=f.relative_to(a.data_root).as_posix();digest=sha(f);assert digest==expected[key]['sha256'];sources[str(f)]=digest
    for f in [Path(__file__),Path(__file__).with_name('config.json'),Path(m.__file__),Path(r.__file__),Path(policy.__file__),run/'config.json']:
        sources[str(f)]=sha(f)
    data={k:v.cuda() for k,v in torch.load(datafile,weights_only=True).items() if k in ['x_train','y_train']}
    vocab=json.loads((cache/'inputs.json').read_text())['wikitext']['vocabulary_size']
    model=model_for(a.case,cache,vocab).cuda();task=ProductionTask(a.case,model,data,cfg['physical'])
    c=dict(task=task,case=a.case,base=cfg['base_lr'])
    _,targets=policy.affine_targets(model,a.case);groups=policy.group_layout(model)
    thresholds_file=next((run/'artifacts/attempt-01').glob('**/thresholds.json'))
    thresholds=json.loads(thresholds_file.read_text())['modules'];sources[str(thresholds_file)]=sha(thresholds_file)
    write(out/'environment.json',dict(config=CFG,source_config=cfg,environment=m.environment(),sources=sources,smoke=a.smoke))
    first_model=None
    for arm_id in CFG['arms']:
        arm=next(x for x in cfg['arms'] if x['id']==arm_id)
        folder=next((run/'artifacts/attempt-01').glob('*/'+arm_id));dest=out/arm_id;dest.mkdir()
        auditfile=folder/f'audit-{cfg["fork"]:04d}.pt';audit=torch.load(auditfile,map_location='cpu',weights_only=False)
        sources[str(auditfile)]=sha(auditfile);start=audit['before']
        if first_model is None:first_model=r.fingerprint(start)
        else:assert r.fingerprint(start)==first_model,'Forks must be identical'
        r.restore(c,start);m.configure_arm(c,arm);r.save(c,dest/'start.pt')
        historical={x['state']:x for x in rows(folder/'boundaries.jsonl')}
        sources[str(folder/'boundaries.jsonl')]=sha(folder/'boundaries.jsonl')
        arrays={};layout={};history={};boundary=[];checks=[];tick=time.monotonic()
        for i in range(n+1):
            assert time.monotonic()-begin<CFG['maximum_seconds_per_case'],'Worker duration exceeded'
            s=c['state'];loss,g=task.gradient();c['current_gradient']=g.clone()
            assert np.isfinite(loss) and bool(torch.isfinite(g).all())
            values,inputs,probe_loss=m.observe(c,cfg,targets)
            if i==0:
                saved=r.fingerprint(model.state_dict());repeat,_,_=m.observe(c,cfg,targets)
                assert all(torch.equal(v,repeat[k]) for k,v in values.items())
                assert saved==r.fingerprint(model.state_dict());del repeat
                error=tensor_error(g,audit['raw_gradient']);assert error['relative'] is None or error['relative']<1e-3,error
                checks.append(dict(state=s,kind='fork_gradient',**error,repeat_probe_exact=True))
            if i==32:
                af=folder/f'audit-{s:04d}.pt';old=torch.load(af,map_location='cpu',weights_only=False);sources[str(af)]=sha(af)
                checks.append(dict(state=s,kind='historic_gradient',**tensor_error(g,old['raw_gradient'])))
                checks.append(dict(state=s,kind='historic_parameters',**tensor_error(m.w.flat(model.parameters()),m.w.flat(old['before']['model'][k] for k,_ in model.named_parameters()))))
                del old
            rep,terms=m.representation_rows(values,history,c)
            for j,(name,v) in enumerate(values.items()):
                v=v.detach().cpu().reshape(-1).numpy()
                if i==0:
                    arrays[name]=np.lib.format.open_memmap(dest/f'activation-{j:02d}.npy',mode='w+',dtype=np.float32,shape=(n+1,v.size))
                    layout[name]=dict(file=f'activation-{j:02d}.npy',coordinates=v.size,shape=list(values[name].shape))
                arrays[name][i]=v
            oldloss=historical[s]['loss']
            record=dict(state=s,loss=loss,probe_loss=probe_loss,gradient_norm=float(g.double().norm()),representations=rep,historic_loss=oldloss,historic_loss_absolute_error=abs(loss-oldloss))
            boundary.append(record)
            with (dest/'boundaries.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if i==n:break
            result=m.apply_update(c,arm,g,inputs,thresholds,groups,targets)
            if i==0:
                checks.append(dict(state=c['state'],kind='historic_first_update',**tensor_error(result['actual_parameters'],m.w.flat(audit['after']['model'][k] for k,_ in model.named_parameters()))))
            if c['state']%16==0:r.save(c,dest/'recovery.pt')
            elapsed=time.monotonic()-tick
            event=dict(case=a.case,arm=arm_id,completed=i+1,total=n,state=c['state'],loss=loss,loss_state=s,seconds=elapsed,remaining_seconds=(n-i-1)*elapsed/(i+1),peak_cuda_bytes=torch.cuda.max_memory_allocated())
            write(out/'progress.json',event)
            if (i+1)%8==0 or a.smoke:print(json.dumps(event),flush=True)
            del result,inputs,values
        for arr in arrays.values():arr.flush()
        r.save(c,dest/'end.pt')
        write(dest/'layout.json',dict(sites=layout,first_state=cfg['fork'],last_state=c['state'],updates=n))
        write(dest/'qualification.json',dict(checks=checks,historic_losses_exact=all(x['historic_loss_absolute_error']==0 for x in boundary),max_historic_loss_absolute_error=max(x['historic_loss_absolute_error'] for x in boundary),seconds=time.monotonic()-tick))
        print(json.dumps(dict(event='arm_complete',case=a.case,arm=arm_id,seconds=time.monotonic()-tick)),flush=True)
        del history,arrays,audit,start,values,inputs,g
    write(out/'sources.json',sources)
    write(out/'capture-complete.json',dict(status='completed',seconds=time.monotonic()-begin,smoke=a.smoke))
if __name__=='__main__':main()
