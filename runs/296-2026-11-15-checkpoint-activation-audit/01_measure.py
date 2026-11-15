"""Activation audit at authenticated saved states; no optimizer updates."""
import argparse, copy, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.eos_tasks.task import configure,model_for
from src.eos_tasks.production import ProductionTask
from src.cifar_eos.core import flat
CFG=json.loads(Path(__file__).with_name('config.json').read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(q,p)
def equal_model(m,s):
    assert all(torch.equal(v.cpu(),s[k]) for k,v in m.state_dict().items())

class Collector:
    def __init__(self,model,case):
        self.model=model;self.case=case;self.hooks=[];self.stats={};self.samples={};self.kinds={};self.calls={}
        self.thresholds=torch.tensor(CFG['absolute_thresholds'],device='cuda',dtype=torch.float32)
        self.hist_edges=np.linspace(-10,3,105)
        self.enabled=True;self.check_only=False;self.check_values={}
        modules=dict(model.named_modules())
        def add(n,kind,pre=False):
            self.kinds[n]=kind
            if pre:h=modules[n].register_forward_pre_hook(lambda m,a,n=n:self.capture(n,a[0]))
            else:h=modules[n].register_forward_hook(lambda m,a,o,n=n:self.capture(n,o))
            self.hooks.append(h)
        if case=='mlp':
            for n in ['1','2','3','4','5']:add(n,'tanh' if n in ['2','4'] else 'linear')
        elif case=='resnet':
            for n,m in modules.items():
                if isinstance(m,torch.nn.ReLU):add(n,'relu')
            add('avgpool','pooled');add('fc','readout')
        else:
            add('encoder','embedding');add('pos_encoder','position_added')
            for i in [0,1]:
                base=f'transformer_encoder.layers.{i}'
                for n,kind,pre in [('norm1','attention_residual',False),('linear1','ff_preactivation',False),('linear2','relu',True),('norm2','block_output',False)]:add(base+'.'+n,kind,pre)
    def reset(self):self.stats={};self.samples={};self.calls={};self.check_values={}
    def begin(self,y):self.calls={};self.mask=y!=-100 if self.case=='transformer' else None
    @torch.no_grad()
    def capture(self,name,value):
        if not self.enabled:return
        self.calls[name]=self.calls.get(name,0)+1
        key=name+('#'+str(self.calls[name]) if self.case=='resnet' and self.kinds[name]=='relu' else '')
        x=value.detach()
        shape=list(x.shape)
        if self.case=='transformer':x=x.transpose(0,1)[self.mask]
        elif x.ndim==4:x=x.permute(0,2,3,1).reshape(-1,x.shape[1])
        else:x=x.reshape(-1,x.shape[-1])
        if self.check_only:self.check_values[key]=x.clone();return
        assert bool(torch.isfinite(x).all()),key
        a=x.abs();n,d=x.shape
        if key not in self.stats:
            z=torch.zeros(d,device=x.device,dtype=torch.float64)
            self.stats[key]=dict(n=0,d=d,kind=self.kinds[name],first_shape=shape,sum=z.clone(),square=z.clone(),abs=z.clone(),maximum=z.clone(),zero=z.long(),near=z.long(),saturated=z.long(),hist=torch.zeros(106,device=x.device,dtype=torch.int64),threshold_counts=torch.zeros(len(self.thresholds)+1,device=x.device,dtype=torch.int64))
            ids=torch.linspace(0,n-1,min(n,CFG['matrix_rows']),device=x.device).long()
            self.samples[key]=dict(values=x[ids].cpu().numpy(),row_ids=ids.cpu().numpy(),rows_in_reference_batch=n)
        s=self.stats[key];s['n']+=n
        s['sum']+=x.sum(0,dtype=torch.float64);s['square']+=x.double().square().sum(0)
        s['abs']+=a.sum(0,dtype=torch.float64);s['maximum']=torch.maximum(s['maximum'],a.amax(0).double())
        s['zero']+=(x==0).sum(0);s['near']+=(a<=.001).sum(0);s['saturated']+=(a>=.99).sum(0)
        bins=torch.bucketize(a.flatten().contiguous(),self.thresholds,right=False)
        s['threshold_counts']+=torch.bincount(bins,minlength=len(self.thresholds)+1)
        logs=torch.log10(a.clamp_min(1e-30)).flatten()
        s['hist'][0]+=(logs< -10).sum();s['hist'][-1]+=(logs>3).sum()
        s['hist'][1:-1]+=torch.histc(logs,bins=104,min=-10,max=3).long()
    def export(self):
        rows=[];arrays={}
        for i,(name,s) in enumerate(self.stats.items()):
            a={k:v.cpu().numpy() for k,v in s.items() if isinstance(v,torch.Tensor)}
            rms=np.sqrt(a['square']/s['n']);energy=a['square'];total=energy.sum();share=energy/total if total else np.zeros_like(energy)
            base=dict(site=name,kind=s['kind'],n_rows=s['n'],units=s['d'],first_shape=s['first_shape'],
                rms=float(np.sqrt(total/(s['n']*s['d']))),zero_entry_fraction=float(a['zero'].sum()/(s['n']*s['d'])),
                dead_zero_fraction=float(np.mean(a['maximum']==0)),near_entry_fraction=float(a['near'].sum()/(s['n']*s['d'])),
                effective_unit_fraction=float(1/(share@share)/s['d']) if total else None,
                top10pct_energy=float(np.sort(share)[-max(1,int(np.ceil(.1*s['d']))):].sum()) if total else None,
                saturated_entry_fraction=float(a['saturated'].sum()/(s['n']*s['d'])) if s['kind']=='tanh' else None)
            assert int(a['threshold_counts'].sum())==s['n']*s['d']
            assert int(a['hist'].sum())==s['n']*s['d']
            base['threshold_entry_fractions']=(np.cumsum(a['threshold_counts'])[:-1]/(s['n']*s['d'])).tolist()
            base['dead_absolute_fractions']=[float(np.mean(a['maximum']<=e)) for e in CFG['absolute_thresholds']]
            base['array_prefix']=str(i);rows.append(base)
            for k,v in a.items():arrays[f'{i}_{k}']=v
            arrays[f'{i}_rms']=rms;arrays[f'{i}_share']=share
            for k,v in self.samples[name].items():arrays[f'{i}_matrix_{k}']=np.asarray(v)
        return rows,arrays
    def close(self):
        for h in self.hooks:h.remove()

def qualify(task,obs,probe):
    model=task.model;x=task.data['x_train'][:probe];y=task.data['y_train'][:probe]
    # Use one intact physical group to qualify exactly the same forward graph.
    x=x[:task.physical];y=y[:task.physical]
    state={k:v.cpu().clone() for k,v in model.state_dict().items()}
    obs.enabled=False;model.train();model.zero_grad(set_to_none=True)
    loss,out=task.loss(x,y);loss.backward();g=flat([p.grad for p in model.parameters()]).clone();out=out.detach().clone()
    obs.enabled=True;obs.check_only=True;obs.begin(y);model.zero_grad(set_to_none=True)
    l2,o2=task.loss(x,y);l2.backward();g2=flat([p.grad for p in model.parameters()]);assert torch.equal(g,g2) and torch.equal(out,o2),dict(gradient_max=float((g-g2).abs().max()),output_max=float((out-o2).abs().max()))
    train={k:v.clone() for k,v in obs.check_values.items()}
    with torch.no_grad():
        obs.begin(y);l3,o3=task.loss(x,y)
        no_grad=all(torch.equal(v,obs.check_values[k]) for k,v in train.items())
        assert no_grad and torch.equal(out,o3)
        model.eval();obs.begin(y);l4,o4=task.loss(x,y)
        eval_exact=all(torch.equal(v,obs.check_values[k]) for k,v in train.items()) and torch.equal(out,o4)
        eval_error=max(float((v-obs.check_values[k]).abs().max()) for k,v in train.items())
    equal_model(model,state);model.train();model.zero_grad(set_to_none=True)
    obs.check_only=False;obs.reset()
    return dict(hooked_plain_gradient_exact=True,grad_nograd_activations_exact=no_grad,train_eval_activations_exact=eval_exact,eval_max_absolute_error=eval_error,sites=len(train))

def measure(task,obs,reference,full):
    obs.reset();total=0.;den=0;probe_result=None;begin=time.monotonic()
    with torch.no_grad():
        for j,(x,y) in enumerate(task.chunks()):
            obs.begin(y);loss,_=task.loss(x,y)
            units=int((y!=-100).sum()) if task.name=='transformer' else len(y)
            total+=float(loss.double());den+=units
            seen=(j+1)*task.physical
            if seen>=reference and probe_result is None:
                probe_result=(*obs.export(),dict(loss=total/den,loss_units=den,seconds=time.monotonic()-begin))
                if not full:break
    final=(*obs.export(),dict(loss=total/den,loss_units=den,seconds=time.monotonic()-begin))
    return probe_result,final

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=list(CFG['source_runs']),required=True)
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    configure();torch.backends.cudnn.deterministic=True
    out=a.out/a.case;out.mkdir(parents=True,exist_ok=False);begin=time.monotonic()
    run=next((a.source_root/'runs').glob(str(CFG['source_runs'][a.case])+'-*'));cfg=json.loads((run/'config.json').read_text())
    cache=a.data_root/'cache/eos-multitask';datafile=a.data_root/('cache/cifar10-eos/data.pt' if a.case=='mlp' else f'cache/eos-multitask/{a.case}_data.pt')
    sources={};inputfiles=json.loads((run/'input_manifest.json').read_text())['files']
    for f in [datafile,cache/'inputs.json',cache/'reference/resnet_cifar.py',cache/'reference/model.py']:
        digest=sha(f);relative=f.relative_to(a.data_root).as_posix();expected=next((r['sha256'] for r in inputfiles if r['path']==relative),None)
        assert expected is not None and digest==expected,(relative,digest,expected);sources[str(f)]=digest
    vocabulary=json.loads((cache/'inputs.json').read_text())['wikitext']['vocabulary_size']
    data={k:v.cuda() for k,v in torch.load(datafile,weights_only=True).items() if k in ['x_train','y_train']}
    model=model_for(a.case,cache,vocabulary).cuda();task=ProductionTask(a.case,model,data,cfg['physical']);obs=Collector(model,a.case)
    write(out/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),config=CFG,worker_sha256=sha(__file__),data_sources=sources))
    records=[];qualification=None;fork_model=None
    for arm in CFG['arms']:
        folder=next((run/'artifacts/attempt-01').glob('*/'+arm))
        paths=[(cfg['fork'],folder/f'audit-{cfg["fork"]:04d}.pt','before'),(cfg['fork']+1,folder/f'audit-{cfg["fork"]:04d}.pt','after'),(cfg['fork']+32,folder/f'audit-{cfg["fork"]+32:04d}.pt','before'),(cfg['fork']+33,folder/f'audit-{cfg["fork"]+32:04d}.pt','after'),(cfg['end']-1,folder/f'audit-{cfg["end"]-1:04d}.pt','before'),(cfg['end'],folder/'end.pt',None)]
        historic={r['state']:r for r in [json.loads(t) for t in (folder/'boundaries.jsonl').read_text().splitlines()]}
        for state,path,key in paths:
            if a.smoke and (arm!='control' or state!=cfg['fork']):continue
            assert time.monotonic()-begin<CFG['maximum_seconds_per_case']
            payload=torch.load(path,map_location='cpu',weights_only=False);saved=payload if key is None else payload[key]
            assert saved['state']==state;model.load_state_dict(saved['model']);model.train();sources[str(path)]=sha(path)
            if state==cfg['fork']:
                if fork_model is None:fork_model={k:v.clone() for k,v in saved['model'].items()}
                else:
                    assert all(torch.equal(v,fork_model[k]) for k,v in saved['model'].items())
                    records.append(dict(arm=arm,state=state,alias='control',source=str(path),sha256=sources[str(path)]));del payload,saved;continue
            del payload,saved
            if qualification is None:
                qualification=qualify(task,obs,cfg['probe']);assert qualification['train_eval_activations_exact'],qualification
                write(out/'qualification.json',qualification)
            full=(state in [cfg['fork'],cfg['fork']+32,cfg['end']]) and not a.smoke
            probe,all_data=measure(task,obs,cfg['probe'],full)
            record=dict(arm=arm,state=state,source=str(path),sha256=sources[str(path)],historical_loss=historic[state]['loss'],historical_probe_loss=historic[state]['probe_loss'],scopes={})
            for scope,result in [('reference',probe)]+([('full_training',all_data)] if full else []):
                rows,arrays,info=result;stem=f'{arm}-{state:04d}-{scope}'
                np.savez_compressed(out/(stem+'.npz'),**arrays);write(out/(stem+'.json'),dict(case=a.case,arm=arm,state=state,scope=scope,sites=rows,**info))
                record['scopes'][scope]=dict(stem=stem,**info)
                old=record['historical_loss' if scope=='full_training' else 'historical_probe_loss']
                relative=abs(info['loss']-old)/max(abs(old),1e-30)
                assert relative<1e-4,(a.case,arm,state,scope,relative)
                record['scopes'][scope]['historical_relative_error']=relative
            records.append(record);write(out/'progress.json',dict(case=a.case,arm=arm,state=state,records=len(records),elapsed_seconds=time.monotonic()-begin,last=record))
            print(json.dumps(dict(case=a.case,arm=arm,state=state,seconds=round(time.monotonic()-begin,1),full=full,loss=all_data[2]['loss'])),flush=True)
    obs.close();write(out/'manifest.json',dict(status='completed',case=a.case,records=records,sources=sources,qualification=qualification,seconds=time.monotonic()-begin,peak_cuda_bytes=torch.cuda.max_memory_allocated(),smoke=a.smoke))
    write(out/'checksums.json',{f.name:dict(bytes=f.stat().st_size,sha256=sha(f)) for f in out.iterdir() if f.is_file() and f.name!='checksums.json'})
if __name__=='__main__':main()
