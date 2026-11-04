"""Run221: fixed-step interventions from authenticated warm checkpoints on Spark."""
import argparse,copy,hashlib,json,math,os,random,signal,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from src.eos_tasks.task import configure,model_for
from src.eos_tasks.production import ProductionTask
from src.cifar_eos.core import flat,tensor_hash


def stamp():return datetime.now(timezone.utc).isoformat()
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def append(path,value):
    with Path(path).open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')
def cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
    return copy.deepcopy(value)
def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b

def lr_at(arm,k,base):
    if arm['kind']=='fixed':return base*arm['ratio']
    if arm['kind']=='cosine':
        ratio=arm['ratio']+(1-arm['ratio'])*(1+math.cos(math.pi*min(k,arm['duration'])/arm['duration']))/2
        return base*ratio
    return base


def modify_gradients_and_decay(model,raw,arm,threshold,lr):
    norm=float(raw.double().norm());factor=min(1.,threshold*arm['factor']/norm) if arm['kind']=='clip' and norm else 1.
    if arm['kind']=='clip':
        for p in model.parameters():p.grad.mul_(factor)
    before_decay=flat(model.parameters()).detach().double()
    if arm['kind']=='decay':
        with torch.no_grad():
            for p in model.parameters():
                if p.ndim>=2:p.mul_(1-lr*arm['weight_decay'])
    after_decay=flat(model.parameters()).detach().double()
    return factor,before_decay,after_decay


def rng():return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate())
def restore_rng(r):
    torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda']);np.random.set_state(r['numpy']);random.setstate(r['python'])
def validate(opt,state,base):
    if state==0:
        assert len(opt.state)==0
        return
    for group in opt.param_groups:
        assert group['lr']==base and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==0
        assert group['foreach'] is False and group['fused'] is False
        for p in group['params']:
            assert p.dtype==torch.float32 and int(opt.state[p]['step'])==state
            assert opt.state[p]['exp_avg'].dtype==torch.float32 and opt.state[p]['exp_avg_sq'].dtype==torch.float32



def run(context,arm,folder,until,stop,progress=None):
    folder.mkdir(parents=True,exist_ok=False);c=context;start=c['state'];begin=time.perf_counter()
    status=dict(status='running',case=c['case'],arm=arm['id'],start_state=start,end_state=until,started_utc=stamp(),identity=c['identity'])
    write(folder/'status.json',status)
    try:
        while True:
            if stop():status['status']='interrupted';save(c,folder/'recovery.pt',arm);break
            state=c['state'];loss,g=c['task'].gradient();gd=g.double();previous=torch.zeros_like(gd) if c['previous_gradient'] is None else c['previous_gradient'].double()
            if not math.isfinite(loss) or not bool(torch.isfinite(g).all()):raise ValueError('Nonfinite raw objective/gradient.')
            gn=float(gd.norm());pn=float(previous.norm());den=float(previous.dot(previous))
            rho=None if gn*pn==0 else float(previous.dot(gd))/(gn*pn)
            value=None if den==0 else float(previous.dot(previous-gd))/den
            row=dict(state=state,loss=loss,raw_gradient_norm=gn,rho=rho,eta_kappa_hat=value,
                parameter_norm=float(flat(c['task'].model.parameters()).double().norm()),manuscript_gradient_index=state+1,
                applied_lr=None if state==until else lr_at(arm,state-c['fork'],c['base']),utc=stamp())
            append(folder/'boundaries.jsonl',row)
            if state==until:
                c['current_gradient']=g.clone();save(c,folder/'recovery.pt',arm);os.link(folder/'recovery.pt',folder/'end.pt');status['status']='completed';break
            validate(c['opt'],state,c['base']);lr=row['applied_lr'];raw_copy=g.clone()
            factor,before,after_decay=modify_gradients_and_decay(c['task'].model,g,arm,c['threshold'],lr)
            input_norm=float(flat([p.grad for p in c['task'].model.parameters()]).double().norm())
            c['opt'].param_groups[0]['lr']=lr
            try:c['opt'].step()
            finally:c['opt'].param_groups[0]['lr']=c['base']
            after=flat(c['task'].model.parameters()).detach().double()
            if not bool(torch.isfinite(after).all()):raise ValueError('Nonfinite parameters.')
            delta=after-before;dn=float(delta.norm());moment=flat([c['opt'].state[p]['exp_avg'] for p in c['task'].model.parameters()]).double()
            mn=float(moment.norm());decay=after_decay-before
            post=dict(state=state,update=state+1,applied_lr=lr,lr_over_base=lr/c['base'],clip_factor=factor,
                optimizer_input_norm=input_norm,raw_gradient_norm=gn,moment_norm=mn,
                raw_gradient_moment_cosine=None if gn*mn==0 else float(gd.dot(moment))/(gn*mn),
                actual_increment_norm=dn,gradient_dot_actual_step=float(gd.dot(delta)),
                raw_gradient_actual_step_cosine=None if gn*dn==0 else float(gd.dot(delta))/(gn*dn),
                decay_increment_norm=float(decay.norm()),adam_component_increment_norm=float((after-after_decay).norm()))
            assert torch.equal(g,raw_copy);append(folder/'metrics.jsonl',post)
            c['previous_gradient']=g.clone();c['state']=state+1
            if c['state']%32==0:save(c,folder/'recovery.pt',arm)
            event=dict(case=c['case'],arm=arm['id'],state=c['state'],loss=loss,loss_state=state,
                elapsed_seconds=time.perf_counter()-begin,completed_updates=c['state']-start,utc=stamp())
            append(folder/'events.jsonl',event);write(folder/'progress.json',event)
            if progress:progress(event)
        if status['status']=='completed':
            status['evaluation']={name:c['task'].evaluate(name) for name in ['valid','test'] if 'x_'+name in c['task'].data}
    except BaseException as error:
        status.update(status='failed',exception=repr(error),traceback=traceback.format_exc());raise
    finally:
        status.update(completed_state=c['state'],updates=c['state']-start,elapsed_seconds=time.perf_counter()-begin,finished_utc=stamp())
        write(folder/'status.json',status)
    return status



def save(c,path,arm):
    validate(c['opt'],c['state'],c['base'])
    payload=dict(schema=1,identity=c['identity'],case=c['case'],arm=arm,completed_updates=c['state'],fork=c['fork'],model=cpu(c['task'].model.state_dict()),optimizer=cpu(c['opt'].state_dict()),previous_gradient=cpu(c['previous_gradient']),rng=cpu(rng()),current_gradient=cpu(c.get('current_gradient')))
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);os.replace(tmp,path)


def prepare(case,cfg):
    random.seed(0);np.random.seed(0);torch.manual_seed(0)
    cache=ROOT/'cache/eos-multitask'; shared=json.loads((cache/'inputs.json').read_text());sp=cfg['cases'][case]
    path=ROOT/'cache/cifar10-eos/data.pt' if case=='mlp' else cache/(case+'_data.pt')
    data={k:v.to('cuda:0') for k,v in torch.load(path,weights_only=True).items() if k.startswith(('x_','y_'))}
    model=model_for(case,cache,shared['wikitext']['vocabulary_size']).to('cuda:0');init=shared['initial'][case][0]
    model.load_state_dict(torch.load(cache/init['path'],weights_only=True));assert tensor_hash(model.parameters())==init['initial_tensor_sha256']
    opt=torch.optim.Adam(model.parameters(),lr=sp['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False,fused=False)
    return dict(task=ProductionTask(case,model,data,sp['physical']),opt=opt,case=case,base=sp['lr'],state=0,fork=sp['start'],previous_gradient=None,threshold=0.,identity={'run':'223','config_sha256':sha(HERE/'config.json'),'manifest_sha256':sha(HERE/'input_manifest.json')})


def load(c,path):
    p=torch.load(path,map_location='cpu',weights_only=False)
    assert p['identity']==c['identity'] and p['case']==c['case'] and p['fork']==c['fork']
    c['task'].model.load_state_dict(p['model']);c['opt'].load_state_dict(p['optimizer']);c['state']=p['completed_updates']
    c['previous_gradient']=None if p['previous_gradient'] is None else p['previous_gradient'].to('cuda:0');restore_rng(p['rng'])
    validate(c['opt'],c['state'],c['base']);return p


def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def compare_history(actual,historical):
    lookup={r['state']:r for r in historical};differences=[]
    for r in actual:
        h=lookup[r['state']]
        for key,hkey in [('loss','loss'),('raw_gradient_norm','gradient_norm')]:
            if r[key]!=h[hkey]:differences.append({'state':r['state'],'field':key,'actual':r[key],'historical':h[hkey],'relative_error':abs(r[key]-h[hkey])/max(abs(h[hkey]),1e-30)})
    return dict(exact=not differences,observations=len(actual),different_scalars=len(differences),max_relative_error=max([x['relative_error'] for x in differences],default=0),first_differences=differences[:8])


def qualify(c,cfg,out,stop,parent=None):
    arm=cfg['control'];start=c['state'];initial=out/'initial.pt';out.mkdir(parents=True,exist_ok=False);save(c,initial,arm)
    run(c,arm,out/'direct',start+2,stop);load(c,initial);run(c,arm,out/'one',start+1,stop)
    load(c,out/'one/end.pt');run(c,arm,out/'resumed',start+2,stop)
    a=torch.load(out/'direct/end.pt',map_location='cpu',weights_only=False);b=torch.load(out/'resumed/end.pt',map_location='cpu',weights_only=False)
    for k in ['model','optimizer','previous_gradient','rng','current_gradient']:equal(a[k],b[k])
    load(c,initial)
    write(out/'qualification.json',dict(status='passed',direct2_vs1plus1_exact=True,start=start,gradient_calls=7,updates=4))


def event_gate(values,sp):
    import importlib.util
    spec=importlib.util.spec_from_file_location('events223',ROOT/'analyses/036-2026-11-02-controlled-adam-diagnostics/02_extract_loss_spikes.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    loss=[r['loss'] for r in values];found=mod.excursions(loss);large=[]
    for e in found:
        if sp['start']<e['peak_state']<=sp['end']:
            ratio=loss[e['peak_state']]/loss[e['before_state']]
            if ratio>1.1:large.append(dict(e,ratio=ratio))
    passed=len(large)>=3 and any(e['ratio']>=2 and abs(e['peak_state']-sp['peak'])<=32 for e in large)
    return dict(passed=passed,events_gt10=large)


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--mode',choices=['leader','follower','calibrate'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--parent',type=Path);p.add_argument('--reference',type=Path);p.add_argument('--arms',default='fixed_half,cosine32_half');p.add_argument('--seconds',type=float,default=21600);a=p.parse_args()
    cfg=json.loads((HERE/'config.json').read_text());sp=cfg['cases'][a.case]
    for item in json.loads((HERE/'input_manifest.json').read_text())['files']:
        if a.case in item['tasks']:
            f=ROOT/item['path'];assert f.stat().st_size==item['bytes'] and sha(f)==item['sha256'],item['path']
    configure();torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    stopped=[False];started=time.monotonic()
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    stop=lambda:stopped[0] or time.monotonic()-started>a.seconds or (HERE/'STOP').exists()
    out=a.output;out.mkdir(parents=True,exist_ok=False)
    status=dict(status='running',mode=a.mode,case=a.case,started_utc=stamp(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,results={})
    write(out/'manifest.json',status)
    try:
        c=prepare(a.case,cfg)
        if a.mode=='calibrate':
            times=[]
            for i in range(3):
                torch.cuda.synchronize();t=time.monotonic();loss,g=c['task'].gradient();c['opt'].step();torch.cuda.synchronize();times.append(time.monotonic()-t)
            status.update(status='completed',mechanics_only=True,seconds=times,peak_allocated_bytes=torch.cuda.max_memory_allocated())
            return
        if a.mode=='leader':
            r=run(c,cfg['control'],out/'prefix',sp['start'],stop);assert r['status']=='completed'
            parent=out/'prefix/end.pt';qualify(c,cfg,out/'qualification',stop)
            historical_root=ROOT/'runs/215-2026-11-02-controlled-full-batch-adam/artifacts/attempt-01'/sp['worker']/('lr-'+format(sp['lr'],'g'))
            hist=rows(historical_root/'boundaries.jsonl');old=json.loads((historical_root/'manifest.json').read_text())
            checks=compare_history(rows(out/'prefix/boundaries.jsonl'),hist)
            same_env=old['environment']['gpu']==status['gpu'] and old['environment']['torch']==status['torch'] and old['environment']['cuda']==status['cuda']
            checks['same_gpu_torch_cuda']=same_env
            if checks['exact'] and same_env:
                run(c,cfg['control'],out/'overlap8',sp['start']+8,stop)
                checks['overlap8']=compare_history(rows(out/'overlap8/boundaries.jsonl'),hist)
                reuse=checks['overlap8']['exact']
            else:reuse=False
            checks['reuse_historical']=reuse;write(out/'historical_qualification.json',checks)
            if reuse:
                values=[r for r in hist if r['state']<=sp['end']]
                status['control']='historical_trace_reuse';write(out/'control_reuse.json',dict(source=str(historical_root),qualification=checks))
            else:
                load(c,parent);r=run(c,cfg['control'],out/'control',sp['end'],stop);assert r['status']=='completed'
                values=rows(out/'prefix/boundaries.jsonl')[:-1]+rows(out/'control/boundaries.jsonl');status['control']='fresh_same_worker'
            gate=event_gate(values,sp);write(out/'event_gate.json',gate)
            if not gate['passed']:raise RuntimeError('Frozen window does not reproduce required multi-spike control. No treatment launched; no window reselection.')
            # READY is published only after same-worker replay and control/event validation.
            write(out/'READY.json',dict(parent_sha256=sha(parent),reference_sha256=sha(out/'qualification/direct/end.pt'),control=status['control'],event_gate=gate))
        else:
            assert a.parent and a.reference;parent=a.parent;saved=load(c,parent)
            loss,g=c['task'].gradient();equal(g.cpu(),saved['current_gradient'])
            run(c,cfg['control'],out/'cross_worker2',sp['start']+2,stop)
            other=torch.load(a.reference,map_location='cpu',weights_only=False);own=torch.load(out/'cross_worker2/end.pt',map_location='cpu',weights_only=False)
            for k in ['model','optimizer','previous_gradient','rng','current_gradient']:equal(other[k],own[k])
            write(out/'cross_worker_qualification.json',dict(status='passed',fork_raw_gradient_exact=True,two_update_replay_exact=True,parent_sha256=sha(parent),reference_sha256=sha(a.reference)))
        for name in a.arms.split(','):
            if not name:continue
            arm=next(x for x in cfg['arms'] if x['id']==name);load(c,parent)
            r=run(c,arm,out/name,sp['end'],stop);status['results'][name]=r;write(out/'manifest.json',status)
            assert r['status']=='completed'
        status['status']='completed'
    except BaseException as error:
        status.update(status='failed',exception=repr(error),traceback=traceback.format_exc());raise
    finally:
        status.update(finished_utc=stamp(),elapsed_seconds=time.monotonic()-started);write(out/'manifest.json',status)
        inventory=[dict(path=f.relative_to(out).as_posix(),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='inventory.json']
        write(out/'inventory.json',dict(files=inventory,status=status['status']))

if __name__=='__main__':main()
