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
    for group in opt.param_groups:
        assert group['lr']==base and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==0
        assert group['foreach'] is False and group['fused'] is False
        for p in group['params']:
            assert p.dtype==torch.float32 and int(opt.state[p]['step'])==state
            assert opt.state[p]['exp_avg'].dtype==torch.float32 and opt.state[p]['exp_avg_sq'].dtype==torch.float32


def prepare(case,cfg,parents,identity):
    random.seed(0);np.random.seed(0);torch.manual_seed(0);cache=ROOT/'cache/eos-multitask'
    shared=json.loads((cache/'inputs.json').read_text());spec=cfg['cases'][case]
    path=ROOT/'cache/cifar10-eos/data.pt' if case=='mlp' else cache/(case+'_data.pt')
    data={k:v.to('cuda:0') for k,v in torch.load(path,weights_only=True).items() if k.startswith(('x_','y_'))}
    model=model_for(case,cache,shared['wikitext']['vocabulary_size']).to('cuda:0')
    parent=torch.load(ROOT/parents[case]['compact_checkpoint'],map_location='cpu',weights_only=False)
    assert parent['task']==case and parent['completed_updates']==1024 and parent['identity']['lr']==spec['lr']
    model.load_state_dict(parent['model'])
    opt=torch.optim.Adam(model.parameters(),lr=spec['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False,fused=False)
    opt.load_state_dict(parent['optimizer']);validate(opt,1024,spec['lr'])
    torch.set_rng_state(parent['torch_rng']);torch.cuda.set_rng_state_all(parent['cuda_rng'])
    context=dict(task=ProductionTask(case,model,data,spec['physical']),opt=opt,case=case,base=spec['lr'],state=1024,
        previous_gradient=parent['previous_gradient'].to('cuda:0'),identity=identity,
        threshold=parents[case]['clipping_threshold_reference_median64'])
    return context,parent


def save(context,path,arm):
    c=context;validate(c['opt'],c['state'],c['base'])
    payload=dict(schema=1,identity=c['identity'],case=c['case'],arm=arm,completed_updates=c['state'],
        model=cpu(c['task'].model.state_dict()),optimizer=cpu(c['opt'].state_dict()),previous_gradient=cpu(c['previous_gradient']),rng=cpu(rng()))
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def load_branch(context,path,arm):
    p=torch.load(path,map_location='cpu',weights_only=False)
    assert p['identity']==context['identity'] and p['case']==context['case'] and p['arm']==arm
    context['task'].model.load_state_dict(p['model']);context['opt'].load_state_dict(p['optimizer']);context['state']=p['completed_updates']
    context['previous_gradient']=p['previous_gradient'].to('cuda:0');restore_rng(p['rng']);validate(context['opt'],context['state'],context['base'])


def run(context,arm,folder,until,stop,progress=None):
    folder.mkdir(parents=True,exist_ok=False);c=context;start=c['state'];begin=time.perf_counter()
    status=dict(status='running',case=c['case'],arm=arm['id'],start_state=start,end_state=until,started_utc=stamp(),identity=c['identity'])
    write(folder/'status.json',status)
    try:
        while True:
            if stop():status['status']='interrupted';save(c,folder/'recovery.pt',arm);break
            state=c['state'];loss,g=c['task'].gradient();gd=g.double();previous=c['previous_gradient'].double()
            if not math.isfinite(loss) or not bool(torch.isfinite(g).all()):raise ValueError('Nonfinite raw objective/gradient.')
            gn=float(gd.norm());pn=float(previous.norm());den=float(previous.dot(previous))
            rho=None if gn*pn==0 else float(previous.dot(gd))/(gn*pn)
            value=None if den==0 else float(previous.dot(previous-gd))/den
            row=dict(state=state,loss=loss,raw_gradient_norm=gn,rho=rho,eta_kappa_hat=value,
                parameter_norm=float(flat(c['task'].model.parameters()).double().norm()),manuscript_gradient_index=state+1,
                applied_lr=None if state==until else lr_at(arm,state-1024,c['base']),utc=stamp())
            append(folder/'boundaries.jsonl',row)
            if state==until:
                save(c,folder/'recovery.pt',arm);os.link(folder/'recovery.pt',folder/'end.pt');status['status']='completed';break
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
            if (c['state']-1024)%16==0:save(c,folder/'recovery.pt',arm)
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


def main():
    cfg=json.loads((HERE/'config.json').read_text());parents=json.loads((HERE/'parent_manifest.json').read_text());manifest=json.loads((HERE/'input_manifest.json').read_text())
    for item in manifest['files']:
        assert (ROOT/item['path']).stat().st_size==item['bytes'] and sha(ROOT/item['path'])==item['sha256'],item['path']
    configure();torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    deadline=datetime.fromisoformat(cfg['compute_deadline_utc']).timestamp();stopped=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    stop=lambda:stopped[0] or time.time()>=deadline or (HERE/'STOP').exists()
    output=HERE/'artifacts/attempts/001-spark';output.mkdir(parents=True,exist_ok=False)
    identity=dict(config_sha256=sha(HERE/'config.json'),manifest_sha256=sha(HERE/'input_manifest.json'),torch=torch.__version__,cuda=torch.version.cuda,
        deterministic=True,tf32=False,physical={k:v['physical'] for k,v in cfg['cases'].items()})
    packet=dict(status='running',run_id='221',started_utc=stamp(),identity=identity,qualification={},results={},skipped_optional=[])
    write(output/'manifest.json',packet);begin=time.perf_counter()
    try:
        for case in cfg['cases']:
            c,parent=prepare(case,cfg,parents,identity);loss,g=c['task'].gradient();reference=parent['current_gradient_reference'].to('cuda:0')
            error=float((g.double()-reference.double()).norm()/reference.double().norm())
            history=json.loads((ROOT/parents[case]['history']).read_text());old_loss=history[-1]['loss'];loss_error=abs(loss-old_loss)/max(abs(old_loss),1e-30)
            assert error<=cfg['bridge_gates']['max_relative_gradient_error'] and loss_error<=cfg['bridge_gates']['max_relative_loss_error'],(case,error,loss_error)
            del c,parent,g,reference;torch.cuda.empty_cache();arm=cfg['arms'][0]
            for label,end,resume in [('direct2',1026,None),('prefix1',1025,None),('resume2',1026,'prefix1')]:
                c,parent=prepare(case,cfg,parents,identity)
                if resume:load_branch(c,output/'qualification'/case/resume/'recovery.pt',arm)
                run(c,arm,output/'qualification'/case/label,end,stop)
                del c,parent;torch.cuda.empty_cache()
            a=torch.load(output/'qualification'/case/'direct2/end.pt',map_location='cpu',weights_only=False)
            b=torch.load(output/'qualification'/case/'resume2/end.pt',map_location='cpu',weights_only=False)
            for key in ['model','optimizer','previous_gradient','rng']:equal(a[key],b[key])
            packet['qualification'][case]=dict(status='passed',gradient_relative_error=error,loss_relative_error=loss_error,
                direct2_vs1plus1_exact=True,updates=4,full_gradient_calls=8)
            write(output/'manifest.json',packet);print(stamp(),'qualification',case,packet['qualification'][case],flush=True)
            del a,b
        for arm in cfg['arms']:
            estimate=(cfg['updates']+1)*sum(cfg['runtime_profile_seconds'].values())*1.25+300
            if not arm['primary'] and deadline-time.time()<estimate:
                packet['skipped_optional'].append(arm['id']);write(output/'manifest.json',packet);continue
            for case in cfg['cases']:
                if stop():raise TimeoutError('Overnight deadline/STOP reached.')
                c,parent=prepare(case,cfg,parents,identity);del parent
                def progress(event):
                    event.update(completed_conditions=len(packet['results']),elapsed_worker_seconds=time.perf_counter()-begin)
                    write(output/'progress.json',event)
                key=case+'/'+arm['id'];print(stamp(),'START',key,flush=True)
                try:packet['results'][key]=run(c,arm,output/'conditions'/case/arm['id'],cfg['end_state'],stop,progress)
                except Exception as error:
                    packet['results'][key]=json.loads((output/'conditions'/case/arm['id']/'status.json').read_text())
                    if isinstance(error,TimeoutError):raise
                write(output/'manifest.json',packet);print(stamp(),'END',key,packet['results'][key]['status'],flush=True)
                del c;torch.cuda.empty_cache()
        packet['status']='completed' if all(v['status']=='completed' for v in packet['results'].values()) else 'completed_with_failures'
    except BaseException as error:
        packet.update(status='interrupted' if isinstance(error,TimeoutError) else 'failed',exception=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        packet.update(finished_utc=stamp(),elapsed_worker_seconds=time.perf_counter()-begin)
        write(output/'manifest.json',packet);write(HERE/'status.json',packet)
        inventory=[]
        for p in sorted((HERE/'artifacts').rglob('*')):
            if p.is_file() and p.name!='inventory.json':inventory.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
        write(output/'inventory.json',dict(status=packet['status'],files=inventory))


if __name__=='__main__':main()
