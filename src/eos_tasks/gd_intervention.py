"""Plain full-batch GD interventions, promoted from frozen Run233 for Run234.

One run per worker process; each configured rate reloads the same warm parent.
"""
import importlib.util
import json
import signal
import time
import traceback
from pathlib import Path

import torch


def cosine(a,b):
    den=float(a.norm()*b.norm())
    return float(a.dot(b))/den if den>0 else None


def gd_optimizer(parameters, lr):
    return torch.optim.SGD(parameters,lr=lr,momentum=0,weight_decay=0,foreach=False)


def save(c,folder,name='recovery.pt'):
    payload=dict(run=CFG['run_id'],config_sha256=w.sha(HERE/'config.json'),input_manifest_sha256=w.sha(HERE/'input_manifest.json'),
        parent_sha256=w.sha(SOURCE/'prefix/end.pt'),completed_updates=c['state'],
        model=w.cpu(c['task'].model.state_dict()),optimizer=w.cpu(c['opt'].state_dict()),rng=w.rng(),
        previous_gradient=w.cpu(c['previous_gradient']),current_gradient=w.cpu(c.get('current_gradient')))
    temp=folder/(name+'.tmp')
    torch.save(payload,temp)
    temp.replace(folder/name)


def load_gd(c,path):
    p=torch.load(path,map_location='cpu',weights_only=False)
    assert p['run']==CFG['run_id'] and p['config_sha256']==w.sha(HERE/'config.json')
    c['task'].model.load_state_dict(p['model'])
    c['opt']=gd_optimizer(c['task'].model.parameters(), c['lr'])
    c['opt'].load_state_dict(p['optimizer'])
    c['state']=p['completed_updates']
    c['previous_gradient']=p['previous_gradient'].to('cuda:0')
    w.restore_rng(p['rng'])


def branch(c,folder,end,stop):
    folder.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    initial=c['state']
    lr=c['opt'].param_groups[0]['lr']
    status=dict(status='running',start_state=initial,end_state=end,started_utc=w.stamp(),optimizer='GD',lr=lr)
    w.write(folder/'status.json',status)
    try:
        while True:
            if stop():
                save(c,folder)
                raise RuntimeError('Bounded stop')
            state=c['state']
            loss,g=c['task'].gradient()
            gd,previous=g.double(),c['previous_gradient'].double()
            assert w.math.isfinite(loss) and bool(torch.isfinite(g).all())
            gn=float(gd.norm())
            den=float(previous.dot(previous))
            record=dict(state=state,loss=loss,raw_gradient_norm=gn,rho=cosine(previous,gd),
                eta_kappa_hat=float(previous.dot(previous-gd))/den if den else None,
                parameter_norm=float(w.flat(c['task'].model.parameters()).double().norm()),
                applied_lr=lr if state<end else None,optimizer='GD',utc=w.stamp())
            w.append(folder/'boundaries.jsonl',record)
            c['current_gradient']=g.clone()
            if state==513:
                save(c,folder,'first_step.pt')
            if state==end:
                save(c,folder)
                (folder/'end.pt').hardlink_to(folder/'recovery.pt')
                break
            before=w.flat(c['task'].model.parameters()).detach().double()
            c['opt'].step()
            after=w.flat(c['task'].model.parameters()).detach().double()
            assert bool(torch.isfinite(after).all())
            delta=after-before
            w.append(folder/'metrics.jsonl',dict(state=state,update=state+1,applied_lr=lr,
                momentum=0,weight_decay=0,optimizer='GD',raw_gradient_norm=gn,optimizer_input_norm=gn,
                actual_increment_norm=float(delta.norm()),gradient_dot_actual_step=float(gd.dot(delta)),
                raw_gradient_actual_step_cosine=cosine(gd,delta),
                gd_step_rounding_residual_norm=float((delta+lr*gd).norm())))
            c['state']=state+1
            c['previous_gradient']=g.clone()
            c['current_gradient']=None
            if c['state']%32==0:
                save(c,folder)
            elapsed=time.monotonic()-started
            event=dict(state=c['state'],loss=loss,loss_state=state,completed_updates=c['state']-initial,
                examples_seen=(c['state']-initial)*5000,elapsed_seconds=elapsed,
                updates_per_second=(c['state']-initial)/elapsed,
                remaining_seconds=(end-c['state'])*elapsed/(c['state']-initial),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),utc=w.stamp())
            w.append(folder/'events.jsonl',event)
            if c['state']%32==0:
                w.write(folder/'progress.json',event)
                print(json.dumps(event),flush=True)
        status.update(status='completed',evaluation={'test':c['task'].evaluate('test')})
    except BaseException as error:
        status.update(status='failed',exception=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        status.update(completed_state=c['state'],updates=c['state']-initial,elapsed_seconds=time.monotonic()-started,finished_utc=w.stamp())
        w.write(folder/'status.json',status)
    return status


def main(here):
    global HERE, ROOT, CFG, PARENT, SOURCE, w
    HERE=Path(here).resolve()
    ROOT=HERE.parents[1]
    CFG=json.loads((HERE/'config.json').read_text())
    PARENT=ROOT/CFG['parent_run']
    SOURCE=ROOT/CFG['parent_worker']
    spec=importlib.util.spec_from_file_location('frozen223',PARENT/'01_worker.py')
    w=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    for item in json.loads((HERE/'input_manifest.json').read_text())['files']:
        p=ROOT/item['path']
        assert p.stat().st_size==item['bytes'] and w.sha(p)==item['sha256'],str(p)
    w.configure()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    out=HERE/'artifacts/attempt-01/worker'
    out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    stopped=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    stop=lambda:stopped[0] or time.monotonic()-start>CFG['runtime_limit_seconds'] or (HERE/'STOP').exists()
    status=dict(status='running',started_utc=w.stamp(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,results={})
    w.write(out/'manifest.json',status)
    parent_cfg=json.loads((PARENT/'config.json').read_text())
    def fresh(lr):
        c=w.prepare('mlp',parent_cfg)
        w.load(c,SOURCE/'prefix/end.pt')
        c['lr']=lr
        c['opt']=gd_optimizer(c['task'].model.parameters(), c['lr'])
        return c
    try:
        c=w.prepare('mlp',parent_cfg)
        saved=w.load(c,SOURCE/'prefix/end.pt')
        _,g=c['task'].gradient()
        w.equal(g.cpu(),saved['current_gradient'])
        result=w.run(c,parent_cfg['control'],out/'qualification2',514,stop)
        assert result['status']=='completed'
        old=torch.load(SOURCE/'qualification/direct/end.pt',map_location='cpu',weights_only=False)
        new=torch.load(out/'qualification2/end.pt',map_location='cpu',weights_only=False)
        for k in ['model','optimizer','previous_gradient','rng','current_gradient']:
            w.equal(old[k],new[k])
        for lr in CFG['gd_lrs']:
            name='gd_lr'+str(lr).replace('.','p')
            qualification=out/(name+'_qualification')
            branch(fresh(lr),qualification/'direct2',514,stop)
            branch(fresh(lr),qualification/'one',513,stop)
            c=fresh(lr)
            load_gd(c,qualification/'one/end.pt')
            branch(c,qualification/'resumed',514,stop)
            direct=torch.load(qualification/'direct2/end.pt',map_location='cpu',weights_only=False)
            resumed=torch.load(qualification/'resumed/end.pt',map_location='cpu',weights_only=False)
            for k in ['model','optimizer','previous_gradient','rng','current_gradient']:
                w.equal(direct[k],resumed[k])
            w.write(qualification/'qualification.json',dict(status='passed',fork_gradient_exact=True,
                two_update_adam_control_replay_exact=True,gd_two_vs_one_plus_one_exact=True,lr=lr))
            status['results'][name]=branch(fresh(lr),out/name,CFG['end'],stop)
            w.write(out/'manifest.json',status)
        status['status']='completed'
    except BaseException as error:
        status.update(status='failed',exception=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        status.update(finished_utc=w.stamp(),elapsed_seconds=time.monotonic()-start)
        w.write(out/'manifest.json',status)
        files=[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=w.sha(p))
               for p in sorted(out.rglob('*')) if p.is_file() and p.name!='inventory.json']
        w.write(out/'inventory.json',dict(status=status['status'],files=files))
