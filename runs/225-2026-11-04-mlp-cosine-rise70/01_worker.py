"""Run225: cosine32 down, hold to650, cosine70 rise to .75x, hold to800."""
import importlib.util,json,signal,time,traceback,math
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
cfg=json.loads((HERE/'config.json').read_text());PARENT=ROOT/cfg['parent_run'];SOURCE=ROOT/cfg['parent_worker']
spec=importlib.util.spec_from_file_location('frozen223',PARENT/'01_worker.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
original_lr=w.lr_at
def lr_at(arm,k,base):
    state=cfg['start']+k
    if state<cfg['return_state']:return original_lr(arm,k,base)
    u=min((state-cfg['return_state'])/cfg['rise_duration'],1.)
    return base*(arm['ratio']+(cfg['return_ratio']-arm['ratio'])*(1-math.cos(math.pi*u))/2)

def main():
    for item in json.loads((HERE/'input_manifest.json').read_text())['files']:
        p=ROOT/item['path'];assert p.stat().st_size==item['bytes'] and w.sha(p)==item['sha256'],str(p)
    w.configure();w.torch.use_deterministic_algorithms(True);w.torch.backends.cudnn.deterministic=True
    out=HERE/'artifacts/worker';out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();stopped=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    stop=lambda:stopped[0] or time.monotonic()-started>cfg['runtime_limit_seconds'] or (HERE/'STOP').exists()
    status=dict(status='running',started_utc=w.stamp(),gpu=w.torch.cuda.get_device_name(),torch=w.torch.__version__,cuda=w.torch.version.cuda,results={})
    w.write(out/'manifest.json',status)
    try:
        parent_cfg=json.loads((PARENT/'config.json').read_text());c=w.prepare('mlp',parent_cfg)
        saved=w.load(c,SOURCE/'prefix/end.pt');assert c['state']==cfg['start']
        loss,g=c['task'].gradient();w.equal(g.cpu(),saved['current_gradient'])
        # Replay the previously authenticated unchanged two-update reference.
        r=w.run(c,parent_cfg['control'],out/'qualification2',cfg['start']+2,stop);assert r['status']=='completed'
        old=w.torch.load(SOURCE/'qualification/direct/end.pt',map_location='cpu',weights_only=False)
        new=w.torch.load(out/'qualification2/end.pt',map_location='cpu',weights_only=False)
        for key in ['model','optimizer','previous_gradient','rng','current_gradient']:w.equal(old[key],new[key])
        w.write(out/'qualification.json',dict(status='passed',fork_gradient_exact=True,two_update_tensor_replay_exact=True,parent_sha256=w.sha(SOURCE/'prefix/end.pt')))
        w.lr_at=lr_at
        for arm in cfg['arms']:
            c=w.prepare('mlp',parent_cfg);w.load(c,SOURCE/'prefix/end.pt')
            c['identity']=dict(run='225',config_sha256=w.sha(HERE/'config.json'),manifest_sha256=w.sha(HERE/'input_manifest.json'),parent_checkpoint_sha256=w.sha(SOURCE/'prefix/end.pt'))
            # Capture the exact pre-update transition gradient, model, moments and RNG.
            gradient=c['task'].gradient
            def capture_transition():
                loss,g=gradient()
                if c['state'] in (cfg['return_state'],cfg['return_state']+cfg['rise_duration']):
                    c['current_gradient']=g.clone();w.save(c,out/arm['id']/('state'+str(c['state'])+'.pt'),arm)
                return loss,g
            c['task'].gradient=capture_transition
            r=w.run(c,arm,out/arm['id'],cfg['end'],stop);assert r['status']=='completed'
            previous=w.rows(SOURCE/arm['parent_arm']/'boundaries.jsonl');actual=w.rows(out/arm['id']/'boundaries.jsonl')
            keys=['loss','raw_gradient_norm','rho','eta_kappa_hat','parameter_norm']
            checked=0
            for a,b in zip(actual,previous):
                if a['state']>cfg['return_state']:break
                assert a['state']==b['state']
                for key in keys:assert a[key]==b[key],(arm['id'],a['state'],key)
                checked+=1
            # Actual Adam updates before650 must also reproduce exactly.
            old_metrics=w.rows(SOURCE/arm['parent_arm']/'metrics.jsonl');new_metrics=w.rows(out/arm['id']/'metrics.jsonl')
            for a,b in zip(new_metrics,old_metrics):
                if a['state']>=cfg['return_state']:break
                w.equal(a,b)
            assert checked==139 and len(actual)==289 and len(new_metrics)==288
            assert all(a['applied_lr']==.00075 for a in actual if 720<=a['state']<800)
            assert actual[650-512]['applied_lr']==.0005 and actual[685-512]['applied_lr']==.000625
            assert all(actual[j+1]['applied_lr']>=actual[j]['applied_lr'] for j in range(650-512,720-512))
            old_transition=w.torch.load(ROOT/cfg['transition_reference'],map_location='cpu',weights_only=False)
            new_transition=w.torch.load(out/arm['id']/'state650.pt',map_location='cpu',weights_only=False)
            for key in ['model','optimizer','previous_gradient','rng','current_gradient']:w.equal(old_transition[key],new_transition[key])
            r['transition650_tensor_replay_exact']=True
            r['prefix_parity']=dict(exact=True,boundaries=checked,fields=keys,actual_update_rows=138)
            status['results'][arm['id']]=r;w.write(out/'manifest.json',status)
        status['status']='completed'
    except BaseException as error:
        status.update(status='failed',exception=repr(error),traceback=traceback.format_exc());raise
    finally:
        status.update(finished_utc=w.stamp(),elapsed_seconds=time.monotonic()-started);w.write(out/'manifest.json',status)
        inventory=[dict(path=f.relative_to(out).as_posix(),bytes=f.stat().st_size,sha256=w.sha(f)) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='inventory.json']
        w.write(out/'inventory.json',dict(status=status['status'],files=inventory))
if __name__=='__main__':main()
