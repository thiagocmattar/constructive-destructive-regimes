"""Run227: two cosine32 decays, to .75x from512 then .5x from650."""
import importlib.util,json,signal,time,traceback,math
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
cfg=json.loads((HERE/'config.json').read_text());PARENT=ROOT/cfg['parent_run'];SOURCE=ROOT/cfg['parent_worker']
spec=importlib.util.spec_from_file_location('frozen223',PARENT/'01_worker.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
original_lr=w.lr_at
def lr_at(arm,k,base):
    state=cfg['start']+k
    if state<cfg['second_start']:return original_lr(arm,k,base)
    u=min((state-cfg['second_start'])/cfg['second_duration'],1.)
    return base*(cfg['second_ratio']+(arm['ratio']-cfg['second_ratio'])*(1+math.cos(math.pi*u))/2)

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
            c['identity']=dict(run='227',config_sha256=w.sha(HERE/'config.json'),manifest_sha256=w.sha(HERE/'input_manifest.json'),parent_checkpoint_sha256=w.sha(SOURCE/'prefix/end.pt'))
            gradient=c['task'].gradient
            def capture_floor():
                loss,g=gradient()
                if c['state'] in (544,650,682):
                    c['current_gradient']=g.clone();w.save(c,out/arm['id']/('state'+str(c['state'])+'.pt'),arm)
                return loss,g
            c['task'].gradient=capture_floor
            result=w.run(c,arm,out/arm['id'],cfg['end'],stop);assert result['status']=='completed'
            actual=w.rows(out/arm['id']/'boundaries.jsonl');metrics=w.rows(out/arm['id']/'metrics.jsonl')
            assert [x['state'] for x in actual]==list(range(512,801)) and len(metrics)==288
            assert actual[0]['applied_lr']==.001 and actual[16]['applied_lr']==.000875
            assert all(x['applied_lr']==.00075 for x in actual if 544<=x['state']<=650)
            assert actual[666-512]['applied_lr']==.000625
            assert all(x['applied_lr']==.0005 for x in actual if 682<=x['state']<800)
            reference_folder=ROOT/cfg['reference_folder']
            reference_rows=w.rows(reference_folder/'boundaries.jsonl')
            for own,reference in zip(actual[:140],reference_rows[:140]):
                for key in ['state','loss','rho','raw_gradient_norm','eta_kappa_hat','parameter_norm']:assert own[key]==reference[key],(own['state'],key)
            for own,reference in zip(metrics[:139],w.rows(reference_folder/'metrics.jsonl')[:139]):w.equal(own,reference)
            old_state=w.torch.load(reference_folder/'state544.pt',map_location='cpu',weights_only=False)
            own_state=w.torch.load(out/arm['id']/'state544.pt',map_location='cpu',weights_only=False)
            for key in ['model','optimizer','previous_gradient','current_gradient','rng']:w.equal(old_state[key],own_state[key])
            result['pre_second_decay_replay']=dict(exact=True,boundaries512_through651=140,updates512_through650=139,state544_tensors_exact=True)
            control=w.rows(SOURCE/'control/boundaries.jsonl')
            for own,reference in zip(actual[:2],control[:2]):
                for key in ['state','loss','rho','raw_gradient_norm','eta_kappa_hat','parameter_norm']:assert own[key]==reference[key],key
            result['common_states_512_513_exact']=True
            status['results'][arm['id']]=result;w.write(out/'manifest.json',status)
        status['status']='completed'
    except BaseException as error:
        status.update(status='failed',exception=repr(error),traceback=traceback.format_exc());raise
    finally:
        status.update(finished_utc=w.stamp(),elapsed_seconds=time.monotonic()-started);w.write(out/'manifest.json',status)
        inventory=[dict(path=f.relative_to(out).as_posix(),bytes=f.stat().st_size,sha256=w.sha(f)) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='inventory.json']
        w.write(out/'inventory.json',dict(status=status['status'],files=inventory))
if __name__=='__main__':main()
