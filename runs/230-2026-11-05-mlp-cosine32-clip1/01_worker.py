"""Run230: combine Run223 cosine32 decay with Run228 global clipping before Adam."""
import importlib.util,json,signal,time,traceback
from unittest.mock import patch
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
cfg=json.loads((HERE/'config.json').read_text());PARENT=ROOT/cfg['parent_run'];SOURCE=ROOT/cfg['parent_worker']
spec=importlib.util.spec_from_file_location('frozen223',PARENT/'01_worker.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
def main():
    for item in json.loads((HERE/'input_manifest.json').read_text())['files']:
        p=ROOT/item['path'];assert p.stat().st_size==item['bytes'] and w.sha(p)==item['sha256'],str(p)
    w.configure();w.torch.use_deterministic_algorithms(True);w.torch.backends.cudnn.deterministic=True
    out=HERE/'artifacts/worker';out.mkdir(parents=True,exist_ok=False);started=time.monotonic();stopped=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    stop=lambda:stopped[0] or time.monotonic()-started>cfg['runtime_limit_seconds'] or (HERE/'STOP').exists()
    status=dict(status='running',started_utc=w.stamp(),gpu=w.torch.cuda.get_device_name(),torch=w.torch.__version__,cuda=w.torch.version.cuda,results={});w.write(out/'manifest.json',status)
    try:
        parent_cfg=json.loads((PARENT/'config.json').read_text());c=w.prepare('mlp',parent_cfg)
        saved=w.load(c,SOURCE/'prefix/end.pt');assert c['state']==cfg['start'];loss,g=c['task'].gradient();w.equal(g.cpu(),saved['current_gradient'])
        r=w.run(c,parent_cfg['control'],out/'qualification2',cfg['start']+2,stop);assert r['status']=='completed'
        old=w.torch.load(SOURCE/'qualification/direct/end.pt',map_location='cpu',weights_only=False);new=w.torch.load(out/'qualification2/end.pt',map_location='cpu',weights_only=False)
        for key in ['model','optimizer','previous_gradient','rng','current_gradient']:w.equal(old[key],new[key])
        w.write(out/'qualification.json',dict(status='passed',fork_gradient_exact=True,two_update_tensor_replay_exact=True,parent_sha256=w.sha(SOURCE/'prefix/end.pt')))
        for arm in cfg['arms']:
            c=w.prepare('mlp',parent_cfg);w.load(c,SOURCE/'prefix/end.pt');c['threshold']=cfg['clip_threshold']
            c['identity']=dict(run='230',config_sha256=w.sha(HERE/'config.json'),manifest_sha256=w.sha(HERE/'input_manifest.json'),parent_checkpoint_sha256=w.sha(SOURCE/'prefix/end.pt'))
            native_gradient_policy = w.modify_gradients_and_decay
            def clip_before_adam(model, raw, schedule, threshold, lr):
                return native_gradient_policy(model, raw, cfg['gradient_policy'], threshold, lr)

            # The arm supplies the original cosine schedule to lr_at. Only the independent
            # gradient policy is replaced, and only during the treatment branch.
            with patch.object(w, 'modify_gradients_and_decay', clip_before_adam):
                result = w.run(c, arm, out / arm['id'], cfg['end'], stop)
            assert result['status'] == 'completed'
            actual=w.rows(out/arm['id']/'boundaries.jsonl');metrics=w.rows(out/arm['id']/'metrics.jsonl');assert [x['state'] for x in actual]==list(range(512,801)) and len(metrics)==288
            cosine_reference = w.rows(SOURCE / 'cosine32_half/metrics.jsonl')
            assert len(cosine_reference) == len(metrics)
            for x, reference in zip(metrics, cosine_reference):
                assert x['state'] == reference['state']
                expected=min(1.,cfg['clip_threshold']/x['raw_gradient_norm']) if x['raw_gradient_norm'] else 1.
                assert x['clip_factor']==expected and x['applied_lr']==reference['applied_lr'] and x['decay_increment_norm']==0
                assert x['optimizer_input_norm']<=cfg['clip_threshold']+2e-7
                assert abs(x['optimizer_input_norm']-x['raw_gradient_norm']*expected)<2e-7
            control=w.rows(SOURCE/'control/boundaries.jsonl');matched=0
            for own,ref in zip(actual,control):
                keys=['state','loss','rho','raw_gradient_norm','eta_kappa_hat','parameter_norm']
                if not all(own[k]==ref[k] for k in keys):break
                matched+=1
            assert matched>=1
            result['common_initial_boundaries_exact']=matched;result['clipped_updates']=sum(x['clip_factor']<1 for x in metrics);result['clip_factor_min']=min(x['clip_factor'] for x in metrics)
            status['results'][arm['id']]=result;w.write(out/'manifest.json',status)
        status['status']='completed'
    except BaseException as error:status.update(status='failed',exception=repr(error),traceback=traceback.format_exc());raise
    finally:
        status.update(finished_utc=w.stamp(),elapsed_seconds=time.monotonic()-started);w.write(out/'manifest.json',status)
        inventory=[dict(path=f.relative_to(out).as_posix(),bytes=f.stat().st_size,sha256=w.sha(f)) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='inventory.json'];w.write(out/'inventory.json',dict(status=status['status'],files=inventory))
if __name__=='__main__':main()
