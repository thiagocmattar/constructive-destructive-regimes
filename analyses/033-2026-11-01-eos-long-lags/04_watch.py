"""Unattended setup, launch, per-worker retrieval/termination and local reduction."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('cloud',HERE/'01_cloud.py')
cloud=importlib.util.module_from_spec(spec);spec.loader.exec_module(cloud)


def main():
    pool=ThreadPoolExecutor(max_workers=12)
    staging={};retrieving={};errors=[]
    started=time.time()
    while True:
        try:
            lease=cloud.loadlease()
            if (HERE/'closeout.json').exists():break
            if any(not p.get('host') and not p.get('deleted_epoch') for p in lease['pods']):
                cloud.connect();lease=cloud.loadlease()
            alive=[p for p in lease['pods'] if not p.get('deleted_epoch')]
            for pod in alive:
                w=pod['worker']
                if not pod.get('host'):continue
                if pod.get('warmup_required') and not (HERE/f'warmup-worker{w}.json').exists() and w not in staging:
                    staging[w]=pool.submit(cloud.warm_one,pod)
                    continue
                packet_file=pod.get('packet_manifest','parallel_packets.json' if pod.get('run_path') else 'packets.json')
                if not (HERE/packet_file).exists():continue
                if not (HERE/f'staging-worker{w}.json').exists() and w not in staging:
                    staging[w]=pool.submit(cloud.stage_one,pod)
            for w,future in list(staging.items()):
                if future.done():
                    try:future.result()
                    except Exception as error:errors.append(dict(epoch=time.time(),worker=w,phase='staging',error=repr(error)))
                    del staging[w]
            futures={p['worker']:pool.submit(cloud.status_one,p) for p in alive if p.get('host')}
            statuses=[]
            for w,future in futures.items():
                try:statuses.append(future.result())
                except Exception as error:errors.append(dict(epoch=time.time(),worker=w,phase='status',error=repr(error)))
            statusmap={s['worker']:s for s in statuses}
            handoff_path=HERE/'handoff_captures.json'
            if handoff_path.exists() and (HERE/'handoff_packets.json').exists():
                handoffs=json.loads(handoff_path.read_text())['captures']
                activation_path=HERE/'handoff_activation.json'
                activation=json.loads(activation_path.read_text()) if activation_path.exists() else dict(activations=[])
                activated={r['worker'] for r in activation['activations']}
                for handoff in handoffs:
                    target=handoff['worker'];s=statusmap.get(target,{})
                    if target not in activated and s.get('progress',{}).get('lag',-1)>handoff['last_lag']:
                        activation['activations'].append(dict(worker=target,source_worker=handoff['source_worker'],
                            parent_run_id=handoff['parent_run_id'],anchor=handoff['anchor'],resume_lag=handoff['last_lag'],
                            first_observed_lag=s['progress']['lag'],epoch=time.time()))
                        cloud.save(activation_path,activation);activated.add(target)
                    old=handoff['source_worker'];record=f'handoff-stop-worker{old}.json'
                    if target not in activated or (HERE/record).exists() or old not in statusmap:continue
                    prior=statusmap[old];pod=next(p for p in alive if p['worker']==old)
                    if old==5:
                        safe=True
                    else:
                        first=prior.get('manifest',{}).get('windows',[{}])[0]
                        progress=prior.get('progress',{})
                        safe=first.get('last_lag',0)>256 or (progress.get('anchor')==3000 and progress.get('lag',0)>256)
                    if safe and prior.get('worker_exit') is None:
                        cloud.stop_completed_first_window(pod,record_name=record,touch_stop=True,
                            reason=f'Qualified Run208 worker{target} owns the checkpoint continuation; preserve the completed independent first profile')
            if (HERE/'parallel_plan.json').exists() and not (HERE/'serial_worker11_stop.json').exists():
                serial=statusmap.get(11,{})
                if serial.get('manifest',{}).get('windows') and serial.get('worker_exit') is None:
                    cloud.stop_completed_first_window(next(p for p in alive if p['worker']==11))
            pilot=statusmap.get(0,{})
            if pilot.get('setup_exit')=='0' and not (HERE/'calibration_decision.json').exists():
                cloud.calibrate()
            if (HERE/'calibration_decision.json').exists():
                ready=[w for w,s in statusmap.items() if s.get('setup_exit')=='0' and
                       not (HERE/f'launch-worker{w}.json').exists()]
                if ready:
                    launch_futures=[pool.submit(cloud.launch,[w]) for w in ready]
                    for future in launch_futures:future.result()
            for pod in alive:
                w=pod['worker'];s=statusmap.get(w,{})
                if s.get('worker_exit')=='0' and not (HERE/f'retrieval-worker{w}.json').exists() and w not in retrieving:
                    retrieving[w]=pool.submit(cloud.retrieve_one,pod)
                if s.get('worker_exit') not in [None,'0']:
                    errors.append(dict(epoch=time.time(),worker=w,phase='production',status=s))
                retry_path=HERE/'setup_retries.json'
                retrying={a['worker'] for a in json.loads(retry_path.read_text()).get('attempts',[]) if not a.get('terminal')} if retry_path.exists() else set()
                failed=s.get('worker_exit') not in [None,'0'] or (s.get('setup_exit') not in [None,'0'] and w not in retrying)
                if failed and not (HERE/f'retrieval-worker{w}.json').exists() and w not in retrieving:
                    retrieving[w]=pool.submit(cloud.retrieve_one,pod,True)
            for w,future in list(retrieving.items()):
                if future.done():
                    try:future.result()
                    except Exception as error:errors.append(dict(epoch=time.time(),worker=w,phase='retrieval',error=repr(error)))
                    del retrieving[w]
            if (HERE/'parallel_packets.json').exists():
                plan=json.loads((HERE/'parallel_plan.json').read_text())
                assigned={p['worker'] for p in lease['pods']}
                assigned.update(role['worker'] for p in lease['pods'] for role in p.get('role_history',[]))
                pending=[w for w in plan['workers'] if w not in assigned]
                for pod in alive:
                    if not pending:break
                    w=pod['worker']
                    if w in plan['reuse_eligible_workers'] and pod['gpu']=='RTX 4090' and (HERE/f'retrieval-worker{w}.json').exists():
                        cloud.reuse(pod,pending.pop(0))
            if (HERE/'handoff_packets.json').exists():
                assigned={p['worker'] for p in lease['pods']}
                assigned.update(role['worker'] for p in lease['pods'] for role in p.get('role_history',[]))
                if 16 not in assigned and (HERE/'handoff-stop-worker5.json').exists():
                    old=next((p for p in alive if p['worker']==5),None)
                    if old and (HERE/'retrieval-worker5.json').exists():
                        cloud.reuse(old,16,'runs/208-2026-11-01-eos-checkpoint-handoffs','handoff_packets.json')
            if any((HERE/f'retrieval-worker{p["worker"]}.json').exists() for p in alive):
                cloud.cleanup();lease=cloud.loadlease()
            cost=sum((p.get('deleted_epoch',time.time())-p['created_epoch'])/3600*p['price_per_gpu_hour'] for p in lease['pods'])
            snapshot=dict(epoch=time.time(),elapsed_seconds=time.time()-started,compute_estimate_usd=cost,
                deleted_workers=[p['worker'] for p in lease['pods'] if p.get('deleted_epoch')],
                staging_workers=list(staging),retrieving_workers=list(retrieving),statuses=statuses,errors=errors[-30:])
            cloud.save(HERE/'scratch/watch-latest.json',snapshot)
            with (HERE/'monitoring.jsonl').open('a') as f:f.write(json.dumps(snapshot)+'\n')
            print(json.dumps(dict(epoch=snapshot['epoch'],compute_estimate_usd=cost,deleted=snapshot['deleted_workers'],
                progress=[dict(worker=s['worker'],setup=s.get('setup_exit'),exit=s.get('worker_exit'),
                    **{k:v for k,v in s.get('progress',{}).items() if k in ['parent','anchor','lag','loss','seconds','peak_allocated_bytes']}) for s in statuses],errors=len(errors))),flush=True)
        except Exception as error:
            errors.append(dict(epoch=time.time(),phase='watch_loop',error=repr(error),traceback=traceback.format_exc()))
            cloud.save(HERE/'scratch/watch-errors.json',errors)
            print(traceback.format_exc(),flush=True)
        time.sleep(40)
    pool.shutdown(wait=True)
    result=subprocess.run([sys.executable,str(HERE/'02_analyze.py')],capture_output=True,text=True,creationflags=cloud.NO_WINDOW)
    cloud.save(HERE/'scratch/automatic-analysis.json',dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr))
    print('All task Pods terminated; local analysis exit:',result.returncode,flush=True)


if __name__=='__main__':main()
