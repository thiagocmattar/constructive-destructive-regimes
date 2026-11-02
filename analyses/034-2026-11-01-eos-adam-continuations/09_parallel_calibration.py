"""Parallel placement of unchanged Adam arms on already calibrated warm workers."""
from concurrent.futures import ThreadPoolExecutor
import copy
import importlib.util
import json
from pathlib import Path
import time

HERE=Path(__file__).parent;ROOT=HERE.parents[1]
RUN=ROOT/'runs/211-2026-11-01-eos-parallel-adam-calibration'
spec=importlib.util.spec_from_file_location('cloud034',HERE/'01_cloud.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
c.PROD=RUN


def prepare():
    assert not (RUN/'config.json').exists(),'Parallel scientific inputs already frozen'
    base=json.loads((c.CAL/'config.json').read_text())
    placements={2000:[18,9,11,14,15],3000:[16,17,10,12,13]}
    jobs=[]
    for anchor,workers in placements.items():
        source=next(j for j in base['jobs'] if j['parent_run_id']=='121' and j['anchor']==anchor)
        for rate,worker in zip(source['rates'],workers):
            job=copy.deepcopy(source);job.update(rates=[rate],worker=worker);jobs.append(job)
    cfg=copy.deepcopy(base)
    cfg.update(run_id='211',phase='parallel_calibration',jobs=jobs,
        inputs_manifest=RUN.relative_to(ROOT).as_posix()+'/input_manifest.json',
        stop_request=RUN.relative_to(ROOT).as_posix()+'/stop-request',
        purpose='Same10 cold-Adam arms at Run121 states2000/3000, parallel placement to avoid delayed setup; no changed model, data, rate, horizon or numerical tolerance')
    RUN.mkdir(parents=True,exist_ok=True)
    (RUN/'01_worker.py').write_bytes((c.CAL/'01_worker.py').read_bytes())
    (RUN/'README.md').write_text('# Run211: parallel placement of unchanged Adam calibration arms\n\nTen independent64-update arms at Run121 states2000/3000 use the same five rates,\nexact source checkpoints and full objectives as Run209. Each fresh arm is placed\non an already calibrated warm4090 worker. This avoids slow setup and preserves\nthe selection experiment; supplemental arms are canonical by this infrastructure\nplan, before their results are observed. Original partial/redundant arms remain\nvisible and do not choose LR. Analysis034 owns the joint records and cost cap.\n',encoding='utf-8',newline='\n')
    c.save(RUN/'config.json',cfg)
    spec=importlib.util.spec_from_file_location('prepare034',HERE/'00_prepare.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.packet(RUN)
    c.save(HERE/'parallel_calibration_plan.json',dict(run_path=RUN.relative_to(ROOT).as_posix(),
        config_sha256=c.sha(RUN/'config.json'),superseded_workers=[1,2],
        canonical_parent='121',canonical_anchors=[2000,3000],placements=placements,
        production_placements={'121-2000':9,'121-3000':11},
        rule='All5 arms at each of these two states use Run211 regardless of outcomes; no outcome-based choice of replay. Same scientific arms, new placement.'))


def one(worker):
    while True:
        pod=next(p for p in reversed(c.lease()['pods']) if p['worker']==worker and not p.get('deleted_epoch'))
        receipt=c.receipt_path(pod,'209')
        if receipt.exists() and json.loads(receipt.read_text())['scientific_status']=='complete':break
        time.sleep(15)
    c.setup(pod,'211')
    while True:
        status=c.status(pod,'211')
        if status.get('setup_exit')=='0':c.launch(pod,'211')
        if status.get('worker_exit')=='0':return c.retrieve(pod,'211')
        if status.get('worker_exit') is not None or status.get('setup_exit') not in [None,'0']:
            c.retrieve(pod,'211',failed=True)
            raise RuntimeError(('Parallel calibration arm failed',worker,status))
        time.sleep(20)


def main():
    if not (RUN/'config.json').exists():prepare()
    cfg=json.loads((RUN/'config.json').read_text())
    with ThreadPoolExecutor(max_workers=10) as pool:receipts=list(pool.map(one,sorted({j['worker'] for j in cfg['jobs']})))
    canonical={}
    for receipt in receipts:
        root=ROOT/receipt['artifact_root']
        for path in root.glob('*-*/result.json'):
            r=json.loads(path.read_text());key=(r['parent_run_id'],r['anchor'])
            for arm in r['arms']:
                assert arm['status'] in ['complete','nonfinite']
                if arm['status']=='complete':assert arm['last_lag']==64
            if key not in canonical:canonical[key]=dict(r,arms=[])
            canonical[key]['arms'].extend(r['arms'])
    assert len(canonical)==2
    for r in canonical.values():
        assert sorted(a['lr'] for a in r['arms'])==[1e-5,3e-5,1e-4,3e-4,1e-3]
        r['arms'].sort(key=lambda a:a['lr'])
    c.save(HERE/'parallel_calibration_ready.json',dict(run_id='211',receipts=receipts,
        canonical_results=list(canonical.values()),config_sha256=c.sha(RUN/'config.json'),epoch=time.time()))
    print('All10 parallel calibration arms retrieved and hash verified',flush=True)


if __name__=='__main__':main()
