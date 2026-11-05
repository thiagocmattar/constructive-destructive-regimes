"""Reproduce planning arithmetic from dated quotes and same-project timings."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    quote=read(HERE/'compute_quote.json')
    price=next(g['securePricePerHr'] for g in quote['gpus'] if g['displayName']=='RTX 4090')
    a100=next(g['securePricePerHr'] for g in quote['gpus'] if g['displayName']=='A100 PCIe')
    runpath=ROOT/'runs/215-2026-11-02-controlled-full-batch-adam/results.json'
    sparkpath=ROOT/'analyses/043-2026-11-03-spark-fixed-step-interventions/results.json'
    measured=[]
    for c in read(runpath)['conditions']:
        if c['environment']['gpu']=='NVIDIA GeForce RTX 4090':
            measured.append(dict(task=c['task'],lr=c['lr'],seconds=c['elapsed_seconds'],updates=1024))
    cloud={task:sum(c['seconds'] for c in measured if c['task']==task)/sum(c['task']==task for c in measured)/3600 for task in ['resnet','transformer']}
    spark={}
    for task in ['resnet','transformer']:
        times=[c['runner_seconds'] for c in read(sparkpath)['comparison'] if c['case']==task]
        spark[task]=dict(branch_seconds_min=min(times),branch_seconds_max=max(times),
            projected_1024_hours=sum(times)/len(times)*16/3600)
    stages=[]
    for name,hours,maximum,costcap,rate,disk in [
        ('A',[5,8],12,12,price,80),('B+C',[38,45],60,55,price,80),
        ('D_optional',[3,8],16,15,price,80),('E_optional',[20,48],60,120,a100,150)]:
        billed_rate=rate+disk*.10/720
        stages.append(dict(stage=name,scenario_gpu_hours=hours,maximum_gpu_hours=maximum,
            usd_per_hour_including_disk=billed_rate,scenario_usd=[h*billed_rate for h in hours],
            maximum_hours_usd=maximum*billed_rate,proposed_cap_usd=costcap))
        assert maximum*billed_rate<=costcap
    core=[sum(s['scenario_usd'][i] for s in stages[:2]) for i in [0,1]]
    output=dict(status='planning_only_not_launched',price_checked_utc=quote['checked_utc'],
        timing_scope='Historical same-project workloads; new HVP/probe overhead unmeasured.',
        measured_cloud=measured,cloud_baseline_hours_per_1024=cloud,spark=spark,
        remaining_eight_resnet_plus_eight_transformer_baseline_gpu_hours=8*sum(cloud.values()),
        branch_4032_updates_cloud_baseline_gpu_hours=1344/1024*sum(cloud.values()),
        stages=stages,core_scenario_usd=core,core_proposed_cap_usd=70,
        optional_caps_excluded_from_core=True,
        sources={p.relative_to(ROOT).as_posix():sha(p) for p in [runpath,sparkpath,HERE/'compute_quote.json',HERE/'03_estimate_compute.py']})
    (HERE/'compute_estimates.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(output))

if __name__=='__main__':main()
