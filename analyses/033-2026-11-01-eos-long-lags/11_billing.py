"""Read posted billing for this task's exact Pod IDs after verified deletion."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).parent

def main():
    close=json.loads((HERE/'closeout.json').read_text());assert close['status']=='complete'
    start=datetime.fromtimestamp(min(p['created_epoch'] for p in close['pods'])-60,timezone.utc).isoformat()
    end=datetime.now(timezone.utc).isoformat()
    def one(pod):
        process=subprocess.run(['runpodctl','billing','pods','--pod-id',pod['id'],'--start-time',start,
            '--end-time',end,'--bucket-size','hour'],capture_output=True,text=True,timeout=45,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if process.returncode:return dict(pod_id=pod['id'],error=process.stderr[:1000],records=[])
        records=json.loads(process.stdout);assert all(r['podId']==pod['id'] for r in records)
        return dict(pod_id=pod['id'],records=records,posted_gpu_usd=sum(r['gpuAmount'] for r in records),
            posted_cpu_usd=sum(r['cpuAmount'] for r in records),posted_disk_usd=sum(r['diskAmount'] for r in records),
            posted_total_usd=sum(r['totalAmount'] for r in records))
    with ThreadPoolExecutor(max_workers=6) as pool:pods=list(pool.map(one,close['pods']))
    record=dict(observed_utc=end,requested_start=start,requested_end=end,pods=pods,
        posted_gpu_usd=sum(p.get('posted_gpu_usd',0) for p in pods),posted_disk_usd=sum(p.get('posted_disk_usd',0) for p in pods),
        posted_total_usd=sum(p.get('posted_total_usd',0) for p in pods),compute_allocation_estimate_usd=close['compute_estimate_usd'],
        billing='provisional; hourly bucket amounts can be partly posted even when bucket end is in the past',
        all_task_pods_deleted=True,task_network_volumes_created=0,unrelated_resources_excluded=True)
    (HERE/'billing.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in record.items() if k!='pods'}))

if __name__=='__main__':main()
