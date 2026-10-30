"""Bounded completion, verified retrieval and cleanup of this task's resources."""
from pathlib import Path
import datetime
import importlib.util
import json
import subprocess
import sys
import time

HERE=Path(__file__).parent;ROOT=HERE.resolve().parents[1]
spec=importlib.util.spec_from_file_location('retrieve',HERE/'04_retrieve.py')
retrieve=importlib.util.module_from_spec(spec);spec.loader.exec_module(retrieve)
c=retrieve.control


def event(phase,**extra):
    record=dict(epoch=time.time(),phase=phase,**extra)
    c.save(HERE/'controller.json',record);print(json.dumps(record),flush=True)


def main():
    lease=c.lease()
    while time.time()<lease['training_deadline_epoch']:
        state=c.status();phase=state.get('manifest',{}).get('status')
        if phase=='failed':raise RuntimeError(state['manifest'])
        if phase=='complete' and state.get('production_exit')=='0':break
        event('waiting_for_production',progress=state.get('progress'));time.sleep(60)
    else:raise TimeoutError('Production deadline; leave evidence for recovery')
    ready=c.ssh('test -f /workspace/constructive/worker-control/package.json && echo ready || echo pending\n').strip()
    if ready!='ready':
        event('packaging');retrieve.prepare()
    while time.time()<lease['stop_epoch']:
        ready=c.ssh('test -f /workspace/constructive/worker-control/package.json && echo ready || echo pending\n').strip()
        if ready=='ready':break
        time.sleep(30)
    else:raise TimeoutError('Archive not ready before compute cutoff')
    event('retrieving');retrieve.retrieve()
    receipt=json.loads((HERE/'retrieval.json').read_text());assert receipt['verified']
    event('checking_scientific_coverage')
    subprocess.run([sys.executable,str(HERE/'02_verify.py'),'--inventory'],cwd=ROOT,check=True)
    # Fresh ownership checks directly before deletion. Only this task's resources.
    pod=c.cli('pod','get',lease['pod_id']);assert pod['name']==lease['pod_name']
    volume=c.cli('network-volume','get',lease['volume_id']);assert volume['name']==lease['volume_name']
    event('cleanup_verified_evidence')
    c.cli('pod','delete',lease['pod_id'])
    for attempt in range(6):
        try:
            result=c.cli('network-volume','delete',lease['volume_id']);break
        except RuntimeError:
            if attempt==5:raise
            time.sleep(10)
    pods=c.cli('pod','list');volumes=c.cli('network-volume','list')
    assert lease['pod_id'] not in json.dumps(pods)
    assert lease['volume_id'] not in json.dumps(volumes)
    completed=time.time();hours=(completed-lease['created_epoch'])/3600
    closeout=dict(completed_epoch=completed,pod_id=lease['pod_id'],volume_id=lease['volume_id'],
        pod_deleted=True,volume_deleted=True,retained_cloud_storage_gb=0,
        elapsed_hours=hours,gpu_rate_usd_hour=.89,compute_cost_upper_estimate_usd=hours*.89,
        storage_cost_upper_estimate_usd=hours*(30*.10+30*.07)/(30*24),billing_status='provisional; posted records may lag',
        evidence_retrieved_and_verified=True,spark_accessed=False,unrelated_resources_modified=False)
    start=datetime.datetime.fromtimestamp(lease['created_epoch'],datetime.timezone.utc).isoformat()
    end=datetime.datetime.fromtimestamp(completed,datetime.timezone.utc).isoformat()
    for kind,args in [('pods',['--pod-id',lease['pod_id']]),('network-volume',[])]:
        try:
            value=c.cli('billing',kind,'--start-time',start,'--end-time',end,'--bucket-size','hour',*args)
            # Volume billing is account-wide; retain only the task-owned matching entries.
            if kind=='network-volume':
                def matching_records(node):
                    if isinstance(node,dict):
                        if any(isinstance(v,str) and v==lease['volume_id'] for v in node.values()):return [node]
                        return [r for v in node.values() for r in matching_records(v)]
                    if isinstance(node,list):return [r for v in node for r in matching_records(v)]
                    return []
                value={'task_filter':lease['volume_id'],'matching_entries':matching_records(value)}
            closeout['posted_'+kind]=value
        except Exception as error:closeout['posted_'+kind+'_error']=repr(error)
    c.save(HERE/'closeout.json',closeout);event('closed',closeout=closeout)


if __name__=='__main__':main()
