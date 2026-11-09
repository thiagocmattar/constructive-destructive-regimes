"""Verify completed H050 retention and task-only teardown, then record provisional cost."""
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cloud',HERE/'48_cloud_conventional.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def main():
    leases=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(c.S.glob('lease-*.json'))]
    aborted=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(c.S.glob('aborted-*.json'))]
    assert len(leases)==7 and {p['name'] for p in leases}==set(c.JOBS)
    assert all(p.get('deleted_epoch') and p.get('retrieval_verified') for p in leases)
    assert all(p.get('deleted_epoch') for p in aborted)
    all_ids={p['id'] for p in leases+aborted}
    present=[p['id'] for p in c.x.cli('pod','list') if p['id'] in all_ids]
    assert not present,('Task resources remain',present)
    arms=[];prestaged=[]
    for p in leases:
        root=c.ROOT/p['output'];v=json.loads((root/'verification.json').read_text(encoding='utf-8'))
        a=json.loads((root/'complete_prefix_audit.json').read_text(encoding='utf-8'))
        assert v['status']==a['status']=='passed'
        assert {r['arm'] for r in v['arms']}==set(p['arms'])
        receipt=c.ROOT/p['run']/(p['name']+'_immutable_prestage.json')
        if receipt.exists():
            staged=json.loads(receipt.read_text(encoding='utf-8'));assert staged['status']=='passed'
            final={r['path']:r for r in json.loads((root/'transfer_inventory.json').read_text(encoding='utf-8'))['files']}
            for item in staged['files']:assert item==final[item['path']]
            prestaged.append(staged)
        arms.extend(dict(worker=p['name'],**r) for r in v['arms'])
    assert len(prestaged)==5
    assert len(arms)==10
    assert sum(a['updates'] for a in arms)==3584
    assert sum(a['boundaries'] for a in arms)==3594
    assert sum(len(a['audits']) for a in arms)==20
    roots=[c.SPARKROOT+'/'+p['output'] for p in leases]
    key=c.SPARKROOT+'/analyses/065-2026-11-07-mlp-frozen-lr-double/scratch/h050-retention-key'
    retained=json.loads(c.spark(f'''python3 - <<'PY'
import hashlib,json,subprocess
from pathlib import Path
reports=[]
for name in {roots!r}:
 root=Path(name);inventory=json.loads((root/'transfer_inventory.json').read_text())
 for item in inventory['files']:
  path=root/item['path']
  with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
  assert path.stat().st_size==item['bytes'] and digest==item['sha256'],str(path)
 physical=int(subprocess.check_output(['du','-s','-B1',str(root)],text=True).split()[0])
 reports.append(dict(root=name,files=len(inventory['files']),logical_inventory_bytes=sum(i['bytes'] for i in inventory['files']),disk_bytes=physical,status='passed'))
removed=[]
for suffix in ['', '.pub', '.hosts', '.hosts.old']:
 path=Path({key!r}+suffix)
 if path.exists():path.unlink();removed.append(str(path))
print(json.dumps(dict(status='passed',roots=reports,temporary_key_files_removed=removed)))
PY
''',timeout=300))
    cost=[]
    for p in leases+aborted:
        hours=(p['deleted_epoch']-p['created_epoch'])/3600
        cost.append(dict(name=p['name'],pod_id=p['id'],hours=hours,quoted_compute_usd_per_hour=p['price_per_hour'],
            provisional_compute_usd=hours*p['price_per_hour'],conservative_total_usd=hours*p['bounded_hourly_usd'],
            kind='executed' if p in leases else 'unused infrastructure attempt'))
    conservative=sum(p['conservative_total_usd'] for p in cost)
    assert conservative<=20
    result=dict(status='passed',utc=c.x.stamp(),runs=[289,290,291],scientific_arms=len(arms),updates=3584,
        full_gradient_boundaries=3594,first_last_update_audits=20,arms=arms,retention=retained,
        task_pods_remaining=present,provisional_costs=cost,immutable_pretransfers=prestaged,
        provisional_compute_usd=sum(p['provisional_compute_usd'] for p in cost),
        conservative_total_usd=conservative,authorized_ceiling_usd=20,
        billing_status='Provisional from live quoted rates and create-to-delete durations; conservative total includes the reserved hourly storage allowance. Provider invoice is authoritative.',
        storage='Raw evidence retained on owned Spark; compact scalar/verification files local; no task-owned Runpod storage remains.',
        qualification_excluded='Seven mechanical preflights and worker qualification passes are excluded from scientific arm/update counts.',
        cleanup='Temporary Spark retrieval key removed. Sleep inhibitor was unavailable; no power settings changed.')
    c.dump(HERE/'conventional_closeout.json',result)
    (c.S/'STOP_WATCHER').write_text('All seven task pods verified and deleted.\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['arms','retention','provisional_costs','immutable_pretransfers']}))


if __name__=='__main__':main()
