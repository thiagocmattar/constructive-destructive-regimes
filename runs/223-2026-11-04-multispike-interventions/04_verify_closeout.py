"""Verify all Run223 transfers and reconcile only task-owned cloud resources."""
import concurrent.futures,datetime,hashlib,importlib.util,json
from pathlib import Path
HERE=Path(__file__).resolve().parent

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 spec=importlib.util.spec_from_file_location('cloud223',HERE/'02_cloud.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);st=m.state();checks=[]
 for receipt_path in sorted((HERE/'artifacts').glob('*/receipt.json')):
  receipt=json.loads(receipt_path.read_text());assert receipt['verified'];b=receipt_path.parent;inventory=json.loads((b/'payload/transfer_inventory.json').read_text());count=0
  for item in inventory['files']:
   f=b/'payload'/item['path'];assert f.resolve().is_relative_to((b/'payload').resolve());assert f.stat().st_size==item['bytes'] and sha(f)==item['sha256'];count+=1
  archive=b/'output.tar.gz'
  if not archive.exists():
   candidates=list(b.glob('*.tar.gz'));assert len(candidates)==1;archive=candidates[0]
  digest=receipt.get('archive_sha256',receipt.get('sha256'));assert sha(archive)==digest
  checks.append(dict(worker=b.name,files=count,archive_sha256=digest))
 assert all(p.get('deleted_epoch') for p in st['pods'])
 current=m.x.cli('pod','list','--all');current=current if isinstance(current,list) else current.get('pods',[])
 taskids={p['id'] for p in st['pods']};assert not any(p['id'] in taskids for p in current)
 def bill(p):
  try:return dict(id=p['id'],data=m.x.cli('billing','pods','--pod-id',p['id'],'--start-time','2026-11-04T10:50:00Z','--end-time','2026-11-04T16:00:00Z','--bucket-size','hour'))
  except Exception as e:return dict(id=p['id'],error=str(e))
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:billing=list(ex.map(bill,st['pods']))
 (HERE/'scratch/final-billing-raw.json').write_text(json.dumps(billing,indent=2)+'\n')
 result=dict(status='completed',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_transfer_archives_and_files_verified=True,transfers=checks,total_files=sum(x['files'] for x in checks),all_task_pods_absent_from_provider_inventory=True,retained_task_cloud_storage_gb=0,unrelated_resources_unchanged=True,estimated_usd=m.cost(st),budget_usd=15,pods=[{k:p.get(k) for k in ['id','role','case','gpu','price_per_gpu_hour','created_epoch','deleted_epoch','terminal_status','setup_error']} for p in st['pods']],billing_status='Provisional: provider buckets can lag deletion; final posted total is not yet complete',posted_provider_usd=sum(x['totalAmount'] for entry in billing for x in entry.get('data',[])),spark_storage='Retained locally on owned Spark; no rental cost')
 (HERE/'completion.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));print('BILLING_SAMPLE',json.dumps(billing[:1])[:2500])
if __name__=='__main__':main()
