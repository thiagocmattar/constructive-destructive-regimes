"""Summarize accepted local evidence and only this task's closed cloud resources."""
from datetime import datetime, timezone
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    rows = load(HERE / 'cohort.json')
    inventory = load(HERE / 'final_pod_inventory.json')
    billing = load(HERE / 'billing_final_snapshot.json')
    posted = {r['pod_id']: r['result']['metadata']['totals']['totalAmount']
              for r in billing['entries']}
    current, failed = [], []
    for row in rows:
        pre = ROOT / 'runs' / row['run'] / 'prelaunch'
        close = load(pre / 'closeout.json')
        lease = load(pre / 'lease.json')
        receipt = load(pre / (lease['attempt'] + '-retrieval.json'))
        assert close['confirmed_absent'] and receipt['verified']
        assert receipt['scientific_verification']['temporal_verified']
        assert datetime.fromisoformat(receipt['accepted_at']) < datetime.fromisoformat(close['deleted_at'].replace('Z', '+00:00'))
        assert close['allocation_hours'] < 4
        current.append({'run': row['run'], 'size': row['size'], **close,
                        'archive_bytes': receipt['archive_bytes'],
                        'inventoried_uncompressed_bytes': receipt['total_bytes'],
                        'archive_sha256': receipt['archive_sha256'],
                        'posted_charges_usd_so_far': posted[lease['pod_id']]})
        old = load(pre / 'infrastructure-01-lease.json')
        failure = load(pre / 'infrastructure-01-failure.json')
        assert failure['scientific_updates'] == 0
        failed.append({'pod_id': old['pod_id'], 'run': row['run'],
                       'created_at': old['created_at'], 'reason': failure['reason'],
                       'scientific_updates': 0, 'posted_charges_usd': posted[old['pod_id']]})
    ids = {r['pod_id'] for r in current + failed}
    assert len(ids) == 6 and not (ids & {r['id'] for r in inventory['pods']})
    estimate = sum(r['estimated_gpu_and_storage_usd'] for r in current) + sum(r['posted_charges_usd'] for r in failed)
    assert estimate < 15
    start = min(datetime.fromisoformat(r['created_at'].replace('Z', '+00:00')) for r in current + failed)
    end = max(datetime.fromisoformat(r['deleted_at'].replace('Z', '+00:00')) for r in current)
    record = {'closed_at': datetime.now(timezone.utc).isoformat(),
              'status': 'All three replays accepted; all six owned workers deleted',
              'completed': current, 'failed_before_training': failed,
              'estimated_total_usd': estimate, 'maximum_authorized_usd': 15,
              'cost_provisional': True,
              'estimate_basis': 'Current-worker GPU/storage allocation from creation through deletion; posted billing for the three failed infrastructure attempts. Startup and billing rounding can change the final charge.',
              'posted_api_charges_usd_so_far': sum(posted.values()),
              'billing_snapshot': 'billing_final_snapshot.json',
              'cloud_wall_hours_including_retries_and_retrieval': (end-start).total_seconds()/3600,
              'retained_task_cloud_storage_gb': 0,
              'retained_local_archive_bytes': sum(r['archive_bytes'] for r in current),
              'retained_local_inventoried_artifact_bytes': sum(r['inventoried_uncompressed_bytes'] for r in current),
              'raw_sketch_data_bytes_per_model': 186646528,
              'local_retention': 'All sketches, exact diagnostics, frozen sources, logs, and initial/midpoint/final recovery checkpoints retained; compressed archives retained too.',
              'unrelated_resources': 'No stop, restart, or delete action was sent to unrelated resources. Final inventory records their live state only.',
              'final_pod_inventory': 'final_pod_inventory.json'}
    (HERE / 'closeout.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: record[k] for k in ['status', 'estimated_total_usd', 'posted_api_charges_usd_so_far', 'cloud_wall_hours_including_retries_and_retrieval', 'retained_task_cloud_storage_gb']}, indent=2))


if __name__ == '__main__':
    main()
