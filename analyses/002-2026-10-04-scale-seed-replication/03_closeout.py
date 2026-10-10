"""Document accepted seed runs and reconcile registries after cloud cleanup."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.stdout.reconfigure(encoding='utf-8')

def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def write(path, value): path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')

def main():
    result = read(HERE / 'results.json')
    generation = read(HERE / 'generation.json')
    record = read(HERE / 'record_generation.json')
    assert generation['results_sha256'] == record['results_sha256'] == sha(HERE / 'results.json')
    assert result['script_sha256'] == sha(HERE / '01_compare.py')
    assert result['pilot_reduction_script_sha256'] == sha(ROOT / 'analyses/001-2026-10-03-model-scale-geometry/01_compare.py')
    assert generation['figure_data_sha256'] == sha(HERE / 'figure_data.csv')
    assert record['observation_sha256'] == sha(HERE / 'observations/001-seed-replication.md')
    count = 0
    for r in result['source_lineage']:
        for x in r['sources'].values():
            assert sha(ROOT / x['path']) == x['sha256']; count += 1
        assert sha(ROOT / r['acceptance']['path']) == r['acceptance']['sha256']
    with (HERE / 'figure_data.csv').open(newline='', encoding='utf-8') as f:
        assert len(list(csv.DictReader(f))) == 6408
    for name, info in generation['figures'].items():
        p = HERE / 'figures' / name
        assert p.stat().st_size == info['bytes'] and sha(p) == info['sha256']
    qa = read(HERE / 'visual-qa.json')
    assert qa['count'] == 4 and qa['all_single_page_text_bounds_ok'] and qa['all_visually_inspected']
    inventory = read(ROOT / 'manuscript/source_inventory.json')
    for f in inventory['files']:
        p = ROOT / 'manuscript/draft' / f['path']
        assert p.stat().st_size == f['bytes'] and sha(p) == f['sha256']
    assert sha(ROOT / 'manuscript' / inventory['source_archive']) == inventory['source_sha256']
    cohort = read(ROOT / 'cache/runpod/seed-cohort.json')
    closeouts = []
    for item in cohort:
        run = ROOT / 'runs' / item['run']; p = run / 'prelaunch'
        acceptance = read(p / 'attempt1-retrieval.json'); close = read(p / 'closeout.json')
        assert acceptance['verified'] and acceptance['scientific_verification']['verified']
        frozen = read(p / 'scientific-freeze.json')
        assert frozen['config_sha256'] == sha(run / 'config.json')
        assert frozen['source_inventory_sha256'] == sha(p / 'source-inventory.json')
        assert read(p / 'local-checks.json')['cuda_fused_adam_audit_passed']
        assert acceptance['files'] == 55
        assert close['confirmed_absent'] and close['deletion_status'] == 204
        assert close['retained_task_cloud_storage_gb'] == 0
        r = next(x for x in result['runs'] if x['run_id'] == run.name[:3])
        lines = [f"# Observation 001: {item['size']} seed {item['seed']} geometry", '',
            'Status: completed, retrieved, independently verified; owned worker and volume deleted.', '',
            '## Sources and reduction', '',
            '[Retrieval receipt](../prelaunch/attempt1-retrieval.json) verifies the complete archive,',
            '55-file inventory, initial tensor hash, independent CPU geometry reconstruction, and',
            'raw gradient norm versus the separate clipping routine. Frozen configuration and source',
            'copies match. Raw evidence lives in `artifacts/attempt-01/full-pass/` (ignored by Git).',
            f"Archive SHA-256: `{acceptance['archive_sha256']}`.",
            f"Input-event SHA-256: `{sha(run / 'artifacts/attempt-01/full-pass/events.jsonl')}`.", '',
            '[Analysis002](../../../analyses/002-2026-10-04-scale-seed-replication/observations/001-seed-replication.md)',
            'owns the cross-run reduction, uncertainty, publication figures and captions.',
            'Generating scripts: Analysis002 `01_compare.py`, `02_record.py`, and `03_closeout.py`.', '',
            '## Measurements', '',
            'All 712 optimizer updates and 1,491,711,416 input tokens are retained. Raw g is measured',
            'before clipping; adaptive d uses bias-corrected actual FP32 moments of clipped gradients',
            'and excludes LR/weight decay. A=sum||x||, B=||sum x||, R=B/A. First cosine is undefined.',
            'Median cosine uses 711 adjacent pairs; other full-pass summaries include update712.', '',
            '| Vector | Median adjacent cosine | A | B | B/A | Median norm | Negative pairs |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for stream in ('raw', 'adaptive'):
            v = r['all_updates'][stream]
            lines.append(f"| {stream} | {v['median_adjacent_cosine']:.6f} | {v['A']:.9g} | {v['B']:.9g} | {v['R']:.6f} | {v['median_norm']:.9g} | {v['negative_cosines']}/711 |")
        lines += ['', f"Median paired ||d||/||g||: {r['median_adaptive_to_raw_norm_ratio']:.6f}.",
            f"Final token-weighted validation loss: {r['final_validation_loss']:.6f}; all 500 documents and tail included.",
            f"Overflow retries: {r['overflow_retries']}.", '', '## Coverage, limits and cleanup', '',
            'Evolution plots omit only final partial update712 (311 sequences), as requested. Full',
            'statistics and raw evidence retain it; Analysis002 also records update711 sensitivity.',
            'A single trajectory supplies no independent seed uncertainty. Cross-seed figures use',
            'three trajectories per size, mean/sample SD or seed range, never independent steps.',
            'Initialization and order seeds vary together; order is matched across sizes within seed.',
            'Absolute norms depend on architecture/dimension. Alignment is not a stability, causal',
            'learning benefit or parameter-displacement measure.', '',
            f"[Closeout](../prelaunch/closeout.json): estimated ${close['estimated_gpu_and_storage_usd']:.2f} including running storage, provisional until billing settles; zero retained task cloud storage."]
        if run.name[:3] in ('005', '008'):
            lines += ['Hardware status queries returned NVML Unknown Error after completed training.',
                'Packaging omitted that redundant query; captured training/calibration hardware',
                'records and all scientific artifacts passed unchanged acceptance checks.']
        obs = run / 'observations'; obs.mkdir(exist_ok=True)
        (obs / '001-seed-geometry.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        (obs / 'INDEX.md').write_text('# Observations\n\n- [001: Seed geometry](001-seed-geometry.md)\n', encoding='utf-8')
        text = (run / 'README.md').read_text(encoding='utf-8')
        a = text.index('Status:'); b = text.index('\n\n', a)
        text = text[:a] + 'Status: completed all 712 updates; artifacts retrieved and independently verified; worker and attached volume deleted.' + text[b:]
        suffix = '\n## Completed evidence\n\n[Observation001](observations/001-seed-geometry.md) records raw/adaptive metrics, artifact verification and cleanup.\n[Analysis002](../../analyses/002-2026-10-04-scale-seed-replication/README.md) compares all three seeds per size.\n'
        if '## Completed evidence' not in text: text += suffix
        (run / 'README.md').write_text(text, encoding='utf-8')
        closeouts.append({'run_id':run.name[:3], 'retrieval_receipt_sha256':sha(p / 'attempt1-retrieval.json'),
            'files':acceptance['files'], 'bytes':acceptance['total_bytes'],
            'archive_bytes':acceptance['archive_bytes'], 'archive_sha256':acceptance['archive_sha256'],
            'closeout_sha256':sha(p / 'closeout.json'), 'cost_usd_provisional':close['estimated_gpu_and_storage_usd']})
    counts = record['primary_replication_counts']
    outcome = '; '.join(f"{v['stream']} {'cosine' if v['metric']=='median_adjacent_cosine' else 'B/A'}: {v['all_seed_blocks_matching']}/3 seed blocks" for v in counts)
    cost = sum(x['cost_usd_provisional'] for x in closeouts)
    p = HERE / 'README.md'; text = p.read_text(encoding='utf-8')
    text = text.replace('Status: all six full passes launched after calibration on parallel RTX5090 workers.',
        'Status: all six new full passes completed; archives accepted; all six workers and volumes deleted.')
    text = text.replace('H002 is being examined across three seeds per size: 1234, 1235, and 1236.',
        'H002 is examined across three seeds per size: 1234, 1235, and 1236.')
    text = text.split('\n## Completed results')[0]
    text += f'\n## Completed results\n\nStrict pilot ordering (14M > 31M > 70M): {outcome}.\nThe main full-pass ordering survives both additional seeds for both streams. Adaptive A/B/norms\nincrease in all seeds; raw B decreases in all seeds, while raw A has a small 31M-to-70M\nreversal in seed1235. Raw phase-specific cosine ordering is not uniform. This is a\nthree-seed reproducibility check, not a fitted predictive scaling law.\n\n[Observation001](observations/001-seed-replication.md) reports every seed, means/sample SD, A/B/norms, phase summaries and limits.\n[Geometry](figures/geometry_across_seeds.pdf), [size summaries](figures/scale_summary_across_seeds.pdf),\n[cumulative A/B](figures/cumulative_across_seeds.pdf), and [norm ratios](figures/norm_ratio_across_seeds.pdf) are publication PDFs.\nAll 6,408 update rows remain in [figure_data.csv](figure_data.csv); exact lineage is in [results.json](results.json).\n\nSix new runs cost an estimated ${cost:.2f} including running storage, provisional until provider billing settles.\nNo task-owned cloud storage remains. [Final review](final-review.json) records verification and preservation.\n'
    p.write_text(text, encoding='utf-8')
    p = ROOT / 'runs/INDEX.md'; text = p.read_text(encoding='utf-8')
    for item in cohort:
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.startswith(f"| [{item['run'][:3]}]"):
                lines[i] = line.replace('Full pass launched; calibrated', 'Completed; 712 updates; verified; cloud closed').replace('| Pending |', f"| [001: Geometry]({item['run']}/observations/001-seed-geometry.md) |")
        text = '\n'.join(lines) + '\n'
    p.write_text(text, encoding='utf-8')
    p = ROOT / 'analyses/INDEX.md'; text = p.read_text(encoding='utf-8')
    text = text.replace('In progress; six workers provisioned | Pending', 'Complete; nine trajectories verified; new cloud closed | [001: Seed replication](002-2026-10-04-scale-seed-replication/observations/001-seed-replication.md), four PDFs')
    p.write_text(text, encoding='utf-8')
    p = ROOT / 'research/INDEX.md'; text = p.read_text(encoding='utf-8')
    prefix = 'Runs 005-010 are preparing' if 'Runs 005-010 are preparing' in text else 'Runs 005-010 completed'
    a = text.index(prefix); b = text.index('\n\n', a)
    text = text[:a] + f'Runs 005-010 completed both additional seed blocks. [Analysis002](../analyses/002-2026-10-04-scale-seed-replication/README.md)\ncompares three seeds per size: {outcome}.\nAll new artifacts are verified; all six workers/volumes removed; estimated ${cost:.2f} provisional.\nInitial/order seeds vary together; order is paired within seed. Three seeds support a\nreplication check, not a predictive scaling law.' + text[b:]
    p.write_text(text, encoding='utf-8')
    p = ROOT / 'research/HYPOTHESES.md'; text = p.read_text(encoding='utf-8')
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith('| H002 |'):
            cells = line.split('|'); cells[3] = ' Replication complete; predictive scaling law still inconclusive '
            cells[5] = f' Strict full-pass pilot ordering: {outcome}. Three seeds, matched order within seed; initial/order seeds co-vary. Raw A/B/norm and phase reversals are retained. No fitted functional form or held-out prediction '
            lines[i] = '|'.join(cells)
    p.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    for dest in ('research/METHODS.md', 'research/findings/README.md'):
        p = ROOT / dest; text = p.read_text(encoding='utf-8')
        link = '../analyses/' if dest.endswith('METHODS.md') else '../../analyses/'
        if 'Analysis002 completed' not in text:
            text += f'\nAnalysis002 completed six additional full passes, yielding three seeds per size.\n[Seed replication]({link}002-2026-10-04-scale-seed-replication/observations/001-seed-replication.md)\nreports {outcome}. Data order is matched across sizes within seed and\nvaries with initialization between seeds. Seed1234 used RTX4090, new blocks RTX5090;\nnumerical hardware effects cannot be separated from seed effects across blocks.\nAll per-run values, reversals, A/B/norms and validation are retained; no predictive\nlaw, stability result or causal learning benefit is established.\n'
        p.write_text(text, encoding='utf-8')
    review = {'reviewed_at':datetime.now(timezone.utc).isoformat(), 'status':'complete',
        'script_sha256':sha(Path(__file__)), 'reduction_and_generating_script_hashes_match':True,
        'source_metadata_and_event_files_rechecked':count, 'figure_data_rows':6408,
        'updates_per_run':712, 'new_run_acceptance_and_cleanup':closeouts,
        'main_plot_updates':'1-711; final partial update712 retained in raw data, CSV and full statistics',
        'publication_pdf_qa':qa, 'primary_replication_counts':counts,
        'validation_coverage':'All 500 documents, packed tail included, target-token weighted',
        'test_record':'Reused unchanged scientific implementation: 27 CPU checks previously passed; 5 CUDA fused-Adam checks passed on each of six workers; independent CPU acceptance on all archives.',
        'estimated_new_cohort_cost_usd':cost, 'cost_provisional':True,
        'retained_task_cloud_storage_gb':0, 'unrelated_resources_preserved':True,
        'manuscript_preservation':{'draft_files_unchanged':len(inventory['files']), 'archive_sha256':inventory['source_sha256']},
        'scientific_limit':'Three seed blocks provide descriptive replication, not a predictive scaling law or causal conclusion.'}
    write(HERE / 'final-review.json', review)
    print(json.dumps({'status':'complete', 'cost_usd_provisional':cost, 'primary':counts}, indent=2))

if __name__ == '__main__': main()
