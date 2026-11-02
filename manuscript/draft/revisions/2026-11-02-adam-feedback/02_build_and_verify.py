"""Verify source-to-figure Adam reductions and claims, then build the review PDF."""
import hashlib
import importlib.util
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import fitz
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DRAFT = HERE.parents[1]
A36 = ROOT/'analyses/036-2026-11-02-controlled-adam-diagnostics'
PRIOR = HERE.parent/'2026-11-02-feedback'
SCRATCH = ROOT/'tmp/pdfs/manuscript-adam-feedback-2026-11-02'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def verify_adam():
    manifest = read(HERE/'figure_manifest.json')
    assert manifest['generator_sha256'] == sha(HERE/'01_prepare_figures.py')
    for name, digest in manifest['sources'].items(): assert sha(ROOT/name) == digest, name
    scientific = read(ROOT/'runs/215-2026-11-02-controlled-full-batch-adam/results.json')
    histories = {}
    for arm in scientific['conditions']:
        folder = ROOT/arm['source']
        boundaries = [json.loads(s) for s in (folder/'boundaries.jsonl').read_text().splitlines()]
        metrics = [json.loads(s) for s in (folder/'metrics.jsonl').read_text().splitlines()]
        assert len(boundaries) == 1025 and len(metrics) == 1024
        assert [b['state'] for b in boundaries] == list(range(1025))
        assert [m['gradient_state'] for m in metrics] == list(range(1024))
        histories[arm['task'], arm['lr']] = boundaries, metrics
    maximum = 0.

    def values(task, rate, row):
        b, m = histories[task, rate]
        if row == 'training_loss_log': return np.arange(1025), np.array([v['loss'] for v in b])
        if row == 'eta_kappa_hat_raw': yy = [v['projected_gradient_change'] for v in m]
        else:
            stream = 'raw' if row in ['rho_raw', 'raw_gradient_norm_log'] else 'moment'
            field = 'cosine' if row.startswith('rho_') else 'norm'
            yy = [v['streams'][stream][field] for v in m]
        return np.arange(1024), np.array([np.nan if y is None else y for y in yy])

    for c in manifest['curves']:
        x, y = values(c['task'], c['lr'], c['row'])
        if c['figure'] == 'overview' and c['row'] != 'training_loss_log':
            pairs = [(int(x[k]), math.fsum(y[k-49:k+1])/50) for k in range(49, len(y))
                     if np.isfinite(y[k-49:k+1]).all()]
            xx, yy = np.array(pairs).T
            assert c['smoothing_window'] == 50 and c['normalization'] == 1
        else:
            selected = np.isin(x, c['x']); xx, yy = x[selected], y[selected]
            if c['figure'] == 'peaks':
                example = next(e for e in manifest['selection'] if e['task'] == c['task'])
                assert c['x'] == list(range(example['start_state'], example['end_state']+1))
                divisor = float(yy[0]) if c['row'].endswith('norm_log') else 1.
                assert c['normalization'] == divisor
                yy = yy/divisor
            assert c['smoothing_window'] is None
        np.testing.assert_array_equal(xx, c['x'])
        np.testing.assert_allclose(yy, c['y'], rtol=3e-14, atol=3e-14)
        maximum = max(maximum, float(np.max(np.abs(yy-np.array(c['y'])))))
    assert len(manifest['curves']) == 66
    assert len([c for c in manifest['curves'] if c['figure'] == 'overview']) == 54
    for (task, rate), (b, m) in histories.items():
        for k in range(1, 1024):
            previous, current = m[k-1]['streams']['raw'], m[k]['streams']['raw']
            q = 1-current['cosine']*current['norm']/previous['norm']
            assert abs(q-m[k]['projected_gradient_change']) < 2e-12
            if q > 2: assert current['cosine'] < 0 and current['norm'] > previous['norm']
    # This independent verifier reconstructs every forecast label, source-clock
    # feature, episode, early-warning lead, AP and Brier score from raw histories.
    module(A36/'17_verify_peak_temporal.py', 'independent_peak_checks').main()
    temporal = read(A36/'peak_temporal_results.json')
    temporal_audit = read(A36/'peak_temporal_verification.json')
    assert temporal_audit['status'] == 'passed'
    text = (DRAFT/'sections/06-adam-diagnostics.tex').read_text(encoding='utf-8')
    details = (DRAFT/'sections/07-gd-diagnostics.tex').read_text(encoding='utf-8')
    claim = ('In Adam, raw-gradient dynamics provide information about impending loss '
             'excursions beyond recent loss history, even though Adam transforms those '
             'gradients before updating the parameters.')
    assert claim in re.sub(r'\s+', ' ', text)
    assert not any(s in text for s in ['AdamW', 'clipp', 'Pythia', 'MiniPile', 'analysis028_'])
    b, m = histories['transformer', .0003]
    for field, displayed in [('raw', '-0.795'), ('adaptive', '0.939')]:
        assert f"{np.median([v['streams'][field]['cosine'] for v in m[-64:]]):.3f}" == displayed
        assert f'${displayed}$' in text
    for task in ['mlp', 'resnet', 'transformer']:
        subset = [(rate, b[-1]['loss']) for (name, rate), (b, m) in histories.items() if name == task]
        assert max(subset)[1] == min(loss for rate, loss in subset)
    for task, shown in [('mlp', '13/17'), ('resnet', '3/3'), ('transformer', '7/7')]:
        r = max((s for s in temporal['episode_summary'] if s['task'] == task), key=lambda s:s['lr'])
        assert shown == f"{r['early_alarm_hits_strong']['eta_kappa_crosses_2']}/{r['strong_episodes']}"
        assert shown in text
    alarm = temporal['forecast']['fixed_alarms']['eta_kappa_above_2']['all']
    assert (alarm['true_positive'], alarm['alarms'], alarm['false_positive']) == (187, 891, 704)
    assert '187 of 891' in text and '21\\%' in text
    for model, shown in [('loss_history', '0.260'), ('projected_change', '0.454'), ('all_diagnostics', '0.395')]:
        assert f"{temporal['forecast']['overall'][model]['average_precision']:.3f}" == shown
        assert shown in details
    table = (DRAFT/'tables/analysis036_forecast.tex').read_text()
    rows = [s for s in table.splitlines() if ' & ' in s and not s.startswith('Case &')]
    assert len(rows) == len(manifest['table_values']) == 4
    for row, values_ in zip(rows, manifest['table_values']):
        cells = [s.strip() for s in row.removesuffix(r'\\').split('&')]
        models = temporal['forecast']['overall'] if values_['task'] == 'all' else temporal['forecast']['per_task'][values_['task']]
        exact = [100*models['loss_history']['prevalence'], models['loss_history']['average_precision'],
                 models['projected_change']['average_precision']]
        assert cells[1:] == [f'{exact[0]:.1f}', f'{exact[1]:.3f}', f'{exact[2]:.3f}']
    assert temporal['forecast']['forward_test_rows'] == 4534
    for task in ['resnet', 'transformer']:
        r = min((s for s in temporal['episode_summary'] if s['task'] == task), key=lambda s:s['lr'])
        assert r['episodes'] == 0
    b, m = histories['transformer', .003]
    assert m[787]['projected_gradient_change'] <= 2 < m[788]['projected_gradient_change']
    assert f"{100*(b[788]['loss']/b[787]['loss']-1):.2f}" == '0.40'
    assert b[789]['loss'] < 1.1*b[787]['loss'] <= b[790]['loss']
    assert max(range(784, 853), key=lambda s:b[s]['loss']) == 794
    assert max(range(788, 795), key=lambda s:m[s]['streams']['raw']['norm']) == 792
    assert max(range(788, 795), key=lambda s:m[s]['streams']['moment']['norm']) == 793
    comments = read(HERE/'comments.json')
    assert len(comments['comments']) == 6 and sha(HERE/'source-comments.pdf') == comments['source_sha256']
    for f in manifest['figures']:
        assert sha(ROOT/f['source']) == sha(ROOT/f['target']) == f['sha256']
        with fitz.open(ROOT/f['target']) as doc:
            assert len(doc) == 1 and not doc[0].get_images()
            for font in doc[0].get_fonts(full=True): assert font[2] != 'Type3' and doc.extract_font(font[0])[3]
    return dict(conditions=9, curves=66, points=sum(len(c['x']) for c in manifest['curves']),
        max_independent_reduction_error=maximum, state_clock_verified=True,
        forecast_table_scalars=12,
        all_displayed_claims_verified=True, temporal_audit_sha256=sha(A36/'peak_temporal_verification.json'),
        annotated_source_sha256=comments['source_sha256'], preserved_comments=6)


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    prior = module(PRIOR/'02_build_and_verify.py', 'prior_manuscript_checks')
    record = dict(math=prior.check_math(), retained_gd_sgd_evidence=prior.check_evidence(), adam=verify_adam())
    tex = ['pdflatex', '--disable-installer', '-no-shell-escape', '-recorder',
           '-interaction=nonstopmode', '-halt-on-error', '-file-line-error',
           '-output-directory='+str(SCRATCH), 'bare_jrnl.tex']
    for i, command in enumerate([tex, ['bibtex', str(SCRATCH/'bare_jrnl')], tex, tex], 1):
        result = subprocess.run(command, cwd=DRAFT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        (SCRATCH/f'build-{i}.txt').write_bytes(result.stdout)
        if result.returncode:
            print(result.stdout.decode(errors='replace')[-6000:]); raise RuntimeError(f'Build pass {i}')
    log = (SCRATCH/'bare_jrnl.log').read_text(errors='replace')
    # Additional float pages can need one more reference pass after BibTeX.
    for extra in range(3):
        if 'Rerun to get cross-references right' not in log: break
        result = subprocess.run(tex, cwd=DRAFT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        (SCRATCH/f'build-extra-{extra+1}.txt').write_bytes(result.stdout)
        assert result.returncode == 0
        log = (SCRATCH/'bare_jrnl.log').read_text(errors='replace')
    problems = [s for s in log.splitlines() if any(t in s.lower() for t in ['overfull', 'undefined', 'rerun to get', 'multiply defined'])]
    assert not problems, '\n'.join(problems)
    prior_figures = read(ROOT/'manuscript/archive/2026-11-02-unused-figures/manifest.json')['active_figures']
    new_figures = [f['target'] for f in read(HERE/'figure_manifest.json')['figures']]
    expected = {str((ROOT/p).resolve()) for p in prior_figures+new_figures}
    present = {str(p.resolve()) for p in (DRAFT/'figures').rglob('*') if p.is_file()}
    assert expected == present
    recorded = set()
    for line in (SCRATCH/'bare_jrnl.fls').read_text().splitlines():
        if not line.startswith('INPUT '): continue
        p = Path(line[6:]); p = p if p.is_absolute() else DRAFT/p
        if p.suffix == '.pdf' and p.resolve().is_relative_to((DRAFT/'figures').resolve()): recorded.add(str(p.resolve()))
    assert expected == recorded
    pdf = SCRATCH/'bare_jrnl.pdf'
    with fitz.open(pdf) as doc:
        text = ''.join(p.get_text() for p in doc)
        assert '??' not in text and 'Pythia-14M:' not in text and 'Checkpoint-matched schedule intervention' not in text
        record['pages'] = len(doc)
        for i, page in enumerate(doc): page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(SCRATCH/f'page-{i+1:02}.png')
    # Poppler renders are the final visual-review inputs.
    result = subprocess.run(['pdftoppm', '-r', '100', '-png', str(pdf), str(SCRATCH/'review')], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    for f in read(HERE/'figure_manifest.json')['figures']:
        result = subprocess.run(['pdftoppm', '-singlefile', '-scale-to', '2000', '-png', str(ROOT/f['target']), str(SCRATCH/Path(f['target']).stem)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        assert result.returncode == 0
    for target in [DRAFT/'bare_jrnl.pdf', HERE/'revised-manuscript.pdf']: shutil.copyfile(pdf, target)
    record.update(status='awaiting_visual_review', pdf_sha256=sha(pdf), active_figures=len(expected),
        build='pdflatex, bibtex, pdflatex, pdflatex; no overfull boxes or unresolved references',
        verifier_sha256=sha(__file__), source_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in
            [DRAFT/'bare_jrnl.tex', *sorted((DRAFT/'sections').glob('*.tex')), DRAFT/'tables/analysis036_forecast.tex']})
    (HERE/'verification.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__': main()
