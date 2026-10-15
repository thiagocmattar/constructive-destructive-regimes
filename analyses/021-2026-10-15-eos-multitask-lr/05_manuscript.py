"""Insert verified new evidence after tab:sign without overwriting later edits."""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPORT = ROOT / 'manuscript/2026-10-11-restatement'
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import file_hash, stamp, write_json


def insert(source):
    marker = '\\input{../../analyses/' + HERE.name + '/figure.tex}'
    if marker in source:
        return source
    anchor = source.index('\\label{tab:sign}')
    end = source.index('\\end{table}', anchor) + len('\\end{table}')
    return source[:end] + '\n\n' + marker + source[end:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--accept-reviewed', action='store_true')
    args = parser.parse_args()
    figure = json.loads((HERE / 'figure_manifest.json').read_text())
    assert figure['status'] in ['awaiting_visual_review', 'passed']
    assert file_hash(ROOT / figure['pdf']) == figure['pdf_sha256']
    assert file_hash(HERE / '04_plot.py') == figure['script_sha256']
    assert file_hash(HERE / 'cohort.json') == figure['cohort_sha256']
    assert file_hash(HERE / 'results.json') == figure['results_sha256']
    assert file_hash(ROOT / figure['arrays']) == figure['arrays_sha256']
    results = json.loads((HERE / 'results.json').read_text())
    assert results['status'] == 'passed' and results['conditions'] == 27
    for row in results['runs']:
        run = ROOT / 'runs' / row['run']
        output = run / 'artifacts/attempt-01-spark/production'
        assert file_hash(run / 'config.json') == row['config_sha256']
        for name, digest in row['source_sha256'].items():
            assert file_hash(output / name) == digest
    evidence = dict(figure=figure['pdf'], figure_sha256=figure['pdf_sha256'],
        results_sha256=figure['results_sha256'], cohort_sha256=figure['cohort_sha256'],
        plot_script_sha256=figure['script_sha256'], arrays_sha256=figure['arrays_sha256'],
        include_sha256=file_hash(HERE / 'figure.tex'), conditions=27, seeds=[0, 1, 2], updates=500)
    tex = REPORT / 'gd-formulation.tex'
    if not args.accept_reviewed:
        before = tex.read_bytes()
        after = insert(before.decode('utf-8')).encode('utf-8')
        # One atomic replacement after checking for a concurrent edit. All
        # surrounding manuscript bytes, including the user's prose, survive.
        assert tex.read_bytes() == before, 'Manuscript changed during insertion.'
        if before != after:
            backup = HERE / 'scratch/manuscript-before-insertion.tex'
            backup.parent.mkdir(exist_ok=True)
            backup.write_bytes(before)
            pending = tex.with_suffix('.tex.tmp')
            pending.write_bytes(after)
            pending.replace(tex)
        subprocess.run([sys.executable, str(REPORT / '00_build.py')], cwd=ROOT, check=True)
    else:
        assert figure['status'] == 'passed', 'Review the standalone figure first.'
        existing = json.loads((REPORT / 'gd-formulation-manifest.json').read_text())
        assert existing['eos_multitask_lr'] == evidence, 'Evidence changed; rebuild and inspect the report.'
        subprocess.run([sys.executable, str(REPORT / '00_build.py'), '--accept-reviewed'], cwd=ROOT, check=True)
    pdf = REPORT / 'gd-formulation.pdf'
    aux = (ROOT / 'manuscript/artifacts/2026-10-11-restatement/gd-formulation.aux').read_text()
    def page_for(label):
        match = re.search(r'\\newlabel\{' + re.escape(label) + r'\}\{\{[^}]*\}\{(\d+)\}', aux)
        assert match, 'Missing compiled label: ' + label
        return int(match.group(1)) - 1
    with fitz.open(pdf) as document:
        table_pages = [page_for('tab:sign')]
        figure_pages = [page_for('fig:eos-multitask-lr')]
        assert figure_pages[0] == table_pages[0] + 1, 'New figure must immediately follow the Table1 page.'
        text = document[figure_pages[0]].get_text()
        assert all(label in text for label in ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)'])
    metadata = REPORT / 'gd-formulation-manifest.json'
    record = json.loads(metadata.read_text())
    record['eos_multitask_lr'] = evidence
    record['scope'] = record['scope'].replace('three-case training results not yet included.',
        'three-case full-batch GD LR comparison (500 updates, three seeds) included after Table1.')
    write_json(metadata, record)
    write_json(HERE / 'manuscript_insertion.json', dict(status='passed' if args.accept_reviewed else 'awaiting_visual_review',
        utc=stamp(), tex_sha256=file_hash(tex), pdf_sha256=file_hash(pdf),
        figure_include_sha256=file_hash(HERE / 'figure.tex'),
        table_page=table_pages[0] + 1, figure_page=figure_pages[0] + 1,
        verification='All surrounding source preserved; figure directly follows Table1 in the rendered report.'))


if __name__ == '__main__':
    main()
