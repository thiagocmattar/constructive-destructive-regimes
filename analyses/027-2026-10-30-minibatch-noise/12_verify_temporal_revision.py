"""Independent checks of the temporal figure and the revised sampling argument."""
from pathlib import Path
import hashlib
import json
import re
import subprocess

import numpy as np
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DRAFT = ROOT / 'manuscript/draft'
BUILD = ROOT / 'manuscript/artifacts/draft'
RAW = ROOT / 'runs/186-2026-10-30-mlp-minibatch-noise/artifacts/attempt-01/production'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prior = read(HERE / 'verification.json')
    assert prior['status'] == 'passed'
    assert digest(HERE / 'results.json') == prior['results_sha256']
    assert digest(RAW / 'inventory.json') == prior['production_inventory_sha256']
    results = read(HERE / 'results.json')
    reduced = read(HERE / 'reduction.json')
    assert digest(HERE / 'reduction.json') == read(HERE / 'scientific_summary.json')['source_reduction_sha256']
    old_figures = read(HERE / 'paper_figure_manifest.json')
    new = read(HERE / 'temporal_figure_manifest.json')
    assert new['script_sha256'] == digest(HERE / '11_temporal_paper_figure.py')
    inventory = {r['path']: r['sha256'] for r in read(RAW / 'inventory.json')['files']}
    used = {}

    def rows(rel):
        data = (RAW / rel).read_bytes()
        assert hashlib.sha256(data).hexdigest() == inventory[rel]
        used[rel] = inventory[rel]
        return [json.loads(s) for s in data.decode().splitlines()]

    histories = {}
    for b in [32, 2048, 5000]:
        for seed in range(3):
            for rep in range(1 if b == 5000 else 8):
                records = rows(f'moving-seed{seed}-state1000-b{b}-rep{rep:02d}/metrics.jsonl')
                assert len(records) == 200 and [r['update'] for r in records] == list(range(1, 201))
                histories[b, seed, rep] = records
    temporal_values = 0
    for curve in new['curves']:
        b, key = curve['batch'], curve['field']
        blocks = []
        for seed in range(3):
            repetitions = [records for (bb, ss, _), records in histories.items() if (bb, ss) == (b, seed)]
            # Direct window averaging independently reproduces the generator's convolution.
            blocks.append([float(np.mean([r[key] for records in repetitions
                for r in records if end - 19 <= r['update'] <= end])) for end in range(21, 201)])
        blocks = np.array(blocks)
        assert curve['updates'] == list(range(21, 201))
        np.testing.assert_allclose(curve['initialization_means'], blocks, atol=2e-15, rtol=0)
        for field, target in [('mean', blocks.mean(0)), ('minimum', blocks.min(0)), ('maximum', blocks.max(0))]:
            np.testing.assert_allclose(curve[field], target, atol=2e-15, rtol=0)
        temporal_values += blocks.size
    assert temporal_values == 3240 and len(new['curves']) == 6

    for summary in new['summaries']:
        b = summary['batch']
        for key in ['cosine', 'rho_full']:
            seed_means = [float(np.mean([r[key] for (bb, ss, _), records in histories.items()
                if (bb, ss) == (b, seed) for r in records[1:]])) for seed in range(3)]
            early = np.mean([r[key] for (bb, _, _), records in histories.items() if bb == b for r in records[1:20]])
            np.testing.assert_allclose(summary[key]['initialization_means'], seed_means, atol=1e-15, rtol=0)
            assert abs(summary[key]['mean'] - np.mean(seed_means)) < 1e-15
            assert abs(summary[key]['mean_updates_2_to_20'] - early) < 1e-15
        for event in summary['first_negative']:
            records = histories[b, event['seed'], event['replicate']]
            for key, field in [('cosine', 'first_negative_sampled'), ('rho_full', 'first_negative_full')]:
                assert event[field] == next((r['update'] for r in records if r[key] is not None and r[key] < 0), None)
    for record in new['references']:
        data = rows(f"reference-seed{record['seed']}/metrics.jsonl")
        assert record['first_negative'] == next(r['update'] for r in data if r['cosine'] is not None and r['cosine'] < 0)
    assert used == new['raw_sources'] and len(used) == 54
    assert [r['first_negative'] for r in new['references']] == [1149, 1227, 1224]
    middle = next(s for s in new['summaries'] if s['batch'] == 2048)
    first = [r['first_negative_full'] for r in middle['first_negative']]
    assert len(first) == 24 and min(first) == 3 and max(first) == 5
    assert f"{middle['rho_full']['mean_updates_2_to_20']:.3f}" == '-0.355'

    outcome_values = 0
    for curve in old_figures['curves']:
        if curve['figure'] != 'batch_size_sgd_outcomes':
            continue
        blocks = np.array([[np.mean([r[curve['field']] for r in results['moving']
            if (r['seed'], r['checkpoint'], r['batch']) == (seed, curve['checkpoint'], b)])
            for b in curve['batches']] for seed in range(3)])
        np.testing.assert_allclose(curve['initialization_means'], blocks, atol=1e-14, rtol=0)
        for field, target in [('mean', blocks.mean(0)), ('minimum', blocks.min(0)), ('maximum', blocks.max(0))]:
            np.testing.assert_allclose(curve[field], target, atol=1e-14, rtol=0)
        outcome_values += blocks.size
    assert outcome_values == 180

    section = (DRAFT / 'sections/05-sgd-noise-experiments.tex').read_text(encoding='utf-8')
    def mean(cp, b, key, seed=None):
        return float(np.mean([r[key] for r in results['moving'] if r['checkpoint'] == cp and r['batch'] == b
                             and (seed is None or r['seed'] == seed)]))
    numerical = []
    for cp, b, key, digits, signed, absolute in [
        (0,32,'mean_rho',3,False,False),(0,32,'mean_full_rho',3,True,False),(0,5000,'mean_rho',3,False,False),
        (1000,2048,'mean_rho',3,False,False),(1000,2048,'mean_full_rho',3,False,False),(1000,5000,'mean_rho',3,True,False),
        (2000,5000,'mean_rho',3,False,False),(2000,32,'mean_rho',3,False,False),
        (2000,5000,'R',4,False,False),(2000,32,'R',4,False,False),
        (2000,5000,'loss_change',4,False,True),(2000,32,'loss_change',4,False,False),
        (0,32,'R',3,False,False),(0,5000,'R',3,False,False),
        (1000,32,'R',3,False,False),(1000,5000,'R',3,False,False)]:
        value = mean(cp,b,key)
        shown = format(abs(value) if absolute else value, ('+' if signed else '') + f'.{digits}f')
        assert shown in section, shown
        numerical.append(dict(checkpoint=cp,batch=b,field=key,value=value,display=shown))
    for seed in range(3):
        assert mean(2000,32,'loss_change',seed) > 0
        assert mean(2000,2048,'final_loss',seed) < mean(2000,5000,'final_loss',seed)
    for cp in [0,1000]:
        batches = [32,128,512,2048,5000]
        assert all(np.diff([mean(cp,b,'R') for b in batches]) > 0)
        assert all(np.diff([mean(cp,b,'loss_change') for b in batches]) < 0)
    assert all(.028 <= mean(2000,b,'R') <= .032 for b in batches)
    frozen = [c for c in reduced['frozen'] if c['batch'] != 5000]
    assert np.ceil(10000*max(abs(c['mean_noise_energy']/c['predicted_variance']-1) for c in frozen))/100 == .50
    assert np.ceil(10000*max(abs(c['mean_noise_projection']/c['predicted_variance']+1) for c in frozen))/100 == .74
    assert f"{max(abs(c['mean_rho']-c['predicted_rho']) for c in frozen):.4f}" == '0.0075'
    outside = sum(not c['mean_rho_conditional_ci95'][0] <= c['predicted_rho'] <= c['mean_rho_conditional_ci95'][1] for c in frozen)
    assert outside == 21 and len(frozen) == 36
    for cp, expected in [(0,'0.986'),(1000,'0.069')]:
        assert f"{np.mean([c['mean_rho'] for c in frozen if (c['checkpoint'],c['batch']) == (cp,2048)]):.3f}" == expected
    ratios = [c['predicted_variance']/c['signal_energy'] for c in frozen if (c['checkpoint'],c['batch']) == (1000,2048)]
    assert f'{min(ratios):.2f}--{max(ratios):.2f}' == '13.37--13.57'

    # Compare all numbered and unnumbered display equations against the last revision.
    method = (DRAFT/'sections/03-methods.tex').read_text(encoding='utf-8')
    before = subprocess.check_output(['git','show','ed1162e:manuscript/draft/sections/03-methods.tex'],cwd=ROOT).decode()
    pattern = r'\\begin\{(equation\*?|align\*?)\}(.*?)\\end\{\1\}|\\\[(.*?)\\\]'
    assert re.findall(pattern,method,re.S) == re.findall(pattern,before,re.S)
    assert 'squared norm of the mean unit gradient' in method
    for name in ['bare_jrnl.tex','sections/04-results.tex','sections/01-introduction.tex']:
        assert (DRAFT/name).read_bytes() == subprocess.check_output(['git','show',f'ed1162e:manuscript/draft/{name}'],cwd=ROOT)
    for term in [r'\mu_k',r'\chi_b',r'q_k','HVP',r'\begin{equation}',r'\begin{table}']:
        assert term not in section
    assert section.count(r'\includegraphics') == 2 and 'batch_size_fixed_parameters.pdf' not in section
    assets = [old_figures['pdfs']['batch_size_sgd_outcomes.pdf'],new['pdf']]
    for asset in assets:
        source,target=ROOT/asset['source'],ROOT/asset['target']
        assert digest(source) == digest(target) == asset['sha256']
        pdf=PdfReader(source)
        assert len(pdf.pages)==1 and abs(float(pdf.pages[0].mediabox.width)-7.16*72)<.001
        fonts=subprocess.check_output(['pdffonts',str(source)],stderr=subprocess.DEVNULL).decode().splitlines()[2:]
        assert fonts and all('Type 3' not in row and re.search(r'\byes\s+yes\s+yes\b',row) for row in fonts)
    log=(BUILD/'bare_jrnl.log').read_text(encoding='utf-8',errors='replace')
    for term in ['Overfull \\hbox','Overfull \\vbox','undefined references','undefined citations','multiply defined','! LaTeX Error']:
        assert term not in log,term
    assert r'\newlabel{eq:noise-cosine-approx}{{19}' in (BUILD/'bare_jrnl.aux').read_text(encoding='utf-8')
    texts=[p.extract_text() for p in PdfReader(BUILD/'bare_jrnl.pdf').pages]
    figure_page=next(i+1 for i,t in enumerate(texts) if 'Fig. 4:' in t and 'Fig. 5:' in t)
    next_section=next(i+1 for i,t in enumerate(texts) if 'VIMECHANISMSOFTRAJECTORYSTABILITY' in re.sub('[^A-Z]','',t.upper()))
    assert figure_page<next_section
    report=dict(status='passed',script_sha256=digest(Path(__file__)),raw_files_checked=len(used),
        temporal_initialization_values_checked=temporal_values,temporal_means_checked=1080,
        temporal_range_endpoints_checked=2160,outcome_initialization_values_checked=outcome_values,
        outcome_means_checked=60,numerical_statements=numerical,
        first_negative_full_at_batch2048=dict(repetitions=len(first),minimum=min(first),maximum=max(first)),
        full_gd_first_negative=[r['first_negative'] for r in new['references']],
        equations_unchanged=True,existing_outcome_figure_unchanged=True,pages=len(texts),
        both_figures_page=figure_page,next_section_page=next_section,overfull_boxes=0,unresolved_references=0,
        pdf_sha256=digest(BUILD/'bare_jrnl.pdf'),temporal_manifest_sha256=digest(HERE/'temporal_figure_manifest.json'),
        sources={name:digest(DRAFT/name) for name in ['sections/03-methods.tex','sections/05-sgd-noise-experiments.tex']})
    (HERE/'temporal_revision_verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({key:report[key] for key in ['status','raw_files_checked','temporal_means_checked','outcome_means_checked','pages','both_figures_page']}))


if __name__=='__main__':
    main()
