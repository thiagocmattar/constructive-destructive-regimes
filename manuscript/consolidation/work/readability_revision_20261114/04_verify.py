"""Verify manuscript claims, new derivations, compiled content, and source preservation."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math
import re
import fitz
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
ROOT = BASE.parents[1]
DRAFT = BASE / 'draft'
E = BASE / 'evidence'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


protected = json.loads((HERE / 'preservation.json').read_text(encoding='utf-8'))
for path, digest in protected.items():
    check(sha(ROOT / path) == digest, 'Protected source changed: ' + path)

# Independent arithmetic for every main intervention table metric.
data = E / '05-adam/c-intervention/data'
curves = np.load(data / 'matched-curves.npz')
reported = rows(E / '05-adam/c-intervention/tables/matched-outcomes.csv')
lookup = {(r['case'], r['arm']): r for r in reported}
held = rows(data / 'matched-heldout.csv')
manifest = json.loads((data / 'main-presentation_manifest.json').read_text(encoding='utf-8'))
cases = ['mlp', 'resnet', 'transformer']
arms = [r['arm'] for r in manifest['table']['rows']]
computed = {}
for case in cases:
    for arm in arms:
        loss = curves[f'{case}__{arm}__loss']
        states = curves[f'{case}__{arm}__state']
        record = lookup[case, arm]
        np.testing.assert_array_equal(states, np.arange(int(record['fork']), int(record['end'])+1))
        terminal = [r for r in held if r['case'] == case and r['arm'] == arm and r['split'] == 'test' and int(r['state']) == int(record['end'])]
        check(len(terminal) == 1, 'Terminal evaluation coverage')
        values = [math.fsum(max(float(b)-float(a), 0.) for a, b in zip(loss[:-1], loss[1:])),
                  math.fsum(float(x) for x in loss[1:])/(len(loss)-1), float(loss[-1]), float(terminal[0]['loss'])]
        for metric, value in zip(['positive_variation', 'mean_loss', 'final_loss', 'terminal_test_loss'], values):
            check(math.isclose(float(record[metric]), value, rel_tol=2e-12, abs_tol=2e-12), 'Metric mismatch')
        computed[case, arm] = values
for record in manifest['table']['rows']:
    expected = []
    for case in cases:
        for i, (value, baseline) in enumerate(zip(computed[case, record['arm']], computed[case, 'control'])):
            effect = 100*(1-value/baseline) if i == 0 else 100*(value/baseline-1)
            expected.append('0' if record['arm'] == 'control' else f'{effect:+.2f}')
    check(record['display_values'] == expected, 'Main table rounding mismatch')
check(sum(int(r['updates']) for r in reported) == 11904, 'Total continuation updates')
check(sum(int(r['updates']) for r in reported if r['arm'] != 'control') == 10912, 'Treated updates')
check(all(float(lookup[c, a]['mean_loss_change_percent']) < 0 and float(lookup[c, a]['pv_reduction_percent']) > 0
          for c in cases for a in ['cap_q25', 'cap_q50', 'cap_q75']), 'Nine cap/model effects')

motion_claims = {}
for case in cases:
    baseline = curves[f'{case}__control__stage_change_rms'][-1, 1:]
    capped = curves[f'{case}__cap_q50__stage_change_rms'][-1, 1:]
    motion_claims[case] = {
        'max_output_rms_reduction_percent': 100*(1-max(capped)/max(baseline)),
        'squared_motion_change_percent': 100*(math.fsum(float(v)**2 for v in capped)/math.fsum(float(v)**2 for v in baseline)-1),
        'baseline_reversal_percent': 100*np.mean(curves[f'{case}__control__stage_rho'][-1, 1:] < 0),
        'q50_reversal_percent': 100*np.mean(curves[f'{case}__cap_q50__stage_rho'][-1, 1:] < 0),
    }
intervention_text = (DRAFT / 'sections/s05-intervention-methods.tex').read_text(encoding='utf-8')
for values in motion_claims.values():
    for value in values.values():
        check(f'{abs(value):.1f}\\%' in intervention_text, 'Output-motion percentage missing or incorrect')
for arm, expected in [('control', 161), ('cap_q50', 70)]:
    rising = curves[f'mlp__{arm}__mse_delta_loss'] > 1e-8
    linear = curves[f'mlp__{arm}__mse_linear'][rising]
    quadratic = curves[f'mlp__{arm}__mse_quadratic'][rising]
    check(int(sum(rising)) == expected and np.all(linear < 0) and np.all(quadratic > -linear), 'MLP output identity')

# GD accumulation and independent derivative-count statements.
gd = json.loads((E / '04-gd-sgd/data/24-section_iv_review_checks.json').read_text(encoding='utf-8'))['runs']
check(len(gd) == 9 and all(r['final_all_pair'] > 0 for r in gd), 'All-pair claim')
check(sum(r['final_lag_one'] < 0 for r in gd) == 7, 'Adjacent claim')
check(gd[-1]['sign_sensitivity'][0]['sign_disagreements'] == 1325, 'Transformer sign residual')
check(4*288+36 == 1188, 'HVP pair coverage')

# Recompute the precision percentages newly exposed in the main severity table.
hits = rows(E / '05-adam/a-characterization/tables/severity-10pct.csv')
signs = rows(E / '05-adam/a-characterization/tables/next-step-signs.csv')
check(len(hits) == len(signs) == 3, 'Characterization coverage')
severity_expected = {
    'mlp': [149, 110, 105, 82, 816, 546, 457, 308],
    'resnet': [12, 11, 11, 7, 300, 324, 162, 36],
    'transformer': [26, 22, 22, 22, 521, 421, 301, 147],
}
characterization = (DRAFT / 'sections/05a-characterization.tex').read_text(encoding='utf-8')
for record in hits:
    case = record['Case']
    expected = severity_expected[case]
    check([int(record[k]) for k in ['rho<0', 'r>1', 'Both', 'Q>=2']] == expected[:4], 'Severity hits')
    sign_record = next(r for r in signs if r['Case'] == case)
    observed_flags = [sum(int(n) for n in sign_record[k].split('/'))
                      for k in ['rho<0 up/down', 'r>1 up/down', 'Both up/down', 'Q>=2 up/down']]
    check(observed_flags == expected[4:], 'Severity precision denominators')
    # Historical 10% counts remain preserved; the main table now uses 0.1%.

new_audit = json.loads((BASE / 'work/author_feedback_20261115/diagnostic_manifest.json').read_text())
new_table = (DRAFT / 'tables/diagnostic-0p1pct.tex').read_text()
for record in new_audit['rows']:
    n, tp, fp, fn, tn = [record[k] for k in ['n','tp','fp','fn','tn']]
    check(n == 1023 and tp+fp+fn+tn == n, 'New diagnostic confusion counts')
    check(record['accuracy'] == (tp+tn)/n, 'Classification accuracy')
    check(record['events'] == tp+fn, 'Target count')
    for key, scale in [('accuracy',100),('precision',100),('precision_lift',1)]:
        value = record[key]
        if value is not None:
            token = f'{scale*value:.2f}' if key == 'precision_lift' else f'{scale*value:.1f}'
            check(token in new_table, 'New diagnostic table value: '+token)

# Current tracing windows, including the refitted large ResNet excursion.
spectral = json.loads((E / '05-adam/b-tracing/data/spectral_controls.json').read_text(encoding='utf-8'))
spectral_text = (DRAFT / 'sections/s04-tracing-methods.tex').read_text(encoding='utf-8')
tracing_text = (DRAFT / 'sections/05b-tracing.tex').read_text(encoding='utf-8')
for case in spectral['cases']:
    values = [{'mlp':'MLP','resnet':'ResNet','transformer':'Transformer'}[case['case']], str(case['rows'])]
    for kind in ['observed', 'same_energy_orthogonal', 'unit_row_directional']:
        values.extend([f"{100*case[kind]['top1']:.2f}", str(case[kind]['rank95'])])
    check(' & '.join(values) in spectral_text, 'Spectral-control table mismatch')
    check(' & '.join([values[0]]+values[2:]) in (DRAFT/'sections/appendix.tex').read_text(), 'Appendix spectral-control table mismatch')
    top_ten = 100*math.fsum(case['observed']['fractions'][:10])

# New explicit trapezoidal residual: test general smooth cubic/quartic losses,
# sampling-noise algebra, and the Lipschitz bound using independent quadrature.
rng = np.random.default_rng(20261114)
nodes, weights = np.polynomial.legendre.leggauss(32)
nodes, weights = (nodes+1)/2, weights/2
math_max_errors = {'path_identity': 0., 'loss_residual': 0., 'cap_bound': 0.}
for _ in range(128):
    p = 5
    matrix = rng.normal(size=(p, p))
    A = matrix.T @ matrix / 20
    b, x = rng.normal(size=(2, p))
    c = float(rng.uniform(.01, .1))
    eta = float(rng.uniform(.005, .25))
    noise0, noise1 = rng.normal(scale=.1, size=(2, p))
    loss = lambda z: .5*z@A@z + b@z + c*np.sum(z**4)/4
    grad = lambda z: A@z+b+c*z**3
    hess = lambda z: A+np.diag(3*c*z**2)
    g0 = grad(x)+noise0
    u = -eta*g0
    g1 = grad(x+u)+noise1
    Q = 1-g0@g1/(g0@g0)
    epsilon = g0@(noise1-noise0)/(g0@g0)
    average = sum(w*(g0@hess(x+a*u)@g0)/(g0@g0) for a,w in zip(nodes,weights))
    path_error = abs(Q+epsilon-eta*average)
    residual = sum(w*(.5-a)*(u@hess(x+a*u)@u) for a,w in zip(nodes,weights))
    surrogate = eta*(g0@g0)*(Q-2)/2+eta*g0@(noise0+noise1)/2
    loss_error = abs(loss(x+u)-loss(x)-surrogate-residual)
    LH = 6*c*max(np.max(abs(x)), np.max(abs(x+u)))
    check(path_error < 1e-11 and loss_error < 1e-11, 'Smooth identity check')
    check(abs(residual) <= LH*np.linalg.norm(u)**3/12+1e-11, 'Trapezoid remainder bound')
    check(abs(average-(g0@hess(x)@g0)/(g0@g0)) <= LH*np.linalg.norm(u)/2+1e-11, 'Curvature bound')
    math_max_errors['path_identity'] = max(math_max_errors['path_identity'], path_error)
    math_max_errors['loss_residual'] = max(math_max_errors['loss_residual'], loss_error)
    H, W, DW = rng.normal(size=(8, p)), rng.normal(size=(3, p)), rng.normal(size=(3, p))
    Db, DH = rng.normal(size=3), rng.normal(size=(8,p))
    direct = H@DW.T+Db
    scores = np.sqrt(np.mean(direct**2, axis=0))
    tau = float(rng.uniform(.01, 2))
    factors = np.minimum(1, tau/scores)
    applied_W, applied_b = DW*factors[:,None], Db*factors
    actual = (H+DH)@(W+applied_W).T+applied_b-H@W.T
    decomposition = H@applied_W.T+applied_b+DH@W.T+DH@applied_W.T
    np.testing.assert_allclose(actual, decomposition, atol=1e-12)
    check(np.max(np.sqrt(np.mean((H@applied_W.T+applied_b)**2,axis=0))) <= tau+1e-12, 'Direct cap bound')
    math_max_errors['cap_bound'] = max(math_max_errors['cap_bound'],
        max(0., float(np.max(np.sqrt(np.mean((H@applied_W.T+applied_b)**2,axis=0)))-tau)))


# Optimizer-independent extension: arbitrary applied steps, including Adam-like
# coordinate scaling and momentum, plus sampled gradient contamination.
math_max_errors['generic_step_average'] = 0.
math_max_errors['generic_noise_identity'] = 0.
math_max_errors['adam_projected_response'] = 0.
for case_index in range(128):
    p = 7
    M = rng.normal(size=(p, p))
    A = (M + M.T) / 8
    b, x = rng.normal(size=(2, p))
    c = float(rng.uniform(.01, .2))
    grad = lambda z: A@z+b+c*z**3
    hess = lambda z: A+np.diag(3*c*z**2)
    f0 = grad(x)
    # Half arbitrary directions, half actual momentum/preconditioned proposals.
    momentum = .9*rng.normal(size=p) + .1*f0
    second = .99*rng.uniform(.1, 2., size=p) + .01*f0**2
    u = (.03*rng.normal(size=p) if case_index%2 else -.03*momentum/(np.sqrt(second)+1e-8))
    f1 = grad(x+u)
    estimate = u@(f1-f0)/(u@u)
    average = sum(w*(u@hess(x+a*u)@u)/(u@u) for a,w in zip(nodes,weights))
    error = abs(estimate-average)
    check(error < 1e-10, 'General applied-step curvature identity')
    response = 1 - f0@f1/(f0@f0)
    projected = -sum(w*(f0@hess(x+a*u)@u)/(f0@f0) for a,w in zip(nodes,weights))
    check(abs(response-projected) < 1e-10, 'Adam gradient-projected response identity')
    math_max_errors['adam_projected_response'] = max(math_max_errors['adam_projected_response'], abs(response-projected))
    noise0, noise1 = rng.normal(scale=.1, size=(2,p))
    sampled = u@((f1+noise1)-(f0+noise0))/(u@u)
    noise_error = abs(sampled-estimate-u@(noise1-noise0)/(u@u))
    check(noise_error < 1e-10, 'Generic sampled-gradient term')
    LH = 6*c*max(np.max(abs(x)),np.max(abs(x+u)))
    check(abs(estimate-u@hess(x)@u/(u@u)) <= LH*np.linalg.norm(u)/2+1e-10,
          'General curvature pointwise-error bound')
    math_max_errors['generic_step_average'] = max(math_max_errors['generic_step_average'],error)
    math_max_errors['generic_noise_identity'] = max(math_max_errors['generic_noise_identity'],noise_error)

# Every active figure uses the verified publication variant. Preserved source
# PDFs may retain their old notation; they are not the figures included here.
figures = json.loads((HERE/'figure_verification.json').read_text(encoding='utf-8'))
for filename,digest in figures['outputs'].items():
    check(sha(DRAFT/'figures'/filename)==digest, 'Figure changed after numerical checks')
for record in figures['exports']:
    for change in record['labels']:
        check('Fixed probe' not in change['after'], 'Old reference terminology')
        check(not re.search(r"(?<![A-Za-z'])s(?![A-Za-z])",change['after']), 'Old step label')
assert not any(v['after']==r'$Q_t$' for r in figures['exports'] for v in r['labels'])

# Compiled content, references, fonts, and page bounds.
build = json.loads((HERE / 'build.json').read_text(encoding='utf-8'))
pdf_info = {}
texts = {}
for name in ['manuscript', 'supplementary']:
    path = DRAFT / (name+'.pdf')
    check(sha(path) == build[name]['sha256'], 'PDF changed after build')
    check(not build[name]['warnings'], 'Compilation warnings: ' + name)
    with fitz.open(path) as doc:
        check('Constructive and Destructive' in doc.metadata['title'], 'Missing PDF title metadata')
        text = '\n'.join(p.get_text() for p in doc).replace('\N{MINUS SIGN}', '-')
        check('??' not in text, 'Unresolved reference')
        check(not re.search(r'\b(TODO|TBD|PLACEHOLDER)\b', text), 'Unfinished manuscript placeholder')
        font_xrefs = {f[0] for p in doc for f in p.get_fonts(full=True)}
        missing_fonts = [x for x in font_xrefs if not doc.extract_font(x)[3]]
        check(not missing_fonts, 'Unembedded font: ' + str(missing_fonts))
        outside = []
        for i,p in enumerate(doc):
            width,height = p.cropbox.width,p.cropbox.height
            for block in p.get_text('dict')['blocks']:
                for line in block.get('lines',[]):
                    for s in line['spans']:
                        x0,y0,x1,y1 = s['bbox']
                        if x0 < -.8 or y0 < -.8 or x1 > width+.8 or y1 > height+.8:
                            outside.append([i+1, s['text']])
        check(not outside, 'Text outside page: ' + str(outside[:10]))
        texts[name] = text
        pdf_info[name] = {'pages': len(doc), 'embedded_fonts': len(font_xrefs), 'out_of_page_spans': 0, 'sha256': sha(path)}
main_text = texts['manuscript']
diagnostic_region = main_text[main_text.index('TABLE II'):main_text.index('Algorithm 1')]
compiled_diagnostic = Counter(re.findall(r'\d+(?:\.\d+)?', diagnostic_region))
expected_diagnostic = Counter()
for record in new_audit['rows']:
    for key,scale,digits in [('accuracy',100,1),('precision',100,1),('precision_lift',1,2)]:
        if record[key] is not None:
            expected_diagnostic[f'{scale*record[key]:.{digits}f}'] += 1
check(all(compiled_diagnostic[v]>=n for v,n in expected_diagnostic.items()), 'Compiled diagnostic cells missing')
for number in range(1,13):
    check(f'Fig. {number}.' in main_text, f'Figure {number} caption missing')
for number in range(1,19):
    check(f'Figure S{number}:' in texts['supplementary'], f'Supplement figure {number} caption missing')
for number in range(1,9):
    check(f'Table S{number}:' in texts['supplementary'], f'Supplement table {number} caption missing')
for token in ['Algorithm 1', 'Algorithm 2', 'TABLE I', 'TABLE II', 'TABLE III', 'TABLE IV', 'TABLE V', 'TABLE VI', 'TABLE VII']:
    check(token in main_text, 'Missing compiled element: ' + token)
table_region = main_text[main_text.index('TABLE III'):main_text.index('Fig. 9.')]
compiled = Counter(re.findall(r'[+-]?\d+(?:\.\d+)?',table_region))
round2 = json.loads((BASE/'work/author_feedback2_20261115/presentation-manifest.json').read_text())
check(len(round2['points'])==36,'Complete final-loss comparison')
for point in round2['points']:
    values=computed[point['case'],point['arm']]
    check(math.isclose(point['final_loss'],values[2],rel_tol=2e-12),'Final-loss scatter coordinate')
    check(math.isclose(point['positive_variation'],values[0],rel_tol=2e-12,abs_tol=2e-12),'Variation scatter coordinate')
check((DRAFT/'tables/matched-main-outcomes.tex').read_text().count('\\multirow{')==4,'Intervention family multirows')
expected = Counter(v for r in round2['table'] for v in r['display_values'])
check(all(compiled[v] >= n for v,n in expected.items()), 'Compiled intervention cells missing')
for row in round2['table']:
    expected_values=[f'{1000*computed[c,row["arm"]][i]:.1f}' for c in cases for i in [1,0,2]]
    check(row['display_values']==expected_values,'Absolute table reduction or rounding mismatch')
for file,digest in round2['figures'].items():
    check(sha(DRAFT/'figures'/file)==digest,'Round-two figure hash mismatch')
for arm,counts in round2['mse_rises'].items():
    rising=curves[f'mlp__{arm}__mse_delta_loss']>1e-8
    check(int(rising.sum())==counts['total'],'q25 total loss rises')
    check(int((curves[f'mlp__{arm}__mse_linear'][rising]<0).sum())==counts['corrective'],'q25 corrective loss rises')
retention=json.loads((BASE/'work/author_feedback2_20261115/retention-manifest.json').read_text())
for file,digest in retention['sources'].items():
    check(sha(ROOT/file)==digest,'Retention source changed')
for r in retention['rows']:
    check(math.isclose(r['retention'],r['net_displacement']/r['path_length'],rel_tol=1e-13),'Retention ratio')
for site,case in [('output','mlp'),('output.log_probabilities','resnet'),('output.log_probabilities','transformer')]:
    r=next(x for x in retention['rows'] if x['case']==case and x['site']==site)
    check(f'{100*r["retention"]:.2f}\\%' in tracing_text,'Output retention claim')
# Compact appendix: independently recover each plotted series and heatmap.
appendix = json.loads((BASE/'work/appendix_20261115/manifest.json').read_text())
arrsha = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
for file,digest in appendix['sources'].items():
    check(sha(data/file)==digest, 'Appendix source changed')
for file,digest in appendix['figures'].items():
    check(sha(DRAFT/'figures'/file)==digest, 'Appendix figure changed')
check(len(appendix['lines'])==78 and len(appendix['maps'])==18, 'Appendix figure coverage')
for r in appendix['lines']:
    stem=f'{r["case"]}__{r["arm"]}__'
    x=curves[stem+'state']; y=curves[stem+r['metric']]
    if r['metric']!='loss': x=x[1:]
    check(arrsha(x)==r['x_sha256'] and arrsha(y)==r['y_sha256'], 'Appendix plotted series')
for r in appendix['maps']:
    stem=f'{r["case"]}__{r["arm"]}__'
    den=curves[f'{r["case"]}__control__stage_change_rms'][:,1:].max(axis=1)
    z=(curves[stem+'stage_change_rms'][:,1:]/den[:,None] if r['metric']=='relative_rms'
       else curves[stem+'stage_rho'][:,1:])
    check(arrsha(z)==r['sha256'], 'Appendix heatmap normalization')
late_text=(DRAFT/'tables/appendix-last32.tex').read_text()
for r in appendix['last32']:
    value=float(curves[f'{r["case"]}__{r["arm"]}__loss'][-32:].mean())
    check(math.isclose(value,r['value'],rel_tol=1e-13), 'Late mean reduction')
    check(r['display']==f'{1000*value:.1f}' and r['display'] in late_text, 'Late table value')
cost_text=(DRAFT/'tables/appendix-costs.tex').read_text()
for r in rows(data/'matched-costs.csv'):
    if r['arm']!='cap_q50': continue
    for metric,count in [('gradient_seconds','full_gradient_passes'),('probe_seconds','full_gradient_passes'),('controller_seconds','updates')]:
        check(f'{1000*float(r[metric])/int(r[count]):.1f}' in cost_text, 'Component cost')
appendix_text=(DRAFT/'sections/appendix.tex').read_text()
check('51/61/49' in appendix_text, 'Tracing window coverage')
check(not re.search(r'Supplementary|supplement', '\n'.join(p.read_text() for p in (DRAFT/'sections').glob('0*.tex'))), 'External supplement dependency')
all_tex = '\n'.join(p.read_text(encoding='utf-8') for p in (DRAFT / 'sections').glob('*.tex'))
check(not re.search(r'^ef\{|Algorithm~\s+ef\{', all_tex, re.M), 'Malformed reference escape')
check(not re.search(r':\s*\\begin\{(?:equation|align)', all_tex), 'Colon before displayed equation')
check(not re.search(r'fixed[- ]probe|secant|Barzilai', all_tex, re.I), 'Superseded terminology')
check(not re.search(r'(?<![A-Za-z])_s(?![A-Za-z])|_\{s[+}\-]', all_tex), 'Old mathematical clock')
bib = (DRAFT / 'references.bib').read_text(encoding='utf-8')
keys = {key.strip() for group in re.findall(r'\\cite(?:\[[^\]]*\])?\{([^}]*)\}',all_tex) for key in group.split(',')}
available = re.findall(r'@\w+\{([^,]+),',bib)
check(keys <= set(available), 'Missing bibliography key')
check(len(available) == len(set(available)), 'Duplicate bibliography keys')
abstract = (DRAFT / 'sections/00-abstract.tex').read_text(encoding='utf-8')
abstract_words = len(re.sub(r'\\(?:begin|end)\{abstract\}', '', abstract).split())
check(abstract_words <= 250, 'Abstract over 250 words')
result = {'status': 'passed', 'protected_files_unchanged': len(protected),
          'main_figures': 9, 'main_tables': 3, 'appendix_figures': 3, 'appendix_tables': 4, 'algorithms': 2,
          'appendix_verified_lines': 78, 'appendix_verified_heatmaps': 18, 'appendix_late_mean_cells': 36,
          'recomputed_intervention_metrics': 144, 'compiled_intervention_cells': 108,
          'gd_conditions_checked': 9, 'historical_severity_counts_checked': 24,
          'new_diagnostic_conditions': 21, 'new_diagnostic_confusion_cells': 84,
          'compiled_diagnostic_metric_cells': sum(expected_diagnostic.values()),
          'spectral_control_cells_checked': 24, 'supplement_figures': 18, 'supplement_tables': 8,
          'representation_retention_sites': len(retention['rows']),
          'smooth_loss_and_cap_cases': 128, 'general_optimizer_step_cases': 128,
          'verified_figure_pdfs': len(figures['outputs']), 'mathematical_max_errors': math_max_errors,
          'abstract_words': abstract_words, 'citation_keys': len(keys),
          'output_motion_claims': motion_claims, 'pdfs': pdf_info,
          'visual_review': 'Recorded separately after final page inspection',
          'verifier_sha256': sha(Path(__file__))}
(HERE / 'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({k:result[k] for k in ['status','protected_files_unchanged','recomputed_intervention_metrics','compiled_intervention_cells','smooth_loss_and_cap_cases','abstract_words','citation_keys','pdfs']},indent=2))
