"""Check revision scope, references, diagnostic claim, and compiled output."""
from pathlib import Path
import hashlib
import json
import re
import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DRAFT = HERE.parents[1] / 'draft'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
baseline = json.loads((HERE / 'scratch/baseline.json').read_text())
allowed = {'manuscript.pdf', '01-introduction.tex', 'README.md'}
changed = [p for p, h in baseline.items() if sha(ROOT / p) != h]
assert set(Path(p).name for p in changed) == allowed, changed
preexisting = json.loads((HERE / 'scratch/preexisting.json').read_text())
assert all(sha(ROOT / p) == h for p, h in preexisting.items())

intro = (DRAFT / 'sections/01-introduction.tex').read_text(encoding='utf-8')
before = (HERE / 'scratch/01-introduction-before.tex').read_text(encoding='utf-8')
assert intro.count('\\item ') == 4
bib = (DRAFT / 'references.bib').read_text(encoding='utf-8')
keys = set(re.findall(r'@\w+\{([^,]+),', bib))
citations = {k for group in re.findall(r'\\cite\{([^}]+)\}', intro) for k in group.split(',')}
assert citations <= keys, citations - keys
alltex = '\n'.join(p.read_text(encoding='utf-8') for p in DRAFT.rglob('*.tex'))
labels = set(re.findall(r'\\label\{([^}]+)\}', alltex))
refs = set(re.findall(r'\\ref\{([^}]+)\}', intro))
assert refs <= labels, refs - labels

table = (DRAFT / 'tables/diagnostic-0p1pct.tex').read_text(encoding='utf-8')
rows = {}
for line in table.splitlines():
    if '&' not in line or not line.rstrip().endswith('\\\\'):
        continue
    parts = line.split('&')
    if len(parts) != 10 or parts[0].strip() == 'Condition':
        continue
    values = []
    for cell in parts[1:]:
        cell = cell.replace('\\\\', '').strip()
        values.append(None if cell == '--' else float(cell))
    rows[parts[0].strip()] = values
joint = rows[r'$\rho_t<-\|g_{t-1}\|/\|g_t\|$']
equivalent = rows[r'$\eta\widehat\kappa_t\geq2$']
assert joint == equivalent
for i in (0, 2, 3, 5, 6, 8):
    assert joint[i] == max(v[i] for v in rows.values() if v[i] is not None)

build = json.loads((HERE / 'build.json').read_text())
assert build['warnings'] == [] and build['pages'] == 22
assert sha(DRAFT / 'manuscript.pdf') == build['sha256']
with fitz.open(DRAFT / 'manuscript.pdf') as doc:
    assert len(doc) == 22
    first = '\n'.join(page.get_text() for page in doc[:2])
    assert 'We make four contributions' in first
    assert 'II. STABILITY ANALYSIS OF SGD' in first
    assert '0.1%' in first
    assert first.count('\u2022') == 4

result = {
    'status': 'passed',
    'changed_draft_files': changed,
    'protected_draft_files_unchanged': len(baseline) - len(changed),
    'preexisting_working_files_unchanged': len(preexisting),
    'source_whitespace_token_count': {'before': len(before.split()), 'after': len(intro.split())},
    'contribution_bullets': 4,
    'resolved_citation_keys': sorted(citations),
    'resolved_section_labels': sorted(refs),
    'diagnostic_table': {'joint_rule_equals_projected_rule': True,
                         'highest_accuracy_and_lift_each_case': True,
                         'joint_accuracy_hit_lift_by_case': joint},
    'pdf': build,
    'introduction_sha256': sha(DRAFT / 'sections/01-introduction.tex'),
    'review': 'Final pages 1-2 inspected individually; all 22 pages inspected in rendered contact sheets. No clipping, overlap, or misplaced floats observed.'
}
(HERE / 'verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(result, indent=2))
