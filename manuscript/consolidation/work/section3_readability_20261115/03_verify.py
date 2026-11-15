"""Verify the conservative Section III edit against its local pre-edit snapshot."""
from pathlib import Path
import hashlib
import json
import re
import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DRAFT = HERE.parents[1] / 'draft'
SCRATCH = HERE / 'scratch'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
old = (SCRATCH / '03-regimes.tex').read_text(encoding='utf-8')
new = (DRAFT / 'sections/03-regimes.tex').read_text(encoding='utf-8')
equations = lambda s: re.findall(r'\\begin\{equation\}.*?\\end\{equation\}', s, re.S)
labels = lambda s: re.findall(r'\\label\{([^}]+)\}', s)
headings = lambda s: re.findall(r'\\subsection\{([^}]+)\}', s)
assert equations(old) == equations(new) and len(equations(new)) == 19
assert labels(old) == labels(new)
expected_headings = headings(old)
expected_headings[-1] = 'Conditions for Loss Decrease and Increase'
assert headings(new) == expected_headings and len(expected_headings) == 5
section_d = lambda s: s.split(r'\subsection{Normalized Displacement and Cumulative Interference}')[1].split(r'\subsection{')[0]
assert section_d(old) == section_d(new)

audit = (HERE / 'PARAGRAPH_REVIEW.md').read_text(encoding='utf-8')
entries = re.findall(r'^## ([^\n]+)\n(.*?)(?=^## |\Z)', audit, re.M | re.S)
assert len(entries) == 20
replayed = old
for key, body in entries:
    assert body.index('**Read**') < body.index('**Diagnosis:**') < body.index('**Proposed solution:**') < body.index('**Rewrite:**')
    before, after = re.findall(r'```tex\n(.*?)\n```', body, re.S)
    assert replayed.count(before) == 1, key
    replayed = replayed.replace(before, after, 1)
assert replayed == new

baseline = json.loads((SCRATCH / 'baseline.json').read_text())
changed = {Path(p).as_posix() for p, value in baseline.items() if sha(ROOT / p) != value}
assert changed == {'manuscript/consolidation/draft/' + p for p in (
    'README.md', 'manuscript.pdf', 'sections/03-regimes.tex')}, changed
preexisting = json.loads((SCRATCH / 'preexisting.json').read_text())
assert all(sha(ROOT / p) == value for p, value in preexisting.items())
all_labels = {label for p in DRAFT.rglob('*.tex') for label in labels(p.read_text(encoding='utf-8'))}
refs = set(re.findall(r'\\(?:eqref|ref)\{([^}]+)\}', new))
assert refs <= all_labels
build = json.loads((HERE / 'build.json').read_text())
assert build['warnings'] == [] and build['sha256'] == sha(DRAFT / 'manuscript.pdf')
log = (SCRATCH / 'latex/manuscript.log').read_text(encoding='utf-8', errors='replace')
assert not re.search(r'Overfull|Underfull|undefined|multiply defined|LaTeX Warning|Package .* Warning', log)
with fitz.open(SCRATCH / 'manuscript.pdf') as before_pdf, fitz.open(DRAFT / 'manuscript.pdf') as after_pdf:
    assert len(before_pdf) == len(after_pdf) == build['pages'] == 22
    changed_pages = [i + 1 for i, (p, q) in enumerate(zip(before_pdf, after_pdf))
                     if p.get_pixmap().samples != q.get_pixmap().samples]
assert changed_pages == list(range(4, 15)), changed_pages
result = {
    'equations_preserved': 19, 'labels_preserved': len(labels(new)),
    'subsection_order_preserved': True, 'subsections': headings(new),
    'normalized_displacement_subsection_unchanged': True,
    'audit_entries': len(entries), 'audit_replays_to_final_source': True,
    'unchanged_draft_files': len(baseline) - len(changed),
    'unchanged_preexisting_modified_files': len(preexisting),
    'changed_draft_files': sorted(changed), 'resolved_reference_keys': len(refs),
    'source_whitespace_token_counts_including_tex': {'before': len(old.split()), 'after': len(new.split())},
    'pdf': build, 'pages_with_changed_rendering': changed_pages,
    'manual_visual_review': 'Pages 4-6 individually and all changed pages 4-14 in contact sheets; no clipping or overlaps. Other pages render identically to the reviewed baseline.'
}
(HERE / 'verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(result, indent=2))
