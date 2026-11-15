"""Verify the Section IV prose revision against its local pre-edit snapshot."""
from collections import Counter
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
old = (SCRATCH / '04-gd-sgd.tex').read_text(encoding='utf-8')
new = (DRAFT / 'sections/04-gd-sgd.tex').read_text(encoding='utf-8')
floats = lambda s: re.findall(r'\\begin\{(?:figure\*|table\*)\}.*?\\end\{(?:figure\*|table\*)\}', s, re.S)
labels = lambda s: re.findall(r'\\label\{([^}]+)\}', s)
headings = lambda s: re.findall(r'\\subsection\{([^}]+)\}', s)
assert floats(old) == floats(new) and len(floats(new)) == 5
assert labels(old) == labels(new)
assert headings(old) == headings(new) and len(headings(new)) == 3
assert Counter(re.findall(r'\d+(?:[.,]\d+)*', old)) == Counter(re.findall(r'\d+(?:[.,]\d+)*', new))
assert re.findall(r'\\cite\{([^}]+)\}', old) == re.findall(r'\\cite\{([^}]+)\}', new)
assert new.index('The complete displacement sums') < new.index('The transition from constructive')

audit = (HERE / 'PARAGRAPH_REVIEW.md').read_text(encoding='utf-8')
entries = re.findall(r'^## ([^\n]+)\n(.*?)(?=^## |\Z)', audit, re.M | re.S)
assert len(entries) == 10
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
    'README.md', 'manuscript.pdf', 'sections/04-gd-sgd.tex')}, changed
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
assert changed_pages == list(range(6, 12)), changed_pages
result = {
    'figure_table_blocks_and_captions_preserved': 5,
    'labels_preserved': len(labels(new)), 'subsections_preserved': headings(new),
    'numeric_tokens_preserved': True, 'citation_keys_preserved': True,
    'audit_entries': len(entries), 'audit_replays_to_final_source': True,
    'unchanged_draft_files': len(baseline) - len(changed),
    'unchanged_preexisting_modified_files': len(preexisting),
    'changed_draft_files': sorted(changed), 'resolved_reference_keys': len(refs),
    'source_whitespace_token_counts_including_tex': {'before': len(old.split()), 'after': len(new.split())},
    'pdf': build, 'pages_with_changed_rendering': changed_pages,
    'manual_visual_review': 'Section IV text inspected individually; final changed pages 6-11 inspected in contact sheet. Closing-line spill corrected. No clipping or overlap observed. Other pages render identically to reviewed baseline.'
}
(HERE / 'verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(result, indent=2))
