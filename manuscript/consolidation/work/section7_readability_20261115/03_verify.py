"""Verify the conclusion revision against its local pre-edit snapshot."""
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
old = (SCRATCH / '06-conclusion.tex').read_text(encoding='utf-8')
new = (DRAFT / 'sections/06-conclusion.tex').read_text(encoding='utf-8')
assert old.split('\n\n')[0] == new.split('\n\n')[0]
assert len(old.strip().split('\n\n')) == len(new.strip().split('\n\n')) == 5
audit = (HERE / 'PARAGRAPH_REVIEW.md').read_text(encoding='utf-8')
entries = re.findall(r'^## ([^\n]+)\n(.*?)(?=^## |\Z)', audit, re.M | re.S)
assert len(entries) == 4
replayed = old
for key, body in entries:
    assert body.index('**Read**') < body.index('**Diagnosis:**') < body.index('**Proposed solution:**') < body.index('**Rewrite:**')
    before, after = re.findall(r'```tex\n(.*?)\n```', body, re.S)
    assert replayed.count(before) == 1, key
    replayed = replayed.replace(before, after, 1)
assert replayed == new
baseline = json.loads((SCRATCH / 'baseline.json').read_text())
changed = {p for p, value in baseline.items() if sha(ROOT / p) != value}
assert changed == {'manuscript/consolidation/draft/' + p for p in (
    'README.md', 'manuscript.pdf', 'sections/06-conclusion.tex')}, changed
preexisting = json.loads((SCRATCH / 'preexisting.json').read_text())
assert all(sha(ROOT / p) == value for p, value in preexisting.items())
build = json.loads((HERE / 'build.json').read_text())
assert build['warnings'] == [] and build['sha256'] == sha(DRAFT / 'manuscript.pdf')
log = (SCRATCH / 'latex/manuscript.log').read_text(encoding='utf-8', errors='replace')
assert not re.search(r'Overfull|Underfull|undefined|multiply defined|LaTeX Warning|Package .* Warning', log)
with fitz.open(SCRATCH / 'manuscript.pdf') as a, fitz.open(DRAFT / 'manuscript.pdf') as b:
    assert len(a) == len(b) == build['pages'] == 22
    changed_pages = [i + 1 for i, (p, q) in enumerate(zip(a, b))
                     if p.get_pixmap().samples != q.get_pixmap().samples]
assert changed_pages == [17, 18, 19], changed_pages
result = {
    'paragraphs_reviewed_and_rewritten': 4,
    'audit_replays_to_final_source': True,
    'section_heading_and_label_preserved': True,
    'unchanged_draft_files': len(baseline) - len(changed),
    'unchanged_preexisting_modified_files': len(preexisting),
    'changed_draft_files': sorted(changed),
    'pdf': build, 'pages_with_changed_rendering': changed_pages,
    'manual_visual_review': 'Conclusion page 17 inspected at full page resolution; changed pages 17-19 inspected together. No clipping or overlap observed. Other pages render identically to baseline.'
}
(HERE / 'verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
for p in [HERE / 'OBSERVATIONS.md', DRAFT / 'README.md']:
    for link in re.findall(r'\]\(([^)]+)\)', p.read_text(encoding='utf-8')):
        if '://' not in link and not link.startswith('#'):
            assert (p.parent / link.split('#')[0]).exists(), (p, link)
print(json.dumps(result, indent=2))
