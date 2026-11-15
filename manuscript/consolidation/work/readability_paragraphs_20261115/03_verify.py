"""Verify editorial scope, sequential audit, references, and the rebuilt PDF.

Requires the ignored pre-edit manifests and paragraph state from this review.
The audit itself is retained in Git and can be inspected independently.
"""
from pathlib import Path
import hashlib
import json
import re
import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DRAFT = HERE.parents[1] / 'draft'
SCRATCH = HERE / 'scratch'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


baseline = json.loads((SCRATCH / 'baseline.json').read_text())
changed = {Path(name).as_posix() for name, value in baseline.items()
           if digest(ROOT / name) != value}
expected = {'manuscript/consolidation/draft/' + name for name in (
    'README.md', 'manuscript.pdf', 'sections/01-introduction.tex',
    'sections/02-related-work.tex')}
assert changed == expected, changed
preexisting = json.loads((SCRATCH / 'preexisting.json').read_text())
assert all(digest(ROOT / name) == value for name, value in preexisting.items())

names = ['01-introduction.tex', '02-related-work.tex']
old = {name: (SCRATCH / name).read_text(encoding='utf-8') for name in names}
new = {name: (DRAFT / 'sections' / name).read_text(encoding='utf-8') for name in names}
equations = lambda text: re.findall(r'\\begin\{equation\}.*?\\end\{equation\}', text, re.S)
labels = lambda text: re.findall(r'\\label\{([^}]+)\}', text)
assert equations(old[names[1]]) == equations(new[names[1]])
assert len(equations(new[names[1]])) == 5
assert all(labels(old[name]) == labels(new[name]) for name in names)
assert new[names[0]].count(r'\item ') == 4
assert new[names[1]].count(r'\subsection{') == 3

state = json.loads((SCRATCH / 'paragraphs.json').read_text(encoding='utf-8'))
assert len(state) == 29 and all(unit['status'] != 'pending' for unit in state)
assert all(new[unit['file']].count(unit['replacement']) == 1 for unit in state)
audit = (HERE / 'PARAGRAPH_REVIEW.md').read_text(encoding='utf-8')
entries = re.findall(
    r'^## (I{1,2}-\d{2})(?: \(coherence review\))?\n(.*?)(?=^## |\Z)',
    audit, re.M | re.S)
assert len(entries) == 36
assert [key for key, _ in entries[:29]] == [unit['id'] for unit in state]
replayed = dict(old)
for key, body in entries:
    assert body.index('**Read**') < body.index('**Diagnosis:**') < body.index('**Proposed solution:**') < body.index('**Rewrite:**')
    before, after = re.findall(r'```tex\n(.*?)\n```', body, re.S)
    name = next(unit['file'] for unit in state if unit['id'] == key)
    assert replayed[name].count(before) == 1, key
    replayed[name] = replayed[name].replace(before, after, 1)
assert replayed == new

bib = (DRAFT / 'references.bib').read_text(encoding='utf-8')
bibkeys = set(re.findall(r'@\w+\s*\{\s*([^,]+),', bib))
citations = {key.strip() for text in new.values()
             for group in re.findall(r'\\cite\{([^}]+)\}', text) for key in group.split(',')}
assert citations <= bibkeys, citations - bibkeys
all_labels = {label for path in DRAFT.rglob('*.tex')
              for label in labels(path.read_text(encoding='utf-8'))}
refs = {ref for text in new.values() for ref in re.findall(r'\\(?:eqref|ref)\{([^}]+)\}', text)}
assert refs <= all_labels, refs - all_labels

build = json.loads((HERE / 'build.json').read_text())
assert build['warnings'] == [] and build['sha256'] == digest(DRAFT / 'manuscript.pdf')
with fitz.open(DRAFT / 'manuscript.pdf') as pdf:
    assert len(pdf) == build['pages'] == 22
    assert all(page.get_text().strip() for page in pdf)
log = (SCRATCH / 'latex/manuscript.log').read_text(encoding='utf-8', errors='replace')
assert not re.search(r'Overfull|Underfull|undefined|multiply defined|LaTeX Warning|Package .* Warning', log)

result = {
    'original_prose_units': len(state), 'audit_entries': len(entries),
    'audit_replays_to_final_sources': True,
    'equations_preserved': 5, 'labels_preserved': sum(len(labels(t)) for t in new.values()),
    'contribution_bullets': 4, 'related_work_subsections': 3,
    'resolved_citation_keys': len(citations), 'resolved_reference_keys': len(refs),
    'unchanged_draft_files': len(baseline) - len(changed),
    'unchanged_preexisting_modified_files': len(preexisting),
    'changed_draft_files': sorted(changed),
    'source_whitespace_token_counts_including_tex': {
        name: {'before': len(old[name].split()), 'after': len(new[name].split())} for name in names},
    'pdf': build,
    'manual_visual_review': 'Final pages 1-4 individually; all 22 pages in contact sheets. No clipping or overlaps observed.'
}
(HERE / 'verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(result, indent=2))
