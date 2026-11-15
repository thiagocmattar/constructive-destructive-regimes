"""Verify Section V prose edits against their local pre-edit snapshots."""
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
NAMES = ['05-adam.tex', '05a-characterization.tex', '05b-tracing.tex']
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
old = {n: (SCRATCH / n).read_text(encoding='utf-8') for n in NAMES}
new = {n: (DRAFT / 'sections' / n).read_text(encoding='utf-8') for n in NAMES}
equations = lambda s: re.findall(r'\\begin\{equation\}.*?\\end\{equation\}', s, re.S)
figures = lambda s: re.findall(r'\\begin\{figure\*\}.*?\\end\{figure\*\}', s, re.S)
labels = lambda s: re.findall(r'\\label\{([^}]+)\}', s)
headings = lambda s: re.findall(r'\\(?:subsection|section)\{([^}]+)\}', s)
for name in NAMES:
    assert equations(old[name]) == equations(new[name])
    assert figures(old[name]) == figures(new[name])
    assert labels(old[name]) == labels(new[name])
    assert headings(old[name]) == headings(new[name])
    assert re.findall(r'\\cite\{([^}]+)\}', old[name]) == re.findall(r'\\cite\{([^}]+)\}', new[name])
    assert re.findall(r'\\input\{([^}]+)\}', old[name]) == re.findall(r'\\input\{([^}]+)\}', new[name])
assert sum(len(equations(s)) for s in new.values()) == 11
assert sum(len(figures(s)) for s in new.values()) == 3
algorithm = lambda s: re.findall(r'\\begin\{algorithm\}.*?\\end\{algorithm\}', s, re.S)
assert algorithm(old[NAMES[2]]) == algorithm(new[NAMES[2]])
svd = lambda s: s.split('For each site separately, collect the')[1].split('Algorithm~')[0]
assert svd(old[NAMES[2]]) == svd(new[NAMES[2]])
numbers = lambda texts: Counter(re.findall(r'\d+(?:[.,]\d+)*', '\n'.join(texts)))
before_numbers, after_numbers = numbers(old.values()), numbers(new.values())
assert not before_numbers - after_numbers
assert after_numbers - before_numbers == Counter({'0.1': 1, '10': 1})

audit = (HERE / 'PARAGRAPH_REVIEW.md').read_text(encoding='utf-8')
entries = re.findall(r'^## ([^\n]+)\n(.*?)(?=^## |\Z)', audit, re.M | re.S)
assert len(entries) == 9
replayed = dict(old)
for key, body in entries:
    assert body.index('**Read**') < body.index('**Diagnosis:**') < body.index('**Proposed solution:**') < body.index('**Rewrite:**')
    before, after = re.findall(r'```tex\n(.*?)\n```', body, re.S)
    name = NAMES[1] if key.startswith('V-A') else NAMES[2] if key.startswith('V-B') else NAMES[0]
    assert replayed[name].count(before) == 1, key
    replayed[name] = replayed[name].replace(before, after, 1)
assert replayed == new

baseline = json.loads((SCRATCH / 'baseline.json').read_text())
changed = {Path(p).as_posix() for p, value in baseline.items() if sha(ROOT / p) != value}
expected = {'manuscript/consolidation/draft/' + p for p in ['README.md', 'manuscript.pdf'] + ['sections/' + n for n in NAMES]}
assert changed == expected, changed
preexisting = json.loads((SCRATCH / 'preexisting.json').read_text())
assert all(sha(ROOT / p) == value for p, value in preexisting.items())
all_labels = {label for p in DRAFT.rglob('*.tex') for label in labels(p.read_text(encoding='utf-8'))}
refs = {ref for s in new.values() for ref in re.findall(r'\\(?:eqref|ref)\{([^}]+)\}', s)}
assert refs <= all_labels
build = json.loads((HERE / 'build.json').read_text())
assert build['warnings'] == [] and build['sha256'] == sha(DRAFT / 'manuscript.pdf')
log = (SCRATCH / 'latex/manuscript.log').read_text(encoding='utf-8', errors='replace')
assert not re.search(r'Overfull|Underfull|undefined|multiply defined|LaTeX Warning|Package .* Warning', log)
with fitz.open(SCRATCH / 'manuscript.pdf') as a, fitz.open(DRAFT / 'manuscript.pdf') as b:
    assert len(a) == len(b) == build['pages'] == 22
    changed_pages = [i + 1 for i, (p, q) in enumerate(zip(a, b)) if p.get_pixmap().samples != q.get_pixmap().samples]
assert changed_pages == list(range(8, 15)), changed_pages
result = {
    'equations_preserved': 11, 'figure_blocks_and_captions_preserved': 3,
    'labels_preserved': sum(len(labels(s)) for s in new.values()),
    'algorithm_preserved': True, 'svd_and_overlap_exposition_preserved': True,
    'headings_citations_and_input_paths_preserved': True,
    'existing_numeric_tokens_preserved': True,
    'added_numeric_tokens': {'0.1': 'Existing 0.001 target restated as percent', '10': 'Existing excursion threshold repeated for comparison'},
    'audit_entries': len(entries), 'audit_replays_to_final_sources': True,
    'unchanged_draft_files': len(baseline) - len(changed),
    'unchanged_preexisting_modified_files': len(preexisting),
    'changed_draft_files': sorted(changed), 'resolved_reference_keys': len(refs),
    'source_whitespace_token_counts_including_tex': {n: {'before': len(old[n].split()), 'after': len(new[n].split())} for n in NAMES},
    'pdf': build, 'pages_with_changed_rendering': changed_pages,
    'manual_visual_review': 'Pages 8-12 individually and all changed pages 8-14 in contact sheets. No clipping or overlap observed. Other pages render identically to reviewed baseline.'
}
(HERE / 'verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(result, indent=2))
