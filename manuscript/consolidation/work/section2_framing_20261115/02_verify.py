"""Verify Section II's preserved mathematics, references and revision scope."""
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
changed = [p for p, h in baseline.items() if sha(ROOT / p) != h]
assert {Path(p).name for p in changed} == {
    'manuscript.pdf', 'README.md', '01-introduction.tex',
    '02-related-work.tex', 'references.bib'}, changed
preexisting = json.loads((HERE / 'scratch/preexisting.json').read_text())
assert all(sha(ROOT / p) == h for p, h in preexisting.items())

before = (HERE / 'scratch/02-related-work.tex').read_text(encoding='utf-8')
after = (DRAFT / 'sections/02-related-work.tex').read_text(encoding='utf-8')
equations = lambda s: re.findall(r'\\begin\{equation\}.*?\\end\{equation\}', s, re.S)
assert len(equations(before)) == 5
assert equations(before) == equations(after)
labels = lambda s: re.findall(r'\\label\{([^}]+)\}', s)
assert labels(before) == labels(after)
assert after.count('\\subsection{') == 3
assert '@@EQ' not in after
intro_before = (HERE / 'scratch/01-introduction.tex').read_text(encoding='utf-8')
intro_after = (DRAFT / 'sections/01-introduction.tex').read_text(encoding='utf-8')
assert intro_after == intro_before.replace(
    'Section~\\ref{sec:background} reviews stability perspectives.',
    'Section~\\ref{sec:background} reviews the related work and conceptual foundations.')

bib_before = (HERE / 'scratch/references.bib').read_text(encoding='utf-8')
bib_after = (DRAFT / 'references.bib').read_text(encoding='utf-8')
assert bib_after.startswith(bib_before.rstrip())
keys_before = set(re.findall(r'@\w+\{([^,]+),', bib_before))
key_list = re.findall(r'@\w+\{([^,]+),', bib_after)
keys = set(key_list)
assert len(key_list) == len(keys)
assert keys - keys_before == {'andreyev2025stochastic', 'gupta2021path', 'lee2026road'}
citations = {k.strip() for group in re.findall(r'\\cite\{([^}]+)\}', after) for k in group.split(',')}
assert citations <= keys, citations - keys
alltex = '\n'.join(p.read_text(encoding='utf-8') for p in DRAFT.rglob('*.tex'))
refs = set(re.findall(r'\\ref\{([^}]+)\}', after))
assert refs <= set(labels(alltex))

build = json.loads((HERE / 'build.json').read_text())
assert build['warnings'] == [] and build['pages'] == 22
assert sha(DRAFT / 'manuscript.pdf') == build['sha256']
with fitz.open(DRAFT / 'manuscript.pdf') as doc:
    text = ' '.join(' '.join(p.get_text().split()) for p in doc)
    assert len(doc) == 22
    assert 'II. RELATED WORK AND CONCEPTUAL FOUNDATIONS' in text
    assert all(s in text for s in ['BB1', 'Lee and Lee', 'SVCCA'])
    assert 'III. CONSTRUCTIVE AND DESTRUCTIVE REGIMES' in text

result = {
    'status': 'passed', 'changed_draft_files': changed,
    'protected_draft_files_unchanged': len(baseline) - len(changed),
    'preexisting_working_files_unchanged': len(preexisting),
    'equations_preserved': 5, 'labels_preserved': labels(after),
    'subsections': re.findall(r'\\subsection\{([^}]+)\}', after),
    'introduction_change': 'Roadmap sentence only',
    'bibliography_change': 'Three appended entries; existing entries unchanged',
    'new_bibliography_keys': sorted(keys - keys_before),
    'resolved_citation_keys': sorted(citations),
    'source_whitespace_token_count': {'before': len(before.split()), 'after': len(after.split())},
    'pdf': build, 'section2_sha256': sha(DRAFT / 'sections/02-related-work.tex'),
    'visual_review': 'Pages 2-4 reviewed individually, all 22 pages reviewed in contact sheets; no clipping, overlap or broken references observed.'
}
(HERE / 'verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(result, indent=2))
