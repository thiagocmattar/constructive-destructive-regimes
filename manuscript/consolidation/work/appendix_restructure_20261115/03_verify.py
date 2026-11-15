"""Verify the appendix restructuring against the frozen local pre-edit files."""
from pathlib import Path
import hashlib
import json
import re
import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DRAFT = HERE.parents[1] / 'draft'
SCRATCH = HERE / 'scratch'
BASE = SCRATCH / 'baseline'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: p.read_text(encoding='utf-8')
old, new = read(BASE/'sections/appendix.tex'), read(DRAFT/'sections/appendix.tex')
equations = lambda s: re.findall(r'\\begin\{equation\}.*?\\end\{equation\}', s, re.S)
normalize = lambda s: re.sub(r'\s+', '', re.sub(r'\\label\{[^}]+\}', '', s))
assert len(equations(old)) == len(equations(new)) == 4
assert list(map(normalize, equations(old))) == list(map(normalize, equations(new)))
tables = lambda s: re.findall(r'\\begin\{table\}.*?\\end\{table\}', s, re.S)
assert tables(old) == tables(new) and len(tables(new)) == 2
assert len(re.findall(r'\\section\{', new)) == 3
assert len(re.findall(r'\\subsection\{', new)) == 5
assert 'app:plots' not in new

# Every edited manuscript block, including the heading repair, replays exactly.
decisions = json.loads(read(HERE/'decisions.json'))
replayed = {}
for item in decisions:
    file, before, after = item['file'], item['before'], item['after']
    if file not in replayed:
        replayed[file] = read(BASE/file) if (BASE/file).exists() else ''
    text = replayed[file]
    assert text.count(before) == 1 if before else text == '', item['id']
    replayed[file] = text.replace(before, after, 1) if before else after
assert all(text == read(DRAFT/file) for file, text in replayed.items())
audit = read(HERE/'PARAGRAPH_REVIEW.md')
entries = re.findall(r'^## ([^\n]+)\n(.*?)(?=^## |\Z)', audit, re.M | re.S)
assert len(entries) == len(decisions) == 36
for (name, body), item in zip(entries, decisions):
    assert name == item['id']
    assert body.index('**Read**') < body.index('**Diagnosis:**') < body.index('**Proposed solution:**') < body.index('**Rewrite:**')
    assert re.findall(r'```tex\n(.*?)\n```', body, re.S) == [item['before'], item['after']]

# Table is an exact column projection of the existing full audit.
def rows(s):
    return [[c.strip() for c in line.removesuffix('\\\\').split('&')]
            for line in s.splitlines() if line.startswith(('MLP &','ResNet-32 &','Transformer &'))]
full = rows(read(DRAFT/'tables/gd-audit.tex'))
compact = rows(read(DRAFT/'tables/appendix-gd-audit.tex'))
assert len(full) == len(compact) == 9
assert compact == [[row[i] for i in (0,1,4,5,6)] for row in full]

# Captions transfer unchanged except the standalone main-paper figure pointer.
old_figures = re.findall(r'\\begin\{figure\*\}.*?\\end\{figure\*\}', old, re.S)
s7 = read(DRAFT/'sections/s07-intervention-trajectories.tex')
new_figures = re.findall(r'\\begin\{figure\}.*?\\end\{figure\}', s7, re.S)
assert len(old_figures) == len(new_figures) == 3
for a,b in zip(old_figures,new_figures):
    caption = lambda s: s[s.index('\\caption'):s.index('\\label')]
    assert caption(a).replace('Sites match Fig.~\\ref{fig:trace-motion}.',
                             "Sites match the main paper's representation-motion figure.") == caption(b)
    asset = lambda s: re.search(r'\\includegraphics\[[^]]+\]\{([^}]+)\}',s).group(1)
    assert asset(a) == asset(b)

baseline = json.loads(read(SCRATCH/'baseline.json'))
changed = {p for p,h in baseline.items() if sha(ROOT/p) != h}
expected = {'manuscript/consolidation/draft/'+p for p in (
    'README.md','manuscript.pdf','supplementary.pdf','supplementary.tex',
    'sections/appendix.tex','sections/04-gd-sgd.tex','sections/05c-interventions.tex')}
assert changed == expected, changed
new_files = {p.relative_to(ROOT).as_posix() for p in DRAFT.rglob('*') if p.is_file()} - baseline.keys()
assert new_files == {'manuscript/consolidation/draft/'+p for p in (
    'tables/appendix-gd-audit.tex','sections/s07-intervention-trajectories.tex')}, new_files
preexisting = json.loads(read(SCRATCH/'preexisting.json'))
assert all(sha(ROOT/p) == h for p,h in preexisting.items())
assert all(sha(ROOT/p) == h for p,h in baseline.items() if '/figures/' in p)

pdfs = {}
for stem, pages in [('manuscript',20),('supplementary',31)]:
    build = json.loads(read(HERE/f'{stem}-build.json'))
    assert build['pages'] == pages and build['warnings'] == []
    assert build['sha256'] == sha(DRAFT/f'{stem}.pdf')
    log = read(SCRATCH/f'latex-{stem}/{stem}.log')
    assert not re.search(r'Overfull|Underfull|undefined|multiply defined|LaTeX Warning|Package .* Warning', log)
    with fitz.open(BASE/f'{stem}.pdf') as a, fitz.open(DRAFT/f'{stem}.pdf') as b:
        changed_pages = [i+1 for i,p in enumerate(b) if i>=len(a) or p.get_pixmap().samples != a[i].get_pixmap().samples]
        assert len(b) == pages
        pdfs[stem] = dict(before_pages=len(a), **build, changed_pages=changed_pages)
assert pdfs['manuscript']['changed_pages'] == [6,16,17,18,19,20]
assert pdfs['supplementary']['changed_pages'] == [1,2,3,27,28,29,30,31]
for p in [HERE/'OBSERVATIONS.md',DRAFT/'README.md']:
    for link in re.findall(r'\]\(([^)]+)\)',read(p)):
        if '://' not in link and not link.startswith('#') and link != 'verification.json':
            assert (p.parent/link.split('#')[0]).exists(),(p,link)
result = dict(
    equations_preserved=4, inline_table_blocks_preserved=2,
    projected_audit_rows=9, projected_audit_cells=45,
    transferred_figure_assets_and_measurement_captions=3,
    audit_decisions=len(decisions), audit_replays_to_final_source=True,
    unchanged_existing_draft_files=len(baseline)-len(changed),
    unchanged_preexisting_modified_files=len(preexisting),
    changed_existing_draft_files=sorted(changed), added_draft_files=sorted(new_files),
    all_existing_figure_assets_preserved=True, pdfs=pdfs,
    manual_visual_review='All changed pages inspected in contact sheets; main pages 17-19 and supplementary pages 28 and 30 inspected at full page resolution. No clipping, overlap, or broken section/table transitions observed. Other retained pages render identically to baseline.')
(HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(result,indent=2))
