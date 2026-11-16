"""Verify the editorial revision against the frozen pre-edit draft and evidence."""
from pathlib import Path
import hashlib
import json
import re
import fitz
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DRAFT = HERE.parents[1] / 'draft'
BASELINE = HERE / 'scratch/baseline'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
result = {}

# Only notation changes are permitted inside the existing numbered equations.
def equations(text):
    return re.findall(r'\\begin\{equation\}(.*?)\\end\{equation\}', text, re.S)

def canonical(text):
    for new, old in [(r'Q_t', r'\eta\widehat\kappa_t'),
                     (r's_{t,j}^{\ell}', r'c_{t,j}'),
                     (r's^{\mathrm{Adam},\ell}_{t,j}', r'c^{\mathrm{Adam},\ell}_{t,j}'),
                     (r's_j', r'c_j')]:
        text = text.replace(new, old)
    return re.sub(r'\s+', '', text)

count = 0
for before in (BASELINE / 'sections').glob('*.tex'):
    after = DRAFT / 'sections' / before.name
    old, new = before.read_text(encoding='utf-8'), after.read_text(encoding='utf-8')
    assert list(map(canonical, equations(old))) == list(map(canonical, equations(new))), before.name
    assert re.findall(r'\\label\{([^}]+)\}', old) == re.findall(r'\\label\{([^}]+)\}', new), before.name
    if not before.name.startswith('s0'):
        count += len(equations(new))
assert count == 44, count
result['main_equations_preserved_modulo_notation'] = count
result['existing_section_labels_preserved'] = True

allowed = {'README.md', 'ADVISOR_ACTION_ITEMS_2026-11-16.md', 'manuscript.pdf', 'supplementary.pdf',
    'figures/00-high-lr-overview-cifar10-mlp-resnet-wikitext2-transformer.pdf',
    'figures/01-three-model-temporal-structure.pdf', 'figures/24-main-intervention-tradeoff.pdf',
    'tables/diagnostic-0p1pct.tex', 'tables/matched-main-outcomes.tex'}
allowed.update('sections/' + x for x in ['00-abstract.tex', '01-introduction.tex',
    '02-related-work.tex', '03-regimes.tex', '05-adam.tex', '05a-characterization.tex',
    '05b-tracing.tex', '05c-interventions.tex', '06-conclusion.tex', 's01-proofs.tex',
    's03-adam-audit.tex', 's05-intervention-methods.tex'])
manifest = json.loads((HERE / 'baseline-manifest.json').read_text())
unchanged, changed = [], []
for path, expected in manifest.items():
    current = ROOT / path
    rel = current.relative_to(DRAFT).as_posix()
    if sha(current) == expected:
        unchanged.append(rel)
    else:
        assert rel in allowed, rel
        changed.append(rel)
result['unchanged_draft_files'] = unchanged
result['changed_draft_files'] = changed

preexisting = json.loads((HERE / 'scratch/preexisting.json').read_text())
for path, expected in preexisting.items():
    assert sha(ROOT / path) == expected, path
result['preexisting_unrelated_edits_preserved'] = len(preexisting)

for name in ['diagnostic-0p1pct.tex', 'matched-main-outcomes.tex']:
    old = (BASELINE / 'tables' / name).read_text(encoding='utf-8')
    new = (DRAFT / 'tables' / name).read_text(encoding='utf-8')
    assert canonical(old.replace('[!t]', '[!tp]')) == canonical(new.replace('[!t]', '[!tp]'))
result['existing_main_table_values_unchanged'] = True
fm = json.loads((HERE / 'figure-manifest.json').read_text())
for path, expected in fm['sources'].items():
    assert sha(ROOT / path) == expected, path
result['figure_source_hashes_verified'] = len(fm['sources'])
for name, expected in fm['figures'].items():
    assert sha(DRAFT / 'figures' / name) == expected, name
result['figure_output_hashes_verified'] = len(fm['figures'])
result['figure_array_checks'] = fm['array_checks']
result['loss_overlays'] = fm['loss_overlays']

# Approximate lexical count, using the same method on each source version.
def words(folder):
    text = '\n'.join(p.read_text(encoding='utf-8') for p in (folder / 'sections').glob('*.tex')
                     if not p.name.startswith(('s0', 'appendix')))
    return len(re.findall(r"\b[A-Za-z]+(?:[-'][A-Za-z]+)*\b", text))
result['main_source_lexical_tokens'] = {'before': words(BASELINE), 'after': words(DRAFT),
    'method': 'English alphabetic tokens in main section TeX, including command names; same method both versions'}

for stem in ('manuscript', 'supplementary'):
    build = json.loads((HERE / (stem + '-build.json')).read_text())
    assert not build['warnings'] and sha(DRAFT / (stem + '.pdf')) == build['sha256']
    doc = fitz.open(DRAFT / (stem + '.pdf'))
    assert len(doc) == build['pages']
    text = '\n'.join(p.get_text() for p in doc)
    assert '??' not in text
    assert all(len(p.get_text().strip()) > 30 for p in doc)
    sheets = HERE / 'scratch/contact-sheets'
    sheets.mkdir(exist_ok=True)
    for start in range(0, len(doc), 6):
        sheet = Image.new('RGB', (1224, 3 * 816), 'white')
        draw = ImageDraw.Draw(sheet)
        for offset, page in enumerate(list(doc)[start:start+6]):
            pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))
            im = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
            im.thumbnail((612, 792))
            x, y = offset % 2 * 612, offset // 2 * 816
            sheet.paste(im, (x, y + 20))
            draw.text((x + 12, y + 3), f'{stem} page {start+offset+1}', fill='black')
        sheet.save(sheets / f'{stem}-{start+1:02}.png')
    result[stem] = build

(HERE / 'verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k not in ('unchanged_draft_files', 'changed_draft_files')}, indent=2))
