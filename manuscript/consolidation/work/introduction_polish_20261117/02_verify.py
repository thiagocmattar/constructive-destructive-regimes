"""Verify the approved opening's scope and prepare page sheets for visual review."""
from pathlib import Path
import hashlib
import json
import re
import fitz
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
DRAFT = HERE.parents[1] / 'draft'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
before = (HERE / 'introduction-before.tex').read_bytes()
after = (DRAFT / 'sections/01-introduction.tex').read_bytes()
marker = b'%%---- stop here'
assert after.split(marker, 1)[1] == before.split(marker, 1)[1]
paragraphs = [p.strip() for p in after.split(marker, 1)[0].decode('utf-8').split('\n\n') if p.strip()][1:]
assert len(paragraphs) == 3 and all('\n' not in p for p in paragraphs)
manifest = json.loads((HERE / 'baseline-manifest.json').read_text())
allowed = {'sections/01-introduction.tex', 'references.bib', 'README.md', 'manuscript.pdf'}
preserved, changed = [], []
for relative, expected in manifest.items():
    if sha(DRAFT / relative) == expected:
        preserved.append(relative)
    else:
        assert relative in allowed, relative
        changed.append(relative)
bib = (DRAFT / 'references.bib').read_text(encoding='utf-8')
assert bib.count('@article{ly2025multifractal,') == 1
assert '10.1038/s41467-025-58532-9' in bib
build = json.loads((HERE / 'manuscript-build.json').read_text())
assert sha(DRAFT / build['output']) == build['sha256'] and not build['warnings']
doc = fitz.open(DRAFT / build['output'])
assert len(doc) == build['pages'] == 20
assert '??' not in '\n'.join(page.get_text() for page in doc)
aux = (HERE / 'scratch/latex-manuscript/manuscript.aux').read_text(encoding='utf-8')
assert list(map(int,re.findall(r'\\newlabel\{eq:[^}]+\}\{\{(\d+)\}',aux))) == list(range(1,45))
for start in range(0, len(doc), 6):
    count = min(6, len(doc)-start)
    sheet = Image.new('RGB', (1224,816*((count+1)//2)), 'white')
    draw = ImageDraw.Draw(sheet)
    for j in range(count):
        pix = doc[start+j].get_pixmap(matrix=fitz.Matrix(1,1))
        im = Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
        x,y = j%2*612,j//2*816
        sheet.paste(im,(x,y+20)); draw.text((x+12,y+3),f'Page {start+j+1}',fill='black')
    sheet.save(HERE/f'scratch/contact-{start+1:02}.png')
result = dict(suffix_preserved_byte_for_byte=True, paragraphs_on_single_source_lines=3,
              changed_existing_files=changed, preserved_existing_files=preserved,
              numbered_equations=44, build=build)
(HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
print(f'Verified stopping-point suffix, single-line paragraphs, 44 equations, {len(preserved)} preserved files, and warning-free PDF.')
