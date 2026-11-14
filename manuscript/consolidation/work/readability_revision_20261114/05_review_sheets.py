"""Render the final PDFs and generate contact sheets; no source mutations."""
from pathlib import Path
import json
import hashlib
import fitz
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
DRAFT = HERE.parents[1] / 'draft'
OUT = HERE / 'scratch/review'
OUT.mkdir(parents=True, exist_ok=True)
records = {}
for name in ['manuscript', 'supplementary']:
    path = DRAFT / (name + '.pdf')
    folder = OUT / name
    folder.mkdir(exist_ok=True)
    with fitz.open(path) as doc:
        files = []
        for i, page in enumerate(doc):
            png = folder / f'page-{i+1:02}.png'
            page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False).save(png)
            files.append(png)
        sheets = []
        for start in range(0, len(files), 4):
            sheet = Image.new('RGB', (1100, 1460), '#ddd')
            draw = ImageDraw.Draw(sheet)
            for j, png in enumerate(files[start:start+4]):
                im = Image.open(png)
                im.thumbnail((540, 700))
                x = (j % 2)*550 + (550-im.width)//2
                y = (j // 2)*730 + 24
                sheet.paste(im, (x, y))
                draw.text((x, y-18), f'{name} / page {start+j+1}', fill='black')
            target = OUT / f'{name}-{start+1:02}-{min(start+4,len(files)):02}.png'
            sheet.save(target)
            sheets.append(target.relative_to(HERE).as_posix())
        records[name] = {'pages': len(doc), 'pdf_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'contact_sheets': sheets}
(HERE/'render_manifest.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({k: v['pages'] for k,v in records.items()}))
