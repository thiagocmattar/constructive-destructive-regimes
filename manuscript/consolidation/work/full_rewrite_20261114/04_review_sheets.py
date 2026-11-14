"""Make numbered contact sheets for a page-by-page visual review (scratch only)."""
from pathlib import Path
import fitz
from PIL import Image, ImageOps, ImageDraw
HERE = Path(__file__).resolve().parent
for name in ['manuscript', 'supplementary']:
    with fitz.open(HERE.parents[1] / f'draft/{name}.pdf') as document:
        paths = [HERE / f'scratch/review/{name}/page-{i+1:02}.png' for i in range(len(document))]
    for start in range(0,len(paths),6):
        group = paths[start:start+6]
        sheet = Image.new('RGB',(1320,1800),'#d8d8d8')
        draw = ImageDraw.Draw(sheet)
        for k,path in enumerate(group):
            img = Image.open(path).convert('RGB')
            img = ImageOps.contain(img,(640,560))
            x = 10+(k%2)*660+(640-img.width)//2
            y = 25+(k//2)*600
            sheet.paste(img,(x,y))
            draw.text((10+(k%2)*660,8+(k//2)*600),f'{name}: page {start+k+1}',fill='black')
        sheet.save(HERE / f'scratch/review/{name}-{start+1:02}-{start+len(group):02}.png')
