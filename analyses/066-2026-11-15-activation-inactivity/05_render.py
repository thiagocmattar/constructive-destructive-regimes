"""Render both deliverables and validate their page bounds and embedded fonts."""
from pathlib import Path
import hashlib,json
import fitz
from PIL import Image,ImageDraw
HERE=Path(__file__).resolve().parent;OUT=HERE/'scratch/render';OUT.mkdir(parents=True,exist_ok=True)
record={}
for name,pages in [('activation-summary',7),('activation-matrices',14)]:
    path=HERE/(name+'.pdf');folder=OUT/name;folder.mkdir(exist_ok=True)
    with fitz.open(path) as doc:
        assert len(doc)==pages
        xrefs={f[0] for p in doc for f in p.get_fonts(full=True)}
        assert all(doc.extract_font(x)[3] for x in xrefs)
        for i,p in enumerate(doc):
            for b in p.get_text('dict')['blocks']:
                for line in b.get('lines',[]):
                    for s in line['spans']:
                        x0,y0,x1,y1=s['bbox'];assert x0>=-1 and y0>=-1 and x1<=p.rect.width+1 and y1<=p.rect.height+1,(name,i,s)
            p.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False).save(folder/f'{i+1:02}.png')
        for start in range(0,len(doc),4):
            canvas=Image.new('RGB',(1500,1120),'#ddd');draw=ImageDraw.Draw(canvas)
            for j in range(min(4,len(doc)-start)):
                im=Image.open(folder/f'{start+j+1:02}.png');im.thumbnail((740,520));x=(j%2)*750+(750-im.width)//2;y=(j//2)*560+25
                canvas.paste(im,(x,y));draw.text((x,y-18),f'{name} / page {start+j+1}',fill='black')
            canvas.save(OUT/f'{name}-{start+1:02}.png')
        record[name]=dict(pages=pages,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),font_count=len(xrefs),outside_page_spans=0)
(HERE/'render-verification.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8',newline='\n');print(record)
