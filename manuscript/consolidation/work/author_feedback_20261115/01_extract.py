"""Preserve annotation text and locations for the author revision audit."""
from pathlib import Path
import hashlib
import json
import fitz

HERE = Path(__file__).resolve().parent
DRAFT = HERE.parents[1] / 'draft'
source = DRAFT / 'manuscript+feedback.pdf'
if not source.exists():
    source = DRAFT / 'archive/2026-11-15/manuscript+feedback.pdf'
records = []
with fitz.open(source) as document:
    for i, page in enumerate(document):
        for annotation in page.annots() or []:
            records.append(dict(id=len(records)+1, page=i+1,
                type=annotation.type[1], comment=annotation.info.get('content', ''),
                rect=list(annotation.rect), vertices=annotation.vertices,
                anchor=page.get_textbox(annotation.rect)))
    pages = len(document)
assert len(records) == 62
(HERE / 'annotations.json').write_text(json.dumps(dict(
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    source_bytes=source.stat().st_size, pages=pages, annotations=records),
    ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(f'Extracted {len(records)} annotations from {pages} pages')
