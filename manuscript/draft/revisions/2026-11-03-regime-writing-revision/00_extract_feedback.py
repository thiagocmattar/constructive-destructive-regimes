"""Preserve the reviewer's comments and highlighted text with stable identifiers."""
import hashlib
import json
from pathlib import Path
import fitz

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / '2026-11-03-adam-spike-anatomy/revised-manuscript+feedback.pdf'


def main():
    comments = []
    with fitz.open(SOURCE) as document:
        for number, page in enumerate(document, 1):
            words = page.get_text('words')
            for annotation in page.annots() or []:
                regions = [fitz.Quad(annotation.vertices[i:i+4]).rect
                           for i in range(0, len(annotation.vertices or []), 4)]
                selected = [w[4] for w in words if any(
                    region.contains(fitz.Point((w[0]+w[2])/2, (w[1]+w[3])/2))
                    for region in regions)]
                comments.append(dict(id=f'C{len(comments)+1:02}', page=number,
                    type=annotation.type[1], comment=annotation.info['content'],
                    selected_text=' '.join(selected), info=annotation.info,
                    rect=list(annotation.rect)))
    assert len(comments) == 12
    record = dict(source=SOURCE.name, source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  comments=comments)
    (HERE / 'feedback.json').write_text(json.dumps(record, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    lines = ['# Extracted author annotations', '', f'Source SHA-256: `{record["source_sha256"]}`.', '']
    for item in comments:
        lines += [f'## {item["id"]} - original page {item["page"]}', '',
                  item['comment'].replace('\r', '\n'), '', '**Selected passage:** '+item['selected_text'], '']
    (HERE / 'feedback.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Preserved {len(comments)} annotations; source unchanged.')


if __name__ == '__main__':
    main()
