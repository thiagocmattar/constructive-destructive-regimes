"""Render all exported PDFs with Poppler and record layout/hash checks."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess
import fitz

HERE = Path(__file__).resolve().parent
poppler = shutil.which('pdftoppm'); assert poppler
records = []
for p in sorted((HERE / 'figures').glob('*.pdf')):
    with fitz.open(p) as document:
        assert len(document) == 1
        page = document[0]; bounds = page.rect
        spans = [s for b in page.get_text('dict')['blocks'] if 'lines' in b
                 for line in b['lines'] for s in line['spans']]
        bad = [s['text'] for s in spans if not bounds.contains(fitz.Rect(s['bbox']))]
        assert spans and not bad, (p.name, bad)
        records.append({'file':p.name, 'pages':1, 'text_spans':len(spans),
            'text_within_page':True, 'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
            'render':f'scratch/qa-{p.stem}.png'})
    subprocess.run([poppler, '-png', '-singlefile', '-r', '110', str(p),
        str(HERE / 'scratch' / ('qa-' + p.stem))], check=True, timeout=120)
assert len(records) == 4
result = {'checked_at':datetime.now(timezone.utc).isoformat(), 'count':len(records),
    'all_single_page_text_bounds_ok':True, 'all_visually_inspected':False,
    'renderer':'Poppler pdftoppm, 110 DPI; PyMuPDF text bounds',
    'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'figures':records}
(HERE / 'visual-qa.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'figures':len(records), 'bounds_passed':True, 'visual_inspection_pending':True}))
