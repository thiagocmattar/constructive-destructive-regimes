"""Sequential paragraph audit: record diagnosis and proposal before each edit."""
from pathlib import Path
import json
import re
import sys

HERE = Path(__file__).resolve().parent
DRAFT = HERE.parents[1] / 'draft'
STATE = HERE / 'scratch/paragraphs.json'
AUDIT = HERE / 'PARAGRAPH_REVIEW.md'

if not STATE.exists():
    units = []
    for section, name in [('I', '01-introduction.tex'), ('II', '02-related-work.tex')]:
        text = (DRAFT / 'sections' / name).read_text(encoding='utf-8')
        pieces = re.split(r'\\begin\{equation\}.*?\\end\{equation\}', text, flags=re.S)
        index = 0
        for piece in pieces:
            piece = re.sub(r'^\\(?:section|subsection|label|begin\{itemize|end\{itemize).*$', '\n', piece, flags=re.M)
            for block in re.split(r'\n\s*\n', piece):
                # Headings and list delimiters stay in place; review prose and bullet bodies.
                lines = [line for line in block.strip().splitlines() if not re.match(
                    r'^\\(?:section|subsection|label|begin\{itemize|end\{itemize)', line)]
                paragraph = '\n'.join(lines).strip()
                if not paragraph:
                    continue
                index += 1
                units.append({'id': f'{section}-{index:02}', 'file': name,
                              'original': paragraph, 'status': 'pending'})
    STATE.write_text(json.dumps(units, indent=2), encoding='utf-8')
    AUDIT.write_text('# Paragraph-by-paragraph review of Sections I and II\n\n'
                    '15 November 2026. Each entry was recorded before its manuscript edit. '
                    'Equations and labels are protected. The sequence follows reading order.\n',
                    encoding='utf-8', newline='\n')

units = json.loads(STATE.read_text(encoding='utf-8'))
pending = next((u for u in units if u['status'] == 'pending'), None)
if len(sys.argv) > 1:
    decision = json.load(sys.stdin) if sys.argv[1] == '-' else json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    if decision.get('revisit'):
        pending = next(u for u in units if u['id'] == decision['id'] and u['status'] != 'pending')
    assert pending and decision['id'] == pending['id']
    assert all(decision.get(k) for k in ('diagnosis', 'solution', 'replacement'))
    target = DRAFT / 'sections' / pending['file']
    text = target.read_text(encoding='utf-8')
    current = pending['replacement'] if decision.get('revisit') else pending['original']
    assert text.count(current) == 1
    entry = ('\n## ' + pending['id'] + (' (coherence review)' if decision.get('revisit') else '') + '\n\n**Read**\n\n```tex\n' + current
             + '\n```\n\n**Diagnosis:** ' + decision['diagnosis']
             + '\n\n**Proposed solution:** ' + decision['solution']
             + '\n\n**Rewrite:**\n\n```tex\n' + decision['replacement'] + '\n```\n')
    with AUDIT.open('a', encoding='utf-8', newline='\n') as f:
        f.write(entry)
        f.flush()
    target.write_text(text.replace(current, decision['replacement'], 1),
                      encoding='utf-8', newline='\n')
    pending.update(decision)
    pending['status'] = 'edited' if pending['original'] != decision['replacement'] else 'retained'
    STATE.write_text(json.dumps(units, indent=2), encoding='utf-8')
    print('Recorded diagnosis and proposal, then applied:', pending['id'])
    pending = next((u for u in units if u['status'] == 'pending'), None)
if pending:
    print(json.dumps(pending, ensure_ascii=False, indent=2))
else:
    print('All paragraphs reviewed:', len(units))
