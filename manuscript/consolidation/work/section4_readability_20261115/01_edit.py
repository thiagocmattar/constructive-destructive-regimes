"""Record each paragraph decision before applying its exact replacement."""
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
TARGET = HERE.parents[1] / 'draft/sections/04-gd-sgd.tex'
AUDIT = HERE / 'PARAGRAPH_REVIEW.md'
decisions = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
if not AUDIT.exists():
    AUDIT.write_text('# Conservative Section IV readability edits\n\n'
                     '15 November 2026. Each diagnosis and proposed solution was recorded '
                     'before its source edit. Unlisted prose was retained. Equations, '
                     'notation, labels, and subsection order are protected.\n', encoding='utf-8')
for d in decisions:
    text = TARGET.read_text(encoding='utf-8')
    assert text.count(d['before']) == 1, d['id']
    entry = ('\n## ' + d['id'] + '\n\n**Read**\n\n```tex\n' + d['before']
             + '\n```\n\n**Diagnosis:** ' + d['diagnosis']
             + '\n\n**Proposed solution:** ' + d['solution']
             + '\n\n**Rewrite:**\n\n```tex\n' + d['after'] + '\n```\n')
    with AUDIT.open('a', encoding='utf-8', newline='\n') as f:
        f.write(entry)
        f.flush()
    TARGET.write_text(text.replace(d['before'], d['after'], 1), encoding='utf-8', newline='\n')
    print('Recorded and applied:', d['id'])
