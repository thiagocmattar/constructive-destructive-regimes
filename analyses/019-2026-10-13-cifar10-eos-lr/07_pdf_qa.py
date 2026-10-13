"""Render every publication page and check PDF text, bounds and identity."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import fitz

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--accept-reviewed', action='store_true', help='Record the agent visual review after inspecting current renders.')
    parser.add_argument('--gd-only', action='store_true')
    parser.add_argument('--simple-gd', action='store_true', help='Review the current six-panel GD figure.')
    args = parser.parse_args()
    assert not (args.gd_only and args.simple_gd)
    manifest_path = 'gd_figure_manifest.json' if args.simple_gd else ('scratch/gd_figure_manifest.json' if args.gd_only else 'figure_manifest.json')
    manifest = json.loads((HERE / manifest_path).read_text())
    scratch = HERE / 'scratch/pdf-qa'
    scratch.mkdir(parents=True, exist_ok=True)
    checks = {}
    for name, item in manifest['figures'].items():
        path = HERE / 'figures' / name
        if not path.exists():
            path = HERE / 'figures/archive' / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        with fitz.open(path) as pdf:
            assert len(pdf) == 1
            page = pdf[0]
            text = page.get_text()
            assert 'CIFAR-10' in text and '20k' in text
            assert 'Unsmoothed' in text or 'unsmoothed' in text
            outside = []
            for block in page.get_text('dict')['blocks']:
                for line in block.get('lines', []):
                    for span in line['spans']:
                        box = fitz.Rect(span['bbox'])
                        if not fitz.Rect(-1, -1, page.rect.width+1, page.rect.height+1).contains(box):
                            outside.append(span['text'])
            assert not outside, outside
            render = scratch / name.replace('.pdf', '.png')
            renderer = 'PyMuPDF'
            poppler = shutil.which('pdftoppm')
            if not poppler:
                windows_poppler = Path.home() / 'AppData/Local/Programs/MiKTeX/miktex/bin/x64/pdftoppm.exe'
                if windows_poppler.is_file():
                    poppler = str(windows_poppler)
            if poppler:
                try:
                    subprocess.run([poppler, '-r', '100', '-png', '-singlefile', str(path), str(render.with_suffix(''))],
                                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
                    renderer = 'Poppler'
                except (subprocess.SubprocessError, OSError):
                    page.get_pixmap(matrix=fitz.Matrix(100/72, 100/72), alpha=False).save(render)
            else:
                page.get_pixmap(matrix=fitz.Matrix(100/72, 100/72), alpha=False).save(render)
            checks[name] = {'status': 'passed' if args.accept_reviewed else 'awaiting_visual_review',
                            'pages': 1, 'page_points': [page.rect.width, page.rect.height],
                            'text_spans_outside_page': outside, 'pdf_sha256': item['sha256'],
                            'renderer': renderer,
                            'render_path': render.relative_to(HERE).as_posix(),
                            'render_sha256': hashlib.sha256(render.read_bytes()).hexdigest()}
    result = {'status': 'passed' if args.accept_reviewed else 'awaiting_visual_review', 'figures': checks,
              'review': 'Agent inspected all current page renders for labels, legends, overlaps, clipping and readability.' if args.accept_reviewed else None}
    destination = 'gd_visual_verification.json' if args.simple_gd else ('scratch/gd_visual_verification.json' if args.gd_only else 'visual_verification.json')
    (HERE / destination).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'figures': len(checks)}))


if __name__ == '__main__':
    main()
