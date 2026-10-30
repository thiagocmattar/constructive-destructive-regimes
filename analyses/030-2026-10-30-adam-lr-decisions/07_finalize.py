"""Local follow-up after verified retrieval; never trains or changes cloud resources."""
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

def main():
    while True:
        if (HERE/'closeout.json').exists():break
        lease=HERE/'scratch/lease.json'
        if lease.exists():
            deadline=json.loads(lease.read_text())['hard_deadline_epoch']
            if time.time()>deadline+300:
                (HERE/'scratch/finalizer-status.json').write_text(json.dumps(dict(status='waiting_for_incomplete_retrieval',at_epoch=time.time())),encoding='utf-8')
                return
        time.sleep(30)
    subprocess.run([sys.executable,str(HERE/'06_analyze.py')],cwd=ROOT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    import fitz
    scratch=HERE/'scratch/renders';scratch.mkdir(exist_ok=True)
    checks=[]
    for path in sorted((HERE/'figures').glob('*.pdf')):
        doc=fitz.open(path);assert len(doc)==1
        page=doc[0];outside=[]
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                for span in line['spans']:
                    if not page.rect.contains(fitz.Rect(span['bbox'])):outside.append(span['text'])
        assert not outside,(path,outside)
        page.get_pixmap(matrix=fitz.Matrix(2.2,2.2)).save(scratch/(path.stem+'.png'))
        checks.append(dict(figure=path.name,pages=1,text_outside_page=outside,visual_review='pending'))
    (HERE/'figure_checks.json').write_text(json.dumps(dict(checks=checks,visual_review='pending'),indent=2)+'\n',encoding='utf-8')
    (HERE/'scratch/finalizer-status.json').write_text(json.dumps(dict(status='analysis_and_previews_ready_for_review',at_epoch=time.time())),encoding='utf-8')

if __name__=='__main__':main()
