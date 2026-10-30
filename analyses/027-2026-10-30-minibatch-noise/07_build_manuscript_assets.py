"""Copy verified publication assets into the self-contained working draft."""
from pathlib import Path
import hashlib
import json
import shutil

HERE=Path(__file__).parent;ROOT=HERE.resolve().parents[1]
DRAFT=ROOT/'manuscript/draft'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert json.loads((HERE/'verification.json').read_text())['status']=='passed'
    review=json.loads((HERE/'figure_review.json').read_text());assert review['status']=='passed'
    figures=json.loads((HERE/'figure_manifest.json').read_text())
    tables=json.loads((HERE/'table_manifest.json').read_text())
    records=[]
    for name,digest in figures['pdfs'].items():
        source=HERE/'figures'/name;target=DRAFT/'figures/analysis027'/name
        assert sha(source)==digest==review['pdfs'][name]
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        records.append(dict(source=source.relative_to(ROOT).as_posix(),target=target.relative_to(ROOT).as_posix(),sha256=digest))
    for name,digest in tables['tables'].items():
        source=HERE/'tables'/name;target=DRAFT/'tables'/name
        assert sha(source)==digest;shutil.copyfile(source,target)
        records.append(dict(source=source.relative_to(ROOT).as_posix(),target=target.relative_to(ROOT).as_posix(),sha256=digest))
    (HERE/'manuscript_assets.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'verified_copies':len(records)}))


if __name__=='__main__':main()
