"""Reduce locally retrieved Run215 histories after detached cloud closeout."""
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
NO_WINDOW=getattr(subprocess,'CREATE_NO_WINDOW',0)


def main():
    while not (HERE/'scratch/production-closed.json').exists():
        time.sleep(30)
    result=subprocess.run([sys.executable,str(HERE/'05_verify_results.py'),'--require-complete'],
        capture_output=True,text=True,creationflags=NO_WINDOW)
    record=dict(verification_exit=result.returncode,stdout=result.stdout[-4000:],stderr=result.stderr[-4000:],
        status='all_nine_arms_verified' if result.returncode==0 else 'requires_review')
    (HERE/'finalization.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
