"""Verified encrypted CLI download when a task host's SSH bulk route is slow."""
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import time
import uuid

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cloud037',HERE/'05_cloud.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def pull(pod,remote,destination):
    c.connect(pod);token=uuid.uuid4().hex[:12]
    digest=c.ssh(pod,f'sha256sum {remote}\n',timeout=60).split()[0]
    log=f'{c.CONTROL}/send-{token}.log'
    c.ssh(pod,f'setsid runpodctl send {remote} > {log} 2>&1 < /dev/null &\necho $!\n',timeout=30)
    code=None
    for _ in range(8):
        raw=c.ssh(pod,f'cat {log}\n',timeout=30)
        text=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',raw)
        match=re.search(r'code is:\s*([A-Za-z0-9-]+)',text,re.IGNORECASE)
        if match:code=match.group(1);break
        time.sleep(1)
    assert code,'Transfer sender did not provide a code.'
    incoming=c.SCRATCH/('incoming-'+token);incoming.mkdir()
    started=time.time()
    with (c.SCRATCH/('receive-'+token+'.log')).open('wb') as stream:
        receiver=subprocess.Popen(['runpodctl','receive',code],cwd=incoming,stdin=subprocess.PIPE,
            stdout=stream,stderr=subprocess.STDOUT,creationflags=c.NO_WINDOW)
        receiver.communicate(b'y\n',timeout=600)
        assert receiver.returncode==0,'CLI transfer failed; inspect ignored receive log.'
    received=incoming/Path(remote).name
    assert c.file_hash(received)==digest
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    received.replace(destination)
    return dict(sha256=digest,bytes=destination.stat().st_size,seconds=time.time()-started,method='encrypted Runpod CLI croc')
