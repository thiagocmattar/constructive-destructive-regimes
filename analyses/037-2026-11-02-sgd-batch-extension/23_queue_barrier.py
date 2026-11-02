"""Wait between forks for independently verified handoffs; never pause a measurement."""
import argparse
import json
import os
from pathlib import Path
import signal
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--deadline-epoch',type=float,required=True)
    a=p.parse_args();stopped=[False];paused=None
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    def ready():
        return all((a.worker/f'state-2000-b{batch}-rep{rep:02d}/summary.json').exists()
            for batch in [5,1] for rep in range(8))
    record=dict(status='waiting',worker=str(a.worker),created_epoch=time.time())
    def save():a.receipt.write_text(json.dumps(record,indent=2)+'\n')
    save()
    try:
        while time.time()<a.deadline_epoch and not stopped[0]:
            if ready():record['status']='ready_without_pause';break
            progress=json.loads((a.worker/'worker_progress.json').read_text())
            if progress['job']['name']=='state-2000-b20-rep05':
                candidates=[]
                for path in Path('/proc').glob('[0-9]*'):
                    try:command=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode()
                    except OSError:continue
                    if '/02_worker.py' in command and str(a.worker) in command:candidates.append(int(path.name))
                assert len(candidates)==1,candidates
                paused=candidates[0];os.kill(paused,signal.SIGSTOP)
                record.update(status='queue_paused',queue_pid=paused,paused_epoch=time.time(),
                    note='Only the queue process is paused. Its active measurement child runs to completion.')
                save();until=min(a.deadline_epoch,time.time()+300)
                while not ready() and time.time()<until and not stopped[0]:time.sleep(1)
                record['status']='ready' if ready() else 'released_at_bound';break
            if progress['job_index']>32:record['status']='queue_already_past_barrier';break
            time.sleep(2)
    finally:
        if paused:
            try:os.kill(paused,signal.SIGCONT)
            except ProcessLookupError:pass
            record['resumed_epoch']=time.time()
        record['finished_epoch']=time.time();save()


if __name__=='__main__':main()
