"""Post hoc loss-relevant output check from the complete captured log probabilities."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--wait',action='store_true');a=p.parse_args()
    for case in ['resnet','transformer']:
        folder=a.root/case;deadline=time.monotonic()+7200
        while not (folder/'stages.json').exists():
            assert a.wait and time.monotonic()<deadline
            time.sleep(5)
        meta=json.loads((folder/'stages.json').read_text())['output.log_probabilities']
        full=np.load(folder/'raw'/meta['path'],mmap_mode='r');labels=np.load(folder/'raw/fixed_targets.npy');labels=labels[labels!=-100].reshape(-1)
        assert full.shape[1]==len(labels)
        target=np.array(full[:,np.arange(len(labels)),labels],dtype=np.float64)
        delta=np.diff(target,axis=0);gram=delta@delta.T;levels=np.sum(target*target,axis=1)
        rows=[json.loads(line) for line in (folder/'boundaries.jsonl').read_text().splitlines() if line]
        captured=[r for r in rows if r['probe_loss'] is not None]
        np.testing.assert_allclose(-target.mean(axis=1),[r['probe_loss'] for r in captured],rtol=2e-6,atol=2e-6)
        np.testing.assert_allclose(np.diag(gram),np.einsum('ij,ij->i',delta,delta),rtol=1e-12,atol=1e-9)
        np.savez(folder/'target_readout.npz',gram=gram,levels=levels,values=target)
        report=dict(status='verified',shape=[len(labels)],states=[r['state'] for r in captured],
            probe_loss_max_error=float(np.max(abs(-target.mean(axis=1)-[r['probe_loss'] for r in captured]))),
            definition='Correct-label log probability for every retained image/token, no averaging in cosine/SVD',
            source_log_probability_file=meta['path'],generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (folder/'target_readout_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(case=case,**report)),flush=True)

if __name__=='__main__':main()
