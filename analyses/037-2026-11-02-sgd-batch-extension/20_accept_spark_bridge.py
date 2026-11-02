"""Independently quantify the additional 200-update GB10/L40S initial replay."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUN=ROOT/'runs/214-2026-11-02-sgd-batch-extension'


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    bridge=RUN/'artifacts/spark-assist/resnet-seed2/initial-full-runtime-bridge'
    cloud=RUN/'artifacts/attempt-04/resnet-seed2/state-0-b50-rep00'
    for folder in [bridge,cloud]:
        inventory=read(folder/'inventory.json')
        for name,entry in inventory.items():
            path=folder/name;assert path.resolve().is_relative_to(folder.resolve())
            assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256']
    summaries=[read(folder/'summary.json') for folder in [bridge,cloud]]
    assert all(s['status']=='complete' and s['updates']==200 and s['valid_cosines']==199 for s in summaries)
    assert summaries[0]['identity']['source']['sha256']==summaries[1]['identity']['source']['sha256']
    for folder,summary in zip([bridge,cloud],summaries):
        rows=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
        assert [r['update'] for r in rows]==list(range(1,201)) and rows[0]['cosine'] is None
        np.testing.assert_allclose(np.mean([r['cosine'] for r in rows[1:]]),summary['mean_rho'],rtol=1e-12)
        np.testing.assert_allclose(rows[-1]['B']/sum(r['norm'] for r in rows),summary['R'],rtol=1e-12)
    models=[torch.load(folder/'final.pt',map_location='cpu',weights_only=False)['model'] for folder in [bridge,cloud]]
    difference=scale=0.
    for name,value in models[0].items():
        difference+=float((value.double()-models[1][name].double()).square().sum())
        scale+=float(models[1][name].double().square().sum())
    relative=(difference/scale)**.5
    errors={key:abs(summaries[0][key]-summaries[1][key]) for key in ['mean_rho','R','actual_R','initial_loss','final_loss']}
    assert relative<1e-4 and max(errors.values())<1e-3
    receipt=dict(status='passed',additional_updates=200,primary_cohort_forks=0,
        full_replay_parameter_relative_l2=relative,absolute_metric_differences=errors,
        parameter_relative_l2_limit=1e-4,metric_absolute_limit=1e-3,
        sources={folder.relative_to(ROOT).as_posix():sha(folder/'summary.json') for folder in [bridge,cloud]},
        verifier_sha256=sha(Path(__file__)),
        limit='This quantifies the initial200-update full-GD bridge, not long chaotic replay or every stochastic path.')
    (HERE/'spark-bridge-acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
