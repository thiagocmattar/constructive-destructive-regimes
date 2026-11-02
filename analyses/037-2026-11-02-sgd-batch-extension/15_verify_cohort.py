"""Check complete replication and reduce published panels directly from fork summaries."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
A27=ROOT/'analyses/027-2026-10-30-minibatch-noise'
BATCHES={'mlp':[32,128,512,2048,5000],'resnet':[1,5,20,50],
         'transformer':[8,32,128,512,59680]}


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')


def main():
    result=read(HERE/'results.json')
    assert result['status']=='passed' and result['verified_forks']==result['expected_forks']==348
    rows=result['forks']
    expected={(task,seed,state,b,rep) for task in ['resnet','transformer'] for seed in range(3)
        for state in [0,2000] for b in BATCHES[task]
        for rep in range(1 if b==BATCHES[task][-1] else 8)}
    keys=[(r['task'],r['seed'],r['state'],r['batch'],r['rep']) for r in rows]
    assert len(keys)==len(set(keys)) and set(keys)==expected
    spark=read(HERE/'spark-qualification.json');bridge=read(HERE/'spark-bridge-acceptance.json')
    assert spark['status']==bridge['status']=='passed' and bridge['additional_updates']==200
    handoff=read(HERE/'spark-handoff.json');assert handoff['verified'] and handoff['status']=='complete'
    spark_raw=read(HERE/'spark-retrieval.json');assert spark_raw['status']=='passed' and spark_raw['verified']
    assert spark_raw['inventory_sha256']==sha(ROOT/spark_raw['path']/'helper_inventory.json')
    for source,digest in bridge['sources'].items():assert sha(ROOT/source/'summary.json')==digest
    imported={job['name']:job['summary_sha256'] for batch in handoff['handoffs'] for job in batch['accepted']}
    assert len(imported)==32
    workers=[]
    for task in ['resnet','transformer']:
        for seed in range(3):
            receipts=[read(p) for p in HERE.glob('worker-acceptance-*.json')]
            receipt=next(r for r in receipts if (r['task'],r['seed'])==(task,seed)
                and r.get('verified_forks')==(50 if task=='resnet' else 66))
            assert receipt['status']=='passed'
            workers.append(receipt)
    derived=[]
    tail=read(HERE/'tail-helper-handoff.json');assert tail['status']=='passed'
    tail_jobs={r['name']:r['summary_sha256'] for r in tail['accepted']};assert len(tail_jobs)==2
    for row in rows:
        folder=ROOT/row['source'];summary=read(folder/'summary.json')
        assert sha(folder/'summary.json')==row['summary_sha256']
        if (row['task'],row['seed'])==('resnet',2) and row['batch'] in (1,5):
            assert imported[folder.name]==row['summary_sha256']
        if (row['task'],row['seed'],row['state'],row['batch'],row['rep']) in {
                ('resnet',2,2000,20,6),('resnet',2,2000,20,7)}:
            assert tail_jobs[folder.name]==row['summary_sha256']
        assert summary['updates']==200 and summary['valid_cosines']==199
        origin=folder.parent/'reference'/f'state-{row["state"]:04d}.pt'
        assert summary['identity']['source']['sha256']==sha(origin)
        for field in ['mean_rho','R','actual_R','initial_loss','final_loss','loss_change']:
            assert row[field]==summary[field]
        if summary.get('derived_from_reference_prefix'):
            derivation=read(folder/'derivation.json')
            assert derivation['training_updates_saved']==200
            assert derivation['metrics_prefix_sha256']==sha(folder/'metrics.jsonl')
            assert derivation['recovery_sha256']==sha(folder/'reference-prefix-recovery.pt')
            measured=(folder/'metrics.jsonl').read_bytes()
            prefix=b''.join((folder.parent/'reference/metrics.jsonl').read_bytes().splitlines(keepends=True)[:200])
            assert measured==prefix
            derived.append(dict(task=row['task'],seed=row['seed'],state=row['state'],batch=row['batch'],
                source=row['source'],derivation_sha256=sha(folder/'derivation.json')))
    old=read(A27/'results.json');qualification=read(A27/'verification.json')
    assert qualification['status']=='passed' and qualification['results_sha256']==sha(A27/'results.json')
    cells=[]
    for task,batches in BATCHES.items():
        for state in [0,2000]:
            for batch in batches:
                source=[r for r in (old['moving'] if task=='mlp' else rows) if
                    (r['checkpoint'] if task=='mlp' else r['state'])==state and r['batch']==batch
                    and (task=='mlp' or r['task']==task)]
                reps=1 if batch==batches[-1] else 8
                assert len(source)==3*reps
                means={field:[float(np.mean([r[field] for r in source if r['seed']==seed]))
                    for seed in range(3)] for field in ['mean_rho','R','loss_change']}
                assert all(sum(r['seed']==seed for r in source)==reps for seed in range(3))
                if task!='mlp':
                    original=next(c for c in result['cells'] if (c['task'],c['state'],c['batch'])==(task,state,batch))
                    for field,values in means.items():
                        np.testing.assert_array_equal(values,original['seed_means'][field])
                        assert original['mean'][field]==np.mean(values)
                cells.append(dict(task=task,state=state,batch=batch,
                    plotted_batch=batch*(1000 if task=='resnet' else 1),repetitions_per_seed=reps,
                    seed_means=means,mean={k:float(np.mean(v)) for k,v in means.items()},
                    minimum={k:min(v) for k,v in means.items()},maximum={k:max(v) for k,v in means.items()}))
    receipt=dict(status='passed',results_sha256=sha(HERE/'results.json'),
        reused_mlp_results_sha256=sha(A27/'results.json'),verifier_sha256=sha(Path(__file__)),
        new_forks=348,primary_fork_adjacent_pairs=348*199,reused_mlp_forks=198,
        figure_continuation_trajectories=546,reference_updates=12000,
        reference_adjacent_pairs=6*1999,derived_initial_controls=derived,
        executed_continuation_updates=69600-200*len(derived),
        additional_runtime_bridge_updates=200,
        total_executed_training_updates=12000+69600-200*len(derived)+200,
        update_count_scope='Production references, explicit continuations and runtime bridge; excludes mechanics/calibration.',
        spark_qualification_sha256=sha(HERE/'spark-qualification.json'),
        spark_bridge_acceptance_sha256=sha(HERE/'spark-bridge-acceptance.json'),
        tail_helper_forks=2,tail_helper_handoff_sha256=sha(HERE/'tail-helper-handoff.json'),
        spark_retrieval_sha256=sha(HERE/'spark-retrieval.json'),
        cells=len(cells),initializations_per_cell=3,stochastic_repetitions_per_initialization=8,
        undefined_cosines=0,worker_acceptances=workers)
    write(HERE/'cohort_acceptance.json',receipt)
    summary=dict(status='passed',cohort_acceptance_sha256=sha(HERE/'cohort_acceptance.json'),cells=cells,
        maximum_retention_roundoff_difference=max(abs(r['R']-r['actual_R']) for r in rows),
        maximum_nominal_displacement_relative_error=max(r['nominal_displacement_relative_error'] for r in rows),
        interpretation_limits=['Three case-specific objectives/rates; differences do not isolate architecture or dataset.',
            'Range across three initialization blocks is descriptive, not a confidence interval.',
            'ResNet blocks span L40S/4090 and seed2 small-group forks use GB10; initialization and numerical-runtime variation are not separated.',
            'Equal updates imply unequal data exposure; endpoint losses are complete training objectives.',
            'Mean adjacent cosine is not an all-lag interaction, stability test or generalization measurement.'])
    write(HERE/'scientific_summary.json',summary)
    with (HERE/'cells.csv').open('w',newline='',encoding='utf-8') as stream:
        names=['task','state','batch','plotted_batch','repetitions_per_seed','metric','seed0','seed1','seed2','mean','minimum','maximum']
        writer=csv.DictWriter(stream,fieldnames=names);writer.writeheader()
        for cell in cells:
            for metric,values in cell['seed_means'].items():
                writer.writerow({**{k:cell[k] for k in names[:5]},'metric':metric,
                    **{f'seed{i}':v for i,v in enumerate(values)},
                    **{k:cell[k][metric] for k in ['mean','minimum','maximum']}})
    print(json.dumps({k:receipt[k] for k in ['status','new_forks','cells','executed_continuation_updates']}))


if __name__=='__main__':main()
