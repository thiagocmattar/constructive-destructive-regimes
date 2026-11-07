"""Prespecified post-Adam gates, stronger neuron caps, and paired norm controls."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
CALIB='runs/258-2026-11-07-mlp-gate-calibration'
SELECTIVE=[(259,'discard_q90','discard','coordinate_q90'),(260,'cap_q90','cap','coordinate_q90'),
           (261,'neuron_q90','neuron','q90'),(262,'neuron_q75','neuron','q75'),(263,'neuron_q50','neuron','q50')]


def main():
    records=[]
    for number,arm,kind,key in SELECTIVE+[(n+5,'uniform_'+a,'uniform',k) for n,a,_,k in SELECTIVE]:
        run=ROOT/f'runs/{number}-2026-11-07-mlp-gate-{arm.replace("_","-")}'
        assert not run.exists();run.mkdir()
        target=next((r['path'] for r in records if r['arm']==arm.removeprefix('uniform_')),None) if kind=='uniform' else None
        cfg=dict(run_id=str(number),analysis_id='065',hypothesis='H047',arm=arm,intervention_label=f'I{number-244}',
            kind=kind,threshold_key=key,calibration_run=CALIB,target_run=target,
            seed=0,fork=448,end=801,first_lr=.001,later_lr=.001,physical_batch=1000,runtime_limit_seconds=600,
            primary_window=[448,800],event_window=[450,500],tail_window=[769,800],target_loss=.02,target_consecutive_states=10,
            intervention='Post-native-Adam first-layer transform; compute all gradients and advance all moments/clocks normally; other layers stay native at LR.001',
            matched_control='Uniform first-layer direction scaled to the paired selective branch actual norm at the same update; may exceed1; equality bounded by FP32 rounding',
            authorization='User explicitly requests all three selective-update methods, especially neuron representation cap; Spark and paired norm controls authorized in context',
            retention='Dense gradients/five representations447..501; rolling metrics448..800; full recovery/fork/native449/intervention449/qualification512/end801; gate audits448,474,800',
            coverage='Same all5000 examples/order, seed0 tanh3072-200-200-10, mean half squared error, physical1000, FP32 TF32off Adam(.9,.999) eps1e-8 no clipping/decay',
            limits='One fork/seed; baseline always unchanged Adam. Thresholds use only416..447. Norm controls follow paired same-update norms and are experimental controls, not standalone online policies.')
        (run/'config.json').write_text(json.dumps(cfg,indent=2)+'\n',newline='\n')
        (run/'.gitignore').write_text('artifacts/\n',newline='\n')
        (run/'01_worker.py').write_text('from pathlib import Path\nimport sys\nHERE=Path(__file__).resolve().parent\nsys.path.insert(0,str(HERE.parents[1]))\nfrom src.eos_tasks.selective_continuation import main\nif __name__=="__main__":main(HERE)\n',newline='\n')
        (run/'test_intervention.py').write_bytes((ROOT/CALIB/'test_intervention.py').read_bytes())
        s=(ROOT/'runs/254-2026-11-07-mlp-layer-lr-global-half/02_remote.py').read_text().replace('254',str(number))
        s=s.replace('layer_lr_continuation.py','selective_continuation.py')
        s=s.replace("['frozen_first_linear.py','selective_continuation.py']","['frozen_first_linear.py','selective_first_linear.py','selective_continuation.py']")
        s=s.replace("['src/eos_tasks/frozen_first_linear.py','src/eos_tasks/selective_continuation.py']","['src/eos_tasks/frozen_first_linear.py','src/eos_tasks/selective_first_linear.py','src/eos_tasks/selective_continuation.py']")
        insertion='''    cfg=json.loads((HERE/'config.json').read_text())
    calibration=ROOT/cfg['calibration_run']
    p=calibration/'config.json';files.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    for folder,names in [(calibration,['thresholds.json','manifest.json'])]+([(ROOT/cfg['target_run'],['updates.jsonl','manifest.json'])] if cfg['target_run'] else []):
        for name in names:
            p=folder/'artifacts/spark/payload/worker'/name
            files.append(dict(path=(folder.relative_to(ROOT)/'artifacts/attempt-01/worker'/name).as_posix(),local_path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
'''
        anchor="    dump(HERE/'input_manifest.json',dict(run_id="
        assert s.count(anchor)==1;s=s.replace(anchor,insertion+anchor)
        (run/'02_remote.py').write_text(s,newline='\n')
        (run/'README.md').write_text(f'''# Run {number}: {arm.replace('_',' ')}

H047 / ongoing Analysis065, I{number-244}. First-layer policy `{kind}` from the
untreated full state448; first changed update448->449, end801. No inherited
freezing or LR reduction. All native Adam LRs remain0.001. All raw gradients,
moments and clocks advance; only first-layer weights/biases are transformed
after native Adam. Later-layer parameters remain native on this trajectory.
Threshold key `{key}` is fixed by Run258 using only updates416-447. Criterion3
is per-neuron preactivation displacement RMS over all5000 fixed examples.
For uniform controls the paired source is `{target}`; per-step first-layer norm
matches that source within measured FP32 rounding, with factor>1 allowed.

Primary metrics: mean loss448-800 and excursion sizes; event450-500, last32
mean769-800, endpoint800, ten consecutive states below0.02, representation
cosines/RMS, applied/removed increment norms and selected fractions. Controls
and all treatments compare with unchanged Adam; single-seed/fork inference.
Dense states447-501 and gate audits448/474/800 support independent reductions;
all-state summaries and full recovery checkpoints retained. Zero-motion factors
are1 for neuron cap and0 for zero-target uniform; no division epsilon.

Owned Spark, no cloud charges. Prior continuations8-10s; extra FP64 sample-space
products are uncalibrated, allow2min/run plus retrieval.600s worker and630s
detached timeout with30s grace; all10 branches max110min including kill grace.
At most30GB total retained including archives. Progress every64 updates;
STOP/signal/time/nonfinite guards. Source identities and size/SHA256 retrieval
must pass before closeout; preserve unrelated work and all earlier figures.
''',newline='\n')
        records.append(dict(run_id=number,label=f'I{number-244}',arm=arm,kind=kind,threshold_key=key,target_run=target,path=run.relative_to(ROOT).as_posix()))
    (HERE/'selective_plan.json').write_text(json.dumps(dict(status='prespecified',hypothesis='H047',calibration_run=CALIB,
        calibration_window=[416,447],primary_window=[448,800],event_window=[450,500],tail_window=[769,800],
        arms=records,threshold_selection='coordinate pooled absolute increment q90; pooled sample-RMS per neuron q90/q75/q50; NumPy linear quantiles',
        uniform_control='Same-step selective first-layer norm schedule; controls may amplify; rounding discrepancy retained',
        source='User requests all3, especially neuron cap',cloud_cost=0,maximum_retained_gb=30),indent=2)+'\n',newline='\n')
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
