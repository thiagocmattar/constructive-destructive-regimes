"""Check the fixed existing ResNet tracing window against the new native control."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
REFERENCE=ROOT/'analyses/063-2026-11-07-architecture-instability-tracing/resnet_390_450_results.json'
STAGES=[('conv1','conv1.call0.output'),('layer1','layer1.4.relu.call1.output'),
 ('layer2','layer2.4.relu.call1.output'),('layer3','layer3.4.relu.call1.output'),
 ('avgpool','avgpool.call0.output'),('output','output.log_probabilities')]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    reference=json.loads(REFERENCE.read_text());path=HERE/'matched/curves.npz';data=np.load(path)
    states=data['resnet__control__state'];loss=data['resnet__control__loss'];lookup={int(s):i for i,s in enumerate(states)}
    rows=[v for v in reference['boundaries'] if v['state'] in lookup]
    errors=[abs(float(loss[lookup[v['state']]])-v['loss']) for v in rows]
    steps=reference['completed_steps'];assert steps==list(range(390,451))
    indices=[lookup[s] for s in steps];comparisons=[]
    for i,(name,old) in enumerate(STAGES):
        for metric in ['rho','change_rms']:
            new=data['resnet__control__stage_'+metric][i,indices]
            before=np.array(reference['curves'][old][metric],dtype=float)
            assert new.shape==before.shape==(61,)
            comparisons.append(dict(stage=name,metric=metric,values=61,max_absolute_error=float(np.max(np.abs(new-before))),
                agrees=bool(np.allclose(new,before,rtol=0,atol=1e-12))))
    exact=all(v==0 for v in errors);agree=all(v['agrees'] for v in comparisons)
    window_losses={arm:data[f'resnet__{arm}__loss'][indices] for arm in ['control','cap_q50']}
    baseline_peak=int(np.argmax(window_losses['control']))
    window_readout={arm:dict(maximum_loss=float(values.max()),
        peak_state=steps[int(np.argmax(values))],
        loss_at_baseline_peak=float(values[baseline_peak])) for arm,values in window_losses.items()}
    window_readout['maximum_loss_reduction_percent']=100*(1-window_readout['cap_q50']['maximum_loss']/window_readout['control']['maximum_loss'])
    window_readout['interpretation']='Secondary descriptive comparison in the previously selected 390-450 tracing window, fixed before this intervention matrix. Whole-horizon outcomes remain primary.'
    report=dict(status='reproduced' if exact and agree else 'different_qualified_native_control',
        loss_states=len(rows),loss_window=[rows[0]['state'],rows[-1]['state']],losses_bitwise_equal=exact,max_loss_error=max(errors),
        stage_window=[390,450],stage_values=sum(v['values'] for v in comparisons),stage_comparisons=comparisons,
        fixed_tracing_window_intervention_readout=window_readout,
        interpretation='This checks measured loss and fixed-probe stage histories in the already selected window; it is not a full-tensor identity claim across the historical runs. No window/setting changes follow from this check.',
        reference=REFERENCE.relative_to(ROOT).as_posix(),reference_sha256=sha(REFERENCE),current_curves_sha256=sha(path),script_sha256=sha(Path(__file__)))
    (HERE/'matched/tracing_overlap.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps({k:report[k] for k in ['status','loss_states','losses_bitwise_equal','max_loss_error','stage_values']}))


if __name__=='__main__':main()
