"""Source-bound standardized tables and observation scaffolds; interpretations reviewed."""
import importlib.util
import argparse
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('events',ROOT/'analyses/036-2026-11-02-controlled-adam-diagnostics/02_extract_loss_spikes.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)

def f(x):return 'undefined' if x is None else f'{x:.4f}'

def interpretation(r):
    s={q['name']:q for q in r['stages']};c=r['curves']
    def correlation(n):return f(s[n]['correlation_with_output'])
    def overlap(n):return f(s[n]['temporal_top10_overlap'])
    def at(n,t):return f(c[n]['rho'][r['states'].index(t)])
    if r['case']=='resnet':
        first=s['conv1.call0.output'];out=s['output.log_probabilities']
        return f'''### What the architectural trace adds

**The reversal is already present at the first convolution, but the output-like
temporal structure becomes clearer deeper in the residual network.** The
alignment-series correlation with output is {correlation('conv1.call0.output')}
at the first convolution, {correlation('layer1.4.relu.call1.output')} at the end
of residual stage 1, {correlation('layer2.4.relu.call1.output')} at the end of
stage 2, and {correlation('layer3.4.relu.call1.output')} at the end of stage 3.
The top-10 temporal-subspace overlap rises from
{overlap('conv1.call0.output')} at the first convolution to
{overlap('layer3.4.relu.call1.output')} at the final residual stage. Thus an
early reversal and an output reversal coexist, but they are not one identical
trajectory transmitted unchanged through every layer.

At update 287, first-convolution cosine is {at('conv1.call0.output',287)},
whereas final log-probability cosine is {at('output.log_probabilities',287)}.
At 302 they are {at('conv1.call0.output',302)} and
{at('output.log_probabilities',302)}, respectively. The full loss rises from
0.3473 at state 286 to 0.4582 at 293 (1.319 times the baseline) and recovers by
302. The outgoing displacement with label 292 enters the peak state 293.

The first-convolution activation coordinate RMS remains between
{first['activation_rms_min']:.6f} and {first['activation_rms_max']:.6f}, while its
step-displacement RMS spans {first['change_rms_min']:.6f} to
{first['change_rms_max']:.6f}. A representation can remain similar in overall
size while its small successive movements grow and reverse. This is not evidence
of the whole representation exploding in norm. These global RMS values do not
exclude rare sample/channel excursions.

There is no nearly one-dimensional full-window output explanation: the leading
output mode contains {100*out['top1']:.2f}% of displacement energy, the top ten
contain {100*out['top10']:.2f}%, and {out['rank95']} modes are needed for 95%.
The first mode changes sign {out['leading_sign_flips']} times across 30 adjacent
coefficient pairs. The stem BatchNorm is more spectrally concentrated (top ten
{100*s['bn1.call0.output']['top10']:.2f}%), but its overlap with the output's top
ten temporal modes is only {overlap('bn1.call0.output')}. Concentration and
transmission of the same temporal patterns are different questions.

Correct-label log-probability changes reverse in
{s['output.correct_label_log_probability']['negative']}/31 updates and their
alignment-series correlation with all-output changes is
{correlation('output.correct_label_log_probability')}. The observed functional
oscillation therefore also appears in the per-example loss readout, rather than
being supported only by unrelated class coordinates. This does not measure
per-example gradient coherence.

**Next discriminating diagnostic:** separate incoming representation changes,
local parameter changes and cross terms at residual-stage transitions. Test
the first convolution with a matched intervention before naming it the cause.
Neither a BatchNorm norm ratio nor the first negative cosine is a causal test.
'''
    emb=s['encoder.call0.output'];out=s['output.log_probabilities']
    rows=[]
    for t in [556,558,559,560,562,563]:
        rows.append('| '+str(t)+' | '+' | '.join(at(n,t) for n in [
            'encoder.call0.output','transformer_encoder.layers.0.self_attn.call0.output',
            'transformer_encoder.layers.0.norm2.call0.output',
            'transformer_encoder.layers.1.norm2.call0.output','output.log_probabilities'])+' |')
    return f'''### What the architectural trace adds

**Strong output reversal is present while the embedding still moves mostly
forward.** At update 558, embedding cosine is {at('encoder.call0.output',558)},
first attention-output cosine is
{at('transformer_encoder.layers.0.self_attn.call0.output',558)}, and final
log-probability cosine is {at('output.log_probabilities',558)}. The embedding
reverses on only {emb['negative']}/49 updates; final output reverses on
{out['negative']}/49. This rules out the descriptive picture of an identical
embedding reversal merely passed unchanged to every later representation.
It does not prove that attention is an independent causal source: its parameters
and its inputs both evolve, and its gradients depend on the downstream network.

| Outgoing update | Embedding | First attention output | First block output | Second block output | Log-probabilities |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The top-10 temporal overlap with output is {overlap('encoder.call0.output')} at
the embedding, {overlap('transformer_encoder.layers.0.self_attn.call0.output')}
at the first attention output, and
{overlap('transformer_encoder.layers.1.norm2.call0.output')} at the second
block output. The first attention boundary is the first measured location where
the output's strong reversal is already visible while the embedding is still
forward-moving. Internal query/key/value projections and attention weights
were not captured, so this trace cannot localize the transition inside attention.

The embedding's leading uncentered mode changes sign only
{emb['leading_sign_flips']} times and contains {100*emb['top1']:.2f}% of energy;
the final output's leading mode changes sign {out['leading_sign_flips']} times
and contains {100*out['top1']:.2f}%. The top ten output modes contain
{100*out['top10']:.2f}% and {out['rank95']} modes cover 95%. Embedding motion
includes a substantial forward component; a leading-mode flip count alone
would miss reversals carried by its other modes. Small late-window coefficient
flips should not be equated with the large spike-associated oscillation.

Embedding activation RMS stays within {emb['activation_rms_min']:.6f} to
{emb['activation_rms_max']:.6f}. First-block norm1 output RMS stays within
{s['transformer_encoder.layers.0.norm1.call0.output']['activation_rms_min']:.6f}
to {s['transformer_encoder.layers.0.norm1.call0.output']['activation_rms_max']:.6f},
despite displacement RMS reaching
{s['transformer_encoder.layers.0.norm1.call0.output']['change_rms_max']:.6f}.
Normalization can constrain representation size while the representation still
rotates or reverses between updates. These global values do not test whether a
few token-wise normalization variances become small; that requires denominator
and Jacobian measurements. This model uses LayerNorm, not RMSNorm.

The full loss rises from 0.8431 at state 557 to 1.6942 at 562 (2.010 times the
baseline) and recovers at 596. Correct-label log-probability changes reverse in
{s['output.correct_label_log_probability']['negative']}/49 updates and correlate
{correlation('output.correct_label_log_probability')} with all-vocabulary
alignment. The result is therefore visible in the token-level loss readout,
not just the many vocabulary coordinates that have tiny predicted probability.

The position-encoding displacement Gram equals 200 times the embedding Gram
to relative error {r['verification']['embedding_positional_gram_relative_error']:.3g},
as required by the fixed positional encoding and embedding multiplier sqrt(200).
That boundary rescales motion without introducing its own temporal pattern.

**Next discriminating diagnostic:** decompose the first attention computation
into changing embeddings, local query/key/value/output parameters and attention
mixing. Measure per-token normalization denominators before assigning an
amplification mechanism. These are candidate follow-ups, not executed causal tests.
'''

def main(case_filter=None):
    for index,case in enumerate(['resnet','transformer'],1):
        if case_filter and case!=case_filter:continue
        r=json.loads((HERE/(case+'_results.json')).read_text());states=r['states'];records={s['name']:s for s in r['stages']};boundary=r['boundaries'];lookup={q['state']:q for q in boundary}
        events=[]
        for event in e.excursions([q['loss'] for q in boundary]):
            mapped={k:(v+boundary[0]['state'] if v is not None and k.endswith('_state') else v) for k,v in event.items()}
            if states[0]<=mapped['peak_state']<=states[-1]:
                mapped['ratio']=lookup[mapped['peak_state']]['loss']/lookup[mapped['before_state']]['loss'];events.append(mapped)
        selected=['conv1.call0.output','bn1.call0.output','relu.call0.output','layer1.4.relu.call1.output','layer2.4.relu.call1.output','layer3.4.relu.call1.output','avgpool.call0.output','fc.call0.output','output.log_probabilities','raw_gradient'] if case=='resnet' else ['encoder.call0.output','pos_encoder.call0.output','transformer_encoder.layers.0.self_attn.call0.output','transformer_encoder.layers.0.norm1.call0.output','transformer_encoder.layers.0.norm2.call0.output','transformer_encoder.layers.1.self_attn.call0.output','transformer_encoder.layers.1.norm2.call0.output','decoder.call0.output','output.log_probabilities','raw_gradient']
        selected.insert(-1,'output.correct_label_log_probability')
        lines=[]
        for name in selected:
            s=records[name];lines.append(f"| `{name}` | {s['shape']} | {s['negative']}/{s['defined']} | {100*s['top1']:.2f}% | {100*s['top10']:.2f}% | {f(s['correlation_with_output'])} | {f(s['temporal_top10_overlap'])} |")
        history=r['historical'];different=sum(not(q['loss_exact'] and q['norm_exact']) for q in history['rows']);maxerr=max(q['loss_relative_error'] for q in history['rows'])
        first=records[selected[0]];output=records['output.log_probabilities'];loss=np.array([lookup[t]['loss'] for t in states]);ploss=np.array([lookup[t]['probe_loss'] for t in states])
        architecture=('Input 1000 x 3 x 32 x 32; stem Conv (16)-BatchNorm-ReLU; three stages with five residual blocks each, widths 16/32/64 and spatial sizes 32/16/8; average pooling to 64; linear output 10. Each block has Conv-BN-ReLU-Conv-BN, a skip path, addition and ReLU. Downsampling skips use average pooling, BN and zero channel padding. BN uses its original fixed 1000-image group; running statistics are disabled. Inputs use fixed full 50k per-channel normalization.' if case=='resnet' else 'Input 64 x 35 fixed token IDs; learned embedding width 200, scaling by sqrt(200), fixed positional encoding; two post-normalization Transformer blocks with two attention heads, feed-forward width 200 and ReLU; linear decoder to 33278 vocabulary coordinates; log-softmax. Dropout is 0. All 2240 probe targets are supervised. Captured attention boundaries exclude internal query/key/value and attention-weight matrices; this is not a complete attention-internals decomposition.')
        text=f'''# {'ResNet / CIFAR-10' if case=='resnet' else 'Transformer / WikiText-2'}: standardized architecture oscillation trace

## TLDR

The first learned representation has negative consecutive motion in **{first['negative']}/{first['defined']}** updates; final log-probability motion is negative in **{output['negative']}/{output['defined']}**. Their alignment-series correlation is **{f(first['correlation_with_output'])}**. This localizes observed motion on the fixed probe and does not identify a unique causal origin.

Full-objective loss over the diagnostic window ranges from **{loss.min():.6g} to {loss.max():.6g}**; fixed-probe loss ranges from **{ploss.min():.6g} to {ploss.max():.6g}**. Their temporal correlation is **{f(r['loss_summary']['probe_full_correlation'])}**; this checks the probe's loss timing, not population-wide activation representativeness. Loss-event details below distinguish an actual excursion from a merely negative cosine. All architecture/optimizer interventions remain TODO.

{interpretation(r)}

## Coverage and trajectory identity

- Run 243, unchanged Adam from the saved Run 223 seed-0 checkpoint; LR 0.003, betas (0.9, 0.999), eps 1e-8, no clipping or decay. FP32 native, FP64 diagnostics, TF32 off.
- Actual execution: {r['manifest']['gpu']}, PyTorch {r['manifest']['torch']}, CUDA {r['manifest']['cuda']}.
- Full objective: {r['manifest']['coverage']['training_units']:,} {'images' if case=='resnet' else 'supervised tokens in 59,680 sequences'} per update; mean cross-entropy over all loss units.
- Fixed probe: {r['manifest']['coverage']['probe_units']} {'images in one original BatchNorm group' if case=='resnet' else 'sequences, 2240 supervised token positions'}. No coordinates within that probe are sampled or averaged away. This is not a census of all training activations.
- Displayed outgoing-update indices {states[0]}--{states[-1]}; {len(states)} SVD time rows, with one extra preceding/following captured state for adjacent motion and endpoint differences.
- Historical exact agreement: **{history['exact']}**. {different}/{len(history['rows'])} state pairs differ in loss or raw-gradient norm; maximum relative loss discrepancy {maxerr:.6g} ({100*maxerr:.4f}%). Same saved starting tensors and policy do not establish bitwise historical continuation across GPU models/runtime kernels.

## Architecture and measurement points

{architecture}

The standard trace contains {len(r['stages'])-1} representation streams plus the full raw-gradient stream. Pre-ReLU tensors are cloned before in-place mutation. Reused module calls have separate call identifiers. Residual-sum inputs and the separate residual branch outputs are distinguished; no local causal contribution is assigned from an output norm ratio alone. Captured fixed inputs are exactly equal across the measurement states; their displacement is zero and motion cosine undefined.

## Standardized landmark table

Rows of a captured representation index fixed images or supervised token positions; trailing dimensions are feature channels/spatial positions or vocabulary coordinates. The SVD matrix instead has optimizer updates as rows and all flattened probe coordinates as columns.

| Stage | One-state captured shape | Negative/defined cosine | Top1 energy | Top10 energy | Alignment r with output | Top10 temporal overlap |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

The raw-gradient row decomposes $g_t$, not gradient differences; its RMS column in the accompanying CSV is gradient RMS. All other rows decompose outgoing activation changes. The reference output is log-probabilities; raw logits remain a separate captured stage. A common class-logit shift can affect raw-logit geometry while canceling from probabilities.

## Loss excursions in this window

The unchanged Analysis 036 recovery detector is applied to the new full-objective continuation, not copied from the historical worker. Events whose peaks lie inside the diagnostic window are:

```json
{json.dumps(events,indent=2)}
```

Indices are completed optimizer updates. The increment entering a peak $p$ is labeled $p-1$. Right-censored events are not claimed recovered. Event timing was selected from the original control before inspecting these representation diagnostics, so this remains a retrospectively selected window rather than an independent discovery sample.

## Algorithm, figures and limits

Apply [the common method](../METHOD.md): fixed probes; unchanged trajectory; outgoing differences; exact uncentered FP64 temporal Grams; fixed-window SVD; signed coefficients; backward architectural comparison of cosine, magnitude and temporal subspaces. No per-step changing SVD basis, sample averaging, random sketch or gradient differencing is used. Temporal-coefficient signs are oriented once by the largest absolute temporal coefficient. Correlations use dependent time points and do not supply independent causal evidence.

**Figure {index}, page 1:** full training/probe loss, raw-gradient/output alignment, every captured representation's consecutive-change cosine, displacement coordinate RMS divided by that stage's own window maximum, and top1/top10 SVD energy with top10 output temporal overlap. Row-wise scaling in the RMS heatmap shows timing, not absolute cross-layer magnitude. Raw zero-input motion is stated separately; no zero-cosine curve is invented.

**Figure {index}, page 2:** architecture landmarks with consecutive cosine, absolute displacement coordinate RMS and the leading fixed SVD coefficient divided by the square root of the number of captured coordinates. Independent layer bases do not identify the same neuron directions; coefficient signs across different layers are arbitrary after their separate fixed orientation.

Strong resemblance across layers can reflect coupled updates driven by the same downstream loss. Even an earliest-layer reversal is not a causal initiating source. Probe selection, one seed, short selected windows, near-degenerate SVD modes and numerical hardware differences limit extrapolation. Normalization/attention mixing cannot be explained solely by elementwise secant arguments. Follow-up causal tests and deeper attention/normalization accounting are separate from this first-pass trace.

## Sources and verification

- [Numerical results and source hashes](../{case}_results.json); [every-stage table](../{case}_stages.csv).
- [Standardized PDF](../figures/{index:02d}-{case}-instability-trace.pdf).
- Scripts 01/02/03 in this analysis generate reductions, figures and this observation; Run 243 owns capture and source identities.
- Observer verifies plain-forward parity, unchanged gradients/RNG/buffers and fixed probe inputs. Independent FP64 contractions check all Gram diagonal norms and adjacent products. Local reduction verifies PSD, trace, full Gram reconstruction and full-gradient boundary norms/cosines. Model checkpoints, input/target identities, exact Grams and raw-cache hashes are retained with transfer verification. Dense activation caches are regenerable from these snapshots and the pinned computation contract.

Rendering and final artifact integrity checks are recorded in the figure and execution verification records. No manuscript edits or causal interventions are part of this result.
'''
        (HERE/'observations'/f'{index:03d}-{case}-trace.md').write_text(text,newline='\n')
        print(json.dumps(dict(case=case,events=events,first=first,output=output,historical_max_loss_relative_error=maxerr)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['resnet','transformer']);a=p.parse_args();main(a.case)
