# 07: High-rate overview with optimizer diagnostics

Requested 7 November 2026 as an additional figure extending Figure 00.
Generator: `../03_optimizer_overview.py`. Independent checks:
`../04_verify_optimizer_overview.py`; provenance, plotted values and results:
`../optimizer_overview_manifest.json`, `../optimizer_overview_values.csv`,
`../optimizer_overview_verification.json`. No new training, cloud cost,
scientific inputs or manuscript changes. Figure 00 and its seven-figure
manifest remain unchanged.

## Sources and reduction

The same high-rate Run215 arms as Analysis055: CIFAR-10/MLP at 0.001,
CIFAR-10/ResNet-32 at 0.003, WikiText-2/Transformer at 0.003. Source JSONL
files are authenticated against Analysis036's frozen spike manifest and
Analysis055's source hashes. Existing event selection is reused unchanged.
The source run fixes initialization, full training objective, ordering,
physical partitions, precision and whole-model parameter scope; one seed
per model. Adam uses beta1=0.9, beta2=0.999, epsilon=1e-8, no clipping or decay.

The first four rows preserve Figure 00's loss, projected change, raw cosine
and initial-relative gradient norm, including their exact values and limits.
The appended rows are:

- Absolute norm of the stored, uncorrected first moment m, from
  `streams.moment.norm`. It is not a norm relative to the initial moment.
- Adjacent first-moment cosine rho^(m), from `streams.moment.cosine`.
- Adjacent adaptive-direction cosine rho^(d), from `streams.adaptive.cosine`.
  Here d = m_hat / (sqrt(v_hat) + epsilon), excluding learning rate.

At parameter state s, the plotted moment/direction is produced using g_s
and used for outgoing update s -> s+1 (recorded update number s+1).
Thus m_s = beta1*m_(s-1) + (1-beta1)*g_s, with m_(-1)=0;
rho^(m)_s = cos(m_(s-1),m_s), and likewise for d. Stored FP32 moments
are read into FP64; norms, cosines and adaptive directions use FP64 arithmetic.
These are same-stream consecutive cosines, not gradient/momentum cross-cosines.
The adaptive direction is distinct from the rounded actual parameter increment.

Coverage per model: losses at 0--1024; gradient and moment norms at 0--1023;
projected change and all adjacent cosines at 1--1023. The terminal state has
no outgoing optimizer update. The first undefined cosine is omitted; no
epsilon is inserted into cosine denominators. All later recorded cosines
are finite and in [-1,1]. No smoothing, subsampling, averaging or new event
detection is applied. Logarithmic axes are used for loss and both norm rows.

## Caption

High-rate full-batch Adam trajectories for CIFAR-10/MLP, CIFAR-10/ResNet-32
and WikiText-2/Transformer. Rows show training loss, projected gradient change,
adjacent raw-gradient cosine, gradient norm relative to its initial value,
absolute first-moment norm, adjacent first-moment cosine and adjacent adaptive
Adam-direction cosine. Filled gray intervals mark rise phases of recovered
excursions above 10%; outlined intervals have unobserved recovery. All curves
are unsmoothed. Moment/direction diagnostics are aligned with the source
gradient state of each outgoing update.

## Observation and limits

The added rows distinguish optimizer geometry from raw-gradient geometry:
the Transformer adaptive-direction cosine stays positive even where raw and
first-moment cosines reverse; the MLP's later adaptive directions also reverse.
This is a descriptive view of three selected high-rate trajectories. It does
not establish stability, generalization, loss decrease from alignment sign,
or a causal role for momentum. Projected change under Adam is not an identified
Hessian curvature or universal threshold. No independent-seed uncertainty
bands are available; consecutive states and excursion intervals are dependent.

## Verification and visual review

All 21,495 plotted values match independent raw-record read-back, including
9,210 added optimizer values. All 12,285 original-row values match the original
plot hashes, with identical coverage. Source hashes and output hashes pass.
The original Figure 00 remains byte-identical to Analysis055's source PDF.
The 7.16 x 9.5-inch, 21-panel vector PDF has embedded fonts, no raster images
and no text outside page bounds. The full-size rendered page was inspected:
labels, panel letters, logarithmic ticks, event shading and legend are legible,
with no clipping or overlap. Scratch render: ignored `tmp/pdfs/analysis058/`.
