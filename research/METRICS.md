# Measurement design

On 2026-10-03 the user specified adjacent-gradient cosine, the cumulative sum
of gradient norms, the norm of the cumulative gradient sum, and their ratio,
recorded every step. Their proposed mathematical and capture contracts are in
[BASELINE_PLAN.md](BASELINE_PLAN.md). Optimizer and training horizon
are resolved in [Run001](../runs/001-2026-10-03-pythia14m-adamw-gradient-baseline/README.md):
AdamW, full pass, capture after accumulation/unscaling and before clipping.
The user explicitly preapproved cloud launches; Run001 is completed and verified,
with per-update measurements, figures, and recovery states retained locally.
Other scientific measurements remain undecided.

Run002 additionally measures adjacent cosine, A/B/R, and individual norms for
the bias-corrected Adam adaptive direction alongside the same raw-gradient
metrics. The user chose to exclude LR and weight decay from that diagnostic
vector. Its precise capture, FP64 evaluation from actual FP32 moments, clipping,
indexing, denominator behavior and norm comparison are in the
[Run002 contract](../runs/002-2026-10-03-pythia14m-adamw-adaptive-geometry/README.md).

Runs 021–023 add the same norm, A, B, B/A and adjacent cosine for actual stored
FP32 Adam first moment m after the accepted optimizer update, using clipped
gradients. This stream excludes bias correction and second-moment division.
FP64 prefix/reduction arithmetic, m/g and d/m norm ratios, undefined cases and
recovery requirements are in the
[Analysis008 contract](../analyses/008-2026-10-08-no-decay-momentum/README.md).
The initial zero-LR update still advances moments and diagnostic clocks.

## Contract for each selected metric

Record its question, symbol and exact formula, units, vector/parameter scope,
sample unit, optimizer boundary, pre/post clipping/decay timing, loss reduction,
precision, cadence, coverage, aggregation, undefined cases, storage, and cost.
Distinguish training-time capture from checkpoint-reconstructible diagnostics.
State whether it is exact, approximate, sampled, or a descriptive association.

## Topics to resolve

- Which loss/performance and stability outcomes should the diagnostic explain?
- Raw-gradient geometry, actual-update geometry, or both? Which temporal lags
  and model/layer scopes are necessary?
- Which norm/energy/decomposition terms are needed, and which are computable
  without per-sample gradients? A batch gradient cannot identify `S1` and `S2`
  separately.
- How will approximation residuals, zero norms, numerical error, and diagnostic
  overhead be reported?
- Which data/probe coverage, seeds, uncertainty estimates, and time weighting
  support the intended claim?
- Which checkpoints, optimizer states, gradient statistics, or histories must be
  retained so later analysis remains possible?

No claim that positive alignment improves loss, or that norm control establishes
stability, is encoded as a measurement rule.

## Operational logging convention

Future runs should record status, completed updates, input and supervised tokens,
wall time, throughput, environment and input identities, and required artifacts.
These are execution/provenance fields. Scientific loss reductions and diagnostic
formulas still need a concrete contract. The artifact envelope is described in
[ARTIFACTS.md](ARTIFACTS.md).
