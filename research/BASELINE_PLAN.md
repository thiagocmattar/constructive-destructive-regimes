# First baseline: adjacent and cumulative gradient geometry

The user's latest correction selects AdamW with the reference learning-rate
schedule and a full MiniPile pass, superseding the earlier SGD selection.
The concrete implementation is [Run001](../runs/001-2026-10-03-pythia14m-adamw-gradient-baseline/README.md).
Seventeen tests pass. The user subsequently explicitly preapproved all Runpod
launches and requested the completed experiment and metric-evolution analysis.
Linux calibration and the full 712-update pass completed on a Secure RTX 4090
(USD0.74/hour). All input tokens/tails were covered; artifacts/checkpoints were
retrieved and verified, figures generated, and all owned cloud resources removed.
Estimated total GPU/storage cost is USD1.30, provisional pending billing coverage.
Operational attempts, capture contract, and observations are recorded in Run001.

## Requested quantities

Let `g_t` be the full model gradient at update `t`, with the first update indexed
`t = 0`. Proposed boundary: gradient of mean supervised-token causal language
modeling loss, accumulated across all microbatches in that optimizer update,
unscaled, before clipping, learning-rate multiplication, momentum/Adam
transformation, or optimizer-applied weight decay. The gradient is evaluated at
the current parameters, along the evolving training trajectory.

Use all unique trainable parameters in a fixed recorded order. A global norm is
the square root of the sum of squared entries across all parameters, not a sum
of parameter-tensor norms or an average of layer cosines. If an expected gradient
is absent, diagnose it rather than silently changing the vector scope.

For every update record:

```text
gradient_norm_t = ||g_t||_2
adjacent_cosine_t = <g_(t-1), g_t> / (||g_(t-1)||_2 * ||g_t||_2)
cumulative_gradient_t = cumulative_gradient_(t-1) + g_t
cumulative_norm_sum_t = cumulative_norm_sum_(t-1) + ||g_t||_2
cumulative_gradient_norm_t = ||cumulative_gradient_t||_2
cumulative_ratio_t = cumulative_gradient_norm_t / cumulative_norm_sum_t
```

Initialize the running vector and scalar to zero before the first training
gradient. First-update adjacent cosine is null. A zero gradient makes its
adjacent cosine undefined; a zero cumulative scalar makes the ratio undefined.
Record null plus a reason, not artificial zero/epsilon-biased values. Exact vector
cancellation with a positive scalar denominator has a defined ratio of zero.

The ratio orientation is proposed as norm-of-sum over sum-of-norms. Keeping both
raw scalar quantities allows the inverse later without rerunning training.
The ratio lies in `[0, 1]` in exact arithmetic. Floating-point residuals must be
checked and reported, not silently hidden. First-update ratio is one when the
gradient is nonzero.

This tracks all recorded gradients cumulatively; it does not assume distant
cross-step correlations vanish. Low ratios can arise from orthogonal directions
as well as opposite directions, so they are not by themselves proof of
destructive learning or instability. Under AdamW, these are raw-gradient
statistics, not a decomposition of actual parameter displacement.

## Constant-memory capture

Keep the cumulative gradient in FP64, the previous gradient in FP32, and the
cumulative scalar/reductions in FP64. Current gradients use the existing model
gradient buffers. Snapshot the previous gradient before clipping/clearing can
mutate it. Detach all diagnostic operations from autograd. Parameter-wise
operations avoid an additional full concatenated gradient allocation.

For approximately 14 million parameters, the two persistent vector buffers total
approximately 168 MB (112 MB cumulative plus 56 MB previous), excluding temporary
reduction buffers and normal training memory. This is `O(parameters)` persistent
diagnostic memory and does not grow with the number of updates.

Write scalar values every update. Do not save a full gradient vector for every
update. Preserve the accumulator, previous gradient, scalar sums, and update
position in recovery checkpoints and save the final accumulator. Exact resume
also requires model/optimizer/scheduler, random states, and data-order position.

## Selected baseline and deferred work

- One dense Pythia-14M model with ordinary GeLU, random initialization, MiniPile,
  and no sparsity intervention. Pin the source-study model/tokenizer/dataset
  revisions only after verifying the intended variants and inputs.
- Proposed source-compatible AdamW variant: peak LR `1e-3`, floor `1e-4`, cosine
  schedule with 1% warmup, betas `(0.9, 0.95)`, epsilon `1e-8`, decay `0.1` with
  the source's exclusions, and clipping `1.0`. Requested diagnostics are captured
  before that clip. The user selected AdamW; the run retains the source's exact
  NeoX pre-step schedule, including a zero-LR first update.
- Deferred: plain SGD without momentum. Its learning rate/decay/clipping
  require an explicit design rather than recycling AdamW's recipe. With constant
  LR and no decay/clipping, the cumulative raw gradient directly determines
  parameter displacement; that equivalence does not hold generally for AdamW.
- Proposed packed input length: 2,048 tokens. Proposed effective batch: 1,024
  sequences per update, matching the reference baseline's scale. Select a smaller
  microbatch plus more accumulation for memory fit; verify the gradient reduction
  is invariant within declared numerical tolerance.
- Selected horizon: one full nonrepeating MiniPile pass. Verified local cache
  bytes imply 712 updates at the selected effective batch. Include the short
  token tail as its own shorter sequence and use a correctly token-weighted
  final update of 311 sequences. Validation likewise includes its short tail.
- Proposed model/data seed: `1234`. One seed is an exploratory baseline, not an
  uncertainty estimate across seeds. Precision and source initialization mapping
  must be pinned and tested in the concrete run.
- Full declared validation coverage before training and at the final checkpoint;
  periodic passes may be added with measured overhead. Keep initial/final model
  states and a midpoint recovery state, including diagnostic accumulators.

If FP16 scaling causes a skipped optimizer update, do not silently count it as
an ordinary diagnostic step: record the event and require the finite-gradient
capture/accepted-update policy to be explicit. A proposed BF16 runtime change
must also be disclosed if matching the source's FP16 recipe is intended.

## Verification and Runpod sequence

1. Implement run-local streaming diagnostics and test aligned, opposite,
   orthogonal, unequal-magnitude, and zero vectors against hand-computed values.
   Compare every streaming prefix with a directly summed stored-gradient reference.
2. Test whole-model aggregation, loss/accumulation normalization, capture before
   clipping, preservation across gradient clearing, and checkpoint/resume continuity.
3. Perform a short production-shaped smoke/calibration. Reset model, optimizer,
   data-order position, and diagnostic state before baseline execution so the
   smoke does not contaminate the recorded trajectory. Measure diagnostic overhead,
   full update time, validation, checkpoint I/O, peak memory, and transfer size.
4. Select one Runpod GPU that fits the measured workload. At the planning read,
   Community RTX 4090 has 24 GB, catalog price USD 0.34/hour, availability LOW,
   and available hosts in the returned CUDA matrix at 13.0. This is a read-only
   snapshot, not a reservation or proof of production memory fit. Recheck stock
   and actual Pod price before provisioning. Source microbatch 32 previously
   exceeded 24 GB; do not transplant it unchanged.
5. Produce the exact ETC, spending ceiling/deadline, storage/recovery/transfer
   plan, and launch command once implementation/calibration and missing choices
   are resolved. No duration or total-price estimate is asserted yet.
6. Run detached, monitor loss/progress/throughput/ETC and finite diagnostics, retrieve
   scalar logs/checkpoints/accumulators, verify hashes, and terminate task-owned
   compute. Analysis can then plot adjacent cosine and the two cumulative norms
   plus their ratio against updates/tokens, without implying causality from geometry.

## Planning verification, not experiment results

Six hand-defined vector cases were checked with independent standard-library
arithmetic during planning. Examples after two gradients:

| Gradients | Adjacent cosine | Norm of sum | Sum of norms | Ratio |
| --- | --- | --- | --- | --- |
| `(1,0)`, `(1,0)` | 1 | 2 | 2 | 1 |
| `(1,0)`, `(-1,0)` | -1 | 0 | 2 | 0 |
| `(1,0)`, `(0,1)` | 0 | sqrt(2) | 2 | 1/sqrt(2) |
| `(2,0)`, `(-1,0)` | -1 | 1 | 3 | 1/3 |

The initial analytic checks have now been supplemented by seven passing tensor
and optimizer-boundary tests. Native Windows production preflight verified the
cache identities but failed before training because Flash Attention is not
compiled into that runtime. Scientific baseline execution is pending.
