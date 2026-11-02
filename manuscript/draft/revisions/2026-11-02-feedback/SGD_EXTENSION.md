# Three-case batch-size comparison

Status: the user resumed this task and requested Runpod execution, after
previously declining the eight-hour proposal. A new aggregate cost/time limit
is pending. The user confirmed the three existing cases. Run214 / Analysis037
retain prepared code, seven passing numerical tests and qualified short real-data
worker queues. No production training or cloud allocation has started.
This extension supersedes the current 2-by-2 SGD figure, not its source data.

## Figure and question

Use a full-width 2-by-3 figure, with columns CIFAR-10/MLP,
CIFAR-10/ResNet-32, and WikiText-2/Transformer. The top row shows mean
consecutive sampled-gradient cosine over 199 pairs; the bottom shows
displacement retention R_200 over a 200-update continuation. Every panel
contains two curves: starting at initialization and starting after 2,000 GD
updates. Use a common two-color legend and distinct markers, shared row limits
([-1, 1] and [0, 1]), column-specific logarithmic batch axes and explicit
full-batch markers. Shade the range across initialization means, not a
confidence interval. No synthetic or interpolated evidence fills missing cases.

The question is whether batch size changes directional coherence and path
retention differently across initial and GD-trained states, and whether that
pattern recurs across these objectives and architectures. Starting state and
batch size vary within each case; initial tensors, objective, learning rate,
200-update horizon, parameter scope and numerical conventions are matched
within each comparison. Equal updates imply unequal data exposure. The cases
also differ in objective, data size, architecture and learning rate, so their
differences cannot be attributed to architecture or dataset alone.

## Evidence audit

- [Analysis027](../../../../analyses/027-2026-10-30-minibatch-noise/REPORT.md)
  supplies all MLP points: three initializations, five batch sizes, eight
  sampling repetitions per stochastic condition and one full-GD control.
  Reuse its source results unchanged at states 0 and 2000, with LR 0.02.
- [Analysis020](../../../../analyses/020-2026-10-14-eos-multitask/observations/001-frozen-real-data-noise.md)
  covers three cases only at frozen initial parameters. Its gradient-sum
  retention is not measured parameter displacement. It cannot provide either
  missing continuation column or the trained-state curves.
- [Analysis024](../../../../analyses/024-2026-10-17-eos-gd-lr-calibration/README.md)
  supplies seed-0 GD trajectories and retained states. Runs120 and126 are
  candidate ResNet/Transformer sources at LR 0.1/0.2. Saved tensors and exact
  data/initialization identities must qualify before reuse. Replicated trained
  states must be generated for any missing seeds; a seed-0 checkpoint is not
  evidence for three initializations.

## Proposed measurement contract

Retain three initialization blocks and eight stochastic repetitions as in the
MLP study. Proposed rates are ResNet 0.1 and Transformer 0.2, held constant
through GD preparation and each SGD continuation. These are already studied
rates with late reversal; selection uses existing GD evidence, not the new
batch-size outcomes. This is a comparison at selected regimes, not an unbiased
estimate over learning rates. Register immutable run inputs before execution.

| Case | Proposed batch sizes | Sampling unit / objective |
| --- | --- | --- |
| MLP (existing) | 32, 128, 512, 2048, 5000 | Images; complete fixed 5k-image objective |
| ResNet-32 (new) | 1000, 5000, 20000, 50000 images | 1, 5, 20 or all 50 fixed 1k-image ghost-normalization groups |
| Transformer (new) | 8, 32, 128, 512, full corpus | Prepared sequences; complete token-weighted training objective |

Verify the Transformer corpus count against its source identity (the prior
control records 59,680 sequences). Sample uniformly without replacement
within each update and independently across updates, pairing batch-size
permutation prefixes within each repetition. For variable supervised-token
counts, use per-sequence loss sums scaled by N / total_supervised_tokens,
averaged over sampled sequences; do not divide by a random batch token count.
ResNet sampling retains its original normalization contexts, including all
members of each sampled group. No change to augmentation, dropout, loss,
parameter scope, momentum, clipping, decay or schedule is introduced.

Record raw FP32 gradients at every update, with FP64 norm/dot/sum reductions
and TF32 disabled, matching the existing experiment. Average the 199 valid
adjacent cosines per fork; calculate R_200 from all 200 updates, resetting the
path at the fork. Check gradient-sum retention against actual parameter
displacement, allowing explicitly measured FP32 update roundoff. Zero norms
produce undefined values, not epsilon-normalized substitutes; report their
coverage and stop automatic figure acceptance if present. Aggregate repetitions
within initialization, then weight the three initialization means equally.

Measure complete-objective loss at every fork's start and end and retain
sampled losses throughout. Use these endpoint losses to check whether any
geometric change accompanies improvement or deterioration. No validation or
generalization benefit is inferred. Full-gradient evaluations at every SGD
state, frozen covariance controls and HVPs are unnecessary for these two rows
and are not part of this proposed extension.

The proposed new ResNet cells comprise 150 forks (30,000 updates); Transformer
cells comprise 198 forks (39,600 updates). Total: 348 forks / 69,600 continuation
updates, plus up to 12,000 reference GD updates if no saved three-seed states
qualify, or 8,000 if both seed-0 references qualify. MLP data are reused.
These counts are workloads, not elapsed-time or cost estimates.

## Execution preparation and acceptance

Before launch, inventory source checkpoints and data hashes, implement the
minimal run-local extension, verify unbiased loss reductions and group
sampling, and smoke-test diagnostic identities and deterministic full-batch
controls. Calibrate on the approved resource before stating an ETC. Any billable
launch needs live pricing and an explicit maximum aggregate cost/duration;
the manuscript-editing request supplies no spending ceiling. Historical leases
and spending caps from completed experiments are not reused.

Use detached execution, per-fork progress and persistent logs. Retain exact
starting states, batch/RNG identities, scalar histories, audit vectors and
rolling recovery including diagnostic accumulators; dense gradient histories
are not required for these statistics. Verify retrieved bytes and SHA-256 before
task-worker cleanup. Preserve unrelated resources and the held older queues.

Accept the figure only after independent raw-to-panel reduction checks,
full-batch replay checks, loss/coverage reporting and visual review at manuscript
size. Write the empirical interpretation after seeing all three cases. Report
case-specific exceptions rather than treating the proposed attenuation pattern
as an acceptance criterion. Rebuild and inspect the manuscript before replacing
the current review PDF.
