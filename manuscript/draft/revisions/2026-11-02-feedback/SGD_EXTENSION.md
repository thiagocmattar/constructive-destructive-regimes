# Three-case batch-size comparison

Status: completed in Runs214/216 and Analysis037. All 348 new continuations,
six references and the additional Spark bridge pass raw and numerical checks;
the requested three-case figure and 16-page review manuscript are verified.
All 13 task Pods/disks are deleted; retained cloud storage 0 GB. Provisional
US$26.30 /4.663 hours, including calibration, retries, disk and retrieval, within
the user's US$40/six-hour envelope ending 2026-11-02 19:34:56 UTC.
The initial design and subsequent implementation decisions remain below.

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
  supplies seed-0 GD trajectories and retained states. Runs120 and 126 are
  candidate ResNet/Transformer sources at LR 0.1/0.2. Saved tensors and exact
  data/initialization identities must qualify before reuse. Replicated trained
  states must be generated for any missing seeds; a seed-0 checkpoint is not
  evidence for three initializations.

## Measurement contract

Retain three initialization blocks and eight stochastic repetitions as in the
MLP study. Proposed rates are ResNet 0.1 and Transformer 0.2, held constant
through GD preparation and each SGD continuation. These are already studied
rates with late reversal; selection uses existing GD evidence, not the new
batch-size outcomes. This is a comparison at selected regimes, not an unbiased
estimate over learning rates. Register immutable run inputs before execution.

| Case | Proposed batch sizes | Sampling unit / objective |
| --- | --- | --- |
| MLP (existing) | 32, 128, 512, 2048, 5000 | Images; complete fixed 5 k-image objective |
| ResNet-32 (new) | 1000, 5000, 20000, 50000 images | 1, 5, 20 or all 50 fixed 1 k-image ghost-normalization groups |
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

Measure raw-gradient geometry at every update using FP32 gradients and FP64 norm/dot/sum reductions
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
These counts are trajectory coverage, not elapsed-time or cost estimates.
All six references are generated fresh. Transformer uses physical accumulation
512 in separately frozen Run216; its complete-gradient comparison against 64
has relative L 2 difference1.53e-6 at the pinned initial tensors. The effective
sampling batches and token-weighted objective are unchanged. The initial full-GD
controls for Transformer seed 0 and ResNet seed 2 reuse the exactly captured
first 200 reference updates, with endpoint losses evaluated separately. Its scalar history,
checkpoint, raw prefix and actual displacement are retained and audited;
the terminal adjacent pair inherits the measured reference scalar because
its previous gradient was not retained. This saves 400 executed updates while
preserving both trajectory measurements. Other controls execute independently.

The user subsequently identified idle Spark capacity. A separate GB10 directory
qualified the unchanged Run214 source/data/config under PyTorch 2.11/CUDA 13:
initial full-gradient relative L 2 difference1.20e-7 and three-update parameter
difference1.59e-9 versus the cloud runtime. Spark executes ResNet seed 2's 32
small-group continuations (1/5 groups at both starting states) from hash-identical
cloud reference checkpoints. An additional 200-update initial full-GD bridge
quantifies longer numerical drift. This adds explicit runtime variation, which
is not isolated from initialization/sampling variation by the plotted range.
It adds no cloud cost. The original suspended Spark queue remains untouched.
With two reused reference-prefix controls and this additional bridge, planned
new production training totals 81,400 updates, excluding mechanics/calibration;
primary coverage remains 348 new continuation trajectories, with 198 MLP
continuations reused for the figure. A qualified extra L40S executes two
trained-state ResNet seed 2 batch 20 forks and hands them off before the main
queue reaches them. A bounded inter-fork queue wait permits the remaining
verified Spark handoffs; the active measurement child is never paused.
Neither scheduling change alters scientific inputs, RNG identities or coverage.

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
