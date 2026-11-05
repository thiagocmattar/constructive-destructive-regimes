# Adam experiment plan, 5 November 2026

Recommendation: use the notebook for small mathematical/smooth-MLP controls,
then run the three-model HVP pilot before committing to the complete cohort.
The pilot is expected to cost **USD 5-8**, with a proposed **USD 12 cap**.
If its measurements are sound and scientifically informative, complete the
replication and optimizer-state comparisons. The recommended complete core
program has an expected **USD 39-48** rental cost and a **USD 70 total ceiling**.
These are proposed spending envelopes, not an authorization or an executed run.
No new training, paid calibration, resource creation or process changes occurred
during this revision. The completed work is retained-data analysis and algebra
verification; see [TODO](TODO.md) and [Analysis057](../../../../analyses/057-2026-11-05-adam-response-memory/README.md).

## What this would establish

The primary comparison is between the measured response z/eta, the one-HVP
directional proxy lambda_B, and lambda_B plus its first-moment correction M.
We ask whether M accounts for disagreement during ordinary oscillation, rise,
peak and recovery, and whether shorter fixed-direction trial steps reduce the
remaining discrepancy. We then test whether changing first-moment direction or
adaptive scaling changes loss from an identical state, and whether the event
relationships replicate across independent initializations. No new optimizer
or forecasting benchmark is required.

The prior-work baseline is the **single-HVP** directional proxy, not construction
of a full Hessian. The published Bai et al. paper includes momentum and stochastic
Transformer experiments; its larger setup specifies **187M non-embedding
parameters**. I verified Eq. 7, Appendix D's proxy and Appendix G against the
[published paper](https://proceedings.mlr.press/v306/bai26o.html). The existing
revision's identities are derivations, not completed empirical calibration.

## Available machines and live prices

Read-only checks around 09:25-09:30 BRT on 5 November:

| Resource | Observed state | Use in this plan |
| --- | --- | --- |
| Notebook | RTX 5070 Ti Laptop, 12,227 MiB total, about 3.7 GiB occupied; 24-core Ultra 9 275HX; 63.5 GiB system RAM | Analysis, quadratics, smooth MLP; its historical complete MLP arms took 15-16 seconds before new HVP/probe costs |
| DGX Spark | SSH works; GB10 at 0% utilization, no active GPU process; about 118 GiB available RAM and 3.4 TiB free disk | Zero-rental pilot or asynchronous runs if waiting overnight/multiple days is acceptable |
| Spark existing work | One unrelated Python process is suspended (state T) | Preserve it; recheck availability immediately before any launch |
| Runpod RTX 4090, 24 GB | Secure USD 0.89/h; EU-CZ-1/EU-RO-1 low stock | Preferred measured cloud reference for ResNet/Transformer |
| Runpod A40, 48 GB | Secure USD 0.59/h; CA-MTL-1 high stock | Capacity fallback only after measuring this exact workload; lower price does not establish lower total cost |
| Runpod L40S, 48 GB | Secure USD 1.09/h; US-MO-1 low stock | Alternative if calibration supports the budget |
| Runpod A100 PCIe, 80 GB | Secure USD 1.79/h; CA-MTL-3 low stock | Optional larger-model/HVP study |

Source: authenticated `runpodctl 3.0.1` live catalog, saved in
[compute_quote.json](compute_quote.json). The installed CLI matches the latest
release checked this session; Runpod MCP tools were unavailable, so the supported
CLI fallback was used. Community RTX 4090 was quoted at USD 0.34/h, but the plan
uses the secure quote and does not assume a community offer can be provisioned.
Capacity is a snapshot, not a reservation. Requote price and inspect placement
at launch. Account authentication succeeded; no credentials are retained here.

The [availability receipt](compute_availability.json) records the checks. The
notebook's remaining VRAM does not prove that full-objective ResNet HVPs with the
fixed 1,000-image BN groups fit. Do not silently reduce those groups to fit:
that would change the objective. Spark is suitable for these small models but
is substantially slower on the full vision/language objectives.

## Timing evidence and budget scenarios

Same-project measurements, not generic GPU specifications:

- Run215 RTX 4090: ResNet 1,024 updates plus recorded diagnostics/evaluations
  took 2,674 s (0.743 h); two Transformer arms took 9,285/9,430 s
  (2.579/2.619 h). Model, objective and accumulation match the proposed core.
- Run222 Spark: 64-update branches took about 977-987 s for ResNet and
  2,282-2,294 s for Transformer. Linear extrapolation is about 4.4 and
  10.2 h per 1,024 updates, before new diagnostics, checkpoint transfers and setup.
- HVP/probe overhead for the new recipe has **not** been measured. The estimates
  allow roughly 15-30% over the corresponding baseline trajectories, plus setup,
  branch work and retrieval. This is a planning scenario, not a confidence bound.
  Replace it with complete-update and HVP timings during the bounded pilot.

| Stage | Proposed work | Rental GPU-hours | Wall-clock ETC after machines are ready | Expected cost | Proposed hard limit |
| --- | --- | ---: | --- | ---: | --- |
| A | Small controls and three highest-rate seed-0 HVP trajectories | 5-8 | 3-5 h with two cloud workers; MLP local | USD 5-8 | USD 12; 12 aggregate GPU-h; 6 h per Pod |
| B+C | Remaining 24 matched trajectories plus state interventions, below | 38-45 | 10-14 h with up to four workers | USD 34-41 | USD 55; 60 aggregate GPU-h; 16 h per Pod |
| Core total | Three seeds x three rates x three models, plus controlled probes/branches | 43-53 | About 13-19 h execution across the two stages, plus review/queue delay | USD 39-48 | USD 70 overall; stage and aggregate limits both apply |
| D, optional | Stochastic fixed-monitor pilot, 12 trajectories | 3-8 | 2-6 h with two workers | USD 3-8 | Separate USD 15; 16 aggregate GPU-h |
| E, optional | 160M calibration and 1B monitoring study | 20-48 on A100 80 GB | 10-24 h with two workers | USD 40-90 | Separate USD 120; 60 aggregate GPU-h; recalibrate first |

All amounts are USD before applicable tax; stage limits include setup, retries,
probes, checkpoint transfer time and disk charges. A budget limit is a stopping
rule, not a promise that every arm completes. Stages D/E are not included in the
USD 70 core ceiling. Under those provisional optional ceilings, the entire staged
program would be at most USD 205; there is no reason to reserve or spend it now.

For core Pods, budget 30 GB container + 50 GB volume disk (80 GB total), both at
USD 0.10/GB/month while running, or about USD 0.0111/h using a 720-hour month.
Thus the quoted 4090 plus disk is approximately USD 0.9011/h. No network volume
is proposed. Runpod lists no data ingress/egress fee; transfer **time** still bills
while a Pod runs. See [current pricing](https://docs.runpod.io/pods/pricing) and
[storage lifecycle](https://docs.runpod.io/pods/storage/types).

The core alternative entirely on Spark is roughly **7-9 days** of serial device
time at the historical rates, with **USD 0 rental** (electricity and machine
occupation excluded). Stage A alone is about **18-24 hours** there. A cheaper
cloud variant reuses the six original low/middle-rate seed-0 arms instead of
rerunning them, saving roughly 7-10 GPU-hours; it leaves hardware/environment
differences entangled with the original seed block. The recommended design
reruns them for a consistent comparison.

## A. Validate the response-curvature connection

First run formal controls on the notebook: rotated anisotropic quadratic with
eigenvalues (0.5, 7), fixed initial loss 0.5, rotations 0, 0.23, 0.71, 1.13 radians,
beta1 in {0, 0.5, 0.9}, beta2=0.999, epsilon=1e-8, eta=0.013, 100 updates.
These settings already pass algebra verification; a new run would preserve full
trajectories and test measurement instrumentation. Constant H removes the
finite-step curvature term; beta1=0 additionally removes the memory correction.
Use the existing smooth tanh MLP as the neural calibration, not an unannounced
activation change to ResNet or Transformer.

Then run seed 0 from the frozen original initial tensors at the highest rate
for each model: MLP 0.001, ResNet 0.003, Transformer 0.003, for 1,024 updates.
ResNet and Transformer use the same RTX 4090 environment; MLP uses the notebook.
Fresh controls and same-environment replay are required. Historical trajectories
remain separate if floating-point drift prevents matching an old event.

At every update, log raw gradient norm/cosine/z, current gradient-moment and
gradient-direction cosine, actual increment/dot product, m/v norms and
norms of adjacent-gradient sum and first-moment residual. Use FP32 training,
TF32 off and FP64 scalar reductions. Unique trainable parameters are counted
once; undefined zero denominators are null, with reasons.

Compute H_t*g_t on the **complete same training objective** at preselected states
0,64,...,960 (16 HVPs per trajectory), sharing it for lambda_B and M. Recover the
next gradient before calculating z_{t+1}. Retain exact clock alignment: at state
s, t=s+1; compute current moments from g_t, then evaluate the actual Adam step.
Do not confuse the vector before and after the moment recursion.

After loss-only event segmentation, select the three largest recovered >10%
events per trajectory, if present, and probe a, p, c plus an ordinary state matched
on stage and baseline loss: at most 12 additional anchors. If an event has no
admissible ordinary match, report it and use fewer anchors. Always retain the
fixed grid, including low-response/non-event states. No event occurrence is
guaranteed; an empty stratum is a result, not a reason to tune the LR afterward.

At these anchors hold d_t, P_t and the objective fixed. Evaluate endpoint gradients
for alpha={1,1/2,1/4}; use alpha=1/8 only as a prespecified follow-up if the first
three are numerically inconclusive. Report the signed and absolute differences
from lambda_B and lambda_B+M, with a scale-normalized error and zero-scale policy.
There is no need to compute a Hessian spectrum.

For the smooth MLP the remaining term can be interpreted as finite-step curvature
variation under the stated regularity. ResNet uses ReLU, and the retained
TransformerEncoderLayer uses its ReLU default. Classical-Hessian path identities
cannot simply be assumed across activation boundaries. There, report total
finite-step discrepancy, including possible nonsmooth and numerical effects;
do not relabel every residual as smooth curvature variation. Compare FP32 and
FP64 scalar reductions and a small higher-precision calibration where feasible.

Proceed to B/C after identities, clocks, exact within-device replay and finite
values pass. Scientifically, report whether correction improves agreement or
reveals interpretable disagreement; do not filter away a negative result.
If the pilot cannot distinguish the terms reliably, revise instrumentation and
the scientific claim before purchasing a larger study.

## B. Independent initialization blocks and event specificity

Complete **27 unique trajectories**: three models x three rates x seeds {0,1,2},
all 1,024 updates. The existing nine trajectories are historical evidence; the
recommended new cohort reruns their seed-0 recipe on the standardized environment
and adds 18 genuinely new initialization conditions. Stage A's three arms count
toward these 27, so 24 remain. Reruns are not extra independent seeds.

Hold data, ordering, objectives, optimizer, precision, horizon and all diagnostics
fixed. Generate and hash each initialization once, then load the identical tensors
for all three learning rates in that seed block. Use fixed rates {1e-4,3e-4,1e-3}
for MLP and {3e-4,1e-3,3e-3} for the other two. Adam (0.9,0.999), epsilon 1e-8;
no clipping, schedule, decay, augmentation or dropout. Use the same hardware and
software per model across all seed/rate conditions. Balance worker assignment so
hardware is not systematically assigned to one LR or seed.

Source identities: Run215 input_manifest.json and Run213 config.json, with their
cache files verified again before use. MLP uses the same prepared 5,000 CIFAR-10
images and mean half-squared error; ResNet uses all 50,000 images, fixed 50 groups
of 1,000 and mean cross entropy; Transformer uses the same WikiText-2 cache,
59,680 sequences and all 2,088,600 supervised targets, token-weighted NLL.
Accumulation partitions are MLP 1,000, ResNet the fixed 1,000-image BN groups and
Transformer 64 sequences. Full evaluation uses every original held-out split
example/token at states 0,128,...,1024, with manifest counts; no truncation.

Retain the 5/10/20% loss-only event definitions and censored candidates. Report
counts, occupancy, absolute/relative increases, loss-excess area, phase profiles,
coefficient stability and all run-level values. Match ordinary windows to event
duration **exactly**, start within 128 updates, and baseline loss within factor
1.25, without overlap with >10% events. Match without replacement, closest
normalized stage/loss distance first with an explicit deterministic tie rule.
If no match exists, retain the event as unmatched and report support; do not relax
calipers to obtain a favorable comparison. Include strong-reversal ordinary
windows in the accounting. This closes the duration-matching gap left by Analysis055.

Fit severity models using the existing eight specifications. Hold out complete
initialization blocks, keeping all rates together; no random split of events.
With only three seeds, report individual blocks and ranges rather than precise
population confidence claims. Add further seeds only if the observed variation
and sparse-event coverage justify a separately bounded design.

## C. Matched optimizer-state effects

Use highest-rate trajectories for all three models/seeds (nine trajectories).
At four loss-defined anchors per trajectory (ordinary, pre-rise, peak, recovery;
at most 36 checkpoints), compare directions from the identical current gradient:

| Direction | First moment | Preconditioner |
| --- | --- | --- |
| Baseline | current bias-corrected m | current P |
| Refresh direction | current raw g | current P |
| Reference scaling | current bias-corrected m | P_ref |
| Both | current raw g | P_ref |

P_ref is the actual bias-corrected preconditioner from 16 completed updates before
the anchor, held fixed. Restrict eligible anchors to s>=16. Compare natural steps
and steps rescaled to the baseline direction norm. The baseline is identical in
both norm settings: seven distinct one-step proposals per checkpoint. Evaluate
the same objective at all endpoints, record delta loss, step norm and cross-angle.
Zero direction norm makes norm matching undefined; retain the natural proposal
and report the missing matched proposal.

Then use pre-rise and ordinary anchors for **32-update continuations** with the
same seven distinct branches: at most 9 x 2 x 7 x 32 = 4,032 updates. In refreshed
branches use the current gradient each step while recording the baseline moment
recursion; in reference-scaling branches hold P_ref fixed throughout. In matched-
norm branches scale the chosen direction to the contemporary baseline proposal
computed at that branch's own state (not to a later control-trajectory gradient).
Keep beta values and their time counters; freezing raw v alone is incorrect
because bias correction would still change P. Compare cumulative loss excess,
mean/final objective and full held-out evaluation at matched budgets. New geometry
after branches diverge limits causal interpretation to the specified intervention.

## D. Stochastic measurements, after the core

Bounded proposal: ResNet-32 and the existing small WikiText-2 Transformer, three
initialization/data-order seeds, two batches, 1,024 updates = 12 trajectories.
Use ResNet batches {128,512} images and Transformer {8,64} sequences; LR 0.001,
Adam (0.9,0.999), no clipping/decay/dropout in this first comparison. Match updates,
not tokens; report the resulting image/token exposure explicitly. Pin source
ordering, RNG streams and initialization hashes before any run.

Use frozen, calibrated BN statistics for this controlled ResNet sampling study.
At each seed's initial weights, make one no-gradient pass over all 50,000 training
images in the fixed 1,000-image groups, with cumulative-average running-statistic
updates and dropout disabled, then freeze those buffers identically for both arms:
otherwise changing batch composition changes the per-example function as well as
gradient sampling. That is an explicit new objective/condition, not reuse of the
full-batch BN objective under a different name. A later train-mode-BN/augmentation
or AdamW/clipping realism arm is a separate comparison, with raw gradient,
optimizer input and actual update logged separately.

Two streams: ordinary training-batch observations, and a fixed monitoring set
(1,000 training images / 512 prepared sequences, sampled once with a recorded seed
and hashes). Evaluate its token/example-weighted loss every update, with fixed
BN statistics and dropout disabled. Report its fraction of the training corpus;
it is not full-training or validation loss. At fixed windows 64..71,256..263,
512..519,768..775, compute same-monitor gradients at all endpoints, allowing
adjacent responses on an unchanged objective. Full held-out coverage remains at
0,256,512,768,1024. Monitor-set selection is independent of event severity.
The moment recurrence applies to the training-gradient stream supplied to Adam;
do not apply it unchanged to the separate monitoring gradients.

The 3-8 GPU-hour quote is an **unmeasured scenario**, allowing 0.1-1.5 seconds per
complete instrumented update plus setup/probes. Benchmark the fixed monitor and
full evaluation before admitting all arms; dense monitor cost may dominate training.

## E. Optional scale and overhead claim

Only if scale remains part of the headline: Pythia-160M architecture for calibration
and Pythia-1B for the larger test, initialized from scratch, using pinned MiniPile
tokens. Exact architecture revisions are saved in
[scale_model_candidates.json](scale_model_candidates.json); no weights downloaded.
Reuse the verified MiniPile dataset revision
18ad1b0c701eaa0de03d3cecfdd769cbc70ffbd0 and tokenizer revision
7386d9a4ae45aef494a6e704910394def3037fc5 only after new cache/hash verification.
Meta-device instantiation gives 162,322,944 total / 85,056,000 non-embedding
parameters for the smaller model and 1,011,781,632 / 805,736,448 for the larger
model. Here non-embedding excludes both input and output embedding matrices,
counting tied tensors once. No weights were downloaded or allocated on a GPU.

Provisional design: three seeds, 1,024 updates each, sequence length 1,024,
effective batch eight sequences (8,192 input tokens; 8,184 within-sequence shifted
targets), FP32 parameters/moments
with BF16 autocast, Adam beta=(0.9,0.999), epsilon=1e-8, constant LR 3e-4 after
64-update linear warmup, no clipping/decay. This is a short dynamics study, not
complete pretraining or a scale law. The packing/tail and label-shift contract
must pin the exact consumed prefix: 8,388,608 input tokens and 8,380,416 supervised
targets per 1,024-update trajectory, with no cross-sequence targets. This is an
explicit prefix budget, not a complete MiniPile pass.
Full MiniPile validation uses all 500 documents with their complete token count
at initialization and every 256 updates. No held-out data selects the LR.

At fixed same-checkpoint/order windows compare instrumentation off, cheap gradient
response, and one-HVP monitoring, with 20 warmup and 50 timed updates per mode,
at both scales. Use a fixed 128-sequence monitor every 16 updates and HVP every 64
for calibration; the monitor has 16-update resolution and can miss short events.
Record wall time, peak allocated/reserved memory, previous-gradient storage,
probe overhead, numerical agreement, and useful phase differences. One FP32
previous-gradient buffer costs 4N bytes; this 1B configuration requires
4,047,126,528 bytes (about 4.05 GB) for that buffer alone.
Single-GPU experiments cannot establish distributed communication scalability.

The A100 80-GB quote assumes 20-48 aggregate GPU-hours, including a bounded initial
calibration, setup, evaluations and retrieval. It is not a measured ETA: 1B training
with these probes could differ materially. Admit it only after measured fit and
throughput; cap the first paid calibration at USD 10 inside the proposed USD 120.
Use at most 150 GB disk per worker and verify checkpoint capacity first. A larger
negative-cosine plot alone would not justify the scale claim.

## Execution, retention and stopping rules

Allocate new run/hypothesis IDs only at launch preparation; do not reuse Run215 or
mutate executed configs. Stage A/B share a single frozen core measurement contract.
The new minimal resume state contains parameters, optimizer moments/time counter,
previous raw gradient, RNG, objective/data position and scalar diagnostic history.
Retain **every 16-update checkpoint**, not an overwritten rolling file, plus all
selected probe/branch parents and final states. Full cumulative-vector diagnostic
streams unused by this revised Adam section need not be added to the new runner;
declare that measurement difference, which does not change Adam's update rule.

For the roughly 14.8M-parameter Transformer, parameters+m+v+previous-gradient alone
are about 237 MB per checkpoint, or about 15.4 GB for 65 checkpoints per trajectory.
Allow roughly 150-200 GB for the full core archive including probe evidence and
other models. Spark has storage headroom. Transfer each completed arm promptly
and verify hashes before reusing its Pod storage; do not retain multiple complete
Transformer arms on a 50-GB volume without checking free capacity. Preserve full
scientific and restart evidence on owned storage, plus compact local records.

Measure after warmup: at least 10 complete updates, five HVPs, one full held-out
evaluation, checkpoint-save duration and representative transfer throughput.
Refresh ETC using actual phase costs and count every arm/worker. Failed starts,
retries and unused idle time count toward the same stage/total limits. Reject a
placement if the projected job cannot fit the remaining aggregate budget.

Use a detached controller and remote workers, persistent logs, source/config
manifest and progress events. Monitor read-only every 10-15 minutes: completed
updates/tokens, loss, timing, memory, disk, event age and refreshed ETC. Warn on
nonfinite values, missing updates, no progress for three expected intervals,
<20% free disk, or budget/ETC overrun. Stop scientific work at 90% of the time/cost
limit, reserving the remainder for checkpointing, transfer and verification.

Copy per-file sizes/SHA-256 inventories, verify all files on owned storage, run
scientific acceptance checks, then delete only task-owned Pods/disks. Pod volumes
survive stopping but not deletion, so copying must precede teardown. Confirm
cleanup with a fresh account read; leave 0 GB task cloud storage unless a bounded
recovery exception is explicitly agreed. Final billing remains provisional until
the provider ledger covers the complete resource window.
