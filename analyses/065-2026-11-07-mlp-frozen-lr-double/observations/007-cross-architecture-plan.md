# H048: transfer selective and adaptive update damping

## Question and authorization

Can a rule motivated by representation oscillation reduce loss excursions without
merely halting learning, across MLP/CIFAR-10, ResNet-32/CIFAR-10 and
Transformer/WikiText-2? The user authorized all runs, additional ideas, and the
larger Transformer embedding LR hypothesis, with **USD 20 total Runpod spending**.
Keep this work in Analysis 065. Runs 269–271 freeze the first comparison.

## Matched comparison

Each worker branches from its own unchanged Adam continuation. Architecture,
data ordering, seed 0, objective, physical batches, precision and Adam moments
remain matched. Full training objectives are retained: 5,000 MLP examples,
50,000 ResNet examples, and 2,088,600 Transformer supervised tokens per update.
ResNet's original groups of 1,000 examples remain intact for batch normalization.
Native FP32, TF 32 off, deterministic algorithms; diagnostic reductions use FP64.

Eight untreated updates calibrate fixed, module-specific thresholds. Branches
start at 448/264/552 and compare losses through 800/576/928, respectively. Every
treatment includes a same-worker untreated control. Native two-step continuation
must equal save/reload continuation exactly; historical cross-hardware controls
are context, not interchangeable causal comparators.

The three selective rules modify the native Adam **parameter increment** after
its ordinary moment/clock update: discard coordinates above the prefix q90,
cap coordinates at q90, or cap each output neuron's frozen-input local affine
displacement RMS at prefix q75. They do not filter the raw gradient or reset
moments. Entry scope is MLP linear 1, ResNet conv 1, and Transformer layer 0 QKV
projection. QKV displacement is not full attention-output displacement. ResNet
also tests a cap throughout its convolutions. Zero thresholds disable that cap.

The adaptive block rule applies a scalar `previous_gradient_norm_EMA / norm`
only when a block's consecutive raw-gradient cosine is negative and its norm
exceeds that preceding EMA. Otherwise the multiplier is 1. EMA beta=.9 updates
after the decision; no future measurements enter. Direct parameter-owning
modules define blocks. This tests damping exceptional reversals while leaving
ordinary learning steps intact. Global half increments provide a simple damping
control. Cheap MLP branches also test global adaptive damping, a secant heuristic,
q50 local caps, and quarter first-layer increments. The secant heuristic is not
an Adam curvature bound. Transformer tests native embedding LR 2x and 4x, and 4x
combined with the adaptive block rule; embedding and decoder are untied.

## Measurement and interpretation

Every update records the full training objective, raw-gradient norm/rho/secant
statistic, block metrics, native and actual update norms, gates, and fixed-probe
representation displacement RMS/cosines. The fixed probes retain all 5,000 MLP
examples, one complete 1,000-example ResNet normalization group, and the first 64
Transformer sequences (2,240 supervised positions). Cosines flatten sample and
feature coordinates together; they are not cosines of sample means. Reduced
probes affect diagnostics only. They are not guaranteed population estimates.

Boundary state s records incoming representation changes s-1→s. Figures shift
these to outgoing update t=s-1, while gradient rho is indexed at t. Three audit
states per arm retain raw/previous gradients, native/actual increments and three
consecutive parameter states. Full fork/end/recovery states and scalar histories
are retained. Dense temporal SVD histories are not collected in this screening
round; no new low-rank claim will be inferred from scalar cosines.

Assess mean training loss, last 32 mean, maximum loss, positive loss variation,
and late-window behavior against each same-worker baseline. Full held-out
evaluation at the final scientific state is secondary, not a tuning target.
Less negative rho alone is not success. One seed and selected continuation
windows limit claims; promising rules need a fresh-window or norm-matched test.
Report failed transfer and delayed spikes rather than selecting only favorable
panels. Additional scientific designs require new run IDs.

## Compute envelope recorded before cloud execution

Live secure RTX 4090 quote on 7 November 2026: **USD 0.89/GPU-hour**. Use at most
three task-owned GPUs, with each lease at most 6hours. Each pod has 30GB container
and 80GB persistent pod volume. At USD 0.10/GB-month running storage, plus a
conservative USD 0.01/hour reserve, the bound is USD 0.91528/hour per pod.
Three full six-hour leases bound USD 16.48; additional/replacement leases must
fit the aggregate USD 18.50 compute/storage guard, leaving USD 1.50 recovery
reserve. The user's absolute ceiling is USD 20. Storage pricing source:
https://docs.runpod.io/pods/pricing (live read before launch).

Prior same-task hardware measurements suggest roughly 20minutes per ResNet
branch and 70minutes per Transformer branch; these are qualified estimates,
not throughput measured for this new implementation. Refresh from current
progress. Use owned Spark for MLP. Jobs detach, emit progress and recovery
states, and have per-job timeouts. A separate local watcher stops only recorded
task-owned pods at the lease deadline or aggregate cost guard. Stops preserve
volume for recovery. Retrieve and verify archive and individual SHA 256/size
identities before deletion; delete task pods promptly after successful retrieval.
Existing unrelated storage is preserved. Target local raw+archives below 30GB;
cloud costs remain provisional until reconciled with billing. Exact leases and
retrieval receipts are recorded alongside the run.

## Execution amendments

The initial comparison remains in Runs 269-271. The Transformer block-shock and
embedding-plus-block-shock arms were not selected for execution after the MLP
shock rules failed severely. All three requested entry-selection methods still
transfer, as do embedding LR 2x/4x. Runs 272-274 add distributed q50 caps and the
embedding 4x-plus-distributed-cap combination. Runs 275-279 test matched norms,
past-axis modification, a native-norm bound, actual functional budgets, and
slow recovery. Runs 282/283 replicate the selected MLP candidates on two more
pinned initializations. Runs 280/281 transfer the slow-recovery rule. Runs 284/285
and the final prelaunch revision of 281 add a probe-loss backtracking comparator.
The latter uses labels and is explicitly a line-search comparator, not a claim
of a new SVD optimizer. No further scientific additions are planned this round.

Run 281's prelaunch draft was never executed. Its final comparison and Run 285
reuse an already-computed unchanged control on the same worker. Before reuse,
require exact full fork (model, optimizer, raw-gradient history and RNG), exact
native two-step fingerprint, matched training/probe settings and common input
hashes. Copied raw control files remain byte-identical and receive explicit
reuse provenance. This saves two full control branches versus running one
fresh control per follow-up. Reuse does not change the rule
that every intervention is compared with unchanged Adam.

Measured Spark Transformer throughput was too slow for the frozen per-arm cap.
Its mechanics remain retained on Spark with verified metadata locally. A fourth
temporary cloud worker was allocated within the same dollar ceiling. A slow
RTX 4090 replacement attempt completed mechanics but its scientific queue was
canceled before starting; all evidence was retrieved before deletion. The
RTX 5090 retry uses unchanged scientific inputs and a fresh native control.
Live quotes were USD 0.89/hour for RTX 4090 and USD 1.19/hour for RTX 5090. Run 280's
Spark mechanics also complete; its science uses the existing faster ResNet pod.

The final lease allocation reserves **USD 19.5525** including conservatively
bounded running storage and a small hourly reserve. The independent stop guard
is **USD 19.60**, below the user's unchanged **USD 20** ceiling. This replaces the
earlier self-imposed USD 18.50 guard while retaining USD 0.40 recovery margin.
Measured rates are approximately 3.5 seconds per ResNet update,7.4-7.8 seconds
on the RTX 5090 Transformer worker, and 10.2-10.9 seconds on the other Transformer
workers. Final additions require about 20 minutes of ResNet training and 100
minutes of Transformer training, plus qualification, replay and retrieval.
Deadlines and all changes are recorded in `cross_architecture_leases.json`;
completed branches stream back while later ones train. Only task-owned pods
are stopped/deleted. Final cost reconciliation supersedes these reservations.

### Queue timing adjustment after measured transfer throughput

The 5090 distributed-cap branch measures about 8.0 seconds per full-objective
update. Reallocate six minutes from the ResNet lease and ten from Transformer B
to Transformer C2, without increasing the aggregate USD 19.5524 reservation.
The revised upper deadlines are 01:16:45 UTC for ResNet, 01:53:00 for Transformer B,
and 02:42:03 for C2 on 8 November (all still 7 November in the user's timezone).
Transformer A remains at 02:12:40 UTC. Expected C2 science completion is about
02:23 UTC, followed by replay and final retrieval; this is an estimate, not a
completion claim. The independent USD 19.60 stop and USD 20 user ceiling remain.

Only the waiting Run 281 scientific driver was replaced before its output
folder existed, giving it the revised cutoff while preserving all scientific
inputs. Its old/new queue records are retained in the owning run. No additional
scientific variants or resources were added by this timing adjustment.


### Retrieval allowance after ResNet closeout

At 2026-11-08T01:12:13.677920+00:00, all ResNet branches and independent audits completed, verified artifacts were retrieved, and its worker was deleted. Eight minutes of released reservation were reassigned to Transformer A verification/retrieval. The updated aggregate upper bound is $19.5600; the $20 ceiling and $19.60 watcher remain unchanged. Scientific inputs are unchanged.

### Retrieval allowance after embedding-worker closeout

At 2026-11-08T01:33:44.683853+00:00, Transformer B had finished all science, independent replay and verified retrieval and was deleted. Its released reservation funds ten additional minutes of replay/retrieval allowance on each remaining worker; scientific inputs and existing training cutoffs are unchanged. The revised aggregate upper bound is $19.5498, within the unchanged $19.60 guard and $20 ceiling.
