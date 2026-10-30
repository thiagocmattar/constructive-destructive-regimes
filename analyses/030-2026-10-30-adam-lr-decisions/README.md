# Adam diagnostics and matched learning-rate decisions

Analysis030 / H024. Authorized on October 30, 2026: implement and run the
matched-checkpoint experiment using Runpod. Status: both sizes passed remote replay
qualification; the18 scientific continuations are running sequentially on the
L40S. First scientific updates and every diagnostic stream were observed; no
completed-arm result or scientific conclusion is available yet.

## Scientific comparison

Question: do optimizer-aware directional diagnostics help identify when an LR
change improves later loss, beyond recent loss and raw-gradient alignment?
Six trained-state blocks use Pythia14M/70M, seeds1234/1235/1236,
at saved update356. Within each block, continue for128 accepted updates at
0.5,1,2 times the LR applied at356. All branches retain identical starting
weights, moments, scaler, RNG and future data order. No new model initialization.
The two sizes sharing a seed form a cluster, not independent seed replications.

The reference rate is held constant. Hold the decoupled shrink factor constant
by dividing the decay coefficient by the LR multiplier, preserving exclusions.
This estimates the effect of adaptive-step length conditional on common
shrinkage, not native fixed-decay scheduler performance. Betas(0.9,0.95),
epsilon1e-8, global clipping1, effective batch1024 packed sequences, microbatch4,
context2048, FP16 dynamic compute and FP32 parameters/moments stay unchanged.
Each arm consumes128*1024*2048=268,435,456 input tokens and268,304,384 targets.
All18 arms consume4,831,838,208 inputs; reused prefixes are not counted as new work.

Reuse pinned MiniPile/tokenizer/model identities and verify original cache and
checkpoint hashes. Full validation includes all500 documents,339 packed sequences
including the tail,693,329 next-token targets, token weighted. Evaluate at local
updates0,16,128. A fixed128-complete-sequence training probe is chosen before
execution from the intersection of blocks not consumed by any arm through484;
evaluate at0,16,32,64,128 in FP32 with math attention. This is a subset objective,
not the full training loss. It is not used to change training.

## Measurements and retention

Every accepted boundary records norms, adjacent cosines and accumulation ratios
for raw unscaled gradients g, clipped optimizer inputs h, actual first moments m,
FP64 adaptive directions d from actual moments, and rounded parameter increments.
All unique trainable parameters are included, with FP64 reductions and prefix sums.
All prefix sums reset at the fork; raw/d adjacent histories retain the last parent
vector, and m starts from the actual parent moment. The first h and actual-step
cosines are undefined. Zero denominators are null. Raw changes are evaluated
pointwise before smoothing; the SGD eta-kappa expression is a projected response,
not an Adam curvature estimate or stability threshold. Incoming g versus prior m
is measured directly before clipping/moment mutation. Log clipping, relative
incoming/retained moment magnitudes, training loss, LR and timing/memory.

Actual increments include LR, decay and FP32 subtraction rounding. Their cumulative
norm is independently compared with distance from the fork at the end. Momentum
and adaptive accumulation ratios are descriptive stream geometry, not displacement.
Diagnostics/evaluation preserve RNG, parameters, outstanding gradients and optimizer
state. Recovery at64 and128 includes all streaming buffers and original-clock RNG,
optimizer, scaler and data position. The common parent is the retained update0 state.

## Analysis fixed before outcomes

Primary: paired full-validation loss at128 for lower/higher LR versus held LR,
reported for every size/seed. Secondary: improvement from16 to128 and fixed-probe
trajectories. Failures/nonfinite losses are outcomes, never silently dropped.
Compare early local updates2--16 diagnostics with later outcomes; update1's
gradient precedes the changed step. Show raw and optimizer geometry alongside
loss effects; do not optimize cosine or assume positive cosine is desirable.

Exploratory prediction uses leave-one-seed-out folds (both sizes and every arm of
the held seed excluded). Fixed ridge penalty10 with training-fold standardization.
Baseline features: initial probe loss, probe loss change0--16, log LR, log parameter
count. Raw extension adds mean raw cosine. Joint extension additionally adds mean
moment cosine, adaptive cosine, incoming-gradient/prior-moment cosine and log raw
norm change (last four versus first four observed norms in2--16). Target: validation
loss128. Report within-block chosen-arm regret relative to observed best arm,
held-LR regret, and all predictions. No tuning against held folds, threshold search,
significance claim or validated-controller claim from six blocks/three seed clusters.
If every block favors the same multiplier, the experiment cannot demonstrate
state-dependent intervention value; retain that negative/inconclusive result.
An additional state-independent comparator chooses a single multiplier using the
training seeds of each fold and applies it to both held-seed states. This tests
information beyond ordinary LR tuning, not merely improvement over holding LR.
It was added during input transfer, before any new training outcomes existed;
the scientific runs and the ridge-feature/penalty contract were unchanged.

These predictors observe each candidate's response during updates2--16. They
test early recognition of a beneficial intervention, not selection from the
common checkpoint alone. A deployment that probes all three candidates for16
updates and then completes the chosen128-update branch would consume160 total
updates, versus128 for a single predetermined rate (25% more training compute,
before evaluation overhead). The regret plot compares final branch losses and
does not demonstrate an equal-compute scheduler improvement. Directional
associations also do not identify a causal mediator or Adam curvature.

## Execution

Run IDs187--204, calibration uses separately labeled mechanics outputs under
the same frozen inputs. No manuscript edits are planned before evidence review.
Spark and unrelated cloud resources remain untouched. Live quote, bounded cost,
calibration, durable storage, monitoring and verified retrieval precede closeout.

Nine local checks pass, including native-update equivalence, direct measurements,
clipping/shrink factors, diagnostic noninterference, exact recovery, complete
synthetic checkpoint-to-artifact execution with validation tail coverage, cloud
ownership/transport checks, and independent ridge calculation/no held-target use.
The production source archive is frozen; subsequent local validation additions
and analysis code do not alter the worker's scientific inputs.

The first unused4090 allocation was removed after matching capacity disappeared;
the RTX6000 Ada replacement failed its10min readiness deadline and was also removed.
No scientific updates or input staging occurred on either worker. The same18 arms
are assigned sequentially to one48GB L40S, live quoteUSD1.09/h in US-MO-1.
The aggregate cap staysUSD15 including retries; the time guard is12h from the first
allocation, with training bounded to10.5h and90min reserved for final retrieval. The
initial6--9h estimate was provisional until the three-update calibration at each size.
Calibration now passes: warmed update times are9.44s/12.69s for14M/70M,
with peak allocated GPU memory6.03/11.42GiB. The conservative queue estimate
including20% timing margin and per-arm allowance is9.56h, versus9.70h available
at qualification. Raw update work at warmed speed is about7.1h; evaluations,
checkpoint writes, inventories and retrieval add time. `launch_validation.json`
records the replay differences and first observed scientific measurements.
No live L40S placement supported a network volume. A200GB persistent pod volume
instead retains checkpoints through a stop, until local hash/scientific acceptance.
Separate guard and retrieval processes manage only this task's IDs.
An additional detached pod-local guard (`08_remote_guard.py`) uses the
container-supplied credential and REST v2 to stop only its own pod at the same
hard deadline. Identity, API read access, the advertised stop action, and its
armed log were verified. The stop action is reserved for the deadline, so this
readback is not an end-to-end shutdown test. No account API key was copied.
This infrastructure-only addition was deployed separately from the frozen
scientific source archive; its hash is in `remote_guard_deployment.json`.
If the deadline interrupts retrieval, recovery is preserved and storage billing
is reported; the guard does not delete unverified evidence.

`06_analyze.py` implements the prespecified reductions and seed-held-out analysis.
`07_finalize.py` waits for verified retrieval/cleanup, then builds the report,
three PDFs and previews. Its automatic layout checks do not replace visual review;
figures and empirical claims remain pending review before manuscript integration.
