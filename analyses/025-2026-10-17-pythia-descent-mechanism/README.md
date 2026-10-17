# Analysis025: causal tests of the Pythia loss-rate separation

Status: authorized on October17 with a USD25 aggregate cap and12h deadline.
Two Secure RTX4090 workers and a new200GB network volume are allocated in
EU-RO-1. Both remote mechanics suites pass; frozen inputs are staging before
GPU qualification. See [execution record](execution.json). Spark jobs remain
untouched. No empirical mechanism result is available yet.

H021 asks whether Adam coordinate preconditioning contributes to the model-size
loss-rate separation around updates200-250, after controlling adaptive-step norm.
This is a proposed optimizer-mediated contribution, not an established explanation
of model capacity. The motivating descriptive evidence is
[Analysis005 observation012](../005-2026-10-04-temporal-lag-profiles/observations/012-loss-separation-200-250.md).
The local observation found smaller models slowing while70M largely maintains
its prior loss decline. The selected interval and follow-up hypothesis are
post-hoc; the new interventions provide the prospective comparison.

## Comparison and scientific inputs

Runs127-135 are9 new baselines: Pythia-14M/31M/70M at seed blocks1234/1235/1236,
350 accepted optimizer updates each. Runs136-153 are18 continuations: for each
baseline, start from its exact state after update175 or225, then take50 updates.
The primary continuation removes coordinate preconditioning and normalizes the
adaptive component to the unmodified parent's per-update norm schedule.
One-step comparisons separately test first-moment averaging and step magnitude.
This explicitly instantiates the prior proposal; no broad hyperparameter search
or feature-freezing intervention is included.

All runs use the pinned MiniPile cache/tokenizer, sequence length2048, effective
batch1024, microbatch4, raw gradient clipping at1, AdamW betas(0.9,0.95), epsilon1e-8,
decay0.1 with the original bias/LayerNorm exclusions, FP32 parameters and dynamically
scaled FP16 training. **The LR clock remains the original712-update schedule**;
stopping training at350 does not compress the schedule. LR, clipping, token order,
future batches, optimizer-state clocks and precision match between each parent
and its continuations. All350 prefix batches are full:358400 sequences,
734003200 input tokens and733644800 supervised next-token targets per baseline.
Each continuation processes another50 parent-calendar updates, not a reset clock.

Architecture/config and exact initial tensors are retained from Runs002-010,
identified in [initial_sources.json](initial_sources.json). No pretrained weights
are introduced and no new random initialization is substituted. Model size varies
architecture/parameter count and initial tensor values. Initialization and order
seeds co-vary between the three blocks, and order matches across sizes within a
block; this does not separate initialization variance from ordering variance.
All new runs use the same GPU class and pinned software. The historical trajectories
are timing/input references, not substitutes for these new controlled replays.

The shared [probe](probe.json) contains128 complete packed sequences,262144 input
tokens and262016 supervised targets. Seed104729 selects from the complement of
the UNION of all three350-update training prefixes. Thus the probe is identical
across all27 conditions and does not occur in any condition's training prefix.
It is a fixed training-distribution objective, not the whole training corpus or
an unbiased estimate of final generalization. Probe data never select gradients,
step lengths or training updates. Full MiniPile validation, including its short
tail (693329 supervised targets), is evaluated at each trajectory's start/end.
It remains separate from the primary fixed-probe outcome.

## Descent measurements

At baseline updates100,150,175,200,225,250,275,300,350, measure the fixed-probe
objective before/after the actual accepted update. For actual rounded full
displacement s=theta_after-theta_before, including decay:

```text
D = F(theta_before) - F(theta_after)
P = -<gradient F(theta_before), s>
R = P - D
D = P - R
alignment = cosine(gradient F(theta_before), -s)
```

P is first-order predicted decrease. R is the exact nonlinear residual relative
to that prediction; it includes curvature and higher-order effects and is NOT
a measured Hessian or directional-curvature estimate. No Hessian approximation
is needed for this first test. Loss decrease is measured on the SAME fixed data
at both endpoints. The source training loss still comes from changing minibatches.

Probe forward/backward use FP32 with math SDPA, TF32 disabled, token weighting
and microbatch1; scalar products/norms use FP64. This objective/precision is fixed
across every counterfactual. Probes preserve model mode, outstanding training
gradients, optimizer state and CPU/CUDA/Python/NumPy RNGs. The native accepted
parameter tensors are restored exactly after every one-step counterfactual.
GPU calibration must qualify the cost/VRAM of this diagnostic precision.

The measured objective and gradient have finite precision; very small nonlinear
residuals can reflect evaluation/derivative error. The reconstruction controls
quantify numerical application differences rather than proving an exact
continuum curvature identity.

All unique trainable parameters are included in fixed named-parameter order.
Each accepted update also records raw-gradient, actual stored first-moment m,
canonical Adam-direction d, and actual full-displacement norms/adjacent cosine.
Raw capture is before clipping; moments use the clipped gradient. d is evaluated
in FP64 from actual updated FP32 moments, including bias correction and epsilon
outside the square root, while excluding LR, decay and subtraction rounding.
Raw/m/d cumulative A/B/B-over-A are retained through the existing tested helper.
No display smoothing is applied to the measurements. First-pair/zero-norm
cosines are null with reasons. Zero counterfactual directions that cannot match
a nonzero target are explicitly undefined and their coverage is reported.

## Controlled direction substitutions

At each baseline measurement, keep theta_before, minibatch, clipped gradient,
updated moment tensors and the Adam step clock fixed. Evaluate seven outcomes:

| Candidate | Adaptive direction before norm matching |
| --- | --- |
| Native AdamW | Actual fused optimizer result; rounding control |
| Reconstructed Adam | m_hat/(sqrt(v_hat)+epsilon) |
| No preconditioning | m_hat |
| No first-moment averaging | current_clipped_gradient/(sqrt(v_hat)+epsilon) |
| Neither component | current_clipped_gradient |
| Half step | Reconstructed Adam with0.5 times baseline adaptive-step norm |
| Double step | Reconstructed Adam with2 times baseline adaptive-step norm |

The four full-length candidate directions use the native proposal's nominal
adaptive-step norm eta*norm(d), WITHIN each model/checkpoint. The identical
decay displacement is then applied before FP32 parameter rounding. A native
versus reconstructed control exposes numerical application differences: require
maximum parameter error<1e-6 and fixed-probe loss difference<1e-5. Candidate
comparisons use the common reconstructed application rule. This is a one-step
direction substitution, not resetting Adam moments or running beta1=0 training.
Equal absolute Euclidean step norms across different architectures are not
asserted to be equivalent interventions.

## Continuations and causal estimand

Each50-update continuation retains actual moments, RNG/data position, original
LR/decay law and clipping, but replaces the adaptive component by normalized
m_hat. Its target norm at each global update is taken from the already completed,
unmodified parent trajectory. This is a controlled preconditioning ablation,
not an ordinary SGD-with-momentum recipe or a deployable optimizer claim.
The norm schedule is set from the parent, never selected using the probe.
Decay coefficients/groups remain identical; after trajectories diverge the
decay vector naturally depends on their respective parameters.

At each checkpoint, the baseline and treatment begin at exactly the same F.
For each size, seed and junction, the primary treatment effect is:

```text
tau = log(F_ablated at junction+50) - log(F_native at junction+50)
size interaction = tau_70M - tau_14M   (31M is an intermediate comparison)
```

Positive tau means removing preconditioning harms progress. Evidence supporting
a preconditioning contribution would combine a repeatable size-dependent tau,
the predicted changes in local P/alignment or R, and reduced70M advantage under
the controlled intervention. Equal harm across all sizes does not explain the
size gap. Lack of an effect, reversed effects, or local accounting inconsistent
with the proposed explanation are retained as refutation/inconclusive evidence.
Compare175/225 junctions and earlier100/150 one-step controls. Loss-matched local
comparisons are secondary and only use overlapping probe-loss coverage.

In an ablated trajectory, the `adaptive` stream remains the canonical Adam
proposal computed from its actual moments; it is not the applied adaptive vector.
The `intervention` stream records the imposed adaptive norm and `actual_parameter_update`
records the actual full rounded displacement. Descent P/D/R always use that
actual displacement. Do not relabel canonical d geometry as the applied update.

Three seed blocks are the replication units. Updates, overlapping intervals,
candidate directions and probe tokens are not independent seed replications.
Report paired effects by seed and their spread; do not claim a scaling law,
asymptotic convergence, stability, a complete capacity mechanism or causal
mediation solely from a correlation. Architecture/parameterization and learned
state differ across sizes; interventions identify conditional optimizer effects.

## Implementation, verification and retention

`00_prepare.py` created the unexecuted config/model-config pairs and probe.
`src/baseline/descent_diagnostics.py` and `causal_training.py` are shared because
all27 runs need them; they reuse the existing native baseline update, initialization,
LR, data and geometry helpers. Original executed sources and artifacts are unchanged.

Local tests cover quadratic D/P/R, independent moment/direction semantics,
decay-group preservation, zero directions, token-weighted tail handling,
diagnostic RNG/gradient/parameter/optimizer noninterference, exact stochastic
recovery, and a complete tiny baseline/continuation/artifact-verification smoke.
A smoke establishes mechanics only. The production-shaped paid GPU calibration
is still pending and will precede the main queue.

`02_worker.py` validates source/input hashes, runs a3-update largest-model
calibration per worker, gates memory and refreshed ETC, then executes assigned
families detached. `03_verify.py` checks coverage, parent hashes, norm schedules,
counterfactual controls, descent identities and scientific completeness.
`tools/artifact_inventory.py` adds per-file size/SHA-256 inventories.

Retain frozen inputs; all scalar/probe events; native environment/config/input
identities; fork175/fork225 recovery checkpoints; and terminal recovery states.
States include parameters, optimizer/scaler, raw/m/d streaming diagnostics,
previous actual displacement, RNGs, update/data position and config/order hashes.
A rolling recovery state is refreshed every50 updates and is removed only after
terminal scientific verification supersedes it. A retry preserves the old attempt
and copies its accepted event prefix into a new attempt. Unexecuted descendant
paths must be resolved/frozen against the accepted parent before launch.
Expected retained checkpoint storage is about104GB, with a120GB allowance;
the new200GB network volume provides working/retry/input headroom.

## Launch proposal and spending gate

[Live read](live_price.json): Secure RTX4090 costsUSD0.89/GPU-hour, with LOW stock
in EU-RO-1, including the two-GPU availability query. The official catalog lists
`runpod-torch-v280`, image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`.
Use two one-GPU workers in that data center, SSH22 only,30GB container disk each,
and a NEW task-owned200GB STANDARD network volume mounted at `/workspace`.
No existing pods/volumes or Spark resources are reused or modified.

`bash setup.sh 0` / `bash setup.sh 1` creates Python3.12 and pins torch2.11.0+cu128, transformers5.12.1 and
numpy2.5.0 to the reference/tested versions. NumPy2.5 requires Python>=3.12;
package availability was checked. Setup/test sources are frozen in the packet.
The dependency lock and actual driver/runtime are recorded on each worker.
Dependency-lock and smoke-test output paths are distinct between workers;
shared source/input files are read-only once staged.

[Launch plan](launch_plan.json): historical native time is9.665 GPU-hours for
4050 updates. LPT balances family work across two workers at4.608/5.057 historical
hours. A1.5 diagnostic allowance plus0.75h setup and1.5h retrieval reserve gives
a qualified estimate of9.84 elapsed hours. New FP32-probe cost is unmeasured;
production calibration must fit the remaining deadline or its worker stops.
This estimate is not a newly measured throughput guarantee.

Proposed approval: **USD25 aggregate, at most12h elapsed, at most2 concurrent
RTX4090 GPUs**, including calibration/retries and storage. GPU rate isUSD1.78/h
for both. [Runpod storage pricing](https://docs.runpod.io/pods/pricing) adds
USD0.07/GB/month for the network volume andUSD0.10/GB/month for running container
disks; estimated combined rateUSD1.8074/h and12h totalUSD21.69, leaving margin.
Storage accrues until deletion. Prices/stock are rechecked at allocation.
No permission is inferred from account balance or prior experiments' budgets.

Launch the separate local spending guard immediately after the first allocation;
it stops only the IDs/names in this task's lease by the aggregate cost/time cutoff.
Main training is bounded to10.5h from first allocation, reserving90min for retrieval.
Use hidden background processes on Windows; remote jobs use setsid and persistent
network-volume logs. Poll read-only every5min: update/tokens/loss, native/probe
times, refreshed ETC, memory, disk and event age. Failed qualification stops that
worker, and its network-volume evidence survives; scientific inputs are not
silently reduced to fit. No duplicate worker start is permitted.

`04_control.py` monitors and retrieves completed runs by individual SCP files,
avoiding an additional100GB local archive. It verifies sizes/SHA-256 and scientific
results locally before marking each accepted. All27 accepted receipts are required
before deleting only this task's workers and new volume. An independent deadline
stop preserves the network volume if retrieval is incomplete. Reconcile final
provider billing, marking estimates provisional until posted.

The user approved this packet and its USD25/12h envelope. Allocation began
October17 at15:35:31 UTC; the hard stop is October18 at03:35:31 UTC. The
independent guard is active. `07_stage_and_launch.py` verifies/extracts the
archive, checks successful setup, starts both detached workers, then runs the
retrieval/cleanup controller. After all27 runs are verified locally and cloud
cleanup succeeds, `08_analyze.py` generates the paired effects, descent-accounting
figures and local observation. Scientific results remain pending execution;
new figure layouts were checked using explicitly synthetic scratch data.

The original source packet and its inventory remain frozen. Setup recovery
added pip with `python -m ensurepip --upgrade`, resumed the original pinned
install lines, created the missing `.pytest_tmp` parent, and reran the unchanged
remote tests:10 passed per worker. Failed/successful setup logs and recovery
scripts are retained locally under ignored scratch with [hashes](setup_evidence.json).
No model, optimizer, data, diagnostic or run-config input changed.
