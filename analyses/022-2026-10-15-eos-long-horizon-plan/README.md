# EOS study: duration, replication and compute plan

**Follow-up:** the user chose a staged3,000-update smallest-LR calibration on
owned Spark instead of launching this larger grid or cloud proposal. See
[Analysis023](../023-2026-10-15-eos-smallest-lr-calibration/README.md).
The priced scenarios below remain planning history, not accepted spending.

**Status: prospective design and verified cost arithmetic, not new training
results.** Requested October 15, 2026. No cloud resources created or changed.
Analysis020 continues; Analysis021's unstarted extra-LR queue remains held.
The user's dictated “Run” / “branch pod” is interpreted as Runpod.
Run120 is still the next unallocated run; this analysis allocates no run IDs.

The recommendation is to treat 500 updates as a pilot and extend the main GD
comparison to the horizons below. Duration should expose the transition and its
persistence, rather than merely match a paper's iteration count. The 3,000-update
larger-model scope remains a partial-trajectory study, not an asymptotic claim.

## What an iteration means

One GD iteration is one parameter update using the complete declared training
objective, as in [Cohen et al., Appendix I/L](https://arxiv.org/html/2103.00065v3).
Physical gradient accumulation batches are not iterations. For ResNet this is
50 fixed 1,000-image ghost-BN groups averaged before one update. For the
Transformer this is 59,680 prepared sequences / 2,088,600 target tokens before
one update. One mini-batch SGD iteration instead processes its sampled units.
Thus equal SGD update counts at different batch sizes do not match data exposure.

The EOS study includes WikiText-2/Transformer, not only CIFAR-10. Appendix L.1
explicitly reports **15,000 GD iterations** for that Transformer experiment.
We preserve our documented reproduction differences rather than claim an exact
replication. See Analysis020's protocol and source review.

## Main full-GD comparison

| Case | Constant learning rates | Updates per run | Seeds | Conditions |
| --- | --- | ---: | --- | ---: |
| CIFAR-10 / tanh 200-200 MLP, first 5k images | 0.004, 0.0066667, 0.01, 0.02 | 20,000 | 0, 1, 2 | 12 |
| CIFAR-10 / ResNet-32, all 50k images | 0.1, 0.2, 0.4 | 3,000 | 0, 1, 2 | 9 |
| WikiText-2 / two-layer Transformer, complete prepared corpus | 0.05, 0.1, 0.2 | 3,000 | 0, 1, 2 | 9 |

This is **30 conditions**, including four already complete MLP seed0 runs;
**26 new trajectories**. Adding the existing intermediate MLP LR is cheap and
specifically tests the delayed transition. Executed Runs059-062 are reused only
after initialization, data, objective and implementation parity checks. They
are not four additional independent seeds. All seeds share the same dataset;
each seed's exact initialization is matched across its LRs. The reused MLP
solver requested two eigenvalues; use the largest with the same residual check
and disclose the solver/runtime differences.

Plain GD has no momentum, clipping, weight decay, schedule, augmentation or
dropout. Keep current loss reductions, normalization, ghost groups and physical
accumulation. Use frozen initial tensors, FP32 parameters/gradients, TF32 off,
and FP64 geometry. Do not change batch-normalization grouping for speed.

**Why these horizons:** retained 20k MLP results first show negative adjacent
cosine at updates 1,156 / 4,565 / 9,467 for LRs 0.02 / 0.01 / 0.0066667;
LR0.004 has none within20k. The central-LR 500-step pilots first cross at109
for ResNet and456 for Transformer. At3k, these central trajectories would have
roughly2,900 and2,500 further updates to examine persistence. First single sign
changes are descriptive events, not sustained-regime onset. Other LRs/seeds
may cross later or not cross at all; retain and report these outcomes.

Within a case, all LRs/seeds receive the same horizon; never extend only an
interesting trace or discard a divergent condition. If3k proves inadequate,
report the limited window and propose a separately recorded extension, rather
than silently change executed inputs. A longer horizon alone does not make
the evidence robust.

One Transformer trajectory at3k processes **6,265,800,000 target tokens**;
the nine conditions process56.39B. This full-corpus cost explains why copying
15k across the whole grid would be a substantial extra experiment.

## Measurements and evidence for the claims

Record unsmoothed full training loss, raw norm, adjacent cosine, A, B, B/A,
energy, exact all-pair interference, lag-one/beyond-lag-one contributions,
parameter displacement and floating-point identity residuals **every update**.
Keep the current convention A=sum(norm(g)), B=norm(sum(g)); never invert B/A.
Record q=eta*kappa_hat=1-rho*norm(g_t)/norm(g_prev), undefined at zero norm.
For deterministic GD q is an exact projected finite-step gradient change;
its interpretation as local directional curvature has a Taylor remainder.
Its algebraic connection to rho is not independent confirmation.

Measure the largest **algebraic** Hessian eigenvalue every100 updates, including
the initial/final states, and verify relative eigenpair residual<=0.005 with
the existing declared solver/fallback. MLP uses its full5k objective; current
ResNet spectra use5k images and Transformer spectra2,500 sequences. Label
these last two as **subset Hessians**, not exact full-objective sharpness.
Include a production-shaped full-objective HVP/eigenvalue timing probe in the
cloud calibration. Before freezing the new configs, resolve an affordable
full/subset audit at common selected states or narrow the sharpness claim;
its cost must fit the allowance, not be presumed zero. Full spectra would cost
more than the present estimates. ReLU also limits a smooth local interpretation.

Prespecified summaries: first negative pair; first ending index of100
consecutive defined pairs with at least80% negative cosines; negative-pair
fraction and mean rho/q in the final1,000 pairs; sampled eta*lambda_max;
final B/A and change over the final1,000 updates; exact lag-one approximation
residual relative to cumulative energy. If a sustained transition has not
occurred, report it as unobserved within the horizon. Keep all three seed
traces and per-seed summaries; a seed range is descriptive, not a reliable
population confidence interval. Do not pool temporal samples as replicates.

Main paper output: the requested3x3 loss / eta*lambda_max / eta*kappa_hat
figure, columns by case and colors by LR, followed by cosine and normalized
displacement comparisons. Use raw traces with clearly labeled optional means;
keep spectra as measured markers. Only replace the current pilot figure after
all conditions pass provenance, numerical, coverage and visual checks. No
manuscript change is made by this planning analysis.

## Separate noise study; avoid the full Cartesian product

First finish the already running Analysis020 SGD screen. Keep the main LR
ablation full-GD; do not multiply every LR by every batch and long horizon.
The proposed next noise stage uses the central LR per case and three seeds,
with the existing three batch sizes and full-GD reference. Its training
horizon/cost are **not yet frozen or included in the GD budget**.

Before moving-trajectory inference, repeat frozen-state batch sampling at
initial, pre-transition and post-transition states selected from the central-LR
GD trajectory using the declared transition criterion. If no transition occurs,
use fixed fractions of the horizon and label them. At each fixed state compute
the complete mean gradient mu, realized xi=g_batch-mu, and replicated batch
variance. For K independent sampled batches, estimate
V_b=sum(||g_batch,k-mean(g_batch)||^2)/(K-1), retaining the actual sampler and
its finite-population correction where applicable. Never infer per-sample
energy/alignment from a batch-mean gradient alone. Keep probe and training RNGs
separate. Existing frozen controls suggest attenuation, not a universal collapse
of rho to zero (the ResNet small-batch control is still strongly aligned).

For moving SGD trajectories, show both update index and cumulative sampled
images/target tokens. Prespecify a primary matched-data-exposure comparison and
an additional matched-update window; unequal batches cannot match both clocks.
Choose the exposure horizon after the screen reveals transitions and measured
probe overhead, then freeze it before confirmatory repetitions. Paired full
gradient references at consecutive sampled states permit the realized noise
term in the identity; a simple observed batch-size trend is not a noise-only
causal comparison because parameters follow different trajectories.

Claims remain conditional: noise may reduce coherent accumulation and adjacent
alignment, but no monotonic effect, loss improvement, generalization result or
stability guarantee is assumed. These controls and the long-lag residual matter
more for a technical paper than adding every possible model/seed combination.

## Execution, runtime and cost

The reproducible estimate is [estimate.json](estimate.json), generated by
`python analyses/022-2026-10-15-eos-long-horizon-plan/00_estimate.py`.
It uses complete500-update production times, not parameter-count heuristics:
approximately2h/ResNet and4.45h/Transformer; MLP20k takes about5-6min.
New work is **348.94 Spark GPU-hours**, about14.5days of sequential training,
or17.4days with a20% planning allowance. This starts after any current Spark
queue; it does not mean the existing500-step queue needs two more weeks.

Runpod catalog read2026-10-15 at11:22UTC: Secure RTX4090 **$0.89/GPU-hour**,
24GB, LOW two-GPU capacity in EUR-IS-1/EUR-IS-2/US-IL-1. A two-GPU5090
configuration was unavailable in that read. No reservation is implied. The
current EOS calibration uses less than8GB reserved, so24GB fits the measured
recipe; verify cloud production/HVP peaks before launch. Do not pay forH100
capacity without a demonstrated throughput benefit.

Use two independent trajectory workers on two4090s, not distributed training
of one tiny model. Keep each full ResNet/Transformer trajectory on one matched
cloud runtime from the frozen initial state; do not splice a Spark prefix onto
a cloud continuation. Benchmark both workers under concurrent load. Legacy
MLP reuse stays explicitly labeled; cheap new MLP conditions can use Spark,
although the conservative cloud cost below also includes them.
Schedule the remaining seed0 LR conditions first, then seeds1/2, keeping whole
trajectories intact. This surfaces unsuitable LR/numerical behavior early;
failure is a reported outcome, not a reason to silently remove a condition.

| Assumed measured speedup after calibration | Two-GPU planning wall time | Priced compute/storage estimate |
| --- | ---: | ---: |
| 2x Spark | 109.3h /4.6days | $196.73 |
| 3x Spark | 73.5h /3.1days | $132.89 |
| 4x Spark | 55.6h /2.3days | $100.97 |

**These speedups are scenarios, not measurements or promised ETCs.** The
calculator schedules whole jobs longest-first, adds20% and2h for setup/
retrieval, and bills the entire two-GPU window conservatively. If cloud is only
as fast as Spark the scenario exceeds the proposed envelope and is rejected.
Sparse Hessians alone will not deliver these speedups: complete-gradient
evaluation dominates GD, especially the Transformer.

Proposed ceiling: **120h fleet wall time, $220 total USD**. At quoted rates,
two GPUs for120h plus100GB network storage for at most7days and40GB total
container disk cost$215.90 before tax/currency conversion. Initial30min on
two GPUs costs$0.89 compute and is included, not additional authorization.
No GPU/data-transfer performance assumption is treated as a current result.
Storage rates follow [Runpod pricing](https://docs.runpod.io/pods/pricing);
capacity, deployment/storage quote and transport must be checked again at launch.

Calibration must measure warmed complete updates, concurrent worker throughput,
existing subset spectra, a full-objective HVP audit, evaluations, saves, memory
and transfer. Only launch the full grid if the refreshed schedule, diagnostic
scope and retrieval allowance fit both limits. If not, report the shortfall
and revise the design with the user; do not spend up to the cap on a knowingly
incomplete grid. A proposed scientific full-Hessian audit needs its own
measured allowance before it becomes a promised output.

Persist logs/manifests/events and recovery state (optimizer, RNGs, previous
gradient, full-prefix accumulators) to the100GB volume. Keep rolling recovery
every100 updates; retain listed checkpoints and final state, not dense gradient
histories. Allow100GB cloud and40GB local retrieval; verify free space first.
Poll read-only every5min; refresh loss, step, throughput, ETC, memory and spend.
Warn on stalled events, nonfinite metrics, failed eigenpairs or forecast overrun.
A detached guard must enforce the deadline/budget; allow2h at the end for
retrieval. Reserve the complete scientific inventory before allocating compute.

Verify byte sizes and SHA256 locally before deleting task-owned workers.
Retain network evidence only until verification, at most7days within this
proposal; complete verified retrieval before that deadline. Do not leave
stopped Pods/disks billing indefinitely or touch unrelated account resources.
If storage must outlive the envelope, arrange that before expiry. Record actual
cleanup and provisional vs posted billing. Changed horizons require new run
IDs/configs; executed500-step inputs stay immutable. Current resume code requires
identical config hashes, so a continuation would need explicit parent lineage
and reviewed guards; the default here is fresh matched trajectories instead.

Cloud spending approval is pending. The request is for the missing envelope
required by `AGENTS.md` and `research/WORKFLOW.md`, not a second approval of
already authorized Spark work. No paid calibration has been started.
