# Controlled full-batch Adam extension

Current user instruction: run the reduced calibration first, one frozen
initialization per case, no Hessian/HVP measurements. The current operational
packet is [FAST_CALIBRATION.md](FAST_CALIBRATION.md); it supersedes the original
21-arm duration estimate. Main training is deferred until calibration review.

## Question and comparison

Describe how raw gradients, first moments and adaptive directions differ along
controlled full-objective Adam trajectories, and whether their geometry changes
between smooth-progress and oscillatory-progress LR conditions. Measure loss
alongside geometry. Determine updates needed to reach historical GD loss
milestones, rather than assume Adam converges faster. These are descriptive,
initialization-conditional questions (H027), without diagnostic-driven LR
changes during a trajectory.

Vary only constant Adam LR within each case. Match the exact seed0 frozen
initial weights, prepared data/order, objective, fixed BN groups, parameter
scope, precision and accumulation partition across rates. Adam moments and clock
start at zero. One initialization per case is the requested study; uncertainty
across initializations is unmeasured. No additional seeds are planned.

| Case | Full objective per update | Numerical partition |
| --- | --- | --- |
| CIFAR-10 / tanh200-200 MLP | Same first5,000 images; half squared error summed over10 outputs, averaged over images | Fixed chunks1,000, qualified legacy MLP reduction |
| CIFAR-10 / ResNet-32 | All50,000 images; mean cross entropy | Same50 fixed ghost-BN groups1,000; train-mode statistics, no running-buffer mutation |
| WikiText-2 / 2-layer Transformer | All59,680 prepared causal sequences,2,088,600 valid targets; token-weighted NLL | Original chunk64; length35, padding excluded |

Reuse Analysis024's prepared bytes and seed0 tensors, verified against input
manifest SHA256 ef6f5b79249c05b1deda802d8b24c55e186c75825bc80f21a0c6f66c829d729f.
Run213's portable input manifest pins source revisions, file hashes, model
sources, vocabulary construction and initial tensor identities. The legacy
WikiText vocabulary uses the train/valid/test first-occurrence union; preserve
and disclose it. Original batchify exclusions are8/6/9 train/valid/test tail
tokens and20/10/10 stream-initial tokens without a target. No extra truncation.

Use ordinary Adam beta=(0.9,0.999), epsilon1e-8, no decay, clipping, AMSGrad,
warmup or schedule, foreach=False and fused=False. This matches Analysis034's
optimizer, not Pythia's AdamW beta2=0.95/clipping/decay recipe. Comparisons with
Pythia also differ in architecture, objective, data, batch noise and precision;
they are not optimizer-only ablations. FP32 parameters, moments and gradients;
TF32 disabled; FP64 diagnostic vectors/sums/reductions. Changed hardware or
accumulation partition needs recorded start-gradient qualification. Matched
initial tensors do not imply bitwise cross-hardware replay.

## Calibration and duration

Analysis034 tested five rates for64 updates from GD-trained states, not
initialization. Selected rates were3e-4 for MLP and the upper boundary1e-3 for
ResNet and Transformer. This establishes neither a sufficiently large upper
rate nor training-from-initialization regimes. Its calibration PDF was rendered
and reviewed; original decisions are unchanged.

Run213 now executes 128 full updates at each of three rates per case:
MLP 1e-4/3e-4/1e-3; ResNet and Transformer 3e-4/1e-3/3e-3.
This is nine arms and 1,152 updates, plus one unapplied terminal gradient per arm.
Independent arms reset the model, Adam and diagnostics. Nonfinite arms and arms
reaching the 10x initial-loss execution guard remain failed/guarded outcomes.
A guard stops an arm; it never reduces LR or substitutes a rate. Per-arm runtime
bound is one hour, within the aggregate 90-minute / USD 20 calibration envelope.
No full main-study training is admitted by this calibration controller.

Report continuous statistics before descriptive labels. On the last64 losses,
report upward-step fraction (rise >1e-5 relative to preceding loss), median
absolute relative loss change, min/max, final mean/initial, full peak and startup
peak. Smooth progress: final mean <=95% initial and upward fraction <=5%.
Oscillatory progress: upward fraction >=25%, median relative change >=0.001
and final mean <=initial. These are operational screening rules, not stability
or convergence definitions. Keep raw loss visible; borderline cases remain
unclassified. Cosine does not define labels, so agreement is not built in.
The final 64 window excludes the first 64-state startup interval at horizon128.

Propose the lowest smooth-progress LR and highest oscillatory-progress LR;
include the completed interior rate nearest their geometric mean. If the grid
does not span the requested regimes, report the missing regime. No automatic
grid expansion, optimizer change, exclusion or replacement experiment.
Reassess provisional labels over the longer trajectory.

After screening, propose three fixed-rate1,024-update trajectories per case,
with separate frozen production IDs:9,216 updates, not an assumption of
convergence. Restart from the same initialization for complete-prefix capture.
Prefix reuse requires explicit provenance/replay qualification and a
continuation plan. Further2,048/3,000-step continuation is a subsequent
horizon/budget decision. Beta2=0.999 has a long startup memory; short loss
improvement does not imply settled moments or asymptotic behavior.

Freeze GD full-objective losses at states500/1000/2000/3000 for all nine
Analysis024 conditions as milestones before new Adam results. Report first
attainment and first64-state-mean attainment, with unachieved thresholds
right-censored at the horizon. Report updates, examples/targets and measured
wall time. Do not select an easier GD rate after seeing Adam. Keep endpoint
loss at matched horizon. Loss matching does not match location or establish
generalization. Speed comparisons remain conditional on rates and one seed.

## Measurement at every optimizer boundary

Let theta_i precede accepted update i (zero-based), g_i be its full raw
gradient, m_i the stored first moment after consuming g_i,
d_i=mhat_i/(sqrt(vhat_i)+epsilon), and Delta_i=theta_(i+1)-theta_i.
First moments use raw gradients directly because there is no clipping.
Every unique trainable parameter contributes once in model order.

For x in {g,m,d,Delta}, record norm, adjacent cosine, A=sum(norm(x)),
B=norm(sum(x)), R=B/A, energy and exact all-pair/lag-one/beyond-lag-one terms.
Prefixes start at initialization and retain all accepted updates. Gradient,
moment and adaptive prefixes describe their named streams. Only Delta provides
actual parameter displacement and path length. Check telescoping against
theta_T-theta_0 and rounding residual relative to -eta*d. Constant LR makes
adaptive and ideal-step retention equal; rounding can change actual-step
retention. Their near-agreement is not independent scientific evidence.

Record stored m and bias-corrected mhat norms, m/g and mhat/g norm ratios,
cos(g_i,m_(i-1)), cos(g_i,m_i), cos(g_i,d_i), cos(g_i,Delta_i),
g_i dot Delta_i, step-rounding residuals and moment age. The first incoming
prior-moment comparison is undefined. Bias correction changes norms and time
weighting although it preserves individual directions. Cumulative retention
for stored m cannot silently represent the bias-corrected stream.

Record q_i=1-<g_(i-1),g_i>/norm(g_(i-1))^2 pointwise before smoothing.
This is exactly normalized projected gradient change. For nonzero norms,
q_i>1 is equivalent to negative cosine, not independent corroboration.
For a deterministic differentiable objective,

    g_i-g_(i-1) = integral_0^1 H(theta_(i-1)+s*Delta_(i-1)) Delta_(i-1) ds.

Thus q observes the integrated Hessian along the applied step through a mixed
projection. With Delta approximately -eta*d, q/eta approximates
g_(i-1)^T H d_(i-1)/norm(g_(i-1))^2 locally, including finite-step remainder
and rounding. It does not estimate g^T H g/norm(g)^2, lambda_max, or an Adam
stability threshold. Negative raw cosine can coexist with decreasing loss.
Cosine alone cannot identify flat, sharp, indefinite or beneficial regions.

No Hessian, HVP, eigenspectrum or curvature-probe measurement is part of the
requested study. The equations above explain an interpretation limit only.
The recorded q is a gradient-change diagnostic and will not be relabeled as
identified curvature or an Adam stability threshold.

Zero denominators yield null, without artificial epsilons. Exact cancellation
at positive A gives R=0. First q and adjacent pairs are undefined. The terminal
theta_T gradient is unapplied: boundaries cover0..T, while accumulated streams
cover0..T-1. Logs distinguish loss state and completed updates. Nonfinite steps
are never silently accepted as ordinary diagnostic updates.

## Evaluation, retention and figures

Record raw full training loss every boundary. To accelerate this training-only
calibration, skip validation/test evaluation entirely; no generalization claim
is made. The later main study will declare full prepared-split evaluation:
MLP's entire test cache, ResNet all10k test images and WikiText valid/test
217,630/245,550 targets. Evaluation never selects LR. Freeze production
evaluation cadence before launch. Show raw full-batch loss/cosine; explicitly
label any trailing64-state summary. Compute pointwise q before smoothing.
Cumulative ratios are unsmoothed.

Planned figures follow Analysis029: (1) three model columns showing loss, raw
cosine, projected change and gradient/moment norms across LRs; (2) adjacent
cosines and cumulative retention for g/m/d/actual steps; (3) incoming-gradient/
prior-moment and raw/adaptive alignment with loss; (4) GD loss milestones versus
updates/exposure. No schedule-response panel or intervention experiment.
Each future publication PDF needs local sources, hashes, exact plotted arrays,
reduction/caption, coverage, uncertainty, caveats, numerical verification and
visual review. No empirical figure comes from the mechanics smoke.

Recovery every64 updates includes model, Adam, four previous and FP64 prefix
vectors/scalars, initial parameters, optimizer clock and Torch CPU/CUDA RNG.
Full-batch order has no shuffle/cursor; no scheduler/scaler. Keep one rotating
recovery, renamed final at completion, all scalar logs, source/environment
identities, and raw audit gradients at0/1/T. Dense gradient history is not
retained; arbitrary retrospective lag profiles cannot be reconstructed.
Preserve all historical runs and evidence.

## Execution packet

Implementation/input verification, eleven focused tests and a fresh two-update
MLP smoke pass. The user accepted the prior cost quote and requested calibration
first. The current packet bounds this stage at USD20 inside the quoted USD50
overall envelope. Live placement, execution and cleanup are recorded separately.

Execution uses the owned RTX5070Ti laptop for all MLP rates and matched Runpod
RTX5090 class for every ResNet/Transformer rate. Actual catalog price was
USD1.19/GPU-hour. The pinned official PyTorch2.11/cu130 runtime and native
runpodctl3.0.1 were read live. All setup/retry cost counts toward the same cap.
Historical Analysis034 prices/resource IDs were not used as current facts.
The controller and independent guard persist local logs; cloud training is
detached. Compact receipts record verified retrieval before worker deletion.

ETC must sum each task's three arms, nine terminal gradients, checkpointing,
setup and transfer. There are no split evaluations in calibration. Four FP64 previous/prefix streams use about
64 bytes/parameter (0.885GB Transformer), before model/moments/temporaries.
Transformer final recovery is approximately1.16GB/arm plus audit vectors.
Reserve10GB retained local storage for this reduced calibration.
Each worker also needs image/packages/cache and atomic checkpoint replacement
space. Production has a separate packet after rate selection.

Run detached with persistent stdout/manifest/events and an external aggregate
cost/deadline guard. STOP, per-arm deadline and SIGINT/SIGTERM request boundary
stops without changing LR. Read-only monitoring every10min reports completed
updates/tokens, loss state/value, throughput, ETC, peak memory and failed/guarded
outcomes. Reserve checkpoint/retrieval time in deadlines. Verify every retrieved
file's bytes/SHA before terminating only task-owned workers; remove their storage
unless explicitly retained. Report retained storage and provisional/posted
billing. No push or manuscript change.
