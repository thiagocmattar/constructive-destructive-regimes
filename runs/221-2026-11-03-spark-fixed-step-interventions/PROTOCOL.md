# Fixed-step highest-LR intervention responses on DGX Spark

User authority: run literature-motivated intervention tests at the highest LR
for all three existing model/dataset cases on the idle Spark until tomorrow
morning. The working plan selects one common fixed checkpoint, not a diagnostic
trigger. H029 / Run221 / Analysis043. Report target2026-11-04 08:00 BRT;
compute stops by07:45 BRT (10:45 UTC). Owned Spark only; zero cloud spending.

The initial planning boundary512 with512 new updates does not fit measured Spark
costs. Before any production, the final boundary is1024 and horizon64, ending1088.
Use authenticated Run215 final weights, both Adam moments, global age and RNG.
Existing scalar histories show the before period; contemporary Spark controls
supply the untreated counterfactual. This selects a trained state, not a new
loss-event boundary. It does not retrospectively intervene on the earlier
703/422/794 peaks. Warm-parent data/gradient and within-Spark replay are qualified.

## Matched inputs and budgets

| Case | Base LR | Every optimizer update | Physical partition |
| --- | ---: | --- | ---: |
| MLP/CIFAR-10 | 0.001 | Same5,000 images; half squared error summed over10 outputs and mean over images; tanh3072-200-200-10 | 1,000 images |
| ResNet-32/CIFAR-10 | 0.003 | Same50,000 images; mean CE; same50 fixed ghost-BN groups, no running-buffer mutation | 1,000-image group |
| Transformer/WikiText-2 | 0.003 | Same59,680 prepared causal sequences/2,088,600 targets; complete token-weighted NLL, zero dropout | 64 sequences, length<=35 |

Seed0; pinned model/data/normalization/order/initialization provenance. Source
metadata retains the historical vocabulary and preprocessing disclosures.
FP32 parameters, raw gradients and Adam moments; FP64 products/reductions;
TF32 off, foreach/fused false, math attention, deterministic algorithms and
cuDNN settings. Original accumulation partitions are retained. Parent tensors
are exact copies of their trusted sources. Source input hashes are reverified
on Spark before production. Branches share the same machine and initial state.

Native Adam betas(0.9,0.999), epsilon1e-8, no pre-existing decay/clipping/schedule.
No cold optimizer starts or changed moment clocks. All methods measure the same
unregularized task objective. Any coupled L2-gradient penalty is excluded.

## Primary grid: ten conditions per case

1. Unchanged Adam, the matched counterfactual.
2. Fixed half LR.
3. Fixed quarter LR.
4-6. One cosine decay from base LR to0.5/0.25/0.1 times base over32 updates,
then hold that absolute floor through the64-update horizon. With k=state-1024,
eta(k)=eta0*[r+(1-r)*(1+cos(pi*min(k,D)/D))/2]. k=0 uses base LR and k=D
reaches the floor. No restart or state-dependent trigger.
7-8. Clip the full accumulated raw gradient globally to C or2C,
C=NumPy median of that case's raw norms at states960..1023. Apply one scalar
factor min(1,threshold/norm(g)) after complete accumulation and before both
moment updates; a zero raw gradient remains zero. Thresholds are fixed before
intervention outcomes. Preserve raw g separately from clipped optimizer input h.
9-10. Decoupled decay lambda0.01/0.1 on tensors with ndim>=2; skip biases and
normalization vectors. After computing the task gradient, apply
p <- (1-eta*lambda)*p to selected weights and then the native task-gradient Adam
update. On identical input gradients this matches native AdamW exactly in the
focused warm-state test, including first/second moments. Decay affects later
states and therefore future task gradients. Record its actual increment separately.

Two optional conditions use the same cosine floor0.25 with durations8 and64.
Admit an entire additional condition across all three cases only after all primary
branches, and only if the measured projected cost times1.25 plus300 seconds fits
before the compute deadline. Admission depends on time, never method outcomes.
All cases retain the same64-update horizon for these optional comparisons.

Ratios, clipping factors and decay strengths are exploratory fixed choices, not
claims of optimal or literature-prescribed settings. Fixed LR controls share
cosine floors but not total LR exposure; a competitive schedule/timing claim
would need matched exposure and tuning controls.

## Literature motivation

[Pascanu, Mikolov and Bengio,2013](https://proceedings.mlr.press/v28/pascanu13.html)
propose global norm clipping to address exploding gradients in RNNs. We apply
that operator to full-objective Adam inputs here as a controlled transfer test,
without assuming its benefit transfers to these models.
[Zhang et al.,2020](https://arxiv.org/abs/1905.11881) motivate adaptive scaling
under gradient-dependent smoothness; their theorem assumptions are not asserted
for these neural objectives.

[Loshchilov and Hutter,2019](https://arxiv.org/abs/1711.05101) distinguish
decoupled weight decay from adding L2 terms to Adam's gradient. The chosen
matrix/tensor mask is explicit; the optimizer continues consuming task gradients
into its moments. Any change in loss/geometry is compared with ordinary Adam.

[Loshchilov and Hutter,2017,SGDR](https://arxiv.org/abs/1608.03983) motivate
cosine LR modulation. This suite uses a single cosine decay without warm
restarts, testing several floors and, if time permits, durations. It does not
claim to reproduce the complete SGDR training recipe or its reported outcomes.

## Qualification and artifact contract

The first mechanics profile stops on metadata-list indexing before any update;
its source/log remain. The corrected mechanics profile executes three full
updates per case (9 total). Post-warmup complete updates cost about0.017/15/35s
for MLP/ResNet/Transformer; no intervention outcomes enter capacity selection.
The primary suite is about9.1h before a planning buffer and deadline monitoring.

Authenticate original and compact parent SHA256. Model and moment tensors must
match exactly after loading. At the same frozen state, compare Spark full loss
and raw gradient with saved original values: relative gradient error<=0.001 and
relative loss error<=0.0001. These are cross-hardware objective/gradient gates,
not a historical trajectory replay guarantee. The before period uses original
hardware; every post-intervention arm, including unchanged, uses Spark.

Before production, each case checks2 direct updates versus1+1 restored updates
for exact weights/moments/raw previous gradient/RNG. All branches must start
from the same authenticated parent and reproduce the same pre-update boundary.
CPU tests cover schedule endpoints/monotonicity/hold, full-gradient clipping,
zero norms, task-only moments and native AdamW parity. Fifteen checks pass.

Retain every raw boundary loss, raw rho, raw gradient norm, projected raw change
labelled $\eta\widehat\kappa_t$, parameter norm, chosen LR and actual increment
norm/dot product. Store clipped-input norm/factor, first-moment norm/alignment,
and separate decay/Adam increment norms. Zero denominators are undefined/null;
nonfinite values fail visibly. Raw gradient diagnostics do not automatically
describe optimizer directions or actual parameter updates.

Every branch writes status/progress/events and an immutable endpoint linked from
its last atomic recovery; checkpoint every16 updates. Checkpoints include model,
moments/age, previous raw gradient, all RNG state and scientific identity. No
all-step dense gradient archive. Detached execution with a hard deadline survives
SSH loss. Failed branches retain their traceback and earlier recovery; the queue
continues other conditions when possible. Retries preserve science and use a
new attempt. Do not modify the old suspended Spark queue or other projects.

## Analysis and figure contract

Show a shared historical before period960..1023 and fresh Spark values1024..1088,
with a marked intervention/hardware boundary at1024. The original history at1024
is retained for bridge reporting rather than silently joined as identical.
Compare every intervention with the same contemporary unchanged continuation.
Use solid, unsmoothed lines; raw rho is its own row, gradient norm stays third;
loss, rho, gradient norm, $\eta\widehat\kappa_t$ and applied LR are the five primary
rows. Independent panel Y ranges retain both full curves. Parameter-norm and
actual-increment/decay responses are available as companions. Each intervention
figure has three model/dataset columns. Preserve source-aligned CSVs/vector PDFs,
local observations, source/reduction/coverage checks and visual review.

Primary reductions: mean unregularized task loss at1025..1088 and endpoint loss,
relative to unchanged; max loss and positive normalized excess area over those
64 states against the shared Spark fork loss; raw rho, gradient norms, projected
change variation, actual increment norms and parameter norms. Report failures,
spikes/censoring, cost, LR exposure and clipping duty. Complete prepared valid/test
splits are descriptive endpoints, never selection inputs. This is one selected
seed/state and a short response window, not population robustness, optimality,
long-term convergence or a generalization claim. Consolidate5-10 factual findings.
