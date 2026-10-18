# Adam preconditioning and the Pythia model-size loss-rate separation

Analysis 025 · Hypothesis H021 · Experiment completed October 17, 2026 · Report updated October 18, 2026

**Main finding.** Removing Adam's coordinate scaling, while preserving the native
adaptive-step norm schedule within each model, worsened the 50-update outcome in
all 18 continuations. The harm increased from 14M to 31M to 70M in every seed block
at both starting checkpoints. At the checkpoint after update 225, the geometric
mean endpoint loss penalties were **26.84%, 57.90%, and 106.49%**, respectively,
relative to native AdamW at update 275. Same-state direction substitutions show
that a favorable first-order loss prediction can be overwhelmed by a much larger
finite-step nonlinear remainder when coordinate scaling is removed.

These results establish a size-dependent response to an optimizer intervention
under this training recipe. They support an optimizer contribution to the observed
loss-rate separation, without establishing its complete mediation or a unique
transition at update 200.

## 1. Question and hypotheses

The motivating observation was that Pythia-70M continued to improve faster around
updates 200–250 while the smaller models slowed. The original comparison used
training losses and the norms and adjacent alignment of gradients and Adam
directions. A larger gradient norm does not by itself imply faster loss reduction:
the applied update's direction, length, and nonlinear effect on the objective also
matter. Likewise, positive alignment between successive optimizer directions does
not establish that either direction decreases loss.

The registered question, **H021**, was whether Adam's coordinate preconditioning
contributes to this model-size separation after controlling adaptive-step norm.
The following distinguishes the primary causal prediction from its descriptive
and mechanistic checks; these are operational components of H021, rather than
new independently registered hypotheses.

| Component | Prediction or question | Role |
| --- | --- | --- |
| Reproduction | Does 70M show faster relative loss reduction over updates 200–250 on a common, fixed objective? | Descriptive check of the motivating pattern. |
| Preconditioning intervention | Removing coordinate scaling at the native adaptive-step norm worsens the next 50 updates, with a larger effect for 70M than 14M and 31M intermediate. | Primary causal test. |
| Local mechanism | Does the intervention change the first-order prediction, the nonlinear remainder, or both? | Same-state diagnostic explaining the immediate response. |
| Timing | Is the dependence specific to the originally selected 200–250 interval? | Limited by two continuation checkpoints and sparse one-step controls. |

No effect, reversed effects, or equal effects across sizes would weaken the proposed
explanation of the size gap. The interval 200–250 was selected **post hoc** from
earlier trajectories; the new interventions and their outcome definitions were
specified before these new runs. The original observation is preserved in
[Analysis 005](../005-2026-10-04-temporal-lag-profiles/observations/012-loss-separation-200-250.md),
and the experiment contract is in [README.md](README.md).

## 2. Experimental design

### Cohort and matched factors

Nine native AdamW baselines crossed three model sizes with three seed blocks
(1234, 1235, 1236), each running for 350 accepted optimizer updates. Each baseline
provided two exact starting states, after updates 175 and 225. From each state,
one intervention continued for 50 updates, covering updates 176–225 or 226–275.
The native parent supplied the calendar-matched control at the same endpoint.
This produced **27 runs: nine baselines and 18 continuations**, totaling
**4,050 accepted updates**.

| Seed block | Size | Baseline run | Branch after 175 | Branch after 225 |
| --- | --- | ---: | ---: | ---: |
| 1234 | 14M | 127 | 136 | 137 |
| 1234 | 31M | 128 | 138 | 139 |
| 1234 | 70M | 129 | 140 | 141 |
| 1235 | 14M | 130 | 142 | 143 |
| 1235 | 31M | 131 | 144 | 145 |
| 1235 | 70M | 132 | 146 | 147 |
| 1236 | 14M | 133 | 148 | 149 |
| 1236 | 31M | 134 | 150 | 151 |
| 1236 | 70M | 135 | 152 | 153 |

The [cohort manifest](cohort.json) maps these IDs to full run directories. Model
sizes contain 14,067,712, 30,494,720, and 70,426,624 trainable parameters. All have
six layers; hidden widths are 128, 256, and 512, with 4, 8, and 8 attention heads.
Architecture and parameterization therefore vary with nominal size.

The baselines reuse the exact randomly initialized tensors from Runs 002–010,
using the same small-init/Wang-init recipe. No pretrained weights were loaded.
Within a seed block, data order is matched across sizes. Across blocks,
initialization and data-order seeds change together, so their variances cannot be
separated. Initial tensor identities are recorded in
[initial_sources.json](initial_sources.json).

| Factor | Shared recipe |
| --- | --- |
| Data | MiniPile, packed to length 2,048; token-weighted mean next-token cross-entropy. |
| Optimizer boundary | One update accumulates 1,024 sequences in microbatches of four. |
| Gradient processing | Dynamic-loss-scale unscaling, then global norm clipping at 1. |
| Optimizer | Fused PyTorch AdamW; betas (0.9, 0.95); epsilon 1e-8 outside the square root. |
| Weight decay | 0.1, excluding bias and LayerNorm parameters. |
| Learning rate | Peak 0.001, minimum 0.0001, 1% warmup then cosine decay; original GPT-NeoX v1 pre-step clock over 712 updates. |
| Horizon | Baselines stop at 350; the 712-update schedule is preserved, without compression or restart. |
| Training precision | FP32 parameters and moments; dynamically scaled FP16 training. |
| Hardware/software | Two RTX 4090 workers; PyTorch 2.11.0+cu128, Transformers 5.12.1, NumPy 2.5.0. |

Every baseline processes 734,003,200 input tokens and 733,644,800 supervised
targets in full batches. A continuation processes another 50 parent-calendar
batches; these replay the corresponding parent data rather than add an independent
sample of 50 updates. Each continuation starts from its parent's parameters,
moments, scaler, RNG states, and data position. Future batches, clipping rules,
moment-update rules, LR clock, and decay coefficients/groups are matched.
Gradients, moment values, and decay vectors naturally diverge after the intervention.

### Common objective and measurement coverage

Let **F(theta)** be token-weighted mean cross-entropy on a fixed probe of 128 packed
sequences: 262,144 input tokens and 262,016 supervised next-token targets. Probe
seed 104729 selects from the complement of the **union of all three seed blocks'
350-update training prefixes**. The probe is identical for all 27 conditions and
absent from every condition's training updates. It never selects directions,
step lengths, or training batches.

This is a held-out portion of the same training corpus, rather than an independent
test population. It supplies a consistent objective for before/after comparisons
that changing-minibatch training losses cannot provide. Full MiniPile validation
was also evaluated at every trajectory's start and end: all 339 sequences,
including the short tail, totaling 693,329 supervised targets. The primary causal
comparison uses the calendar-matched probe endpoints; branch-final validation
must not be compared directly with the baseline's later update-350 validation.

Baseline one-step measurements occur at updates **100, 150, 175, 200, 225, 250,
275, 300, and 350**. Probe evaluation and differentiation use FP32, math SDPA,
TF32 disabled, microbatch one, and FP64 scalar dot products/norms. Diagnostics
preserve RNGs, model mode, outstanding training gradients, and optimizer state;
the exact accepted native parameters are restored after counterfactuals.

### Directions and the controlled intervention

For the raw unscaled accumulated gradient g_t, let c_t be its clipped version.
Adam's first moment averages c_t; its second moment averages the coordinatewise
squares of c_t, using betas 0.9 and 0.95. Let m_hat_t and v_hat_t be those moments
divided by 1 - beta1^t and 1 - beta2^t, respectively. For scheduled learning rate
eta_t, define:

```text
d_t = m_hat_t / (sqrt(v_hat_t) + epsilon)
native adaptive component = -eta_t * d_t
```

Here d excludes learning rate, weight decay, and parameter-subtraction rounding.
It is distinct from the stored first moment m, the raw gradient g, and the actual
full parameter displacement. All vector measurements cover unique trainable
parameters, counting tied parameters once. Adjacent alignment rho_d is the cosine
between d_t and d_(t-1); it differs from alignment with the fixed-probe gradient.

The sustained intervention removes coordinate scaling but preserves the parent's
adaptive-step length at each global update:

```text
r_t = eta_t * ||d_t from the unmodified native parent||
branch adaptive component = -r_t * m_hat_t(branch) / ||m_hat_t(branch)||
```

The branch updates its own moments throughout. Its target r_t comes from the
already completed parent, rather than its own hypothetical Adam step or the probe.
The same decay rule is applied to the branch's own parameters. Norm matching is
within each model, before FP32 rounding; it does not match full rounded update
norms or impose equal norms across architectures. This tests removing coordinate
scaling at the native Adam scale, rather than the best achievable performance of
momentum with a retuned scalar learning rate.

At each baseline measurement, seven one-step candidates additionally hold the
pre-step state, minibatch, clipped gradient, updated moments, and clock fixed:

| Candidate | Adaptive direction/length |
| --- | --- |
| Native AdamW | Actual fused optimizer step, including its rounding. |
| Reconstructed Adam | d_t; numerical application control. |
| No preconditioning | m_hat_t, normalized to native adaptive-step norm. |
| No first-moment averaging | c_t / (sqrt(v_hat_t) + epsilon), at native adaptive-step norm. |
| Neither component | c_t, at native adaptive-step norm. |
| Half step | Reconstructed Adam at half the native adaptive-step norm. |
| Double step | Reconstructed Adam at twice the native adaptive-step norm. |

Candidate applications include the same decay displacement before FP32 rounding.
Replacing the current direction does not reset moment history or implement a
sustained beta1=0 optimizer. Undefined zero directions are explicitly recorded;
none occurred in these candidate measurements.

### Outcomes and loss accounting

The primary effect for size, seed, and starting checkpoint j is:

```text
tau = ln F_ablated(j + 50) - ln F_native(j + 50)
paired size interaction = tau_70M - tau_14M
```

Both trajectories start from the same state and probe loss. Positive tau means
that removing preconditioning gives a worse endpoint. We report the mean and
range of tau over three seeds. The corresponding geometric mean relative loss
penalty is 100 * (exp(mean(tau)) - 1).

For a one-step displacement s = theta_after - theta_before, including decay and
rounding, the probe accounting is:

```text
D = F(theta_before) - F(theta_after)       # measured decrease
P = -<gradient F(theta_before), s>        # first-order predicted decrease
R = P - D                               # nonlinear remainder
D = P - R
```

Positive D denotes improvement; negative D denotes a loss increase. Tables of
local effects use arithmetic means of D/F_before, P/F_before, and R/F_before.
R includes curvature, higher-order effects, and numerical error. It is an exact
residual of the measured accounting, rather than a measured Hessian curvature
or an edge-of-stability statistic.

## 3. Observations

### Native 70M loss reduction is faster over the motivating interval

The new baselines reproduce the larger model's faster relative improvement on
the fixed objective. For updates 200–250:

| Size | Mean ln(F_250 / F_200) | Geometric mean loss reduction |
| --- | ---: | ---: |
| 14M | -0.028793 | 2.84% |
| 31M | -0.030683 | 3.02% |
| 70M | -0.046614 | 4.55% |

The 70M model improves more than either smaller model in each paired seed block.
This is a finite-window observation, rather than a claim of asymptotic convergence.
Individual sampled native updates remain noisy and do not uniformly rank the
three sizes by one-step decrease.

### Removing preconditioning harms every continuation, with greater harm at larger size

| Starting checkpoint | Endpoint | Size | Mean tau | Seed range of tau | Geometric mean loss penalty vs. native |
| ---: | ---: | --- | ---: | ---: | ---: |
| 175 | 225 | 14M | 0.208094 | 0.181416–0.233604 | 23.13% |
| 175 | 225 | 31M | 0.350102 | 0.320622–0.399874 | 41.92% |
| 175 | 225 | 70M | 0.522908 | 0.486341–0.562303 | 68.69% |
| 225 | 275 | 14M | 0.237750 | 0.200716–0.265057 | 26.84% |
| 225 | 275 | 31M | 0.456766 | 0.415563–0.504590 | 57.90% |
| 225 | 275 | 70M | 0.725101 | 0.660009–0.778978 | 106.49% |

These percentages compare the ablated endpoint against the native endpoint;
they are not percentage changes from the starting checkpoint. All 18 tau values
are positive. In fact, every ablated branch also ends above its own starting probe
loss, while its native control improves over the same 50 updates.

The size ordering holds within every paired block, rather than arising only after
averaging:

| Starting checkpoint | Seed | tau, 14M | tau, 31M | tau, 70M | tau_70M - tau_14M |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 175 | 1234 | 0.181416 | 0.329811 | 0.520082 | 0.338666 |
| 175 | 1235 | 0.233604 | 0.320622 | 0.562303 | 0.328698 |
| 175 | 1236 | 0.209262 | 0.399874 | 0.486341 | 0.277078 |
| 225 | 1234 | 0.265057 | 0.415563 | 0.660009 | 0.394952 |
| 225 | 1235 | 0.247477 | 0.504590 | 0.736315 | 0.488839 |
| 225 | 1236 | 0.200716 | 0.450144 | 0.778978 | 0.578262 |

Thus the primary prediction is supported for these states and this norm schedule.
The later checkpoint has larger mean harm at every size, but different starting
states and only two continuation checkpoints prevent interpreting this as a
precisely dated onset.

### Local accounting exposes a large nonlinear penalty

At update 200, the 70M means across three seeds are shown below. Entries are
percentages of the pre-step probe loss; negative D/F means that loss increased.

| Direction or length | First-order prediction P/F (%) | Remainder R/F (%) | Actual decrease D/F (%) |
| --- | ---: | ---: | ---: |
| Native AdamW | 1.199 | 1.116 | 0.083 |
| No preconditioning, matched norm | 5.010 | 141.298 | -136.288 |
| No first-moment averaging, matched norm | 5.012 | 14.752 | -9.740 |
| Neither component, matched norm | 15.070 | 223.879 | -208.809 |
| Half-length reconstructed Adam | 0.599 | 0.281 | 0.318 |
| Double-length reconstructed Adam | 2.398 | 4.351 | -1.953 |

The unpreconditioned momentum direction has a more favorable linear projection
than native AdamW in this comparison. Nevertheless, its nonlinear remainder is
far larger and overwhelms that prediction. The immediate benefit of coordinate
scaling here therefore cannot be described simply as a larger gradient norm or
a better first-order projection. At the tested finite step length, the scaled
direction avoids much of the loss increase incurred by the substitute.

The half/double controls also show that step magnitude matters: halving the native
adaptive length gives a larger mean one-step decrease here, while doubling it
raises loss. These local controls do not establish a better sustained LR schedule.
Removing first-moment averaging is harmful in this one-step comparison as well;
no sustained first-moment ablation was run.

The preconditioning sensitivity already exists at earlier measured states.
For 70M, the norm-matched unpreconditioned direction gives mean one-step loss
increases of **210.16% at update 100** and **125.65% at update 150**. Together with
the effect of branching after 175, this prevents attributing the findings to a new
dependence that first appears at update 200. The update-200 table is an illustrative
mechanistic readout; all measured states and candidates remain in the results.

## 4. Conclusions and interpretation limits

**The causal conclusion is conditional:** for these trained states, removing
Adam's coordinate preconditioning while retaining each parent's adaptive-step
norm schedule causes worse 50-update fixed-probe outcomes. The effect increases
with model size in all three paired seed blocks at both checkpoints. This supports
the primary intervention prediction of H021.

**The local mechanism is a finite-step directional effect.** The substituted
momentum direction can have a favorable linear prediction and still produce a
large loss increase because the nonlinear remainder dominates. Coordinate scaling
changes where the same adaptive-step length is spent in parameter space. These
measurements connect direction geometry to an observed objective response more
directly than adjacent alignment or vector norms alone.

**The original timing and full mediation questions remain open.** Native 70M
improves faster over updates 200–250, and its dependence on coordinate scaling is
greater under the intervention. This does not quantify how much of the native
size gap is mediated by preconditioning, establish a unique switch near update
200, or show that parameter count itself causes the effect. Architecture, learned
state, parameterization, and native step scale differ across sizes.

The main limits are:

- **Three replication blocks.** Seeds, rather than updates, candidates, forks, or
  probe tokens, are the replication units. Both forks share parent histories;
  they do not provide six independent capacity replications. Ranges describe the
  observed seed spread, rather than population confidence intervals.
- **A specific intervention and scale.** Matching norm within a model does not
  equate intervention severity across architectures. A scalar-LR momentum method
  retuned for its own direction could behave differently. Decay and rounding also
  contribute to full parameter displacements.
- **Restricted objective and horizon.** The fixed probe is disjoint from all
  training prefixes but belongs to the same corpus. These 50-update effects do
  not establish generalization, long-run convergence, or formal stability.
- **Residual interpretation.** R is not a Hessian measurement, a pure quadratic
  term, or an edge-of-stability test. Native one-step probe loss can rise even
  while the longer native interval improves.
- **Remaining variation.** Initialization and data ordering co-vary across seed
  blocks. Both workers use RTX 4090s and the same package lock, but driver patch
  versions differ; each branch remains paired with its own parent's worker.

## 5. Evidence, verification, and reproduction

All 27 runs passed local size/SHA-256 retrieval checks and scientific verification.
Coverage includes 567 baseline candidate records (nine baselines × nine measured
updates × seven candidates), plus 36 actual branch-step probe records, totaling
603. The native/reconstructed Adam controls passed at all 81 baseline states:
maximum parameter error was **1.1921e-7** and maximum probe-loss gap **5.0291e-8**,
below the required 1e-6 and 1e-5 bounds. The large intervention effects substantially
exceed these application-control discrepancies.

Retained local evidence totals **103,516,215,158 bytes**, including raw scalar and
probe histories and 45 recovery states: each baseline's updates 175, 225, and 350,
and each continuation's terminal state. Checkpoints include optimizer/scaler,
RNGs, and data-order position. Large raw artifacts are retained outside Git in
their owning run directories; tracked manifests, receipts, and reductions identify
them. The complete reduction reproduces the saved [results.json](results.json)
from the verified scalar histories.

Reproduction inputs and outputs:

- **Data identity:** `JeanKaddour/minipile` revision
  `18ad1b0c701eaa0de03d3cecfdd769cbc70ffbd0`; Pythia tokenizer revision
  `7386d9a4ae45aef494a6e704910394def3037fc5`. Cache hashes, exact initial tensors,
  model configs, and source identities are recorded in
  [input_packet.json](input_packet.json), [initial_sources.json](initial_sources.json),
  [source_packet.json](source_packet.json), and each run's config/input manifest.
- **Probe identity and exclusions:** [probe.json](probe.json).
- **Accepted reductions and provenance:** [results.json](results.json), including
  all seed effects, size interactions, native probe values, and scalar-source hashes.
- **Scientific verification:** [03_verify.py](03_verify.py); training and probe
  semantics are implemented in
  [causal_training.py](../../src/baseline/causal_training.py) and
  [descent_diagnostics.py](../../src/baseline/descent_diagnostics.py).
- **Figure and reduction generator:** [08_analyze.py](08_analyze.py). From the
  repository root, with retained artifacts available, run
  `python analyses/025-2026-10-17-pythia-descent-mechanism/08_analyze.py`.

The three publication PDFs were rendered and visually inspected. No display
smoothing is used. Seed curves/points show the available replication; lines
between measured updates do not imply intermediate probe evaluations.

| Figure | Contents |
| --- | --- |
| [Native descent accounting](figures/native_descent_accounting.pdf) | Fixed-probe loss evolution and normalized D, P, and R on native trajectories. |
| [Preconditioning effects](figures/preconditioning_effects.pdf) | All paired seed effects and means for the two 50-update continuation tests. |
| [Direction substitutions](figures/direction_substitutions.pdf) | One-step effects of direction and length substitutions at updates 175, 200, 225, and 250. |

The local [observation record](observations/001-preconditioning-effects.md)
provides figure captions, coverage, reductions, and caveats. This report concerns
the completed experiment; no manuscript claim was changed.

## 6. Execution and cost closeout

Training finished on October 17 at 18:23:36 BRT. Retrieval, verification, and
cleanup finished at 19:05:20 BRT. Both task-owned pods and the 200 GB network
volume were deleted; provider read-backs confirmed their absence. All retained
experiment artifacts are local. Posted task charges total **USD 11.416828**,
within the approved USD 25 / 12-hour envelope; posted billing remains subject to
provider reconciliation. See [billing.json](billing.json),
[execution.json](execution.json), and [closeout.json](closeout.json).
