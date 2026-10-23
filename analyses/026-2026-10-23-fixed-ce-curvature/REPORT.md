# Fixed cross-entropy and attained curvature

Analysis 026 · H022 · Completed October 23,2026 · Runs 154-184

**The fix substantially lowers ordinary Hessian curvature at a matched training
budget. Most of that reduction accompanies a lower attained loss. The
matched-loss controls give little evidence of additional flattening at the
same loss.** The experiment therefore supports the user's intuition about the
states reached during training, with a narrower interpretation than changing
the mathematical loss landscape itself.

The [publication figure](figures/fixed_ce_curvature.pdf) presents the complete
comparison. [Results](results.json), [verification](verification.json),
[figure data](figure_data.json), [checkpoint-gradient audit](checkpoint_gradient_audit.json),
and the original [prospective protocol](PROTOCOL.md) provide the numerical evidence.

## Primary result

Ten paired initializations compared native FP32 CE (A), fixed FP32 CE (B), and
native CE evaluated in FP64 on the same FP32 model logits (C). Each condition
ran 10,000 full-batch AdamW updates. Every curvature measurement used the same
fully FP64 model copy and fixed differentiable CE on all 480 training pairs.

For each seed, the primary endpoint is the geometric mean of the largest
algebraic training-Hessian eigenvalue at states 9000,9250,9500,9750,10000. The
paired fixed/native geometric ratio is **0.0148854**, with a model-based 95% paired t
interval of **[0.00307713,0.0720074]** in the original ratio scale. That is a
**98.51% estimated reduction**, with an interval corresponding to 92.80-99.69%
reduction. All 10 paired ratios are below 1. The entire interval is below the
predeclared 0.8 boundary, supporting a reduction greater than 20% in this metric.
The enumerated two-sided sign-flip sensitivity p-value is 0.001953125.

The uncertainty unit is the initialization seed, with 9 degrees of freedom for
the paired t interval in log space. Five repeated checkpoints do not provide
five independent replicates. The sign-flip calculation is a sensitivity check;
it does not make this a randomized assignment experiment or remove its
distributional assumptions. No seeds or checkpoints were replaced or dropped.

| Seed | Native FP32: late lambda | Fixed FP32: late lambda | FP64-loss control: late lambda | Fixed/native | FP64-loss/native |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 7.96349 | 0.0612764 | 0.0619099 | 0.00769467 | 0.00777423 |
| 1 | 0.0770083 | 0.0548413 | 0.0605847 | 0.712149 | 0.78673 |
| 2 | 6.5766 | 0.0603658 | 0.0629921 | 0.00917888 | 0.00957822 |
| 3 | 15.1227 | 0.0636675 | 1.56718 | 0.00421006 | 0.103631 |
| 4 | 10.4804 | 0.0605739 | 0.0525764 | 0.00577974 | 0.00501665 |
| 5 | 3.23886 | 0.0555222 | 0.0639384 | 0.0171425 | 0.019741 |
| 6 | 8.09081 | 0.0597257 | 0.0547189 | 0.00738192 | 0.00676309 |
| 7 | 51.402 | 0.0586049 | 0.0636775 | 0.00114013 | 0.00123881 |
| 8 | 0.0695235 | 0.0594338 | 0.0660993 | 0.854874 | 0.950747 |
| 9 | 16.8884 | 0.0597523 | 0.0597973 | 0.00353808 | 0.00354074 |

## Precision control

The FP64-loss control also lowers the primary curvature endpoint in 10/10 seeds.
Its ratio to native FP32 is 0.0210346,95% CI[0.00414967,0.106625]. This corroborates
a role for loss arithmetic while keeping model forward/backward arithmetic,
parameters, optimizer moments, data and optimization hyperparameters in FP32.

The two accurate-loss arms are not numerically interchangeable across every
trajectory. Fixed/FP64-loss has geometric ratio 0.707663,95% CI[0.344099,1.45535].
In seed 3, the FP64-loss late endpoint is about 24.6 times the fixed-loss endpoint;
the other nine fixed/control ratios range 0.868-1.152. That outlier remains in
every reported aggregate. Equivalence between B and C is not established.
The supported control result is agreement in direction relative to native CE.

## Curvature at matched loss

The fixed/native late-window loss ratio is 0.0168866, close to the raw curvature
ratio 0.0148854. After the predeclared descriptive normalization by the common
objective, the paired geometric ratio of lambda/F is 0.881492. This is much
closer to 1 and is not a parameterization-invariant flatness measure.

All 100 candidate first-crossing comparisons satisfy the predeclared 10%
multiplicative loss-matching rule: five levels, ten seeds, and two comparisons
against native FP32. The table reports descriptive medians of within-seed
curvature ratios. Different levels within the same seed are correlated.

| First-crossing loss | Fixed/native matched pairs | Fixed/native median | Fixed/native range | FP64-loss/native median |
| --- | ---: | ---: | ---: | ---: |
| 1e-02 | 10 | 0.9863 | 0.849-2.252 | 0.9731 |
| 1e-03 | 10 | 0.9944 | 0.761-1.167 | 0.9849 |
| 1e-04 | 10 | 1.0524 | 0.649-1.374 | 0.9858 |
| 1e-05 | 10 | 1.0877 | 0.812-1.313 | 0.9882 |
| 1e-06 | 10 | 1.0308 | 0.875-1.219 | 0.9748 |

There is no consistent substantial curvature reduction in these matched-loss
controls. The large matched-update difference therefore mainly tracks the
ability to reach and remain at lower loss. This is an interpretation supported
by the controls, not a formal causal mediation result or proof of equivalence
at every loss value. Parameter norms also differ: mean final norms are 19.574,
17.563 and 17.811 for A/B/C, and rescaling symmetries remain.

In exact arithmetic, the two CE formulas define the same objective and have
the same Hessian at fixed parameters. The [linked paper, v 1](https://arxiv.org/pdf/2610.31642v1)
changes numerical evaluation. Our common FP64 evaluator distinguishes changed
attained parameters from a precision artifact in the Hessian calculation.

## Ordinary curvature, Adam scaling and alignment

The geometric mean of the ordinary late-window endpoint across seeds is 3.98541
for native FP32,0.0593246 for fixed FP32 and 0.0838317 for the FP64-loss control.
Gradient-direction and actual-update curvature are also retained in the figure
and per-state evidence. Their descriptive arithmetic averages are affected by
brief large native excursions and should not be equated with the primary
geometric endpoint.

Adam-scaled curvature moves differently. For
`D = diag(1/(sqrt(vhat)+epsilon))`, fixed/native
`lambda_max(D^(1/2) H D^(1/2))` has paired geometric ratio **2.40923**.
The corresponding FP64-loss/native ratio is 2.28404. These are descriptive
local operators, using stored moments; they are not the full Adam dynamical
Jacobian or an Adam stability criterion. Smaller ordinary curvature does not
by itself establish better-conditioned or more stable Adam dynamics.

The late mean raw consecutive-gradient cosine, averaged over the ten seeds,
is-0.2585 for native CE and-0.6651 for fixed CE. Negative-pair fractions are
67.92% and 96.96%, respectively. Lower loss and lower ordinary curvature thus
coexist with more negative raw-gradient alignment in this AdamW experiment.
This is consistent with the manuscript's need to distinguish raw gradients,
momentum, adaptive directions and actual displacement. The plain-GD local
alignment threshold must not be applied directly to these AdamW trajectories.

At final evaluation, all ten fixed-loss and all ten FP64-loss conditions reach
100% accuracy on all 481 held-out pairs. Nine native conditions also reach 100%;
native seed 7 ends at 95.2183%, during a large endpoint loss increase. This is a
descriptive endpoint result in the finite modular-addition domain, not a
generalization law or a claim that low curvature causes higher accuracy.
Its final common loss is 0.0823382; the last pre-update logged loss refers to
state 9999 and must not be mistaken for the final state 10000. Terminal dots in
figure(a) include the actual final full-FP64 loss values.

## Mechanism qualification and exploratory audit

**The predeclared hard-collapse exposure criterion is not met:0/10 native seeds
qualify, versus 8 required.** The largest observed native collapse fraction at
any training state is 2/480 examples (0.4167%, seed 6), far below the 10% threshold.
The experiment does not constitute a qualified reproduction of the paper's
specific hard-collapse/slingshot mechanism.

An exploratory audit was specified after the first seed showed curvature
separation without exact collapse. It recomputes three gradients at the same
retained states 0,9000,9250,9500,9750,10000 for all 30 conditions, with no updates
and a verified unchanged parameter hash. It is explicitly post-hoc and does
not replace the primary endpoint or exposure criterion.

On the 50 late checkpoints reached by fixed CE, native CE has median relative
parameter-gradient error **0.1699** against FP64 CE on the same FP32 model;
fixed CE's median error is **1.38e-7**. Native logit-gradient relative error has
median 0.3165, while fixed CE's is 1.58e-7. The selected late checkpoint audit
finds no exactly zero native target gradients. This supports an explanation
involving inaccurate small gradients before they become exactly zero.
It does not establish a unique causal mechanism for the entire trajectory.
This audit's reference isolates CE arithmetic and is not a fully FP64 model
gradient; the curvature evaluator separately uses the full FP64 model.

## Design, verification and limits

The paper-inspired task is `(a+b) mod31`:480 training pairs and 481 held-out
pairs, a fixed seed 314159 split, and one causal width 64 Transformer block with
four attention heads, feed-forward width 256, ReLU, pre-LayerNorm residual
sublayers and a final LayerNorm. There are 53,760 unique trainable parameters.
All ten initial states are serialized once on CPU and shared exactly across
arms. Initialization varies across seeds; the data split and order do not.
No pretrained model or language tokenizer is involved.

All conditions use full-batch mean CE, unfused AdamW, LR1e-3, betas(0.9,0.98),
epsilon 1e-8, decay 1 on every parameter, no clipping, schedule, dropout,
augmentation or mixed precision, and TF32 disabled. Attention softmax remains
native and matched. Only hard-target CE is varied; distillation and the paper's
separate soft-target fix are outside scope. This is not an exact replication
of an unspecified author architecture or an extension to Pythia/CIFAR.

Verification covers 300,000 contiguous scientific updates, exact paired initial
and final tensor identities, all retained production file sizes/SHA256 hashes,
690 regular curvature states,840 ordinary and 660 Adam-scaled residual-checked
eigenpairs, directional Rayleigh bounds, and stored Taylor decompositions.
Every Hessian uses all 480 training examples, never a subset. Matrix-free
`which=LA` selects the largest algebraic eigenvalue; negative eigenvalue
magnitudes and global flatness in all directions are not measured.
ReLU is piecewise smooth, and finite-step Taylor remainders are retained.

The raw metric key `update` contains actual-displacement geometry. Its exact
integer clock is recovered as `gradient_state + 1`; all 10,000 pre-update states
are contiguous. Training displacement differences are formed in FP32 and then
reduced in FP64. The maximum accumulated-displacement discrepancy against the
exact FP64 difference of stored endpoints is 3.58e-10 relative. The actual-update
curvature uses the FP64 checkpoint difference directly. Terminal update
curvature and initial Adam-scaled curvature remain undefined, not zero.

The primary uncertainty covers ten initializations on one fixed dataset split,
architecture, optimizer and horizon. Five late spectra are a checkpoint summary,
not a measurement of every intervening curvature spike. Lambda/F and matched
loss do not remove every parameter rescaling effect. Full dense gradient and
parameter histories were not retained; scalar geometry cannot reconstruct
arbitrary temporal lags. No universal curvature/stability/convergence claim is
supported by this experiment.

## Execution and reproducibility

The approved owned-GPU alternative was used after read-only Runpod stock
changed. The detached queue ran from 18:15:18.84 to 18:55:37.01 UTC
(15:15:18.84-15:55:37.01 Sao Paulo): **40 min 18 s** including its calibration,
training, diagnostics and inventory work. The GPU was an owned RTX5070 Ti Laptop,
with PyTorch 2.11.0+cu 128, SciPy 1.18.1 and NumPy 2.5.0. Maximum reported PyTorch
allocation was 0.1124 GB. GPU rental is **USD0**; no cloud resource was created.

Run 154 retains two 600-update mechanics/calibration attempts using seed 1000,
excluded from inference. The 30 production conditions retain 1,008,678,749 bytes;
all calibration and production artifacts together occupy 1,061,488,148 bytes.
Checkpoints, optimizer/RNG state, scalar histories, sparse next-state pairs,
logs and sources remain locally under the owning runs and outside Git.

The [cohort](cohort.json), [frozen inputs](source_inventory.json),
[artifact inventory](artifact_inventory.json), [output checks](output_verification.json)
and [figure manifest](figure_manifest.json) identify the evidence. Reproduction:

```text
python analyses/026-2026-10-23-fixed-ce-curvature/08_analyze.py
python analyses/026-2026-10-23-fixed-ce-curvature/10_checkpoint_audit.py
python analyses/026-2026-10-23-fixed-ce-curvature/09_plot.py
python analyses/026-2026-10-23-fixed-ce-curvature/12_verify_outputs.py
```

The final PDF was rendered with Poppler and visually reviewed. Source inputs,
failed/negative qualification, every seed and the precision-control outlier are
preserved. The manuscript draft and unrelated research work are unchanged.
