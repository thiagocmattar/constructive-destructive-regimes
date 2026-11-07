# Selective first-layer Adam updates and representation displacement caps

## Conclusions first

**The third method is useful in this fork, but its effect depends on the budget.**
The q75 neuron-displacement cap (I18) is the best of the five selective policies
on mean loss over 448-800: **-10.42% versus unchanged Adam**.
It reduces the window maximum from 0.055063 to 0.033354
(-39.43%) and cumulative positive loss variation
by 79.37%.
It modifies 15.37% of
neurons per update on average; this is not a fixed set of neurons.

Its mean loss is **4.63% lower than its
paired uniform control with the same first-layer update-norm schedule**. Selection
therefore matters beyond that particular uniform scaling schedule in this run.
However, q75 still reverses hidden and output motion on all 353 measured updates,
and its last-32-state mean is 5.68%
above unchanged Adam. This is amplitude damping with improved window-average
loss, not elimination of oscillation or a uniform learning advantage.

Discarding large coordinates worsens mean loss (+1.38%).
Coordinate capping gives a modest improvement (-2.26%).
The stricter q50 neuron cap is worse than q75 and worse than its own uniform
control. In fact, that q50 norm control has the lowest full-window mean among
these ten new branches (-11.56% versus baseline).
**The evidence does not establish that selective capping is generally superior
to uniform damping.** All results are one seed and one untreated fork.

## Results

Losses are full training-objective values on all 5,000 examples. Negative mean
changes favor the intervention. The tail is states 769-800 (32 states), not the
endpoint alone. Tables are direct reductions from `selective_comparison.json`.

| Arm | Mean loss 448-800 | Change vs baseline | Maximum (state) | Tail mean 769-800 | Loss at 800 |
|---|---:|---:|---:|---:|---:|
| Unchanged Adam | 0.022808 | +0.00% | 0.055063 (522) | 0.013458 | 0.030250 |
| I15: discard coordinates q90 | 0.023122 | +1.38% | 0.034269 (463) | 0.018412 | 0.018361 |
| I16: cap coordinates q90 | 0.022292 | -2.26% | 0.036619 (450) | 0.015423 | 0.016409 |
| I17: neuron cap q90 | 0.021962 | -3.71% | 0.036266 (450) | 0.015114 | 0.012501 |
| I18: neuron cap q75 | 0.020432 | -10.42% | 0.033354 (450) | 0.014222 | 0.013530 |
| I19: neuron cap q50 | 0.021637 | -5.13% | 0.032288 (448) | 0.018238 | 0.025972 |
| I20: norm control for I15 | 0.021840 | -4.24% | 0.034572 (450) | 0.015886 | 0.016296 |
| I21: norm control for I16 | 0.022345 | -2.03% | 0.036765 (450) | 0.015406 | 0.014959 |
| I22: norm control for I17 | 0.022317 | -2.15% | 0.038823 (529) | 0.015079 | 0.012936 |
| I23: norm control for I18 | 0.021424 | -6.07% | 0.034785 (450) | 0.014984 | 0.014727 |
| I24: norm control for I19 | 0.020171 | -11.56% | 0.032288 (448) | 0.014184 | 0.012191 |
| I10: first-layer LR 0.25x | 0.021197 | -7.06% | 0.032288 (448) | 0.015583 | 0.015132 |

The q50 selective/control maxima occur at the common untreated state 448;
that inherited maximum cannot be lowered by a subsequent intervention.
All new branches have higher tail means than unchanged Adam. A smaller maximum
or a better single endpoint alone would miss this tradeoff.

| Selective policy | Mean loss vs its norm control | Mean fraction modified | Mean applied/proposed first-layer norm | Positive loss variation | Hidden/output negative cosines | First sustained target start |
|---|---:|---:|---:|---:|---:|---:|
| I15: discard coordinates q90 | +5.87% | 13.21% | 0.6237 | 0.318742 | 279/308 of 353 | 724 |
| I16: cap coordinates q90 | -0.24% | 4.05% | 0.9676 | 0.197758 | 353/353 of 353 | 630 |
| I17: neuron cap q90 | -1.59% | 3.89% | 0.9770 | 0.149978 | 353/353 of 353 | 675 |
| I18: neuron cap q75 | -4.63% | 15.37% | 0.9126 | 0.075172 | 353/353 of 353 | 635 |
| I19: neuron cap q50 | +7.27% | 57.88% | 0.6190 | 0.312568 | 299/323 of 353 | 688 |

Baseline positive loss variation is 0.364412; hidden/output
negative-cosine counts are 350/351 of 353. The sustained target is ten consecutive
states below 0.02; baseline first starts at 585.
Event-window summaries are in the JSON/CSV; target-confirmation states are in the JSON.

For the norm controls, maximum absolute paired-norm discrepancy is
6.27e-09 (maximum relative discrepancy
1.25e-07); every update passes its independently
measured rounding bound. Amplification occurs on 56/353 updates for I20, 86/353 updates for I21, 108/353 updates for I22, 29/353 updates for I23, 46/353 updates for I24. The largest actual neuron-cap excess is 1.63e-07, within the sample-space rounding bound at every update.

## Question and matched comparison

H047 / Runs 258-268 extend the same Analysis 064/065 study. The question is whether
selective suppression of large proposed first-layer movements preserves learning
better than reducing all movements. The user requested all three methods, with
particular emphasis on the representation-based criterion.

Every treatment starts from the identical **untreated full state 448** and runs
through 801. State 448 is evaluated before the first changed update 448->449.
There is no inherited freeze, LR reduction, optimizer reset, or change at 500.
All Adam LRs stay 0.001, betas (0.9, 0.999), epsilon 1e-8, no clipping/decay,
foreach/fused false, FP32 and TF32 off. Same seed 0, original 5000 CIFAR-10 images
and order, physical batch 1000, tanh MLP 3072-200-200-10, mean half squared error.

All gradients are computed normally. Native Adam advances every moment and
clock. We then modify only the proposed increments of first-linear weights and
biases; later-layer increments remain native along each branch's trajectory.
Thus these are **post-Adam update interventions**, not raw-gradient filtering.
All arms are compared with unchanged Adam; previous first-layer 0.25x LR is a
supplementary reference in the numerical record.

## Three policies

Let $u_{W,t}$ and $u_{b,t}$ be the first-layer increments that native Adam would
actually apply at the current state, including its FP32 rounding. First-layer
weights have 200 neuron rows and 3072 input columns; biases have 200 coordinates.
Concatenating weights and biases gives 614600 scalar parameter coordinates.

1. **Discard large coordinates:**
   $\widetilde u_i=u_i\mathbf1\{|u_i|\leq X\}$. Larger proposed increments become
   zero; smaller ones retain their original values.
2. **Cap large coordinates:**
   $\widetilde u_i=\operatorname{sign}(u_i)\min(|u_i|,X)$. Large increments retain
   their sign with bounded magnitude.
3. **Cap each neuron's preactivation displacement:** use the fixed input matrix
   $X_{\rm data}\in\mathbb R^{5000\times3072}$ and compute

   $$D_t=X_{\rm data}u_{W,t}^{\mathsf T}+\mathbf1u_{b,t}^{\mathsf T},
   \qquad s_{t,j}=\sqrt{\frac1{5000}\sum_{n=1}^{5000}D_{t,nj}^2}.$$

   $D_t$ has 5000 sample rows and 200 neuron columns. There is no sample averaging
   before the sum of squares and no centering. Set
   $c_{t,j}=\min(1,\tau/s_{t,j})$, with $c_{t,j}=1$ when $s_{t,j}=0$, then multiply
   both the weight-row increment and bias increment for neuron $j$ by $c_{t,j}$.

The third policy limits the first-linear functional displacement, not the
post-tanh activation value, output displacement, or loss. First-layer motion
can remain nonzero. The tanh and later trainable layers still transform that
motion, and later-layer parameter updates can independently affect the output.
No SVD axis or future trajectory enters the selection rule.

Threshold calculations and sample-space matrix products use FP64. Applied
parameters remain native FP32. For the representation cap we retain intended,
actual and rounding-error displacements and check, neuron by neuron,

$$s^{\rm actual}_{t,j}\leq\tau+
\operatorname{RMS}_n[(X_{\rm data}e_{W,t}^{\mathsf T}+\mathbf1e_{b,t}^{\mathsf T})_{nj}],$$

where $e$ is the actual increment minus the intended capped increment. This is
the triangle-inequality allowance for storage rounding, not an arbitrary
stability tolerance. A tiny numerical excess over the ideal cap is recorded.

## Threshold calibration and arms

Run 258 reconstructs unchanged Adam from initialization through 448, matches all
449 boundary losses/norms and the full 448 checkpoint, and uses only outgoing
updates **416-447** for calibration. Future losses and intervention trajectories
do not set the thresholds. NumPy's linear quantile convention is fixed in advance.

- Coordinate threshold: pooled absolute native first-layer increments across
  32 updates and 614600 coordinates; q90 gives **0.00011478690430521965**.
- Neuron thresholds: pooled sample-RMS values across 32 updates and 200 neurons;
  q90 gives **0.1515489535027608**, q75 **0.09891533373818674**, and q50
  **0.05222294052352068**. Each arm uses one common, fixed budget across neurons.

These are fixed thresholds, not rules that always suppress a fixed fraction
of coordinates or neurons. The modified fraction can vary over training.

| Arm | Run | Policy | Threshold | Paired uniform control |
|---|---:|---|---:|---|
| I15 | 259 | Discard coordinates | Coordinate q90 | I20 / Run 264 |
| I16 | 260 | Cap coordinates | Coordinate q90 | I21 / Run 265 |
| I17 | 261 | Neuron displacement cap | Neuron q90 | I22 / Run 266 |
| I18 | 262 | Neuron displacement cap | Neuron q75 | I23 / Run 267 |
| I19 | 263 | Neuron displacement cap | Neuron q50 | I24 / Run 268 |

## What the norm controls match

For a selective arm, save the actual first-layer increment norm $r_t$ at each
outgoing update $t$. Its paired control computes its own native Adam proposal
$u_t^{\rm control}$ and scales the entire first-layer block by
$a_t=r_t/\|u_t^{\rm control}\|$, with a zero increment when both norms are zero.
The applied norm matches $r_t$ within independently measured FP32 rounding.
The factor may exceed 1; amplification counts are reported.

This matches the **absolute first-layer norm schedule**, including biases,
across the diverging trajectories. It does not merely match the initial LR or
a shrinkage fraction at different states. Only the first-layer norm is matched:
later-layer increments follow each branch's own gradients and Adam state.
These are paired experimental controls, not standalone online policies: their
same-update norm target comes from the corresponding selective branch.

## Endpoints, capture and limitations

Prespecified primary window 448-800; event 450-500; last 32 states 769-800. Report
mean/max/endpoint loss, largest upward loss change, cumulative positive loss
variation, and the first ten consecutive states strictly below loss 0.02.
Targets not reached are censored. A peak still rising at 800 is horizon-censored.
No stopping or threshold adjustment uses a treatment's outcome.

The representation probe is the complete 5000-example training set. Outgoing
differences are $\Delta A_t=A_{t+1}-A_t$, with cosine against $\Delta A_{t-1}$ after
flattening all sample/coordinate entries. Coordinate RMS is $\|\Delta A_t\|_F/
\sqrt P$, where $P=5000\times200$ for hidden representations and $5000\times10$
for outputs. Full raw gradients have 656810 coordinates, before Adam; they are
not differenced. Eta kappa hat is $1-\langle g_t,g_{t-1}\rangle/\|g_{t-1}\|^2$,
a projected-gradient diagnostic under Adam rather than a curvature threshold.

Dense raw gradients and five representation arrays retain states 447-501.
All-state scalar diagnostics cover outgoing updates 448-800. Full native and
intervention first-step states, fork 448, qualification 512, recovery and end 801
checkpoints are retained. Three audits at 448/474/800 retain first-layer proposals,
applied increments, neuron scores and scaling factors for independent CPU checks.

This is one seed and one selected fork. Comparisons concern training, not
validation/generalization. No uncertainty intervals or universal stable threshold
are inferred. Low-dimensional SVD concentration does not imply a small number
of problematic parameter coordinates, and no new SVD fit is performed here.
The separate input-covariance causal test remains a TODO.

## Figures, provenance and reproduction

**Figure 04 caption.** Three columns compare coordinate discard q90, coordinate
cap q90, and neuron displacement cap q90 with the same unchanged Adam baseline.
Rows show full loss, eta kappa hat, full raw-gradient cosine/norm, first-hidden
change cosine/RMS, output-change cosine/RMS, applied/proposed first-layer norm,
and modified fraction. Every curve uses states 448-800 with no smoothing. Loss
axes are independent; the other axes share scales. Blue is the unchanged baseline.

**Figure 05 caption.** Page 1 compares the three neuron budgets q90/q75/q50;
page 2 compares the two coordinate policies. Each column overlays unchanged
Adam, its selective intervention, and its uniform norm control. Rows and scales
follow Figure 04. The gray control can amplify its own native first-layer
proposal to match the selective branch's absolute norm. Fraction modified means
coordinate fraction for coordinate rules and neuron fraction for neuron rules;
it is not a fraction of examples. For uniform scaling it is one whenever the
scalar differs from one. Baseline fractions/norm ratios are exactly 0/1 by policy.

Sources: Runs 258-268, qualified unchanged Run 244/Analysis 064 evidence,
and the verified prior `layer_lr_comparison.json` for the baseline and I10.
`selective_plan.json` records all choices; `selective_comparison.json` records
source SHA-256 identities, full curves, window summaries and paired checks.
`selective_metrics.csv` is the compact numerical table. Each run retains its
input manifest, resolved policy, raw evidence, retrieval receipt and verification.

Reproduce reductions with `11_reduce_selective.py` and the two new PDFs with
`12_figures_selective.py`. `09_prepare_selective.py` and `10_execute_selective.py`
record preparation/orchestration; do not relaunch executed run IDs.
`13_closeout_selective.py` reads task processes/storage without modifying Spark.

Verification passes: calibration quantiles and independent sample-space audits;
full fork equality; all 65 native continuation boundaries and full state 512;
native first-step Adam moments/clocks/RNG and later parameters; all-step gate
bounds/norm matches; independent NumPy policy audits at 448/474/800; dense CPU
gradient and five-representation reductions. All 720 branch input-file checks
and 270 retrieved files (including calibration) pass size/SHA-256 checks.
All 74,130 plotted values, repeated baselines and three rendered PDF
pages pass review; Figures 01-03 are unchanged. Verification records are
`selective_figure_verification.json` and `selective_visual_verification.json`.

An initial CPU audit treated a uniform factor numerically different from one
by about 4e-16 as a changed step, while the native GPU operands gave exactly one.
The corrected audit checks the discontinuous binary flag from its recorded
native operands and still verifies the continuous factor/norm independently
with NumPy. No executed policy, threshold or trajectory was changed or rerun.

## Compute and retention

Owned Spark/GB10 only, no cloud charge. Calibration took
5.48 s; ten branches took
379.94 s total
(35.07-
43.55 s each),
with peak allocated memory 539.4 MB.
These are diagnostic-rich worker times including qualification, repeated FP64
sample-space checks and artifact writes, not a clean training-overhead benchmark.
The neuron policy itself requires a 5000-by-3072 times 3072-by-200 product plus
RMS reduction per update; no extra backward pass is needed.

All task workers exited. Retained run directories and task source archives use
21.907 GB on Spark; local run directories use
21.907 GB including compressed copies. Raw evidence,
full recovery states and archives are retained, each machine below the 30 GB
allowance. No unrelated resource was modified. See `selective_compute_closeout.json`.
