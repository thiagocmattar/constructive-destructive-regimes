# Adam spikes arise from output-sensitive feature displacement and step overshoot

The selected peaks show little abrupt growth in parameter or hidden-activation
magnitudes. Their distinguishing changes are in the hidden representation and
its alignment with the output mapping. Static parameter recombinations assign
most of the loss increase to the two hidden weight matrices. Direct measurements
along the incoming Adam steps show a lower-loss region before the full step:
the steps are locally descending but overshoot that region. Tanh attenuates
activation changes; its bounds do not enforce monotone optimization.

This is a mechanistic explanation supported by exact replay and static probes
for this selected seed/window, not proof of a universal Adam instability or a
unique causal allocation across interacting parameter groups.

## Selection and exact control replay

The user's literal last local maximum before state512 is state503, with local
pre-rise502 and first return to/below that loss at505. It sits in the larger
fixed-anchor excursion beginning492, peaking493, and recovering508. We also
inspect the nearby larger peak474 and the unchanged control's next peak522.
All four events were selected from retained losses before internal diagnostics.

| Event | Before / peak / return states | Before loss | Peak loss | Return loss |
| --- | --- | ---: | ---: | ---: |
| Nearby larger excursion | 465 / 474 / 485 | 0.0281879 | 0.0536730 | 0.0277188 |
| Last complete pre512 excursion | 492 / 493 / 508 | 0.0262139 | 0.0294607 | 0.0244672 |
| Last local pre512 peak | 502 / 503 / 505 | 0.0275415 | 0.0286617 | 0.0274330 |
| Secondary continued-control excursion | 512 / 522 / 529 | 0.0230571 | 0.0550626 | 0.0226791 |

State means completed updates, not a one-based manuscript gradient index.
Run236 replays the pinned Run223 initialization through535 on the same Spark.
All536 full-objective losses and raw-gradient norms match the retained control
exactly. Model, optimizer, RNG and both gradient histories match exactly at512.
The passive observer also preserves gradients and RNG. This is the unchanged
Adam trajectory, not a newly fitted or altered control.

Model:3072-200-200-10, tanh after both hidden layers, biased linear output;
656,810 parameters. Seed0, fixed first5,000 CIFAR-10 training images, complete
MSE objective `mean_i(0.5 * sum_c (f_ic - onehot_ic)^2)`, physical batches1,000,
Adam(.9,.999), LR.001, epsilon1e-8, no decay/clipping. FP32 vectors and optimizer,
FP64 diagnostic reductions, TF32 off. Training-only diagnostic coverage; no
test-set inference is made. No per-sample gradients are inferred from batch means.

## Parameter magnitudes and biases

For the larger pre512 event465→474, loss rises90.41%, but weight Frobenius norms
change only+0.358% (W1),+0.075% (W2),+0.347% (W3). Bias norms change+0.565%,
+0.298%,+0.031%. These are gradual changes, not a coincident norm explosion.
For the literal502→503 peak, weight norm changes are even smaller:
+0.037%,+0.007%,+0.032%. Exact singular-value diagnostics likewise show no abrupt
spectral-norm blowup; the output-head spectral norm is about0.85–0.87.

The magnitude-only probes make this more than a correlation:

| Static465→474 comparison | Full training loss |
| --- | ---: |
| Actual pre-rise model | 0.0281879 |
| Actual peak model | 0.0536730 |
| Peak model, all six tensor norms restored to pre-rise norms | 0.0536103 |
| Pre-rise directions, only tensor norms grown to peak norms | 0.0280355 |
| Peak hidden layers, original output weight and bias | 0.0527713 |
| Pre-rise hidden layers, peak output weight and bias | 0.0279378 |

Restoring all parameter norms leaves99.75% of the excess peak loss. The analogous
fraction is100.38%,101.12%,99.70% at493,503,522; small values above100% mean the
norm-restoration operation slightly worsens that model. Norm growth alone does
not reproduce any of the four spikes. These probes preserve directions or
norms of entire tensors, not every neuron's individual norm or all matrix
singular values. They evaluate static recombinations, without retraining.

All64 combinations of before/peak values of W1,b1,W2,b2,W3,b3 were evaluated
on all5,000 examples. Exact Shapley allocation shares their interactions:

| Peak | W1 share | W2 share | W3 share | All biases, signed share |
| --- | ---: | ---: | ---: | ---: |
| 474 | 73.98% | 24.76% | 1.28% | -0.021% |
| 493 | 70.73% | 27.32% | 1.94% | +0.005% |
| 503 | 73.72% | 25.94% | 0.38% | -0.044% |
| 522 | 76.23% | 22.69% | 1.09% | -0.014% |

Thus biases and the head's own parameter changes make small contributions in
this allocation. The six-group choice and reference endpoints matter; these
percentages are not unique physical causes. Indeed, at503 replacing W1 alone
or W2 alone in the pre-rise model improves loss, while their joint changes
increase it. The interaction must not be omitted.

## Tanh works, but most of the relevant changes remain unsaturated

For465→474, hidden activation RMS changes0.892576→0.892798 (layer1) and
0.396262→0.396169 (layer2): magnitudes are nearly constant. Nevertheless,
the full hidden2 activation array moves by7.16% of its pre-rise norm. Norm
and vector change are different measurements: a vector can move with almost
unchanged norm. At502→503, hidden2's relative change is5.32% while its RMS
increases only0.075%.

About63.4% of first-layer activation entries already satisfy|h1|>.95.
But99.67% of the measured h1 change energy comes from entries that were
*not* saturated by that threshold. Only0.212% of h2 entries exceed.95;
its mean tanh derivative is0.843, so this layer remains highly responsive.
The same pattern holds across all four events. Threshold comparisons use
native FP32 values, matching the observer; no silent precision recategorization.

Tanh attenuates the observed changes: for465→474,
`||delta_h1|| / ||delta_z1|| = 0.342` and
`||delta_h2|| / ||delta_z2|| = 0.889`.
The head's fixed-matrix gain is also below one:
`||W3_before delta_h2|| / ||delta_h2|| = 0.668`.
These are full-vector norm ratios. Comparing per-coordinate RMS across a200-unit
hidden layer and a10-output layer would incorrectly suggest gain above one.
There is no evidence here of exploding forward activation norms.

The changes matter because they lie in directions read by the head. Its
10-dimensional row space contains only4.68% of baseline hidden2 activation
energy, but63.69% of the hidden2 *change* energy. Across the other events the
latter fraction is65.45–68.93%. The isotropic reference is10/200=5%; isotropy is
only a geometric comparison, not an assumption about actual activations.
About91–93% of output-change energy lies in one class-space direction, with
sample-dependent amplitudes. This is not a uniform output-bias shift: the
sample-coherent fraction is only about0.3–0.4%.

## Role of the last linear layer

The last layer is uncapped, so tanh does not directly restrict each output to
[-1,1]. For a fixed head, however, a finite bound still holds:
`||f(x)|| <= ||W3||_2 sqrt(200) + ||b3||`.
Neither bounded outputs nor bounded hidden activations guarantee that a finite
optimizer update reduces squared error. Here the head's spectral norm is below
one, so attributing the spikes to an exploding final-layer gain is unsupported.

For each rise and recovery we decompose the output change exactly:

`delta_f = W3_before delta_h2 + delta_W3 h2_before
           + delta_W3 delta_h2 + delta_b3`.

The affine-head identity is evaluated in FP64 on retained FP32 features and
parameters; native FP32 output rounding is measured separately. Squared-loss
change is allocated using the old residual plus symmetric sharing of cross
terms. The hidden-feature term accounts for98.75%,98.07%,99.48%,98.94% of the
loss rises at474,493,503,522 respectively. This is a different, explicitly
defined decomposition from parameter-group Shapley values. It agrees with the
fixed-head replacements: changing hidden features drives the observed spikes,
while the old head already reads those changes into prediction errors.

## The incoming Adam steps overshoot a lower-loss region

We evaluate the full objective at25 points on each actual incoming displacement:
`theta(alpha) = theta_before_step + alpha * actual_Adam_increment`,
with alpha0..1.5 in increments of1/16 and exact native endpoints at0 and1.
This changes the scale of one frozen parameter displacement, not optimizer
moments or an entire training trajectory.

| Incoming step | Loss before step | Best sampled alpha | Loss there | Actual full-step loss |
| --- | ---: | ---: | ---: | ---: |
| 473→474 | 0.0477048 | 0.4375 | 0.0258118 | 0.0536730 |
| 492→493 | 0.0262139 | 0.3750 | 0.0243767 | 0.0294607 |
| 502→503 | 0.0275415 | 0.5000 | 0.0230138 | 0.0286617 |
| 521→522 | 0.0489077 | 0.5000 | 0.0213565 | 0.0550626 |

All four raw-gradient dot actual-step products are negative. For473→474,
the first-order prediction is-0.0945439, while the measured loss change is
+0.00596816; the exact nonlinear remainder is+0.1005121. That remainder includes
all higher-order effects and is not presented as a measured quadratic Hessian
term. The local descent direction is overshot by the full finite step. The
sampled alpha minimum is not a continuously optimized step size or a validated
training policy. Momentum/gradient alignment at these steps is mostly positive;
the observed peak is not explained by an immediately uphill aggregate direction.

## What happens during recovery?

Loss returns without reversing the gradual parameter-norm growth. For the
large event, W1 norms are14.3112→14.3624→14.4359 and W2 norms
8.81935→8.82597→8.82940 at465/474/485. Hidden activation RMS stays similar.
The hidden2 recovery displacement partially opposes its rise displacement
(cosine-0.825 for474, -0.845 for493, -0.389 for503, -0.908 for522).
This is a partial correction of the representation's output-sensitive motion,
not a return to identical weights/features or a collapse in parameter size.

## Figures and reproducibility

Figure01, `../figures/01-internal-time-course.pdf`: unchanged Adam full-objective
loss and all-layer weight norms, bias norms, activation RMS, saturation and
mean tanh slopes over448..535. Magnitude changes are relative to448 where
labeled. The blue band marks502..505; the dotted line marks512. All5,000
examples and every hidden/output coordinate are included. No smoothing or
uncertainty bands; dependent time points from one seed.

Figure02, `../figures/02-mechanism-probes.pdf`: Shapley shares of loss rise over
six parameter groups; norm-restored and norm-only static networks; head-row-space
energy of baseline features versus feature change; and full-objective loss
along actual incoming peak steps. Peak indices are474,493,503,522. Norm-probe
losses are normalized by the event's pre-rise loss; line-search losses use the
loss immediately before the peak's incoming update. The dotted5% reference
assumes isotropic feature energy only as a geometric benchmark. No retraining.

Generator `../01_report.py`, numerical tables and hashes `../results.json`,
plotted values `../plotted_values.csv`. Raw source: Run236
`artifacts/spark/payload/worker/{internals,counterfactuals,step_curves,steps}.jsonl`
and15 paired parameter/activation snapshots. Source/cache/checkpoint identities
are pinned by its42-entry manifest. `02_feature_probe.py` and
`03_norm_direction_probe.py` record the declared post hoc Spark analyses with
separate source hashes and retrieval receipts; no executed Run236 input changed.

Run236 `03_verify.py` independently checks42 retrieved artifacts, all source
hashes,536 exact loss/norm pairs, state512 tensor/RNG equality,15 full snapshots,
MSE identities and Shapley via all720 permutations rather than the worker's
subset weighting. Analysis `04_verify_probes.py` verifies feature geometry on
CPU and all32 static norm-probe losses on all5,000 examples; maximum CPU/Spark
loss difference5.59e-9. Full plot read-back covers1,372 plotted points; embedded
fonts, vector-only PDFs, text bounds and final renders are verified. Scratch
PNGs remain ignored. Floating-point tolerance adjustments affect verification
only (FP32 saturation threshold ties and stacked-bar subtraction rounding).

Run236 completes535 replay updates in12.89 seconds total, including88 observer
states,512 hybrid evaluations and100 step-curve evaluations. Peak CUDA allocation
351,147,520 bytes. Post hoc probes add32 norm evaluations and feature algebra,
with no training updates. All task processes exited; checkpoints, arrays, raw
logs and a size/SHA-verified archive remain local and on Spark. Storage and
zero cloud cost are recorded in Run236 `completion.json`. No manuscript edit,
external publication, resource deletion or generalization claim.
