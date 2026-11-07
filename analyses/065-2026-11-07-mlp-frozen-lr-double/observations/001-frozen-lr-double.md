# MLP interventions against the unchanged baseline

This observation records I1 and I2. The subsequent [I3 extension](002-frozen-lr-0p02.md)
adds LR 0.02 to the same analysis and current figure; all original measurements
below are retained. I3 is now the third baseline-comparison column.
The subsequent [halving extension](003-lr-halving.md) adds I4-I8; the current PDF
uses four pages, with matched scales and unchanged baseline in every column.

## Conclusion

Both **I1: freeze linear 1** and **I2: freeze linear 1 + double LR at 500** suppress
the baseline's large loss excursions and representation motion. I2 reintroduces
small output reversals while retaining the large amplitude reduction. Over
448-800, their mean training losses remain **22.27% and 20.58% above unchanged
Adam**, respectively. Smoother training is not uniformly faster optimization.

Analyses 064 and 065 are one ongoing intervention study. Preserve 064 as the
first-stage record and continue iterations in 065. An intervention comprises
all changes from baseline, including inherited changes. Every intervention is
compared against unchanged Adam; I2-versus-I1 comparisons are supplementary.

## Comparison and execution

The seed-0 CIFAR-10 MLP is 3072 -> 200 tanh -> 200 tanh -> 10 linear. All arms use
the same 5000 images/order, physical batch 1000, mean per-example half squared
error, FP32 arithmetic and TF32 off. Baseline Adam uses LR 0.001, betas
(0.9, 0.999), epsilon 1e-8, and no clipping or weight decay.

I1 freezes first-linear weights and biases from state 448. I2 inherits that
freeze and changes LR to 0.002 at state 500; **500 -> 501 is the first changed
update**. No optimizer reset or unfreeze occurs. Native Adam computes updates
and advances moments/clocks; first-layer parameters are restored before the
next forward. Their applied increment is exactly zero. Full raw gradients
remain available even for frozen parameters. This is not paused-moment freezing.

The unchanged baseline is reused from the exactly qualified retained evidence
in 064. Run 245 reconstructs the frozen parent from its full state 448 and
matches all 54 boundary/gradient states through 501 and all five representation
tensors at 499-501. The restored start at 500 differs only in LR. First-step
Adam moments/clocks match the frozen LR-0.001 parent; doubling actual later-layer
increments agrees within FP32 rounding (maximum error 1.49e-8). No complete
historical checkpoint at 500 is claimed. First-layer parameters remain exactly
fixed through 801; later layers update and all parameters stay finite.

## Results

The 500-550 response window includes 51 outgoing updates and 50 within-window
loss transitions. The unchanged baseline is the reference in every column pair.

| Quantity, 500-550 | Unchanged baseline | I1: freeze | I2: freeze + double LR |
|---|---:|---:|---:|
| Mean loss | 0.0289606 | 0.0281841 | 0.0280714 |
| Maximum loss | 0.0550626 | 0.0283354 | 0.0283517 |
| Loss at 550 | 0.0273520 | 0.0280555 | 0.0278545 |
| Upward loss transitions | 25/50 | 0/50 | 1/50 |
| Negative output-change cosines | 49/51 | 7/51 | 36/51 |
| Maximum output displacement RMS | 0.155872 | 0.00140750 | 0.00358001 |

I1 and I2 reduce maximum output RMS by 99.10% and 97.70% versus baseline.
In I2, only the first loss transition after the LR change rises, by 0.00001632
(0.0576%); subsequent loss transitions through state 800 decrease.
The second hidden representation reverses on 0/51 and 1/51 updates under I1/I2,
versus 49/51 in baseline. Zero first-layer motion has undefined cosine.

| Quantity, 448-800 | Unchanged baseline | I1 | I2 |
|---|---:|---:|---:|
| Mean training loss | 0.0228078 | 0.0278867 | 0.0275019 |
| Mean loss relative to baseline | Reference | +22.27% | +20.58% |
| Loss at 800 | 0.0302502 | 0.0271296 | 0.0263062 |
| Positive loss variation | 0.364412 | 0.00271393 | 0.00273026 |
| Negative output-change cosines | 351/353 | 12/353 | 58/353 |

Supplementary incremental comparison: I2 lowers loss at 800 by 3.04% and mean
loss over 500-800 by 1.63% relative to I1. Its increased small reversals do not
restore baseline-sized spikes. Baseline repeatedly reaches lower inter-spike
losses, explaining its lower long-window mean despite a worse endpoint.

## Measurements and figure caption

[Figure 01](../figures/01-frozen-first-linear-double-lr-cifar10-mlp.pdf) puts the
interventions side by side in groups of four; each overlays unchanged baseline with matching
axis scales. Page 1 covers 448-800: loss; projected-gradient diagnostic;
gradient cosine; full and later-layer gradient norms; output-change cosine;
output coordinate RMS; and later-layer LR. Page 2 traces all five learned
representations over 500-550, comparing change cosine and coordinate RMS.
No smoothing or per-example averaging precedes the representation reductions.
The extended figure uses log axes for loss, gradient norms, output RMS and LR;
the projected-gradient diagnostic uses a symmetric-log axis linear within +/-1.
Layer RMS uses a symmetric-log axis linear within +/-0.0001, preserving exact zeros.

At state $t$, $g_t$ is the raw gradient of the full mean training objective.
We plot $\rho_t=\cos(g_t,g_{t-1})$ and
$\eta\widehat\kappa_t=1-\langle g_t,g_{t-1}\rangle/\|g_{t-1}\|^2$.
The latter is a projected-gradient diagnostic under Adam, not a measured
curvature or a stability threshold. Full gradients contain 656810 parameters;
later-layer statistics restrict to the **same 42210 parameters in every arm**.
In row (c), solid denotes full gradients and dashed denotes this later subset.
Full-gradient alignment can obscure later-layer reversals because it includes
derivatives for the large frozen block. Neither statistic is Adam-update alignment.

For a representation $A_t$, define outgoing displacement
$\delta_t=\mathrm{vec}(A_{t+1}-A_t)$, cosine
$\cos(\delta_t,\delta_{t-1})$, and RMS $\|\delta_t\|/\sqrt P$.
Each hidden matrix has 5000 sample rows and 200 feature columns, so $P=10^6$;
outputs have 5000 rows and 10 columns, so $P=50000$. All coordinates are retained.
State 801 supplies the outgoing update at 800. At 500, state-based metrics are
still shared by I1/I2 but outgoing displacement already reflects the new LR.
No new SVD is fitted here; dense captures preserve that option.

## Coverage, uncertainty and provenance

This is one deterministic seed, fork and finite horizon, with no validation or
generalization assessment, LR sweep or matched later-layer freeze. The first
block contains 93.57% of parameters. Results do not establish a unique initiating
cause, a universal stable LR, or that later layers cannot generate reversals.
Direction, amplitude and loss must be interpreted together.

Sources: [064 comparison](../../064-2026-11-07-mlp-freeze-first-linear/comparison.json),
Run 244 retained boundaries/representations and
[Run 245 configuration](../../../runs/245-2026-11-07-mlp-frozen-lr-double/config.json),
native boundaries/representations, dense arrays, qualification and inventory.
Generating scripts: [01_reduce.py](../01_reduce.py) and [02_figures.py](../02_figures.py).
[comparison.json](../comparison.json) pins source hashes; [metrics.csv](../metrics.csv)
contains the four declared windows. Independent FP64 reductions check every
dense layer cosine/RMS at 500-550 and raw/later-gradient statistics at 500-551.
The previous baseline/freeze window summaries reproduce; both columns' baseline
arrays match exactly. Both PDF pages were rendered and reviewed.

Run 245 completed in 7.76 s on owned Spark (6.46 s for 301 intervention updates),
peak allocated GPU memory 456.7 MB. All 66 source files and 22 retrieved files
pass identity checks. Worker exited; 2.021 GB is retained remotely, with verified
local copies. No cloud resources or charges; figure revision requires no new run.
