# Observation 001: Loss and raw-gradient cancellation across sizes

## Question and provenance

H003 records the user's expectation that larger models have a less destructive
regime, possibly through dimensionality, explaining their better performance.
The pilot existed when this question was posed. This is an exploratory reduction,
not a preregistered test. Different meanings of intensity yield different trends;
the proposed causal explanation is untested.

Primary cohort: Run 002 attempt 02 (14M), Run 003 attempt 01 (31M), Run 004 attempt 01
(70M). Sources are their completed `artifacts/<attempt>/full-pass/events.jsonl`,
summaries, resolved configurations, input identities, environments, verification
records and verified retrieval receipts. Exact paths, SHA-256, pinned model/data/
tokenizer revisions, cache/order identities and matched contract are in
`../results.json`. Completed source hashes are checked against Analysis 001's
frozen results. Run 001 attempt 04 supplies a separate 14M replay sensitivity only.

Actual unique trainable parameter counts: 14,067,712; 30,494,720; 70,426,624.
The primary comparison matches seed 1234, random SmallInit/WangInit recipe,
token cache and permutation bytes, six layers, packed length 2048, effective
batch 1024/microbatch 4, token-weighted causal cross entropy, AdamW groups/betas/
epsilon/decay, clipping 1, LR schedule, precision and measurement boundary.
Width 128/256/512 and heads 4/8/8 vary. Initialization recipe matches; tensor shapes,
values and width-dependent scales differ. This is matched in tokens and updates,
not compute or a pure intervention on dimension. Exact configuration values and
source hashes are retained rather than inferred from model names.

Each model consumes one complete MiniPile pass: 712 accepted optimizer boundaries,
728,375 packed sequences, 1,491,711,416 input tokens and 1,490,983,041 supervised
targets. Updates 1-711 have 1024 sequences; update 712 has 311. No tokens or short
tails are dropped. Initial/final validation covers all 500 fixed documents,
693,668 inputs and 693,329 supervised targets including its short tail. There
are no intermediate fixed-validation curves. Training loss uses changing batches,
not repeated evaluation of a fixed training objective.

## Metric and aggregation contract

At boundary t, g_t is the whole-model accumulated, unscaled raw gradient of
supervised-token mean next-token loss, before clipping, Adam, LR multiplication
or optimizer decay. All unique trainable parameters count once. Weights/gradients
are FP32 under dynamically scaled FP16 training; cumulative vectors and reductions
are FP64. These are raw gradients along an AdamW trajectory, not plain-SGD
experiments or actual parameter displacements. The first scheduled LR is zero.

rho_t = dot(g_(t-1), g_t)/(norm(g_(t-1))*norm(g_t)). The first raw cosine is null;
zero norm denominators would also be null with a reason. All remaining 711 pairs
are defined. Local "destructive" means rho<0, without assuming harm to learning.
For equal weighting over unsmoothed finite pairs:

```text
F_minus = mean(1[rho < 0])                 # frequency
D_minus = mean(max(-rho, 0))              # unconditional negative intensity
M_minus = mean(-rho | rho < 0)            # conditional negative magnitude
D_minus = F_minus * M_minus
```

M_minus is undefined with no negative pairs. Exactly zero is neither positive
nor negative. These statistics are dimensionless. Correlated updates are not
independent replicates; no seed confidence interval or significance test is used.

Figures use centered 31-update means. Loss is supervised-token weighted; cosine,
F_minus and D_minus use equal finite-pair weights. Edge windows shrink and nulls
are omitted. The first displayed smoothed cosine therefore uses neighboring
defined pairs, while the first raw cosine remains null. This is retrospective
smoothing, not an online detector. All 2136 raw update records and plotted
reductions are in `../figure_data.csv`.

Full statistics include updates 1-712. Both figures display 1-711 to keep the batch
size fixed, preserving 712 in raw data and summaries. Sensitivity excludes 712
and repeats smoothing at 15/31/61 updates. Fixed quarters 1-178, 179-356, 357-534,
535-712 are descriptive, not fitted regime boundaries. The pair at a quarter's
first update includes the previous gradient; counts are 177, 178, 178, 178.

## Results and sensitivity

| Model | Mean cosine | Median cosine | F_minus | D_minus | M_minus | Last 31 training loss | Full final validation loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14M | 0.028246 | 0.007220 | 49.23% | 0.160561 | 0.326168 | 5.231147 | 5.182392 |
| 31M | -0.057773 | -0.086252 | 56.96% | 0.175342 | 0.307822 | 4.586877 | 4.611768 |
| 70M | -0.123108 | -0.160566 | 67.65% | 0.200192 | 0.295919 | 3.984634 | 4.063201 |

From 14M to 70M, negative frequency rises 18.42 percentage points and unconditional
intensity rises 24.7%, while conditional negative magnitude falls 9.3%. Thus larger
models have lower measured losses alongside more frequent local cancellation.
This does not identify cancellation as either causing or preventing better learning.

In the last quarter, F_minus is 2.81%, 17.98%, 48.31% and D_minus is 0.002440,
0.023681, 0.092977. The size difference is especially clear late in training.
Ordering is not uniform across phases: quarter 2 D_minus is 0.305018, 0.322443,
0.278723. The curves do not establish exactly two discrete latent regimes or
a phase transition.

Excluding the partial final boundary gives F_minus 49.15%, 57.04%, 67.61%,
D_minus 0.160653, 0.175589, 0.200381, and M_minus 0.326831, 0.307822, 0.296398. These
aggregate orderings survive the batch sensitivity. The fraction of centers with
negative smoothed cosine at 31 updates is 53.87%, 61.18%, 78.48%; this ordering holds
at 15 and 61 as well. Minimum smoothed cosine ordering depends on width: 31M is
deepest at 15 updates, 14M at 31 or 61. A deepest-trough claim is smoothing-sensitive.

Run 001's same-seed 14M replay gives F_minus 53.16%, D_minus 0.176398, M_minus 0.331797,
median cosine -0.043422, validation 5.182962. Its D_minus slightly exceeds 31M's
0.175342, so replacing Run 002 removes strict monotonicity of unconditional
intensity. Frequency still increases and conditional magnitude still decreases.
This is numerical replay sensitivity, not independent seed evidence. It reinforces
the need for the ongoing seed replication before making a robust size claim.

## Adjacent versus cumulative interaction

Saved scalar norms permit this algebraic reduction, up to floating-point error:

```text
E = sum_t norm(g_t)^2
L = 2 * sum_(t>=2) rho_t * norm(g_(t-1)) * norm(g_t)
B2 = norm(sum_t g_t)^2
J = B2 - E = 2 * sum_(u<t) dot(g_u, g_t)
J - L = signed aggregate of all nonadjacent pair interactions
```

Norm-weighted L differs from the equal-pair cosine mean. Individual omitted
lags, per-sample energy versus intra-batch alignment, and actual AdamW displacement
cannot be recovered by this reduction.

| Model | E | Lag-one L | All-pair J | Exact B2 | Lag-one E+L | Relative error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 14M | 166.025428 | +3.344271 | +11435.220533 | 11601.245962 | 169.369699 | -98.54% |
| 31M | 160.068830 | -4.819859 | +10081.846721 | 10241.915551 | 155.248971 | -98.48% |
| 70M | 171.037709 | +1.681282 | +8710.800142 | 8881.837851 | 172.718991 | -98.06% |

Net whole-trajectory raw-gradient interaction is positive at every size despite
many negative adjacent pairs and B/A<1. Lag-one truncation severely underestimates
the squared resultant norm in this cohort. This tests the manuscript's raw-gradient
norm subexpression, not its complete constant-step SGD parameter-norm identity
under these AdamW runs. Low B/A alone is not evidence for net negative interaction;
orthogonality also lowers that ratio.

## Captions, checks, and limits

`loss_and_raw_cosine.pdf`: Training loss (left) and cosine between adjacent
whole-model raw gradients before clipping (right), for matched Pythia 14M/31M/70M
MiniPile trajectories, seed 1234. Lines are centered 31-update means over full-sized
updates 1-711; edge windows shrink. Loss is supervised-token weighted and cosine
uses equal finite-pair weights. Zero marks local reinforcement/cancellation.
Update 712 is omitted only from display. Color/line style redundantly identify
size. One trajectory per size supplies no seed uncertainty band.

`negative_alignment.pdf`: Negative adjacent-pair frequency (left) and mean
max(-rho, 0) (right) in centered 31-update windows of the same cohort, colors/styles
and coverage. The right statistic combines incidence and magnitude, rather than
conditioning on negative pairs. Neither quantity identifies harmful learning.
The undefined first raw cosine is omitted from window reductions.

Generated by `../01_analyze.py`; checked by `../02_verify.py`. Five mathematics
checks cover sign factorization/undefined cases, weighted smoothing, exact temporal
identities, and a quadratic counterexample with negative alignment and decreasing
loss. Source hashes, 2136-row reductions, manuscript/live scientific input preservation,
PDF text bounds and visual renders were checked; see `../verification.json`.

Limits: one pilot seed, architecture and initialization co-variation, same nominal
LR rather than separately tuned size recipes, changing parameters/batches,
whole-model aggregation possibly hiding layer effects, no actual-update loss
analysis or causal intervention. No stability, generalization mechanism, universal
scaling law or independent dimension effect is established. Runs 005-010 and
Analysis 002 remain untouched. Manuscript implications are in `../REPORT.md`.
