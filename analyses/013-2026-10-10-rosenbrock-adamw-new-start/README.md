# Analysis013: Rosenbrock AdamW from (0, 1)

Completed and verified, 2026-10-10. **[Open the single comparison figure](figures/rosenbrock_learning_rates.pdf).**
The user selected start **(0, 1)**. Repeat Analysis012 at the same three LRs,
weight decay 0.1, momentum, 2,000 updates and numerical settings; change only
the initial point. New attempts and records preserve the previous cohort.

Same latest layout: a large full-domain contour/trajectory panel above eight
metric panels, with all three LRs overlaid. Cosine panels (c,d) use trailing
100-update means with complete finite windows, first at update 101. Norm/A/B
panels (e,f,g) use linear Y; loss uses log Y; ratios are B/A. No zoom or thinning.
The starting marker and label use the actual saved initial coordinates.

| LR | New run | Final objective | Final point | Final gradient norm |
| --- | --- | ---: | --- | ---: |
| 1e-2 | [038](../../runs/038-2026-10-10-rosenbrock-adamw-new-start-lr-1e-2/README.md) | 0.000972263060 | (0.969356332, 0.940228143) | 0.307249 |
| 3.162277660e-3 | [039](../../runs/039-2026-10-10-rosenbrock-adamw-new-start-lr-midpoint/README.md) | 0.000117958119 | (0.989636698, 0.979705757) | 0.162892 |
| 1e-3 | [040](../../runs/040-2026-10-10-rosenbrock-adamw-new-start-lr-1e-3/README.md) | 0.000011047136 | (0.996694358, 0.993365021) | 0.00998323 |

**LR 1e-3 has the lowest final objective here**, changing the ordering from
Analysis012, where the midpoint had the lowest final objective. All three now
reach the optimum neighborhood within the horizon, with ongoing loss fluctuations.
None hits the joint loss/gradient threshold or passes its final-100 sustained
criterion. These are fixed-budget outcomes, not proven convergence or a universal
learning-rate ranking.

| LR | Final loss, old start (-1.2, 1) | Final loss, new start (0, 1) | Raw negative pairs, old | Raw negative pairs, new |
| --- | ---: | ---: | ---: | ---: |
| 1e-2 | 0.00104246819 | 0.000972263060 | 66.6333% | 69.8849% |
| 3.162277660e-3 | 0.000115598949 | 0.000117958119 | 44.9725% | 59.6798% |
| 1e-3 | 0.234111865 | 0.000011047136 | 0% | 25.0125% |

The largest change is at LR 1e-3: the new path reaches a much lower objective
despite gaining negative raw adjacent pairs. New momentum negative fractions
are 0.5503%, 0%, 0% in table order. Prefix B/A remains high even where late
raw cosine means are negative, because it incorporates the whole history and
its vector magnitudes. It does not identify the local loss response by itself.

Matched settings: f=(1-x)^2+100(y-x^2)^2, optimum (1,1), CPU float64, exact
analytic gradients, zero initial moments, betas (0.9,0.95), epsilon 1e-8,
decoupled weight decay 0.1 on both coordinates, no clipping, constant LR and
2,000 updates without early stopping. Initial f is 101 versus 24.2 previously;
the intervention changes initial objective, direction and decay displacement.
No data, stochastic seed, validation split, cloud resource or billing.

[Method](METHOD.md) · [Observation and caption](observations/001-initial-point-comparison.md) ·
[Reductions and paired comparisons](results.json) · [Scientific verification](verification.json) ·
[Figure provenance](figure_manifest.json) · [Figure verification](figure_verification.json).

All 6,000 independent PyTorch fixed-gradient updates agree within 3.34e-16
point error. Analytic objective/gradient, exact replay, all-stream geometry,
momentum identity, decay displacement and artifact size/hash checks pass.
All three previous controls are checked read-only for source/config/history/
summary integrity and matching except initial point. Every rolling mean agrees
with its direct window. The nine-panel, single-page PDF passed integrity checks
and visual review, retaining Analysis012's landscape domain and LR encodings.

Reproduce from the root with `python analyses/013-2026-10-10-rosenbrock-adamw-new-start/01_verify.py`,
then `02_plot.py` and `03_record.py` at the same path. Preparation used
`00_prepare.py 0 1`; preparation/execution refuse existing run inputs/attempts.
Do not rerun numerical execution into completed attempts. The numerical source
is unchanged; plot code reads the stored starting point instead of hardcoding it.
Older figure provenance retains its original source hashes and Git revisions.

Raw histories total 2,837,664 bytes and remain ignored locally; compact configs,
summaries, final states, inventories, manifests and receipts are versioned.
Simulation/saving totals about 0.184 seconds; verification and plotting add
seconds. All evidence is retained locally. Two deterministic starts of one
function/recipe supply no sampling uncertainty, general stability, causal
alignment mechanism, asymptotic guarantee or LLM conclusion.
