# Observation 001: Cumulative slopes, ratio plateaus and loss

Question: H006 asks whether A/B slopes become equal, whether a flat B/A tracks
loss convergence, and whether it implies that new information is destroyed.
The exact user wording and exploratory reduction choices are in
[design.json](../design.json).

Sources: Run 002 attempt02 full-pass events and scientific verification, matched
to the original comparison figure's embedded rows and SHA-256. Source paths,
sizes, before/after hashes, reduced values and software versions are retained in
[results.json](../results.json). Source evidence was unchanged.

Coverage: one Pythia-14M / MiniPile / AdamW trajectory, seed 1234. All 712 updates
are in [curve_data.csv](../curve_data.csv); publication figures show full-sized
updates 1-711. The final 311-sequence batch is retained in sensitivity. Whole-model
raw gradients are measured before clipping; the adaptive stream is the
bias-corrected preconditioned direction from actual optimizer state, excluding
LR and decay. It is not the first moment or actual parameter update.

Reduction: exact backward differences of A/B and their quotient, complete
trailing 31-update means for norm increments, and trailing 31-point OLS slopes
for loss and B/A. Recent slope ratios are ratios of sums of increments.
Loss means are supervised-token weighted; slope regression uses equal update
weights. Full definitions, null handling, smoothing 15/31/61, terminal32/64/100/128
windows and partial-batch sensitivity are in [METHOD.md](../METHOD.md).

Result: at 612-711, raw mean A/B slopes are 0.322505/0.148944 and adaptive slopes
1236.298989/965.709908. Ratios rise from 0.324905 to 0.338631 and 0.693394 to 0.704758.
Recent slope ratios are 0.461833 and 0.781130. Both B increments and all prefix
alignments are positive throughout this interval.

Raw B/A is almost unchanged across424-484, 0.308506 to 0.308516, while B grows
by 0.132705 per update and training-loss OLS slope is-0.00088518. The plateau
condition is proportional growth, not zero resultant growth. Terminal loss
progress is slower, but no convergence or information-storage conclusion follows.

## Figure captions and legends

[cumulative_slopes.pdf](../figures/cumulative_slopes.pdf): Columns compare raw
gradients and adaptive Adam directions on separate norm scales. Top: original
A (orange solid) and B (blue dashed) curves. Middle: faint exact increments and
trailing 31-update means; dotted black gives the mean B increment required to
keep B/A flat. Bottom: cumulative B/A (green solid) versus recent slope ratio
(purple dashed). All panels show updates 1-711.

[ratio_slope_and_loss.pdf](../figures/ratio_slope_and_loss.pdf): Top shows changing-batch
training loss (faint gray) and trailing 31-update token-weighted mean (black).
Middle shows the trailing 31-point OLS loss slope. Bottom compares raw (blue
solid) and adaptive (orange dashed) OLS ratio slopes. Slope axes use symmetric
logarithmic scaling with zero reference lines. No intermediate validation curve
exists; loss is training loss, not a fixed-objective convergence test.

Generating scripts: [01_analyze.py](../01_analyze.py) and
[02_verify.py](../02_verify.py). Source/hash checks, scalar algebra, aggregation
and four synthetic cases passed; both PDF renders were visually inspected.

Uncertainty and caveats: one correlated trajectory, no seed confidence bands.
Smoothing and window sensitivity are descriptive. The near-minimum intervals
are retrospective. Preconditioning changes scales, cumulative ratios have
history-dependent responsiveness, and Adam diagnostic vectors exclude LR/decay.
A plateau does not quantify stored information or establish asymptotic convergence.
See [the full answer](../REPORT.md).
