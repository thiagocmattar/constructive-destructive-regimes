# Observation 001: exact long-lag anchored gradient profiles

Question: how far does the previously observed odd/even gradient oscillation
persist beyond lag 10, for the same nine EOS conditions and three saved starts?

Sources: Runs 206/207/208 frozen configs/input manifests, per-worker inventories, exact
per-window results and sparse full vectors; Run205's verified source histories
and end states. `checkpoint_provenance.json` hashes all 27 source windows.
`handoff_captures.json` and activation records trace the accelerated resumes.
`02_analyze.py` audits/reduces the evidence and generates the four PDFs;
`03_record.py` writes the report/summary. Exact values are in `lag_profiles.csv`.

Coverage: 27 seed 0 profiles, 25 primary, common lag 256, maximum measured lag
2048; 36,096 anchored cosines.
The 17 primary initially negative lag-1 profiles include 4 satisfying
the declared final-pair envelope threshold and 13 still above it.
See REPORT.md's table for each LR/anchor and stop. Failed Run062 replay starts
at 2000/3000 remain flagged and excluded from primary plots.
The source cohort contains 21 direct saved starts and six reconstructed starts;
four reconstructions met the earlier historical checks and enter the primary
set. Qualification does not imply bitwise equality of later continuations.

For example, the early Transformer at LR 0.2 has local half-gap 0.225 at lag 512, rising to 0.600 at 1024 and ending at 0.326 at 2048. More lags reveal a rebound, not monotonic damping.

Raw cosine reduction: no random projection, averaging of positions, smoothing or centering.
FP64 dot products/norms of FP32 full-objective gradients. The envelope used
for stopping is median(|rho_(2m)-rho_(2m-1)|/2) over the last 32 pairs; it is
a descriptive reduction, not a validation or optimization criterion. CPU/GPU
reductions agree within 2.89e-15; 95 retained
full-vector audits pass. Each plotted artist is compared to its exact source
arrays. Numerical validation is in verification.json; visual review is separate.

Caption, `common_lag256.pdf`: Whole-model raw full-objective gradient cosine
from states 500/2000/3000 for the 25 primary Analysis024/032 conditions.
Columns show MLP, ResNet-32 and Transformer; rows show anchors. Color identifies
learning rate. Solid/dashed curves join measured even/odd lags separately,
without smoothing. Each lag uses one anchored pair; all panels share lag 256
coverage. This figure displays the common measured range before adaptive
extension, without a stability or generalization interpretation.

Caption, `long_lag_profiles.pdf`: Same 25 primary anchored profiles with
base 2 logarithmic lag axis and declared adaptive terminal horizons. Initially
positive lag-1 controls stop at 256. Initially reversing profiles extend through
512/1024/2048 until the final 32-pair median half-gap is<=0.1 or a bound is
reached. Dotted vertical lines mark 256. Curves ending at different lags have
different measured coverage; persistence at a bound is not evidence of eventual
decay. Exact endpoint gaps and stopping reasons are in the report.

Caption, `all_profiles_with_reconstruction_flags.pdf`: Same longer profiles
including all 27 windows. The unqualified Run062 historical reconstructions at
anchors 2000/3000 are explicitly labeled [U] and faded. They are exploratory
continuations and do not become historical matches because this run completes.

Caption, `alternating_gap.pdf`: Declared local attenuation score for the 25
primary profiles: the rolling median of half the absolute difference between
an even-lag cosine and its preceding odd-lag cosine, over 32 complete pairs
(64 lags). The horizontal dashed line marks 0.1 and the vertical dotted line
marks common lag 256 coverage. This companion is a derived summary of exact
raw cosines, not an additional sample or independent observation. Rebounds
remain visible. A small gap need not mean zero cosine offset or permanent
decorrelation. The raw figures retain every cosine without smoothing.

Uncertainty/caveats: one initialization per case and one dependent pair per lag,
no confidence interval. Adaptively selected horizons are not fixed-coverage
comparisons beyond 256. Full-objective deterministic GD differs from minibatch
Pythia histories. Selected Transformer compute chunks preserve the objective
but change FP32 accumulation; hardware/rounding can change future trajectories.
Damped alternation may leave a nonzero cosine offset. Anchored geometry alone
does not measure gradient-norm or parameter-oscillation amplitude decay. It
does not establish that consecutive gradients stop reversing, even within the
measured range. It also
does not measure stability, generalization, optimizer utility or full temporal
displacement accounting. No new held-out evaluation or manuscript claim.
