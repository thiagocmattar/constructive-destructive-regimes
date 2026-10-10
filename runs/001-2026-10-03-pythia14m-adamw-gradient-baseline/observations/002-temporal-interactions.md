# Observation 002: adjacent versus full temporal interactions

The adjacent-only reconstruction understates the final squared raw-gradient
resultant norm by 98.58%. The aggregate adjacent interaction is slightly negative
(−3.7242), while the sum over all cross-step pairs is positive (+11,404.7797).
This shows that adjacent signs and the full temporal aggregate describe different
features of this trajectory. The residual is aggregate evidence for all lags
greater than one; it does not identify any particular omitted lag.

This secondary reduction uses only the requested full-pass scalar records; it adds no training intervention or gradient-history requirement. Source SHA-256: `ec6aec5633aed84285976934720d8b3040c1227c0d45e0fb0eb48b2921a0c6ac`. Script: `../05_temporal_reduction.py`; all prefix values: `../figures/temporal_data.json`. Coverage: all 712 accepted updates of seed 1234, including the smaller final update. No smoothing, filtering, or uncertainty band.

For each prefix define E = sum ||g_u||², L = 2 sum of adjacent rho_u ||g_(u-1)|| ||g_u||, and J = ||sum g_u||² − E. Then J − L is the signed sum of all nonadjacent pair interactions (twice each pair dot product), up to floating-point reduction error. E + L is an adjacent-only estimate of the squared raw-gradient resultant norm; it is kept signed and never clamped before analysis.

At the final update: exact squared resultant norm 11572.491773; energy E 167.712081; all interactions J 11404.779692; adjacent interactions L -3.724169; omitted nonadjacent interactions J−L 11408.503860. The adjacent-only value is 163.987913, a relative squared-norm error of -0.985830. There are 0 prefixes with a negative adjacent-only squared-norm estimate.

Caption (`temporal_interactions.pdf`): Exact cumulative raw-gradient squared norm compared with energy plus lag-one interactions, and the signed adjacent/nonadjacent interaction sums along Run001. Values come from the saved whole-model norms and adjacent cosines, without reconstructing every pair separately. One trajectory, no replicate uncertainty.

Interpretation: a ratio ||sum g|| / sum ||g|| below one alone does not establish net negative temporal interaction; orthogonality also reduces the ratio. J supplies the net sign for raw gradients. This reduction examines the cumulative raw-gradient norm term only. Under AdamW and a varying learning rate, it is not a reconstruction of parameter norm, displacement, or the manuscript's full constant-step SGD identity, and it does not establish stability or causality.
