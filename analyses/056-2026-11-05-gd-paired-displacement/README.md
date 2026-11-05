# Analysis056: local reversal and accumulated GD displacement

Requested by the author in `manuscript/draft/revisions/2026-11-05-human-input/`.
This is a retrospective analysis of existing evidence, with no new training,
scientific input changes, hypothesis, remote execution or cloud spending.

## Comparison and measurement contract

Reuse all nine Analysis024 trajectories: three fixed learning rates per model,
one shared initialization within each model, 3,000 full-objective plain GD
updates. Data preparation, objectives, parameter scope, FP32 training and FP64
geometry are unchanged. Learning rate varies; other factors are matched within
each model. These comparisons are conditional on one initialization and do not
isolate architecture from dataset/objective differences.

For raw whole-model gradients `g_t` applied at updates `t=1,...,T`, define
`A_T=sum ||g_t||`, `B_T=||sum g_t||`, `E_T=sum ||g_t||^2`,
`R_T=B_T/A_T`, `R_perp=sqrt(E_T)/A_T`, and `Q_T=B_T^2/E_T`.
`R_perp` is the algebraic zero-cross-term reference with observed lengths,
not a fitted random-walk model. `Q_T-1` has the sign of the all-pair interaction.
Do not confuse this cumulative Q with the Adam directional statistic elsewhere.

At even prefixes, pair `(g_1,g_2),...,(g_(T-1),g_T)`. Reconstruct pair norms
from unsmoothed adjacent cosines and norms, using the stable expression
`(a-b)^2+2ab(1+rho)`. Let `H_T` be the sum of these norms. Then
`C_T=H_T/A_T`, `D_T=B_T/H_T`, and `R_T=C_T D_T`.
As a pairing-offset sensitivity check, use `(g_2,g_3),...,(g_(T-2),g_(T-1))`
and retain `g_1,g_T` as singleton boundary segments. Both partitions cover the
same entire prefix and have the same numerator. This is not a cropped-window
displacement estimate, which cannot be recovered by subtracting two norms.

Zero path/energy makes the corresponding ratio undefined. If all paired sums
vanish, C is zero and D is undefined; the unnormalized identity B=0 still holds.
Cosines outside [-1,1] by more than numerical tolerance are errors; any tiny
clamping is counted. No smoothing is used before nonlinear reductions.
Floating-point parameter updates need not equal -eta times the logged gradient
exactly; compare recorded actual displacements with the gradient-sum identity.

For each partition, compute full-objective two-update loss changes at the
matching boundaries. Retain all pairs, including increases. Summaries cover
the complete 3,000-update prefix and the fixed last-1,000-update window;
no post-hoc event selection. Changes below relative 1e-6 of endpoint loss
scale are reported separately in a sensitivity check. Loss is training-only;
this analysis does not establish generalization, asymptotic convergence or a
causal benefit from reversal.

Reuse all nine GD backward profiles from Analysis035, at endpoint states
756/2256/3256 with 256 prior gradients each. These are separate recomputed
continuations from saved states, not later gradients from Analysis024's exact
floating-point paths. Check raw CPU/GPU products and show endpoint coverage.
Reuse Analysis037's three initialization blocks and eight sampling repetitions
for model-specific moving-SGD trends; no frozen-endpoint sampling control is
claimed. A same-state frozen control cannot identify negative endpoint cosine.

## Reproduction

1. `python analyses/056-2026-11-05-gd-paired-displacement/01_analyze.py`
2. `python analyses/056-2026-11-05-gd-paired-displacement/02_verify.py`
3. `python analyses/056-2026-11-05-gd-paired-displacement/03_plot.py`

See [the observation](observations/001-paired-displacement.md) for results,
figure contracts, uncertainty and limitations; `results.json` and compact CSVs
retain reductions. Raw source artifacts and generated array caches stay ignored.
