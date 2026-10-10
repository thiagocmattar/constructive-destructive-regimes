# What the slopes say about interference and learning

The useful part of the intuition is to examine how much an incoming gradient
adds to the cumulative resultant. The proposed equivalence between a flat
cumulative ratio, destroyed contributions and absence of learning does not hold.
Run 002 also does not show a terminal zero ratio slope.

## A and B do not acquire equal terminal slopes

For the last 100 plotted updates, 612-711:

| Stream | Mean slope of A | Mean slope of B | Recent slope ratio: sum of B increments / sum of A increments | Cumulative B/A, first to last |
| --- | ---: | ---: | ---: | ---: |
| Raw gradient | 0.322505 | 0.148944 | 0.461833 | 0.324905 to 0.338631 |
| Adam adaptive direction | 1236.298989 | 965.709908 | 0.781130 | 0.693394 to 0.704758 |

Slopes are norm units per optimizer update; absolute scales differ because Adam
preconditions the vectors. Least-squares A/B slopes agree closely:
0.322288/0.148993 for raw gradients, 1236.449291/965.847624 for Adam.
The slopes remain separated.

The ratios rise by 1.373 and 1.136 percentage points, respectively. Their OLS
slopes are \(+1.3911\times10^{-4}\) and \(+1.1461\times10^{-4}\) per update.
Both recent slope ratios exceed the historical cumulative ratios. New vectors
increase B at a greater relative rate than the prior accumulated history.

Equal fractional slopes would be the condition for a flat ratio. These also
differ over the same interval:

| Stream | Mean log increment of A | Mean log increment of B |
| --- | ---: | ---: |
| Raw gradient | 0.00106969 | 0.00148912 |
| Adam adaptive direction | 0.00140136 | 0.00156572 |

All 100 B increments are positive in both streams. Every incoming vector has
positive alignment with its prior cumulative sum in this interval; mean prefix
cosines are 0.465932 raw and 0.780745 adaptive. This gives no support to the claim
that everything new is canceled. It does not quantify model information storage.

See [the A/B and derivative figure](figures/cumulative_slopes.pdf).

## What a flat ratio actually means

The exact finite-difference relation is

\[
\Delta(B/A)_t
=\frac{\Delta B_t-(B_{t-1}/A_{t-1})\Delta A_t}{A_t}.
\]

If cumulative B/A is 0.3, a flat ratio means that B grows by 0.3 times the growth
of A. It can keep growing substantially. Equal absolute slopes would raise a
ratio below 1.

Moreover, increasing A divides each new ratio change by a progressively larger
denominator. A visually flattening cumulative ratio can reflect the accumulated
history rather than an approach to a minimum. This mathematical effect is why
recent \(\sum\Delta B/\sum\Delta A\) is a more direct diagnostic of ongoing
accumulation.

Even a zero B increment would only show zero change in the norm of the resultant.
The vector can turn without changing its norm, and actual AdamW parameter
updates follow a different vector and include LR and decay.

## A near-flat interval still has growing B and declining loss

Raw B/A reaches its minimum at update 454. In the centered 61-update interval
424-484:

| Quantity | Value |
| --- | ---: |
| B/A at 424 / at 484 | 0.308506 / 0.308516 |
| OLS ratio slope | \(-4.0771\times10^{-6}\) per update |
| Mean A increment | 0.432178 |
| Mean B increment | 0.132705 |
| Recent slope ratio | 0.307060 |
| OLS training-loss slope | \(-8.8518\times10^{-4}\) per update |

This is close to the plateau condition: the recent slope ratio approximately
matches cumulative B/A. Yet B rises at every step, all incoming prefix alignments
are positive, and training loss has a decreasing trend. This is a within-run
counterexample to interpreting a near-flat ratio as total destruction or cessation
of learning. The interval was selected around the observed minimum and is
descriptive, not a held-out detector evaluation.

The adaptive ratio minimum occurs much earlier, at update 102. In its surrounding
window 72-132, training loss falls with OLS slope \(-0.011938\) per update.
The two streams' turning points differ along the same loss trajectory.

## Loss progress slows, but convergence is not established

Mean training loss falls from 5.270867 in updates 512-611 to 5.244120 in 612-711.
The latter interval has OLS slope \(-3.0562\times10^{-4}\) per update.
It is much slower than the first-quarter slope of approximately \(-0.02116\).

Terminal window sensitivity:

| Window ending 711 | Raw ratio slope | Adaptive ratio slope | Training-loss slope |
| --- | ---: | ---: | ---: |
| 32 updates | \(+9.9037\times10^{-5}\) | \(+1.0796\times10^{-4}\) | \(-1.7998\times10^{-4}\) |
| 64 updates | \(+1.1423\times10^{-4}\) | \(+1.1081\times10^{-4}\) | \(-4.0401\times10^{-4}\) |
| 100 updates | \(+1.3911\times10^{-4}\) | \(+1.1461\times10^{-4}\) | \(-3.0562\times10^{-4}\) |
| 128 updates | \(+1.4887\times10^{-4}\) | \(+1.1739\times10^{-4}\) | \(-3.1769\times10^{-4}\) |

Both ratio slopes remain positive with 15/31/61 smoothing and when the final
partial update is included. Short-window loss slopes are more sensitive:
the final 15/31/61-update slopes are \(+0.0018045\), \(+0.0000326\) and
\(-0.0004114\), respectively. With the partial update included, the 32-update
loss slope also becomes positive. Thus local training-loss flatness depends
on the window and batch coverage.

Fixed full-validation loss falls from 11.032736 to 5.182392, with all 500 documents
included. There are no intermediate fixed-validation evaluations. The figures
show slowing training progress; they cannot identify a convergence time or
validate a stopping criterion. See [ratio slopes and loss](figures/ratio_slope_and_loss.pdf).

## Consequence for the paper

A supported statement is:

> Cumulative geometry becomes less responsive late in training, while recent
> resultant growth remains positive. In this trajectory, the adaptive stream
> has greater marginal resultant growth per unit vector norm than raw gradients.
> A plateau of the cumulative ratio describes a balance between current and
> historical accumulation and does not by itself establish convergence or
> absence of learning.

Keep "information being stored" as a separate hypothesis requiring a functional
measurement, such as improvement on repeated fixed evaluation. Norm cancellation
is not an information measure. The reported adaptive/raw difference describes
transformations within AdamW; it does not isolate momentum, preconditioning or
their causal benefit.

The next useful test would compare recent slope ratios and loss responses under
matched LR interventions from the same checkpoint, with repeated fixed validation
and multiple seeds. This analysis creates no new training commitment or launch.
All source-run evidence and manuscript sources are preserved.

## Requested per-step panel grid

The [3x2 scatter grid](figures/loss_difference_vs_slopes.pdf) shows
\(L_t-L_{t-1}\) against exact slopes of A, B and cumulative B/A.
Raw gradients are left; Adam adaptive directions are right. Each panel includes
710 unsmoothed points, updates 2-711, colored by update. Negative loss differences
indicate a lower recorded batch loss. The
[data](loss_difference_grid_data.csv) retain all 712 rows and the final partial
batch; [observation 002](observations/002-loss-difference-grid.md) records the
caption, checks and temporal-alignment caveat.

The same grid is also available at [lag 2](figures/loss_difference_vs_slopes_lag02.pdf),
[lag 5](figures/loss_difference_vs_slopes_lag05.pdf) and
[lag 20](figures/loss_difference_vs_slopes_lag20.pdf).
Every variable uses the same lagged endpoints in each figure. See
[observation 003](observations/003-matched-lag-grids.md) for definitions and coverage.
