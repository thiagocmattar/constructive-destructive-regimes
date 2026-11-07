# Temporal and same-state comparison of gradient and output-change alignment

The user requested a new figure overlaying the two alignments from Figure03,
plus a scatter plot of one against the other. This is a reduction of the
existing Run238/239 measurements; no new training or remote computation.

Figure: [Gradient/output alignment comparison](../figures/04-gradient-output-alignment-comparison.pdf).
Generator: [05_alignment_comparison.py](../05_alignment_comparison.py).
Values: [control_coherence_448_800.csv](../control_coherence_448_800.csv), columns
`state`, `rho` and `output_change_cosine`.
Verification: [alignment_comparison_verification.json](../alignment_comparison_verification.json).

## Definitions and pairing

For every integer $t=448,\ldots,800$, compare

$$
\rho_t=\cos(g_t,g_{t-1}),\qquad
\rho_t^{(\Delta F)}=\cos(F_{t+1}-F_t,F_t-F_{t-1}).
$$

Here $g_t$ is the raw full-objective mean gradient over all5000 fixed examples,
including all weights and biases. $F_t$ concatenates all10 raw output values for
all5000 examples in the same order. Pair the statistics at the same plotted$t$,
without shifting either curve, smoothing, subsampling or selecting a lag to
maximize agreement. The gradient statistic compares states$t-1,t$; the output
statistic compares the incoming and outgoing output displacements around$t$.
This preserves Figure03's definitions and does not claim identical time support.

## Caption

**Temporal and comparative alignment in the unchanged CIFAR-10 MLP Adam control.**
(a) Overlay of adjacent raw-gradient cosine (blue solid) and consecutive
output-change cosine (red dashed), including every state448--800. (b)353 paired
points with gradient cosine on the horizontal axis and output-change cosine on
the vertical axis. Point color denotes state; the gray dashed diagonal indicates
equality. Both scatter axes span the full cosine range and have equal aspect.
The annotation reports descriptive Pearson correlation across these paired
states. The footer specifies the exact indexing of both measurements.

## Observations and limits

Same-state Pearson correlation is0.7762948; mean absolute difference between
the two alignment values is0.0656419. Most points occupy the negative-negative
region, with many near(-1,-1); excursions from that region do not agree exactly.
The temporal overlay makes their similar oscillatory pattern visible while
the scatter exposes departures from equality.

These are dependent time points from one deterministic trajectory and one seed.
The correlation is descriptive, with no IID significance test or confidence
interval. Similarity does not establish that the two quantities are equivalent
or that either causes the other. The plotted pairing is fixed by the existing
definitions; no claim is made about an optimal lag.

## Sources and verification

Same seed0, architecture, data, Adam settings, full-example coverage and precision
as [observation003](003-control-coherence-448-800.md). Undefined pairs do not occur
in this window. The generator verifies the source CSV against its existing
verified SHA256, then independently checks all353 values of each series against
Run238's raw `boundaries.jsonl` and Run239's raw `output_alignment.jsonl`.
All706 temporal values and353 scatter-coordinate pairs are checked against the
actual Matplotlib artists. Source/generator/PDF hashes and statistics are saved
in the local verification JSON. The PDF was rendered and visually reviewed;
the time colorbar and plot elements remain vector graphics.
