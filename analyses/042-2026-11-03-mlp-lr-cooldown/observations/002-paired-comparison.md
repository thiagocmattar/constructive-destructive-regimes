# Observation002: paired columns, raw rho and independent Y ranges

The author requested three unchanged-versus-intervention columns, solid lines,
independent Y scales and restoration of rho. This is a visualization revision
of the completed Run220 evidence, without new training or a new finding set.

Source: unchanged, always_half, hold8 and hold8_cosine8 boundaries.jsonl and
decisions.jsonl under Run220 artifacts/attempts/002-audit-continuation/conditions.
The eight source files match their hashes in the original raw_inventory.csv.
Source hashes, sizes, figure hashes and actual Y limits are retained in
../paired_comparison_verification.json. Generating script: ../02_paired_comparison.py.
Values: ../paired_comparison_values.csv. Previous figures/reductions remain intact.

Caption: Each column compares the same unchanged Adam reference (gray) with
half LR (blue), eight-update cooldown (green), or cooldown plus eight-update
cosine recovery (purple). Rows show unsmoothed training loss, raw adjacent-gradient
$\rho_t$, raw gradient norm, $\eta\widehat\kappa_t$ and applied LR/base. All strokes
are solid. Loss and gradient norm use log scales; the other rows use linear
scales. Both trajectories in each panel use one common Y scale. Panels autoscale
independently using only their displayed pair; there is no shared Y-axis group,
clipping, rescaling of the diagnostic, normalization of the amplitudes, smoothing
or decimation. Y ticks must be consulted when comparing variation across columns.
Horizontal reference lines show rho=0 and the diagnostic threshold1.8.

Definitions: at parameter state s, manuscript t=s+1. $\rho_t=\cos(g_t,g_{t-1})$
is the stored FP64 cosine of consecutive raw full-objective gradients along the
actual trajectory, before Adam transformations. It is not moment or update
alignment. The available $\eta\widehat\kappa_t$ is normalized raw projected change,
consistent with1-rho_t*norm(g_t)/norm(g_(t-1)), without LR-ratio rescaling.
No Adam directional-curvature or safety interpretation is introduced.

Coverage: full states657..1024 (368 observations/367 decisions per condition),
and the original fixed zoom673..745 (73 observations). The unchanged reference
is intentionally repeated in all three columns. Four unique conditions give
1,472 unique boundary rows; the paired table has2,208 rows. All13,230 plotted
scalars across both figures exactly equal their corresponding source values.
The raw rho and available statistic satisfy the independent norm/cosine identity
at every source row. Every trajectory has solid style; the PDF drawing records
also contain no dashed paths. Y axes are explicitly unshared, and recorded
limits differ across the full-view diagnostic panels. All finite raw points
are retained. At1024 the LR drawing value only closes [1023,1024); the table
records no applied terminal decision. Undefined rho would require a gap, but
all1,472 source rho values are defined and in[-1,1].

Visual verification: both vector PDFs were rendered with PyMuPDF and inspected.
Three distinct headers, pairwise legends, five rows, independent tick labels and
rho are legible; there are no label collisions or clipped trajectories. Gradient
norm stays in the third row. The orange one-step series is absent.

Uncertainty/limits: the earlier selected seed0/state657, finite horizon,
retriggering, rate-exposure and missing-control limitations still apply.
Independent axes change visual amplitudes, not numerical outcomes. No causal,
stability, generalization, optimal-duration or transfer claim follows from this
layout. No new computation beyond reading/reducing and drawing existing scalars;
no changes to scientific training code, raw artifacts or manuscript.
