# Observation001: complete trajectories at all three Adam rates

## Sources and generator

Run215 original nine arms, pinned by [manifest](../manifest.json); source
config, input manifest, initial tensors and raw histories remain unchanged.
[01_analyze.py](../01_analyze.py) audits sources and derives scalar rows;
[02_plot.py](../02_plot.py) generates the publication PDFs:

- [Low rates](../figures/adam_full_trajectories_low.pdf).
- [Middle rates](../figures/adam_full_trajectories_middle.pdf).
- [High rates](../figures/adam_full_trajectories_high.pdf).

## Reduction and coverage

Each figure has four rows and three model columns. All 1,025 losses,1,024 raw
norms and 1,023 defined consecutive cosines/projected changes per arm are shown.
The separately captured terminal gradient at 1024 is consistently excluded
from the prospective vector stream; terminal loss is included. Norms divide
by their state 0 value, not their first displayed spike-window value. This
normalization differs from the winning Analysis036 zoom; raw norms are unchanged.

No averaging, smoothing, thinning, quantile clipping or hidden startup interval.
Loss/norm use log axes, cosine/Q linear. Q limits are shared across rates within
each model; other vertical limits include all values but are independently
scaled, so raw heights across PDFs should not be compared without their ticks.
All scientific curves are green; no momentum stream is plotted.

Gray bands use recovered fixed-reference events with peak increase >10%, from
pre-rise state to peak. Dotted unfilled outlines show above-threshold observed
rises without recovery, retained as censored. Across the cohort 129 recovered
events and two above-threshold censored candidates are eligible for this marking.
All 812 candidates remain in the source dataset; this display cutoff does not
redefine them. Shared event boundaries and multiple within-event maxima retain
the source definition.

## Caption

Complete full-objective Adam trajectories at the [lowest/middle/highest] tested
rate in each model. Rows show training loss, projected raw-gradient change,
consecutive raw-gradient cosine and norm relative to initialization. Every
recorded observation is shown without smoothing. At parameter state s, vector
diagnostics refer to g_(s+1), before update s+1. Loss/norm axes are logarithmic.
Gray bands mark pre-rise-to-peak intervals of recovered >10% excursions;
dotted outlines retain rises with unobserved recovery. The Q=2 and rho=0
horizontal references describe diagnostic geometry, not an Adam stability
criterion. One initialization per model, matched across rates.

## Observation, uncertainty and caveats

The complete paths show ordinary intervals with negative cosine and substantial
norm growth. Low-rate ResNet/Transformer have zero eligible >=10%/eight-update
future rises despite186/215 Q>2 quiet observations. Highest-rate ResNet changes
to a later constructive regime with continued loss reduction; MLP and Transformer
retain recurring excursions. These are observed trajectories, not proof of
necessity, causality or a general operating threshold.

There are no independent-seed confidence intervals. One frozen initialization
per model and differing qualified hardware limit interpretation. The repeated
temporal observations are dependent. Full objectives and precision inherit
the protocol; no held-out generalization or validation improvement is assessed.
All 36 curves /36,855 points match raw sources; embedded-font/vector/text checks
and Poppler visual review pass. See Observation003 for the full QA receipt.
