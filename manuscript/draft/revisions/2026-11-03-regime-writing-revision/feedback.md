# Extracted author annotations

Source SHA-256: `1ca460a9d7b4aef38a50e5f2aabdba302211b13724d3ecf4cfc8dadd1e4d34c9`.

## C01 - original page 8

The argument can be sharper here, right? What is the motivation? Why restate the equations? My take is that adam takes an adaptive step, but the diagnostics from the equations X, Y (ref them here) still is very informative about what is happening on the loss landscape. About the curvature, constructive-destructive regimes, etc. we're measuring different points in space between steps. Those are fed into the adaptive optimizer, so they influence the movement. The goal here is to do a qualitative argument and tell the reader why and how the diagnostic we're proposing can help in adam optimization.

**Selected passage:** Although dt determines the step, the raw gradients still reveal variation in local slope along the adaptive trajectory. Their consecutive alignment and magnitudes distinguish directional reversal from a sharp increase in local slope. The first moment shows how this changing input propagates through the optimizer’s memory. The projected-change diagnostic remains exactly com- putable as ηbκt = 1 ∥gt∥ ∥gt−1∥. (27) Under Adam, this is normalized projected raw-gradient change along the realized path; it is not the SGD directional-curvature estimate. A value above two requires negative cosine and a sufficiently large consecutive norm ratio. Thus, reversal and norm growth explain the diagnostic’s increase algebraically, rather than supplying independent confirmations of instability. The reference at two recalls the GD analysis, but does not identify an Adam stability boundary.

## C02 - original page 9

what is this word? aim for clarity

**Selected passage:** interspersed

## C03 - original page 9

remove. don't use the excess negative about and afirmative statement.

**Selected passage:** its presence alone does not rank the final loss.

## C04 - original page 10

remove

**Selected passage:** Their location in the middle or later part of training is an outcome of this maximum- amplitude rule, not an additional temporal filter.

## C05 - original page 10

remove. its confuse

**Selected passage:** with model- dependent timing relative to the raw gradient.

## C06 - original page 10

this is very good. make it clearer and sharper.

**Selected passage:** In the Trans- former, the first crossing of two within the excursion is at state 788, when loss is only 0.40% above its pre-rise level. The +10% crossing follows at 790; raw-gradient and momentum norms reach their event maxima at 792 and 793, before loss peaks at 794. The diagnostic thus precedes the pronounced spike and its maximum, while the first small loss increase has already occurred.

## C07 - original page 10

use the constructive / destructive regime nomenclature.

**Selected passage:** The descent separates the models. ResNet and Transformer move toward positive gradient and momentum cosines, lower projected change and a comparatively smooth loss decline. The MLP continues to oscillate: raw cosine remains mostly nega- tive while momentum cosine changes sign, and loss and norms decrease through repeated rebounds. Recovery therefore need not require every raw-gradient pair to become constructive.

## C08 - original page 10

I think this is not clear enough. State variables, explain the intuition before the math. Think on the reader.. we're trying to define precisely and also explain the intuition. So it should be easy for the reader to understand. They shouldn't have to think hard.

**Selected passage:** To examine repetition without selecting on gradients, write Ls := L(θs) for the recorded training loss. Starting at initial- ization, let ai be the earliest available state with Lai+1 > Lai. Hold the pre-rise reference Lai fixed. When a return is observed within T updates, define the first recovery and peak by ci := min{s : ai < s Ls ≤Lai}, pi := min arg max ai<s<ci Ls. (28) An event is included only when recovery is observed by T = 1024. Search resumes at ci; partial declines and additional local maxima before this return belong to the same event. Thus ai is the last pre-rise state, ai + 1 the first elevated state, pi the loss peak, and ci the first recovered state. Peak increase is 100(Lpi/Lai −1); rise and recovery times are pi−ai and ci− pi, in optimizer updates. The three loss ratios are 1, Lpi/Lai and Lci/Lai

## C09 - original page 10

negation. remove.

**Selected passage:** This is an operational amplitude filter chosen for interpretation, not a separation inferred from a density minimum or an optimizer stability threshold.

## C10 - original page 10

illustrations?

**Selected passage:** illustrations

## C11 - original page 10

please note.. the text should be insightful and explain / provide insights for the reader. it doesn't need to be overly descriptive.

**Selected passage:** Figure 7 places the illustrations within the complete distri- bution of recovered excursions. Small fluctuations coexist with long tails; larger rates populate more severe excursions, al- though the distributions overlap and do not undergo a uniform shift. To focus the anatomy analysis on pronounced changes, we retain events with peak increase strictly greater than 10%. This is an operational amplitude filter chosen for interpretation, not a separation inferred from a density minimum or an optimizer stability threshold. The density retains all amplitudes so that the selection remains visible in context. The filtered sample contains 129 recovered events. Two further above- threshold candidates without observed recovery are excluded. For Table II, pre-rise diagnostics are measured at θai, hence use gradient index ai + 1. Recovery entries first average each event’s diagnostic over the arrival states pi + 1, . . . , ci, then summarize those means across events with equal event weight. They are not values only at ci. The norm-ratio columns instead use the endpoints ∥gpi+1∥/∥gai+1∥and ∥gci+1∥/∥gpi+1∥. The gradient norm’s own maximum need not coincide with the loss peak.

## C12 - original page 12

THis is a very good table, but it require some improvements:
- more spacing between columns
- you can use power scientific notation for \eta
- use the entire column / horizontal text space for the table;
- centralize everything (c)
- break line on dataset / (line break) model

Explain / define every column on the legend. Although it may be repetitive, some readers do "dynamical reading", so they should be able to understand the raw, before, rise, gradient norm ratio, etc on the table legend.

Use max 2 significant digits (i.e. for >100 we don't need decimals; same for >10. for <10 and > 1, use 1 decimal; for <1 use 2 decimals.).

**Selected passage:** TABLE II: Anatomy of recovered spikes with peak increase greater than 10%. Cells report mean ± sample SD across events; maximum peak increase has its own subcolumn. Before denotes a single diagnostic at state ai; recovery denotes the event’s mean over states pi + 1, . . . , ci. Norm ratios compare loss-peak/pre-rise and recovered/loss-peak endpoints. N counts included events; dashes indicate no qualifying event. Two initialization events have undefined pre-rise diagnostics, leaving 46 and 38 observations for those entries in the MLP’s middle and highest LR rows. No smoothing or averaging across initializations. Dataset/model η N Peak increase (%) Time (updates) ηbκ (raw) ρ (raw) Gradient norm ratio Mean ± SD Max Rise Recovery Before Recovery Before Recovery Rise Recovery CIFAR-10 MLP 0.0001 8 19.2±6.1 30.9 8.4±2.9 4.4±2.4 1.81±0.38 1.67±0.08 −0.77±0.31 −0.89±0.08 5.32±1.71 0.30±0.07 0.0003 47 42.6±45.8 203.1 5.5±4.4 4.9±4.0 1.35±0.25 1.55±0.18 −0.53±0.29 −0.71±0.24 4.46±3.03 0.33±0.13 0.001 39 94.6±107.5 317.6 6.1±5.9 7.4±7.5 1.36±0.40 1.44±0.26 −0.46±0.38 −0.55±0.30 5.23±3.75 0.31±0.14 CIFAR-10 ResNet-32 0.0003 0 – – – – – – – – – – 0.001 10 104.6±181.7 602.1 7.5±2.2 17.3±11.0 1.34±0.39 1.34±0.25 −0.36±0.42 −0.36±0.26 4.32±2.87 0.43±0.24 0.003 7 179.1±309.2 858.8 6.3±1.3 19.3±20.2 1.74±0.38 1.10±0.37 −0.65±0.28 −0.13±0.42 5.02±4.22 0.34±0.24 WikiText-2 Transformer 0.0003 0 – – – – – – – – – – 0.001 6 33.1±25.5 80.6 6.3±1.9 23.7±9.8 2.06±0.27 0.96±0.16 −0.84±0.10 −0.02±0.12 9.05±3.77 0.09±0.05 0.003 12 82.4±70.3 231.4 6.2±2.7 31.2±16.8 2.17±0.43 0.91±0.19 −0.79±0.22 0.09±0.21 11.43±6.86 0.08±0.08
