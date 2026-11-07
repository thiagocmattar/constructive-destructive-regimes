# Matched intervention figure consolidation

Local consolidation of completed experiments, requested 6 November 2026.
No new training, hypothesis, run, cloud spending, or manuscript changes.
Source analyses, scientific inputs, and raw evidence remain unchanged.

## Numbered figures

| Figure | Comparison | Result in the retained window |
| --- | --- | --- |
| [00: High-LR overview](figures/00-high-lr-overview-cifar10-mlp-resnet-wikitext2-transformer.pdf) | Original Analysis055 overview, copied exactly | Context for the three full-batch Adam cases |
| [01: Immediate half LR](figures/01-persistent-half-lr-cifar10-mlp-resnet-wikitext2-transformer.pdf) | Three models, each against unchanged LR | Mean loss improves for MLP/Transformer, worsens for ResNet |
| [02: Floors and abrupt return](figures/02-lr-floors-and-abrupt-return-cifar10-mlp.pdf) | Two cosine floors; immediate-half history followed by abrupt return | Half floor suppresses late excursions; return restores large spikes |
| [03: Return styles](figures/03-lr-return-styles-cifar10-mlp.pdf) | Cosine70 return against unchanged, held-half and abrupt return | Gradual return delays but increases the large spike |
| [04: Two-stage decay](figures/04-two-stage-decay-cifar10-mlp.pdf) | Second decay against unchanged and held-0.75x | Lower mean/tail loss and smaller first post-650 spike |
| [05: Clipping](figures/05-gradient-clipping-cifar10-mlp-resnet.pdf) | MLP caps 0.5/1; ResNet cap 1; matched unclipped controls | Model- and window-dependent tradeoffs; raw/input norms differ |
| [06: Pythia schedules](figures/06-lr-schedules-minipile-pythia14m-70m.pdf) | Original, held and accelerated decay; 14M/70M | All requested diagnostics available; alignment recovery does not rank validation loss |
| [07: High-LR optimizer overview](figures/07-high-lr-optimizer-overview-cifar10-mlp-resnet-wikitext2-transformer.pdf) | Figure 00 extended with momentum norm and adjacent momentum/Adam-direction cosines | Original four rows preserved; three additional rows use the same Run215 trajectories |

Captions, coverage and limitations: [observation index](observations/INDEX.md).
The four-row Figure 02 omits rho; its third column uses only the immediate-half
history, not the cosine-half return history. These layout assumptions were
stated during preparation. Figure 03 deliberately retains the source's cosine
initial reduction and three comparisons. Figure 05 retains both MLP caps in one
column. No new immediate-0.75x experiment is invented.

## Figure contract

Eight numbered publication PDFs, 00--07, with provenance and local observations.
00 preserves Analysis055's high-LR overview exactly. Figures 01--06 use
7.16-inch width, Times/STIX fonts, embedded vector text, compact labels,
unsmoothed full-objective curves, and gray pre-intervention context.

01: immediate permanent half LR only, three model columns, five rows.
02: cosine floors 0.75x/0.5x against unchanged, plus immediate half at 512
followed by abrupt 0.75x return at 650 against held-half. Three columns,
four rows: loss, projected gradient change, gradient norm, applied/base LR.
03: preserve Analysis046's three comparisons of cosine70 restoration.
04: preserve Analysis050's two comparisons of the two-stage decay.
05: MLP caps 0.5 and 1 against unchanged; ResNet cap 1 against its fresh
same-Spark control. Include post-clipping norms distinctly from raw norms.
06: Pythia-14M/70M schedule comparison, including recovered raw diagnostics.
07: Figure 00's four rows followed by absolute stored first-moment norm,
adjacent first-moment cosine, and adjacent adaptive Adam-direction cosine.
Seven rows, three columns, unsmoothed; see [Observation007](observations/007-optimizer-overview.md)
for optimizer timing, vector definitions and full coverage.

The five-row order is loss, eta*kappa_hat, rho, raw-gradient norm, applied/base
LR. Under Adam/AdamW, eta*kappa_hat denotes the dimensionless measured projected
gradient change, not a validated curvature estimate or stability threshold.
Zero-norm denominators remain undefined. No new smoothing or event selection.
Pythia minibatch losses and diagnostics use the update index recorded by the
source; unlike the other figures, these are not full-objective losses. Figure 06
shows unsmoothed values, including unequal-size final update 712, rather than
the centered 31-update smoothing in Analysis004. The first undefined cosine
and projected change are omitted without epsilon substitution. Its LR divisor
is the inherited midpoint applied LR, and the entire pre-fork schedule is shown.

The source experiments determine initialization, fixed objectives, optimizer
state, precision and data order. This analysis changes presentation only;
comparisons retain their original warm states and controls. Most experiments
use one selected initialization/window. No confidence intervals, pooled model
effect, causal mediation, generalization or scheduler benefit is inferred.

Reproduce with `python analyses/058-2026-11-06-intervention-consolidation/01_figures.py`
and verify with `02_verify.py` in the same folder. See observations for exact
captions, coverage, outcomes and limitations; manifests record source and PDF
hashes. Scratch renders live under ignored `tmp/pdfs/analysis058/`.

Generate the additional Figure 07 with `03_optimizer_overview.py` and verify it
with `04_verify_optimizer_overview.py`. Its separate `optimizer_overview_manifest.json`,
`optimizer_overview_values.csv`, and `optimizer_overview_verification.json` preserve
the original seven-figure manifest and verification record.

## Completed verification

All 69,354 plotted scalars in Figures 01--06 match the raw records independently;
48,006 also match the original analysis plot tables exactly. Checked 12,144
projected-change identities and 63 hashed source files. Figure 00 is byte-identical
to its authenticated original. All seven one-page PDFs have embedded fonts,
vector curves, 7.16-inch width and text within page bounds. All seven renders
were inspected, including full-size reviews of Figures 01--06; the final layout
preserves the full Pythia LR range. See [verification](verification.json),
[source/curve manifest](manifest.json), [plotted values](plotted_values.csv),
and [full-objective loss summaries](loss_summary.json).

Figure 07 (7 November): all 21,495 plotted scalars match frozen Run215 records;
12,285 values in the original four rows also match Figure 00's source plot hashes.
The 9,210 added values cover all available momentum norms and adjacent moment/direction
cosines. Its one-page 7.16 x 9.5-inch PDF passes embedded-font, vector and text-bounds
checks, and the full-size render was inspected. Figure 00 remains byte-identical.
