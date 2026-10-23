# Observation 002: FP32 training, eta-scaled curvature and the manuscript proxy

October 23,2026. Presentation revision requested by the user; no new training,
model evaluation, curvature estimation, cloud resource or rental. The original
three-arm evidence, executed inputs and eight-panel PDF are preserved.

## Question and sources

Show the native and fixed FP32 CE comparison with fewer panels, including
eta*lambda_max and the user's eta*kappa. The working draft distinguishes
directional kappa from its gradient-only estimate, so both appear, explicitly
labeled, in a four-panel figure.

- [Cohort](../cohort.json): all ten seeds of each FP32 arm,20 runs and200,000
  updates. The CE64-loss training arm is omitted only from this figure.
- Owning-run `artifacts/attempt-01/config.json`, `metrics.jsonl.gz` and
  `curvature.jsonl`: all60 source files checked against the size/SHA256 entries
  in the existing [artifact inventory](../artifact_inventory.json). Their exact
  paths and identities are also in the new [figure data](../fp32_figure_data.json).
- [Original verified figure data](../figure_data.json) and
  [results](../results.json): loss, lambda and directional kappa checked for exact
  agreement; terminal losses include the native seed7 spike at state10000.
- [Draft definitions](../../../manuscript/draft/sections/03-methods.tex):
  `eq:kappa` and `eq:kappa-estimate`.
- [Script](../13_plot_fp32.py), [manifest/checks](../fp32_figure_manifest.json),
  [publication PDF](../figures/fixed_ce_curvature_fp32.pdf).

## Definitions, coverage and reduction

The learning rate is eta=0.001, read and checked in every optimizer config.
Both displayed training arms use FP32 models, logits, loss arithmetic, gradients
and AdamW moments. The already completed Hessian measurements use the common
full-FP64 model/objective evaluator for both arms. Omitting the CE64 training
arm does not change the precision of the shared diagnostic.

1. Loss: arithmetic mean of the common CE64-on-FP32-logits training objective
   in100-state bins, using every pre-update state0..9999. Each seed has100
   complete bins. Separate dots at10000 show the actual terminal common full-FP64
   objective, preserving the native seed7 endpoint excursion. The native FP32
   optimization loss is not used as a supposedly precise loss reference.
2. Maximum curvature: eta times the largest algebraic eigenvalue of H, at all23
   regular states0,500,...10000 plus9250/9750. Every H uses the full480 training
   pairs; all eigenpair relative residuals are at most0.005. Lines connect
   measurements and are not additional spectra.
3. Measured directional curvature: eta*kappa=eta*g^T H g/||g||^2, using the common
   full-FP64 objective gradient and Hessian at those same23 states. It describes
   the reference loss landscape at the attained FP32 parameters, rather than the
   potentially inaccurate FP32 training-gradient direction. This measured quantity
   was already retained; no reconstruction or finite-difference estimate is used.
4. Manuscript proxy: q_t=eta*kappa_hat_t=1-rho_t*||g_t||/||g_(t-1)||, from each
   arm's actual saved full-batch FP32 training-gradient norms/cosines, reduced in
   FP64. Here manuscript g_t is evaluated at theta_(t-1); stored state s corresponds
   to manuscript t=s+1. The plotted x coordinate is the later gradient's parameter
   state s=1..9999, with its previous gradient at state s-1. There are9999 adjacent
   pairs per run and no gradient at theta10000 was logged. No terminal pair or
   initial zero is fabricated. No zero gradient norm occurs. The proxy already
   includes eta algebraically; it must not be multiplied by0.001 again.

For panel(d), take consecutive100-pair arithmetic means within each seed,
retaining the final99-pair bin. The plotted coordinate is the arithmetic mean
of the represented states. Panel(a) uses the analogous bin centers. Thin lines
show all ten seed histories per arm; thick lines are pointwise seed medians
after binning. Thus they are medians of within-seed arithmetic means, not a pooled
mean over seeds. No seeds, gradients or curvature measurements are excluded.
Panels(a-c) use log Y because all displayed values are positive. Panel(d) uses
symmetric-log Y with linear region[-0.1,0.1], retaining negative means and
large excursions without clipping. The dotted q=1 line marks zero adjacent
alignment for individual nonzero gradient pairs; a binned mean crossing1 does
not imply every pair in that bin has negative alignment.

## Observation and interpretation

The original paired late-window fixed/native lambda ratio is unchanged:
0.0148854,95% CI[0.00307713,0.0720074]. Multiplication by the same eta preserves
every paired ratio. The matched-loss controls remain in the original report;
they still give little evidence of additional flattening at comparable loss.
The simplified figure makes the attained-loss and ordinary-curvature separation
visible without the precision-control and ancillary diagnostic panels.

Both requested quantities are available. The two kappa quantities behave very
differently under AdamW, consistent with the need to keep their definitions
separate. Descriptive summaries from the retained evidence are:

| Summary | Native FP32 | Fixed FP32 |
| --- | ---: | ---: |
| Median eta*lambda across ten seeds and five late states | 0.00394966 | 0.0000592093 |
| Median measured eta*kappa across those50 measurements | 0.00288985 | 0.0000462087 |
| Mean q over last1000 pairs, averaged over ten seeds | -0.195673 | 1.686775 |
| Fraction of last1000 pairs with q>1, averaged over seeds | 67.92% | 96.96% |

These pooled descriptive medians are not the predeclared geometric endpoint or
independent50-replicate inference. Native q's arithmetic mean is sensitive to
large negative excursions, even while most late pairs have q>1; the figure's
thick median across seed-bin means answers a different question. The displayed
binning suppresses individual-pair spikes, and raw logs preserve every pair.

The draft's q approximately equals eta*kappa minus sampling noise under a
plain-SGD update and a local first-order approximation. Here gradients are
full-batch, so there is no mini-batch sampling noise, but the optimizer is
AdamW. Momentum, adaptive scaling, weight decay, finite-step remainders and
native-gradient numerical error remain. Consequently q is an observable
consecutive-gradient proxy; it is not a verified directional-Hessian estimator
or an AdamW stability test. The GD references eta*lambda=2, eta*kappa=1 and
eta*kappa=2 are not applied as AdamW stability thresholds. The chart omits
those GD threshold lines.

## Caption and verification

Fixed cross-entropy under FP32 AdamW training of a width64 modular-addition
Transformer:480 training pairs, ten paired initializations,10,000 updates and
eta=1e-3 per arm. Native CE is orange/dashed; fixed CE is blue/solid. Panels show
(a) accurate training loss, (b) eta*lambda_max, (c) measured eta*kappa and
(d) the gradient-only eta*kappa_hat proxy. Thin curves are individual seeds;
thick curves are seed medians. Loss and proxy curves use100-state/pair means
(final proxy bin99); terminal loss dots are retained. Hessian panels use all23
measured states of the same full-FP64 objective on all480 pairs. Proxy axis is
symmetric log, and q=1 is an alignment reference, not a stability threshold.

All60 input identities,20 contiguous clocks,200,000 losses,199,980 adjacent
pairs and460 ordinary spectra are checked. Multiplication by eta is from the
optimizer configs, raw q is checked against the independent projected-change
algebra and q>1 against rho<0, and every one of88 plotted seed/median curves is
checked against its generating array. The PDF is rendered with Poppler and the
actual render reviewed for labels, legend, bounds and all four panels. Manifest
hashes bind the script, figure data, PDF and reviewed render. Ten initializations
on one fixed task/split/optimizer do not establish transfer to other settings.

Reproduce:

```text
python analyses/026-2026-10-23-fixed-ce-curvature/13_plot_fp32.py
```

After inspecting the newly generated Poppler render, record that review with
the same command plus `--accept-reviewed`.
