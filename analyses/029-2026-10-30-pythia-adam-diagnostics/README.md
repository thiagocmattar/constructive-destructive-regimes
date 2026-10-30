# Pythia diagnostics: learning rate, optimizer stages and model size

Analysis029, October 30, 2026. Requested cross-run visualization and scientific
assessment. This is a local reanalysis of existing events, not a new training
run or a cloud launch. The manuscript and original analysis outputs are preserved.
Status: complete. Four rendered PDFs reviewed;22 event-file hashes and all303
curve descriptors /192,399 finite plotted values pass independent checks.

## Comparison and metric contract

- **Learning rate:** Analysis009's seven Pythia-14M runs, one common seed1234
  initialization/order, warmup then constant LR. Three original rates use GB10;
  four intermediate rates use RTX4090. Comparisons are grouped by hardware in
  interpretation; every rate uses solid curves and a distinct color.
- **Schedule:** Analysis004's original/held/accelerated branches, Pythia14M/70M,
  seed1235, common checkpoint356 within each size. All branches use RTX5090;
  optimizer state and future batches are matched. LR also changes AdamW decay.
- **Size:** Analysis002's 14M/31M/70M by seeds1234/1235/1236. Data exposure and
  recipe match within each seed; architectures and initialization shapes vary.
  Seed1234 uses RTX4090 and the other seeds RTX5090. This is a matched size
  comparison with three seed blocks, not a scaling law.

All use the pinned MiniPile packed cache, 1,491,711,416 input tokens in712 updates,
1,024 sequences per full update, context2,048, microbatch4, AdamW beta=(0.9,0.95),
epsilon1e-8, clip norm1, decay0.1 with bias/LayerNorm exclusions. FP16 compute,
FP32 parameters/moments, FP64 diagnostic reductions. Raw gradients precede
clipping. Adaptive directions use actual bias-corrected moments and exclude LR
and decay. All unique trainable parameters are measured after accumulation.
Validation is token weighted over all500 documents/693,329 targets, including
the packed tail. Models start from scratch; they are not pretrained checkpoints.

The SGD expression requested by the user is evaluated pointwise as

`1 - rho_t * ||g_t|| / ||g_(t-1)||`.

It equals `-<g_(t-1), g_t-g_(t-1)> / ||g_(t-1)||^2`. Under Adam this is a
normalized projected gradient change, **not a Hessian-curvature estimate**.
Its crossing of1 is exactly equivalent to negative raw cosine for positive
norms; that agreement is an identity, not validation. It is not an Adam
stability threshold. The preceding LR is zero for the first observed pair
(update2); the expression remains defined, but division by that LR to obtain
a finite kappa is invalid. We never perform that division.

For the seven runs with momentum capture, reconstruct the clipped input from
`h_t=(m_t-beta1*m_(t-1))/(1-beta1)`. Norms and adjacent momentum cosine determine
`||h_t||^2` and `<h_t,m_(t-1)>`, hence their cosine, without full vector histories.
Compare its reconstructed norm with the raw norm times the logged global clip
coefficient. Global positive clipping preserves input/momentum cosine. This
reconstruction is exact for ideal EMA arithmetic, subject to stored FP32 moment
roundoff. It is checked against clipping records; it is not a new Hessian or
counterfactual measurement. Moment capture is absent from the size/schedule
cohorts, so it is not reconstructed for them.

Display uses full-sized updates1--711; the partial final update712 remains in
all raw reductions and endpoint validation. Nonpositive/undefined denominators
produce missing values, not epsilon offsets. Smooth each pointwise diagnostic
with a complete trailing25-update arithmetic mean; training loss uses target
token weights. Never multiply a smoothed cosine by a smoothed norm ratio.
Schedule smoothing starts at the affected pair358 and uses no pre-branch
history. Size bands show min/max across the three separately smoothed seed
curves, with their mean as the bold line; they are not confidence intervals.
Stream cumulative ratios are unsmoothed and include their complete prefix.
They describe the named vector stream, not actual AdamW displacement.

## Outputs

- [LR response](figures/pythia_lr_response.pdf): loss, raw cosine, projected
  gradient change, raw/moment norms and incoming-gradient/prior-moment alignment.
- [Optimizer streams](figures/pythia_optimizer_streams.pdf): adjacent cosine
  and cumulative retention for raw gradients, moments and adaptive directions.
- [Schedule response](figures/pythia_schedule_response.pdf): imposed LR,
  raw cosine, projected change, adaptive cosine and full-validation loss.
- [Size response](figures/pythia_scale_response.pdf): loss, raw/adaptive cosine,
  projected change and raw/adaptive retention, with all three seed blocks.

[REPORT.md](REPORT.md) interprets the findings and prioritizes a further
experiment. [Observation001](observations/001-figures.md) supplies figure
captions, sources, reductions, uncertainty and caveats.

Reproduction: run `01_analyze_plot.py`, then `02_verify.py`. The first rechecks
source hashes and writes figures, reduced results and exact plot arrays; the
second independently checks reductions and reconstructed mathematics. Original
event files must be available at their recorded paths. Scratch renders and
registry-before snapshots remain ignored in `scratch/`. No old operational
script is run and no historical cloud-status paragraph is treated as a live read.
The generated `plot_arrays.npz` is also ignored by the repository's array-file
rule; it is retained locally and regenerated by the first script before running
the verifier. Source hashes, curve descriptors, numerical summaries, scripts
and all four publication PDFs are committed.
