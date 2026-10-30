# Pythia diagnostic figures

October 30, 2026. Generator: [01_analyze_plot.py](../01_analyze_plot.py).
Independent checks: [02_verify.py](../02_verify.py). Source hashes, exact plotted
arrays and curve descriptors are in [results.json](../results.json) and
`plot_arrays.npz`; [verification.json](../verification.json) records the checks.
The [metric contract](../README.md) applies to all four figures.

## Source coverage

- Analysis009: Runs021--027, seven full712-update trajectories. The archived
  comparison arrays are checked against each original event-file hash. Run027
  uses its original compact retrieval copy, with the same archived hash.
- Analysis004: Runs011--016 and their common prefixes from005/007. The six
  continuations contain updates357--712; the first raw gradient357 precedes
  any changed parameter update. The affected raw pairs begin at358.
- Analysis002: Runs002--010, three sizes crossed with three seeds. Run002 uses
  accepted attempt02, the others attempt01. Original event hashes are checked
  against the source-lineage registry.

There are22 unique event files,15,664 record rows after joining schedule
prefixes, and15,642 defined projected-change observations. Prefix copies across
schedule branches are shared evidence, not additional independent trajectories.
All full batches have1,024 packed sequences; the last batch712 has311. Main
diagnostic curves use1--711; endpoint validation includes the full pass. No
training targets or validation tails are discarded from the source evidence.
Complete windows leave the first24 loss/norm means and the first25
adjacent-pair means undefined. The size figure's thin seed curves are also
smoothed; they do not interpolate those absent initial means. The LR figures
add unsmoothed thin traces that retain the initial observations.

## Figure 1: learning-rate response

[Publication PDF](../figures/pythia_lr_response.pdf).

**Caption.** Pythia-14M trained from one common initialization and data order
(seed1234) at seven peak rates, with the same1% warmup and then constant LR.
Panels show training loss, raw consecutive cosine, the normalized projected
gradient-change expression, raw and first-moment norms, and the reconstructed
cosine between the incoming gradient and prior moment. Thin traces are
unsmoothed; bold curves are complete trailing25-update means, with target-token
weighting for loss. Loss and norms use logarithmic Y axes. Norms are averaged
before log display. The norm-one reference is the clipping threshold, and the
projected-change reference1 denotes zero projected gradient overlap; it is not
a stability boundary. The1e-4/1e-3/1e-2 runs used GB10 and intermediate rates
used RTX4090. All LR curves are solid and identified by color.

**Observation.** The0.01 run develops high raw norms, low momentum persistence
and negative input/history alignment while its raw cosine is only weakly
negative late. The input/memory comparison helps explain why raw adjacent
cosine alone misses this aspect of optimizer behavior. Its reconstructed
clipped norm matches the logged clipping result to max relative error3.40e-7
across all4,977 pairs. This is EMA accounting, not evidence that momentum
rotation itself caused the poor loss. Single-seed and cross-hardware limits
remain. The first undefined pair is missing without a denominator offset.

## Figure 2: optimizer stages

[Publication PDF](../figures/pythia_optimizer_streams.pdf).

**Caption.** The same seven Pythia-14M trajectories viewed through raw gradients,
actual uncorrected first moments and bias-corrected adaptive directions.
Top: adjacent cosine, with unsmoothed thin traces and trailing25-update means.
Bottom: unsmoothed norm-of-cumulative-sum divided by cumulative sum of norms
for the named vector stream. First moments use clipped gradients; adaptive
directions exclude LR and decoupled decay. Every ratio includes the complete
prefix from initialization; none is claimed to be actual parameter displacement.
Raw cosine uses[-1,1] support; moment/adaptive axes display[0,1], since all
recorded values are positive. All data fall inside the displayed limits.

**Observation.** Raw reversal coexists with positive moment and adaptive
alignment. The high-rate run has much greater moment rotation, though its
moment cosine never becomes negative. Adaptive alignment stays high despite
poor loss. Accumulation ratios differ substantially between streams; high
ratio is not identified with performance. The raw and moment panels are not
independent optimizer runs. Uncertainty is one initialization, not variability
among the serial observations.

## Figure 3: schedule intervention

[Publication PDF](../figures/pythia_schedule_response.pdf).

**Caption.** Pythia14M and70M branch from their respective shared update356
checkpoint, seed1235, into original cosine decay, held LR and faster decay.
Rows show the imposed rate, raw cosine, projected gradient change, adaptive
cosine and full-validation loss. Diagnostic traces begin at affected pair358;
the bold trailing25 mean first appears at382 and uses no pre-branch history.
The imposed LR is unsmoothed from357. Validation markers are the complete
evaluations at356/445/534/623/712; connecting lines guide the eye rather than
representing measurements between them. Rate and validation axes are logarithmic;
Y limits are shared within each row. The two size panels show separate parent
states, not an intervention on size. No momentum trace is inferred for these
runs because its historical scalar capture is absent.

**Observation.** Faster decay changes raw response and produces earlier
positive alignment, but held-rate branches achieve lower final validation
loss at both sizes. The70M positive recovery is temporary, despite a fixed LR
floor later. The pointwise projected-change/cosine crossings express the same
algebraic sign relation; separately smoothed curves may cross at different
updates. They are not two independent tests. These are matched schedule
effects at one state and seed per size. LR-scaled weight decay varies jointly
with adaptive-step length; mediation is not isolated. Smoothing differs from
Analysis004's original centered31 convention, so the earlier numerical
recovery times should not be read from these trailing25 curves.

## Figure 4: size and seed replication

[Publication PDF](../figures/pythia_scale_response.pdf).

**Caption.** Pythia14M/31M/70M under the original common LR recipe, with three
matched seed blocks per size. Panels show training loss, raw cosine, projected
gradient change, adaptive cosine and raw/adaptive accumulation ratios. Loss
is a trailing25 token-weighted mean; cosines and projected change are
trailing25 arithmetic means computed separately for each trajectory. Thin
curves show each seed, bold curves their equal-weight mean, and bands their
min/max at each update. Bands are descriptive seed ranges, not confidence
intervals. Ratios are unsmoothed within each trajectory before seed aggregation.
Loss uses log Y; the other panels use linear Y. Colors identify size throughout.

**Observation.** Size differences persist across seed blocks, particularly in
adaptive alignment and retention, but raw alignment varies with training phase.
The larger model has lower loss while retaining lower directional coherence
late. The full-pass projected-change ordering increases with size in3/3 seed
blocks; its relationship to the previously reported raw-cosine ordering uses
the same underlying vectors. Architecture/initialization vary with size, and
seed1234 used4090 versus5090 for the other blocks. These data do not isolate a
universal parameter-count effect or establish a predictive scaling law.

## Scientific and visual review

All303 curve descriptors and192,399 finite plotted values pass independent
reconstruction, including token weighting, pointwise-before-smoothing order,
affected-window boundaries and seed-band endpoints. The input/memory
reconstruction is independently checked by parallel/transverse components and
against direct synthetic vectors; it is not silently applied to cohorts that
lack moment measurements. The PDF checks find four single-page documents,
embedded fonts, in-bounds text and no plotted values outside their axes.

All four final Poppler renders were inspected. Panel labels sit above the
axes, footnotes are absent, LR/schedule colors use solid lines, and dotted
references identify constants rather than conditions. Raw traces are faint
enough to expose variability without masking the smoothed curves. No independent
seed uncertainty is implied by their density. Original figures and the
manuscript remain unchanged; these are candidate analysis figures for review.
