# IEEE figure review

Reviewed 2026-10-30. All five PDFs passed individual rendering at 150 dpi,
embedded-font inspection and manuscript-scale inspection. Numeric records are
unchanged by layout refinement: all 300 plotting records retain their arrays.
Hashes and the completed checklist are in [figure_review.json](figure_review.json).
Scientific contracts and captions are in [Observation001](observations/001-controlled-minibatch-noise.md).

Common decisions: vector PDF, 7.16-inch width, Times/STIX typography, 9-point
labels and 8-point ticks, embedded fonts without Type 3 glyphs. Batch colors are
consistent throughout; all curves are solid. Panel letters sit outside the data
regions. There are no embedded footnotes. Captions explain sampling and reduction.
The five-color comparison relies on color, as requested; a grayscale print can
reduce discrimination among stochastic batches, while the full-GD control remains
black. No inferential claim depends on visually separating nearly overlapping lines.

## Frozen calibration

Three panels are sufficient to distinguish two exact moment identities from one
approximation. Normalizing both measured and predicted moment axes by the same
state-dependent `S + V_b` keeps the full range legible without changing agreement.
Checkpoint markers supplement batch colors. Confidence intervals condition on
each saved state and resample sequences, not dependent adjacent pairs. Repeated
full-batch points overlap deterministically and provide no additional replication.
The diagonal view can hide small bias, so the text explicitly reports maximum
errors and the 21/36 cosine cells whose intervals exclude the approximation.

## Moving geometry

The 3-by-3 organization makes the scientific comparison explicit: checkpoint
columns, sampled cosine/full-gradient cosine/displacement rows. The complete
gradient is evaluated along each SGD path; it is not confused with the black
full-GD trajectory. Only cosine rows receive a complete 20-pair trailing mean;
the displacement ratio remains unsmoothed. The mean/range reduction gives each
initialization equal weight. Three-seed ranges are descriptive and cannot support
population uncertainty claims. Height was reduced to 5.3 inches without changing
fonts or arrays so the appendix companion fits with the loss figure in IEEE layout.

## Raw moving geometry

Same axes, colors and reductions, without temporal smoothing. This checks that
the displayed effects are not introduced by smoothing and exposes seed-dependent
transition variability. It remains a mean across repetitions, not a display of
every individual stochastic path. Those paths are retained in raw evidence and
the conditional summaries. The dense raw display belongs in the appendix.

## Full-objective loss

Three columns with logarithmic Y axes retain all 201 parameter states, including
the initial and final losses. The initially clipped axis label was fixed. Tick
labels now report their actual values to three significant digits and avoid
overlapping or misleading rounded duplicates. Each column has its own Y range;
comparisons of visual slopes across columns require care. All values use the
complete 5,000-image objective. The figure guards against reading objective
improvement from alignment or displacement alone. It does not report generalization.

## Proxy/HVP validation

The upper row displays the observed proxy; the lower row displays its measured
noise correction. Both rows share the same horizontal limits and ticks. Curves
provide 20-pair temporal context, but filled dots and open HVP rings use the same
raw probe pairs, avoiding a smoothed-versus-instantaneous comparison. Near-overlap
is meaningful but not an error estimate: the companion table reports individual
raw-pair median and 90th-percentile discrepancies. HVPs cover all 5,000 examples
in the sampled direction; they are not maximum Hessian eigenvalues. The correction
uses expensive full-gradient references and is identified as an offline diagnostic.

## Manuscript placement

Frozen calibration and smoothed geometry appear in the main empirical section.
Raw geometry, loss and proxy validation appear in Appendix E. The compiled
23-page draft has resolved references and no overfull boxes. Final appendix
floats are flushed before a separate, balanced bibliography; resetting the
available column space fixes the space retained from the final double-column
float. The current manual reference break should be revisited if citations change.
