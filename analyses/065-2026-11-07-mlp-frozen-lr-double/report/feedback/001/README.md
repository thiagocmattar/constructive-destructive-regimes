# Feedback 001: applied to report Revision 001

The [archived annotated PDF](001-local-displacement-cap-report-feedback.pdf) is an
exact copy of the supplied `local-displacement-cap-report+feedback.pdf`.
All seven pages' extracted report text match the pre-revision report at commit
`6622278`. The original input remains untouched. [annotations.json](annotations.json)
preserves every comment, author, timestamp, annotation ID and location;
[manifest.json](manifest.json) records hashes, evidence sources and verification.

The current report is labeled **Revision 001** and has eight pages. All ten
complete comments are addressed. One further annotation is an unfinished
sentence; it is preserved without inventing a request. No experiments were run.
Numerical results, generated result tables and companion figures are unchanged.

## Comment-by-comment disposition

Numbers follow PDF annotation extraction order. Locations below refer to the
revised report; original locations are retained in the JSON.

| No. | Original page | Request | Resolution and revised location |
|---|---:|---|---|
| 1 | 1 | Explain `vec`, the 10-row by 5-column example, and matching samples. | Page 1, Section 1: flattening produces 50 coordinates without averaging. The representation cosine is one cosine over all matched sample/feature coordinates, not a mean of sample cosines. |
| 2 | 1 | Make MLP/ResNet/Transformer tracing and low-dimensional SVD findings precise. | Page 1, Section 1: first-projection MLP reversal persists through hidden layers despite tanh attenuation; ResNet's output-like timing is stronger in later stages (correlation 0.578 to 0.991), not evidence of monotonically growing displacement; Transformer reversal appears at the first measured attention output while embeddings can still move forward. The earlier event-window output fits need 5/16/15 modes for 95% energy, rather than a universal 10-16-dimensional claim. Windows, finite-rank limitation and possible rotation remain explicit. |
| 3 | 1 | Clarify what a fixed probe holds constant. | Page 1, Section 1: same input examples, order, token contexts and normalization groups across network states; representations change as parameters change. |
| 4 | 1 | State online/changing-input analysis as future work. | Page 5, Section 7: changing probes, reusing training-batch inputs and recalibrating caps require validation. Consecutive representation differences still need matching inputs or a validated estimator accounting for data changes. |
| 5 | 1 | Unfinished note: "Or maybe we coul". | Preserved verbatim. No substantive request can be inferred beyond the complete neighboring comment; no additional change claimed. |
| 6 | 2 | Explain practical computation and whether two network forwards are necessary. | Page 2, Section 3: one additional probe pass captures inputs; local affine products compute proposed changes without a second full-network pass. The next state supplies actual representation diagnostics. Two full forwards would include upstream changes and measure a different quantity. |
| 7 | 2 | Make additional computation explicit even if it is a limitation. | Page 2, Section 3 and page 8, Section 8: extra forward/input storage, FP64 products and the local rounding-check product are explicit; measured timings retain their limitations. No universal 2x training overhead is claimed. |
| 8 | 2 | Define H/W/ones/b dimensions, n by m, and the per-output score. | Page 2, Section 3: shapes are explicit; rows are examples/positions, columns output units. Squaring and averaging across rows yields one RMS score per unit; the MLP example specifies every shape and all 200 scores. |
| 9 | 2 | Define threshold calibration, dimensions and proportional scaling using update steps. | Page 3, Section 4: one scalar threshold per module is the linearly interpolated quantile of 8 times m scores. The first capped step is t0; preceding steps are t0-8 through t0-1. The example uses steps 441-448 before step 449. Per-unit scaling is min(1, threshold/score); a score twice the threshold halves the increment. No "fork state" terminology remains. |
| 10 | 2 | Clarify whether updates causing large activation sensitivity are dampened. | Page 3, Section 4: dampening responds to the joint proposed weight/bias movement on current probe inputs, not to weight magnitude or a Jacobian sensitivity measurement. |
| 11 | 2 | Clarify whether all incoming weights are dampened and whether this is per column. | Page 3, Section 4 and Algorithm 1: output column j corresponds to row j of W (m by d). One factor scales every incoming weight increment in that row and the corresponding bias increment. |

## Evidence and checks

Implementation statements were checked against `src/eos_tasks/representation_policy.py`
(`capture`, `affine_scores`, `apply_policy`) and
`src/eos_tasks/representation_continuation.py` (calibration and step placement).
Tracing statements use Analysis 063's event-window report/results table and the
ResNet observation. The report retains localization-versus-causality and
fixed-basis-versus-rotating-motion distinctions.

The two-pass LaTeX build passes with eight pages, a numbered algorithm, no
unresolved references, no overfull boxes and no out-of-page text. All eight
revised pages were rendered at 120 dpi and visually checked. All source-data,
table and figure hashes remain unchanged. The three standalone figures retain
their prior verified visual review; the report preserves log loss and linear
eta-kappa-hat. Archive hash and all 11 annotations were verified after copying.
