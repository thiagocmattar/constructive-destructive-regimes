# Critical review of the Analysis024 figures for an IEEE paper

The review covers the six isolated figures and their use in the new manuscript
Section IV. The numerical evidence remains the completed nine-condition,
one-initialization calibration. This revision changes presentation and indexing,
not training inputs, observations, smoothing windows or scientific Y values.

## Presentation decisions

IEEE's [graphics size guidance](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/resolution-and-size/)
gives 7.16 inches for two-column graphics. Its
[format guidance](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/file-formatting/)
recommends embedded fonts and approximately 9-10 point type at final size.
The revised vector PDFs are drawn at 7.16 inches rather than relying on a
large reduction of the former 11.4-inch artwork. They use embedded Times New
Roman text, STIX mathematics, 9.5-point axis labels, 9-point headings,
8.5-point ticks and 8-point legends. Small legend type remains a compromise
for three rates in each column; it was checked in the compiled manuscript.

All data curves are solid and use the author's requested color-only encoding.
Dashed lines indicate scalar reference values only. This preference does not
provide a redundant grayscale encoding. Panel letters are in the headings or
above the lower axes, never over data. Embedded footers are removed; seed,
coverage, smoothing and interpretive qualifications belong in figure captions
and experimental methods. Loss ticks are simplified to readable decimal values
while retaining logarithmic spacing. No observations are floored or clipped.

## Figure-by-figure assessment

| Figure | Critical assessment | Revision and manuscript role |
| --- | --- | --- |
| Training loss, log Y | The unsmoothed trace is necessary to retain visible fluctuations. Log axes emphasize relative changes and can visually compress spikes; they do not establish a spike threshold. Objective units and limits differ across models. | Keep unsmoothed, use simple numeric log ticks, and state independent scales in the caption. Main Fig. 1 establishes progress before interpreting geometry. |
| Raw cosine | Essential for distinguishing an isolated sign event from a persistent negative phase. Dense late traces and exact overlap near +1 make this less readable as the main comparison. Missing-looking lines can be coincident curves. | Keep every defined pair and the zero reference. Appendix D / Fig. 8 supplies the raw evidence and explains overlap; it is not used as a separate statistical replication. |
| Smoothed cosine | A 50-pair trailing mean exposes sustained behavior, but shifts and broadens transitions and omits the incomplete initial window. Its crossing cannot date the first raw reversal. | Retain complete-window arithmetic means, a common [-1.06,1.06] scale, and caption the reduction. Use it above unsmoothed R_t in Main Fig. 2. |
| Smoothed eta*kappa_hat | q>1 and negative cosine are exactly equivalent before smoothing. A mean of pointwise q is not q formed from mean gradients/norms. A plateau near two is consistent with near-opposing, comparable-norm gradients; it is not an independently measured maximum eigenvalue. | Retain pointwise-then-average reduction, a common untruncated scale and the reference at one. Main Fig. 3 pairs it with gradient magnitudes. The text separates it from the spectral endpoint table and retains finite-step and subset caveats. |
| Smoothed gradient norms | A norm increase adds magnitude information but alone establishes neither divergence nor reduced learning. Averaging norms before log display differs from averaging logarithms or taking the norm of a mean gradient. Scales differ across parameterizations/tasks. | Keep complete 50-gradient arithmetic means before log display. Main Fig. 3 uses within-model comparisons; no cross-model magnitude ranking or zero-denominator substitution is introduced. |
| Unsmoothed R_t | This is the direct cumulative quantity needed for the displacement argument. It retains complete-prefix geometry across all lags, but depends on the entire preceding path. Low R_t does not prove a negative all-pair sum, small absolute displacement, poor generalization or less information learned. | Keep every update, a shared [0,1] reference range with display margins, and no smoothing/reset. Main Fig. 2 pairs cumulative and local geometry without treating them as equivalent. |

## Scientific argument and limits

The main presentation has three steps: show continued loss reduction; compare
adjacent reversal with cumulative displacement; then interpret magnitude and
the projected change. A generated table reports all nine final losses, late
cosines, final R values, late projected changes and final spectral references.
Late means cover 1,000 observations, while the spectral entries are endpoints;
the text explicitly avoids equating these temporal reductions.

The clean spectral comparison is the MLP, where the training gradient and
Hessian use the same 5k objective. ResNet and Transformer spectra use fixed
subsets. Even with matched coverage, maximum pointwise eigenvalue and
finite-step directional gradient change are different quantities. Differences
cannot be attributed uniquely to subset error, gradient direction or Taylor
remainder without additional measurements. No such attribution is claimed.

Rates were chosen using prior calibration, and all trajectories use one paired
initialization per model. The nine rows are not nine independent replicates.
There are no across-initialization confidence intervals, significance tests,
universal phase boundaries or prospective warning/control results. Full-batch
evidence isolates the zero-sampling-noise case; it cannot establish the
manuscript's noise approximation on moving SGD trajectories. Training loss
does not establish generalization, and the retained geometry is not measured
in information units. Diagnostic runtime/memory overhead is not isolated by
an execution benchmark.

The new section is consistent with the restricted displacement argument.
The older introduction, optimizer discussion and conclusion remain a separate
revision concern: this insertion does not resolve every legacy stability or
learning interpretation elsewhere in the paper.

## Integrity, placement and review

All 54 standalone Y arrays are unchanged. Gradient diagnostics are now
displayed at the manuscript's one-based gradient index t (evaluation at
theta[t-1]); loss and R retain state/completed-update indices. The former
gradient-state X coordinates therefore increase by exactly one, with no
interpolation, additional pair or endpoint gradient inferred.

The two manuscript composites reuse existing arrays: smoothed cosine above
raw R_t, and smoothed norms above smoothed projected change. Their letters
are unique (a)-(f), the model headers/legends occur once, and all six isolated
PDFs remain available. The composites are 3.65 inches high to keep Main
Figs. 2 and 3 together before the next section without shrinking the font.

Verification covers source/config identities, exact diagnostic/aggregation
identities, every artist array, old/new Y hashes, the explicit one-index
change, copied-PDF hashes, table values, embedded fonts, PDF bounds, complete
LaTeX references and page-level rendering. The results and figure manifests
retain the hashes. The new observations and manuscript-asset manifest supply
the claim-to-source map. No new training, sampling or cloud resource is used.
