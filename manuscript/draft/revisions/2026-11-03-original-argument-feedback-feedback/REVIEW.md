# Review report - original argument / feedback / feedback

This revision addresses all 30 annotations in the preceding revision's newly supplied feedback PDF. Passages marked remove/delete are deleted. The original introduction, front matter, conclusion, prior revisions and empirical assets remain preserved.

## What changed

The paper now presents the same evidence more directly. The estimator and loss-rise condition are shorter, the GD overview follows its setup without an extra heading, and Adam uses the existing raw-gradient definitions explicitly. The individual-spike discussion follows the requested sequence: destructive regime, gradient growth, diagnostic rise before the peak, and recovery. The aggregate discussion uses percentages and states the explanatory strength of the cosine/norm relationship.

Recovery is tied to the loss immediately before the rise. The new illustration makes the four event instants explicit. The earliest-maximum rule is unchanged; only its notation is simpler.

## Annotation dispositions

| Comment | Location | Action |
| --- | --- | --- |
| C01 | II-D | Deleted both examples about absent spikes. |
| C02 | III | Deleted both sentences after the constructive/destructive definitions. |
| C03 | III-A | Changed sampling contribution to noise contribution. |
| C04 | III-A | Replaced the highlighted colon with a direct so transition. |
| C05 | III-A | Replaced the forgetting qualification with the iid overwriting mechanism. |
| C06 | III-B | Collapsed the estimator to one line and removed the dot-product form. |
| C07 | III-B | Deleted the complexity and extra-computation list; retained the one-gradient implementation sentence. |
| C08 | III-C | Deleted the normalization qualification. |
| C09 | III-E | Kept only the cosine inequality in the box. |
| C10 | IV | Deleted the marked calibration and setup qualifications. |
| C11 | IV | Removed the first subsection heading; the overview follows the setup. |
| C12 | Figure 1 | Removed the legend parenthesis; all 27 curves are numerically unchanged. |
| C13 | IV-A | Changed destructive alignment to destructive regime. |
| C14 | V; Appendix A | Identified rho explicitly as raw-gradient cosine, removed the repeated equation, and corrected the appendix to show Adam direction d and raw-gradient projection g separately. |
| C15 | V | Explained that successive raw gradients reveal changing local loss slope across the states visited by Adam. |
| C16 | V | Removed the standalone training-setup subsection. |
| C17 | V | Replaced interspersed with with. |
| C18 | V-A | Deleted the selected-peak temporal-filter qualification. |
| C19 | V-A | Deleted the momentum-timing qualification. |
| C20 | V-A | Condensed the case analysis around negative raw cosine, norm growth, diagnostic increase before the peak, and the distinct descents. |
| C21 | V-B | Defined recovery as the first loss at or below the fixed pre-rise loss L(a_i), not the first elevated loss. |
| C22 | V-B | Used membership in argmax and a plain earliest-maximum tie rule; event extraction is unchanged. |
| C23 | V-B | Deleted the amplitude-filter qualification. |
| C24 | Figure 7 | Added a compact schematic labeling a_i, a_i+1, p_i and c_i with the fixed baseline. |
| C25 | V-B | Deleted the marked recovery and non-universality paragraph. |
| C26 | V-B | Replaced narrative frequencies with 92.1%, 100%, 16.5%, 63.6%, 22.5% and 14.0%; retained table counts. |
| C27 | V-C | Deleted the event-descriptor and online-predictor caveat. |
| C28 | V-C | Deleted the regression-interpretation caveats. |
| C29 | V-C | Stated the joint explanatory role of the destructive regime and gradient growth, retaining the per-model R-squared ranges on the log response. |
| C30 | V-C | Deleted the clipping qualifications; retained its literature motivation and future-intervention proposal. |

## Figures and evidence

**GD overview:** [generator](01_prepare_figures.py), [manifest](curvature_comparison_manifest.json). The legend change requested in C12 is the only change. All 27 curves and 53,838 plotted observations match the preceding figure exactly, including 279 retained spectral measurements. Loss and spectra remain unsmoothed; the raw-gradient diagnostic retains the trailing 50-pair mean. Coverage and uncertainty remain those of Analysis024. The prior PDF and generator remain intact.

**Spike boundaries:** [generator](01_prepare_figures.py), [manifest](spike_definition_manifest.json). The explicit synthetic sequence illustrates the mathematical definition; it is not an empirical event and does not enter any statistic. It shows the fixed pre-rise baseline, first elevated state, maximum and first recovery. Straight segments connect the illustrative samples, with no smoothing or uncertainty interval. Caption and labels identify it as a schematic.

**Quantitative claims:** event percentages are recomputed from the unchanged 129-row anatomy sample. Pre-rise quantities use the 127 defined measurements; the two initialization events remain in the other summaries. Existing density, micro-plots, table and regression fits are unchanged.

## Verification

The 13-page manuscript compiles with no undefined references, duplicate labels or overfull boxes. All 934 retained mathematical cases pass, including the adaptive-direction identity; figure/data checks and all 37 protected-file hashes pass. All 13 pages and both changed figures were visually inspected. The new schematic, density and anatomy table appear in sequence on page 10, followed by the regression on page 11. See [verification](verification.json) and [visual review](review.json).

Reproduce with `python manuscript/draft/revisions/2026-11-03-original-argument-feedback-feedback/01_prepare_figures.py`, then `python manuscript/draft/revisions/2026-11-03-original-argument-feedback-feedback/02_build_and_verify.py`. The build reruns the preserved numerical and source checks before rendering the paper. No training or new empirical measurement was performed.

## Revision chain

[Original argument restored](../2026-11-03-original-argument-restored/README.md) -> [original argument / feedback](../2026-11-03-original-argument-feedback/README.md) -> **original argument / feedback / feedback**. The annotated source and the preceding review PDF are preserved byte for byte.
