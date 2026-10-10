# Observation005: squared cumulative geometry for raw gradients and momentum

Date: 2026-10-10. User-requested display of A squared, B squared and their difference, with raw gradients and momentum in separate columns. This figure transforms existing scalar measurements; no training or new hypothesis is introduced.

## Sources and generation

[Figure](../figures/squared_cumulative_geometry.pdf), [exact plotted series and metadata](../squared_cumulative_geometry_data.json), [verification](../verification.json). Generator: `../../13_plot_squared_geometry.py`. Existing palette, source and axis helpers: `../../10_plot_lr_geometry.py`, with the unchanged Analysis008 `08_plot_simplified.py` value extraction.

Source: `../comparison_data.json`, SHA256 a02a1ba7ace7a7011063767e6e7fb1c25548be0a12be06b6dad6826e5e7b2739, containing verified scalar histories for all seven Runs021-027. Source, generator, helper and PDF hashes are recorded in the metadata. The original source and the preceding four PDFs are preserved.

## Definitions, layout and caption

For either vector stream v, at optimizer update t:

- A_t = sum from s=1 to t of ||v_s||.
- B_t = ||sum from s=1 to t of v_s||.
- Row1 displays A_t squared, the square of the cumulative sum of norms.
- Row2 displays B_t squared, the squared norm of the cumulative vector sum.
- Row3 displays A_t squared minus B_t squared.

The3x2 layout has raw pre-clipping gradients g in the left column and actual stored, uncorrected Adam first moments m in the right column. Panel labels are row-major(a)-(f). Momentum is measured after the accepted optimizer update and before preconditioning; it uses clipped gradients.

All seven learning rates overlay each panel. The existing reversed-plasma palette runs from yellow at the lowest LR to dark purple at the highest, uniformly sampled in log10(LR). Dashed curves identify original Spark conditions, solid curves the new RTX4090 conditions. Every Y axis is linear with an independent range, zero lower limit and automatic upper limit. Scientific-notation multipliers are explicit. Compare numerical axis values when comparing magnitudes across panels.

All curves are unsmoothed, with no normalization or token weighting of the geometric quantities. Display covers full-sized updates1-711; the partial update712 remains in source data and its squared endpoints are included separately in the figure metadata. The zero-LR first update is included, following the existing diagnostic clock. One seed1234, frozen initialization and token order; hardware differs across original/new groups and LR scales AdamW decay.

Caption: Squared cumulative raw-gradient and momentum geometry across seven warmup-only learning rates in Pythia-14M. Columns distinguish raw gradients from stored first moments. Rows show the squared sum of norms, squared norm of the sum, and their difference. Colors identify LR and line styles identify hardware group. Independent linear Y ranges retain each panel's trajectory shape; all curves are unsmoothed.

## Verification and interpretation limits

All42 plotted series and both-stream endpoints at update712 were checked against source measurements. A was independently reconstructed by accumulating individual recorded vector norms. A squared minus B squared was also checked using (A-B)*(A+B). All plotted values are finite and all differences are nonnegative. The PDF has one page and six panels; its latest110dpi render was inspected for legibility, complete labels, and absence of clipping or overlap.

The third row is the nonnegative deficit in squared resultant norm relative to perfectly aligned vectors with the same magnitudes. A squared is not the sum of squared individual norms, so this difference is distinct from the signed all-pair interaction J in Observation004. It does not quantify wasted learning or a causal benefit of momentum. One seed and mixed hardware limit cross-condition inference; these are unweighted vector-stream sums, not realized AdamW parameter displacement. No uncertainty band or new performance claim is introduced.
