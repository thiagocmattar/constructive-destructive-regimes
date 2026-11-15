# Second author-feedback revision — 15 November 2026

- **Read next:** [manuscript.pdf](../../draft/manuscript.pdf), 18 pages; [supplement](../../draft/supplementary.pdf), 27 pages.
- **Scope:** 61 annotations in [feedback2](../../draft/archive/2026-11-15/manuscript+feedback2.pdf), preserved byte-for-byte. Both feedback rounds now reside in the archive; no locked duplicate remains at the draft root.
- **Scientific inputs:** unchanged training trajectories, intervention arms, initialization, windows, and reference sets. No training or cloud spending. All 416 protected original files remain unchanged.
- **Review approach:** the existing author-supplied review skill's gap → contribution → evidence approach informed the comparator paragraph. The explicitly requested Socratic sub-agent reviewed the definitions before and after editing. Its final corrections concerned temporal-basis notation, increment versus parameter scaling, and q25-specific verification.

## Annotation resolution

| IDs | Resolution |
|---|---|
| 1 | Adam moments use separate aligned lines. |
| 2–7 | Removed the long post-equation qualification and redundant figure narration; moved the full Adam grid into S3. Retained the precise gradient-projected meaning in the derivation and caption. |
| 8 | Explained possible basin movement and momentum delay as interpretations; cited the relevant SGD catapult evidence, rather than an inapplicable no-free-lunch theorem. |
| 9–13 | Clarified before-update classification and incoming/outgoing timing. Named the destructive loss-spike condition; explained why its two table rows are algebraically equivalent. Simplified accuracy, hit rate, and lift. Exact preceding-step sign equivalence is confined to quadratic plain GD and strict positivity. |
| 14 | Removed the extended severity, horizon, warning-memory, and confusion-table presentation from the compiled supplement; original evidence and historical figures remain preserved. |
| 15–19 | Defined training states, windows, measured sites, row count, and the relationship between gradient and functional-change cosine before Algorithm 1. Replaced measured-stage terminology with site; residual architectural labels are Group 1/2/3. |
| 20–24 | Added representation retention, its exact equation, and one interpretation paragraph with first-site/output results from the full retained Grams. No truncated-mode approximation. Low retention is displacement cancellation, not a direct measure of learning quality. |
| 25–32 | Organized Fig. 6 interpretation from output backwards toward input sites. Distinguished pre/post nonlinearity and forward attenuation from the analysis reading order. Specified the 15-step and five-step lead relative to the plotted 10% markers. Architecture-dependent explanations remain hypotheses. Annotation 29 contains only a period. |
| 33–39 | Defined a mode as the site/window-specific spatial right singular vector; clarified its coordinates, singular value, temporal coefficient, and rank bound. Removed the toy vector and long rank-percentage paragraph. Defined final network output and temporal overlap before the algorithm and figures. Completed the two Socratic review passes. |
| 40–48 | Renamed the intervention section; strengthened the working hypothesis; used proposed update/site/cap consistently. Explained one local linear product or convolution plus reduction, separately from reference evaluation and implemented verification costs. |
| 49–50 | Rebuilt Fig. 8 with absolute mean loss and upward variation. Rebuilt Table III with Intervention/Parameter value, three absolute training outcomes per model, one decimal in explicitly stated ×1000 units, column minima in bold, and an above-table caption only. |
| 51–55 | Kept direct/upstream decomposition in S5. Positioned the cap against relative representation-change correction, AGC, and FOOF without an untested superiority claim. Added comparator citations. Used t0/tf and mathematical P. |
| 56–58 | Focused the main results on training loss and smoothness; held-out audit stays in the supplement. Reduced intervention subsections to method and comparison. Both main plotted metrics and all main table metrics now have lower-is-better direction. |
| 59–61 | Replaced the main q50 motion illustration with q25, retained every quantile in the table, used logarithmic losses for all models, removed peak markers, reduced labels, and matched the tracing figure typography. q25 is an illustration chosen after inspecting training outcomes, not the prespecified optimum. |

## New retained-data reduction

- Generator: [01_retention.py](01_retention.py); [all-site CSV](representation-retention.csv); [source hashes and full eigendecomposition checks](retention-manifest.json).
- Incoming windows: MLP 451–501 (51 changes), ResNet 390–450 (61), Transformer 553–601 (49).
- Exact calculation: `sqrt(sum(K)) / sum(sqrt(diag(K)))`, where `K[i,j] = dot(delta[i],delta[j])`.
- MLP Gram slice is `[3:54,3:54]`; ResNet and Transformer drop the retained predecessor with `[1:,1:]`. No coordinate averaging, centering, or mode truncation.
- First-site/output retention: **3.25/1.71%, 3.72/2.44%, 24.33/3.51%**. All saved representation sites are reported in the CSV; parameter/gradient matrices are excluded.
- Symmetry, PSD tolerance, window dimensions, and direct-versus-eigen numerator agreement are checked. Zero total motion is undefined.
- Interpretation: small net displacement relative to traveled distance, combined with observed negative cosine, supports back-and-forth functional motion. Small retention alone can also result from orthogonal movement and does not measure retained knowledge.

## Revised intervention artifacts

- Generator: [04_figures.py](04_figures.py); [numeric manifest](presentation-manifest.json).
- Sources: frozen `matched-results.json` and `matched-curves.npz` in evidence/05-adam/c-intervention/data; all 36 arms and original windows are preserved.
- Follow-up author request: Table III now groups its four intervention families with LaTeX multirow. Fig. 8 uses exact absolute final loss horizontally and positive variation vertically. The horizontal axis is logarithmic; table values remain untransformed loss metrics scaled by 1000. Mean loss remains in the table as a trajectory-wide summary, not the primary convergence comparison. Final loss is the fixed-budget endpoint; the supplement retains final-32 means to expose phase sensitivity.
- Fig. 8 uses final loss and positive variation. Its symmetric-log vertical scale retains zero (linear below .001 for MLP and .01 otherwise); points are not jittered, only label locations move.
- Table III contains 108 training-metric cells, independently recomputed from complete loss curves; ×1000 units allow one decimal without rounding every small loss to zero. Bold uses full-precision minima.
- Fig. 9 uses q25 and unchanged Adam, all post-fork updates and measured sites. RMS normalization is shared within each site across arms, using its unchanged-Adam maximum. The common color scale retains all values; no smoothing.
- q25 maximum output RMS reductions: **92.7/33.0/45.1%**. Squared output movement changes: **−97.6/−2.5/−59.7%**. Output reversal frequencies: **49.7/59.7/88.5%**.
- The MLP output-loss identity holds for every recorded q25 step. Among 140 rises above 1e-8, 139 have a negative linear term exceeded by the quadratic term; one has a positive linear term. All 161 control rises have the former pattern. The main claim explicitly retains this exception.
- Coverage: one initialization per model, selected finite windows, unequal effective intervention strengths. q25 improves mean/variation/final loss in each case but is not universally best. These are training-dynamics comparisons; no seed-level uncertainty or generalization improvement is inferred.

## Reproduction and verification

- `01_retention.py` recomputes the new scalar reduction. `04_figures.py` regenerates the two revised main intervention PDFs and Table III.
- `02_revise.py` and `03_interventions.py` are historical one-time editorial scripts; **do not rerun them over subsequent author edits**. Later wording/layout corrections are retained in the final TeX.
- Current full build pipeline: `../readability_revision_20261114/02_figures.py`, `03_build.py`, `04_verify.py`, and `05_review_sheets.py`.
- Checks cover 416 protected originals; retained tracing arrays; 144 underlying intervention metrics; 108 compiled main-table values; diagnostic counts; curvature/cap identities; citation/reference resolution; embedded fonts; page bounds; and final rendered pages.
- Figures, tables, main PDF, and supplement stay inside consolidation. No push is authorized by this round.

## Loss-scale clarification

- MLP uses half-squared error; ResNet and Transformer use mean cross-entropy. For the Transformer, perplexity is exp(cross-entropy in nats), not log(cross-entropy). A logarithmic plotting axis changes only the display. Positive variation is computed from original loss differences; exponentiating it would not produce a useful perplexity statistic.
- Primary reference for the perplexity identity: https://huggingface.co/docs/transformers/v4.56.2/perplexity
- All 36 final-loss/variation coordinates are checked against full-precision independent curve reductions. All 108 table cells and the four multirow groups are checked. No experiments or source evidence changed.
