# Adversarial review, round 2: final methods check

Date: 5 November 2026. Reviewer: independent methods/theory agent.

Reviewed artifact: [revised-manuscript.pdf](../revised-manuscript.pdf), 18 pages.

SHA-256: `d078731c253ffed9da6d0120795beb205a40bc00b0a785cf1419762e65e1d44f`.

Scope: final active TeX, the rebuilt PDF's extracted text, visual inspection of PDF pages 5, 11, 13 and 17, the intervention table, selected machine-readable evidence, and the implemented Pythia schedule. This is a final mathematical and reporting review, not a repeat of every raw-artifact audit or every-page visual QA.

## Disposition

**No remaining scientific correctness blocker was identified within the manuscript's stated diagnostic scope.** The required round-1 method-reporting repairs are closed. I would not retain a methods-based rejection on the grounds raised in the previous round.

This is not an acceptance forecast. The journal-level significance of the integrated empirical account remains an editorial judgment: the constituent mathematics is elementary or classical, and the full-batch evidence is conditional on one initialization per model. Those limitations are now visible and do not require further experiments to make the claims that the paper actually states.

## Round-1 closure

| Issue | Final disposition |
| --- | --- |
| R1: probe exclusion described too broadly | **Closed.** Section VI-A now limits non-overlap to the union of the three 350-update training prefixes in the intervention study. Same-corpus scope remains explicit. |
| R2: missing Pythia LR recipe and 350-update schedule clock | **Closed for scientific reporting.** Peak `1e-3`, floor `1e-4`, linear 1% warmup and cosine decay are specified. Native component-intervention trajectories explicitly use the first 350 updates of the original 712-update clock. |
| N1: smoothness versus actual architectures | **Closed.** Appendix B names ReLU for ResNet and Transformer and declines the classical-Hessian identity across activation boundaries. |
| N2: nonlinear loss-sign inference near the approximate boundary | **Closed.** Section III-E requires the predicted change to dominate approximation error. |
| N3: “smoother” versus a more-positive measured cosine | **Closed in the intervention interpretation.** The relevant sentence now states more-positive raw-gradient alignment and the specific training-loss outcome. |
| N4: malformed GD numerical range | **Closed.** The secant means now lie between 1.97 and 1.99. |

For exact reproducibility, the archived LR implementation should remain part of the accompanying code/config evidence: `src/baseline/recipe.py` uses a pre-step scheduler clock and a clamped cosine. The concise prose alone does not uniquely determine rounding and indexing. This is a nonblocking reproducibility detail, not a residual error in the reported comparisons.

## Final checks and improvements

The raw-gradient versus actual-displacement reminder is present in the Pythia methods. Consequently, the all-pair raw interaction result is no longer liable to being read as an AdamW displacement identity.

The actual-step clipping measurements strengthen the paper materially. For the selected MLP window, mean actual increment norm rises from 0.04179 to 0.05707 and its maximum from 0.09366 to 0.18368 despite an input-gradient cap. ResNet's maximum actual step instead falls from approximately 0.600 to 0.158. These values agree with the selected machine-readable evidence; this review did not rerun their raw-file reductions. They directly substantiate why identifying the vector stream matters, beyond merely restating that Adam transforms gradients.

The GD paired-factor paragraph now explains a positive analytical payoff: the Transformer has the strongest within-pair cancellation of the three high-rate paths, yet the largest overall path fraction because its residuals accumulate more coherently. This makes the purpose of measuring both factors apparent. The same paragraph retains the separate objective and pair-offset checks.

The local-loss and Adam loss-integral formulas remain consistent. Smoothness, sampling and finite-step qualifications are preserved. The 794-step result describes negative first-order contributions throughout selected rise intervals; it does not imply that every individual step raises loss. The first-moment recurrence, alternating reference, HVP discrepancy and bound remain correctly restricted to an analytical comparison.

The inspected PDF pages preserve equation signs, denominators, subscripts and table captions. In particular, the anatomy table explicitly identifies its retained raw projected-change column with the newly named `z` statistic. No actionable mathematical rendering problem was observed on the inspected pages.

## Remaining significance and replication risks

1. **Integration is the contribution.** The displacement expansion, paired factorization and secant interpretation are not a new convergence theory. The revision acknowledges close work and must retain that positioning in any submission letter or response.
2. **Conditional evidence is not population replication.** Many Adam events and matched branches share a seed or parent. The stronger three-block SGD/Pythia scope cannot be transferred to the single-initialization full-batch cases.
3. **Interventions are selected examples.** Their outcomes establish concrete distinctions within their measured windows, not a universal ranking of clipping, LR schedules or optimizer components. Training, complete validation and fixed-probe outcomes remain separate.
4. **Parameter geometry is coordinate-dependent.** The evidence does not establish representation loss, forgetting, or an invariant measure of useful learning. Functional and causal claims would require other measurements.
5. **Practical value is interpretive.** The paper supplies a more precise diagnosis and assessment of intervention hypotheses. It has not demonstrated a validated controller, forecasting advantage, or measured overhead advantage over the one-HVP alternative.

These are honest boundaries rather than newly discovered correctness failures. Within them, the manuscript now presents a coherent, reviewable methods-and-evidence argument. I recommend ending this revision cycle after the existing final build/evidence checks and reporting the residual significance uncertainty without soliciting an artificial unanimous acceptance vote.
