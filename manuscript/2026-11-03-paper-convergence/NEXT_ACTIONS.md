# Bounded work queue for closing the paper

All actions below are proposed. This consolidation launches nothing and allocates no run, analysis or hypothesis ID. At the reviewed snapshot the next available IDs were Run217, Analysis039 and H029; recheck the registries when work is actually selected.

## Decision board

| Priority | Work package | Counterfactual or comparison | Deliverable and stopping decision |
| --- | --- | --- | --- |
| P0 | Freeze the core claim and prior-work distinction | Existing secant, gradient-product, directional-smoothness and Adam-spike accounts | One claim-to-evidence paragraph and explicit BB equivalence. No new model or run. |
| P0 | Make existing evidence visible | Current paper versus existing noise/HVP, Pythia schedule and historical forecast records | Choose one compact bridge; do not add all old figures. |
| P1 | Check incremental explanatory information | Current 127 defined severity points: cosine alone, peak norm ratio alone, signed product, and both components | Report every LR/model group and uncertainty limits. If direction adds little, reduce the scatter's narrative weight. |
| P1 | Audit existing prediction before repeating it | Archived blocked forecasts: identical rows, loss/norm baselines, cosine and projected change | Confirm feature availability, old target and component comparisons. Identify the smallest missing independent validation. |
| P1 | Audit actual-step recoverability | Raw projection versus s/y products at the same full-objective states | If required products are absent, capture them in a selected future fork rather than infer them from unrelated scalars. |
| P2 | Optional local action-effect pilot | Untreated Adam versus half LR, raw clipping, applied-step clipping, Armijo | Paired spike/progress/cost results, including failures and delayed spikes. Stop if no useful tradeoff emerges. |
| P2 | Quantify diagnostic cost | Identical workload with capture disabled, minimal capture, and optional full multi-stream capture | Wall time, variability and peak allocated memory. Verify unchanged numerical updates where practical. |
| P3 | Independent confirmation if the claim requires it | Frozen online geometry policy versus norm-only/loss-only and fixed policies on new seeds | Full-trajectory benefit with ordinary periods and costs retained. Negative result narrows the paper. |
| P4 | Submission pass | Chosen venue's current requirements and the original rejection | Final text/figures, reproducibility package and resubmission eligibility check. No publication without the author's request. |

## Existing-data work before more training

### Current anatomy and component ablation

Use the current fixed-pre-rise dataset and retain its original 812 candidates, 806 recovered, six right-censored records, and 129 recovered >10% events. The signed-product analysis has 127 defined cosines. Do not substitute the historical trailing-minimum episode population.

Compare log peak/pre-rise loss against pre-rise cosine, peak/pre-rise gradient-norm ratio, their product, and a model with both components. Use the same eligible events, response transformation and group boundaries. The peak norm ratio is an explanatory variable measured at the realized peak; call this a severity-description analysis throughout. Keep groups with small counts and show their counts.

Randomly splitting dependent events into train/test is not independent replication. Whole-trajectory or chronological sensitivity checks may assess fragility, but the current one-initialization-per-case dataset cannot supply seed-level confirmation. A nonlinear interaction should not be credited merely because its comparator is an artificially weak linear model; match model capacity and selection budget, and show the simple transparent fits as well.

### Historical forecast audit

Read `archive/PEAK_TEMPORAL_PROTOCOL.md`, `archive/PEAK_TEMPORAL_ANALYSIS.md`, and `archive/paper_evidence/PROTOCOL.md` under Analysis036. Reuse the recorded rows/predictions before fitting a replacement. The existing forecast predicts a >=10% rise relative to current loss within eight updates, in quiet states, using expanding-prefix fits and five later blocks. It is a retrospective study with chronological out-of-sample predictions, not a prospective collection on unseen seeds.

The existing comparison already includes gradient level/growth, momentum level/growth, cosine and richer loss histories. The pooled primary AP values 0.260/0.303/0.454 compare loss history, its norm extension and its projected-change extension. The robustness package also reports about 0.423 for its cosine extension and 0.447 for richer loss plus projected change. All inspected later blocks are now development evidence.

Check whether direction adds to a comparably expressive norm/growth baseline and whether the incremental advantage survives within-run/task reductions. Do not treat pooled AP as transfer across tasks, as an accuracy percentage, or as causal benefit. A fixed `Q > 2` alarm has only 21.0% state precision and 37.8% recall on that historical target; a useful interpretation is not yet a useful operating point.

### Curvature and Adam consistency

The project already contains sampled-versus-full gradients and 1,188 directional HVP probe pairs in Analysis027. Reuse that validation before proposing a duplicate HVP campaign. It does not settle the corresponding Adam step question.

For actual-step comparison, define `s = theta_next - theta`, `y = g_next - g`, retain `s^T y`, `s^T g`, `||s||` and the endpoint gradients' identities. Under a fixed smooth objective, `s^T y/||s||^2` is the path-average curvature along the applied step. The raw statistic remains the raw projection. Zero denominators and non-descending steps are undefined cases for the normalized step-response statistic and must stay visible.

If empirical pointwise/HVP validation is extended, separate starting-point curvature, integrated path average and actual loss change. A mismatch between those quantities is scientifically interpretable, not automatically an implementation failure. ReLU and numerical/normalization conventions must be accounted for.

## Optional reduced intervention pilot

The authoritative existing proposal is [Analysis038](../../analyses/038-2026-11-03-adam-spike-intervention-design/PROTOCOL.md). It proposes eight arms per model, 128 updates each, from states 32 updates before the formal pre-rise. The following five-arm subset is a **new closing recommendation** for author evaluation; the original protocol remains intact.

| Arm | Retain from Analysis038 | Purpose |
| --- | --- | --- |
| Control | A0 unchanged Adam | Contemporary untreated event and progress reference |
| Smaller step | A1 half LR | Direct step-scale comparator |
| Input clipping | A3 full-objective raw-gradient norm cap | Effect of modifying the input to Adam's two moments |
| Proposal clipping | A4 complete Adam-step norm cap | Effect of directly restricting motion while preserving ordinary moments |
| Safeguard | A7 full-objective Armijo with rollback | Reference with no accepted increase in the checked objective, charged for extra evaluations and stalls |

Keep the original fork states 657/383/755, LRs 0.001/0.003/0.003, seed0, objectives, moment histories, FP32/FP64 measurement contract and 128-update horizons. Use prior-only 64-state medians for clipping thresholds. Reconstruct each prefix and qualify the untreated replay before comparing treatments; do not reset Adam at a trained fork. The overwritten intermediate Run215 checkpoints cannot be recovered from scalar histories.

Scope choices for a future approval packet:

- MLP first: 657 prefix updates + 5 x 128 branch updates = **1,297 nominal updates**.
- All three models with five arms: 1,795 prefix + 1,920 branch = **3,715 nominal updates**.
- Original eight-arm proposal: 1,795 prefix + 3,072 branch = **4,867 nominal updates**.

These are work counts, not wall-time estimates or authorization. Armijo adds up to eight full-objective trial forwards per accepted update; validation, qualification, saving, transfers and retries are additional. Use the original rollback, direction fallback, finite-trial stop and complete-state retention rules. A monotone accepted loss is inherent to this safeguard; useful progress per evaluation/time is its real comparison.

The launch packet must identify each config/run ID, data/model identities, initialization and seed plan, image/token exposure, update budget, checkpoint policy, measured or qualified runtime, storage and maximum spending/duration when applicable. Do not borrow a historical cloud price or earlier session's spending envelope. The full-objective Transformer uses 2,088,600 supervised targets per update; do not describe the work count as cheap without timing it.

Use Analysis038's fixed untreated calendar-state baseline and shared target window to compare peak excess and excess area. Also report full continuation loss, delayed/replacement events, censored outcomes, final/mean loss, time-to-target, held-out coverage and cost. Its >=50% excess reduction with <=1% loss penalty is an operational pilot screen, not proof of population benefit. Preserve the continuous results even when the screen fails.

The original beta2 and numerator ablations remain useful mechanism probes. Defer them until the closing study needs to distinguish second-moment adaptation from step scale; they are not prerequisites for a diagnostic paper. Add independent seeds for the selected claim, not every exploratory arm.

## What would test the user's conditional clipping idea

Separate the action from the alarm. At a decision state, evaluate the current gradient and compare it with the retained preceding gradient. Then a past/current-only candidate is

```text
geometry_gate = (rho_current < 0) and (norm_current / norm_previous > 1)
```

The gate is unavailable when a denominator is zero. It describes the response to the preceding step and affects the outgoing update. It must not use a future peak norm, centered average, recovery time or an outgoing product unavailable before applying that update.

Freeze the clipping magnitude rule and action first. Compare: untreated; always-on clipping; norm-only gate; loss-only gate; geometry gate; and a simple fixed/held LR policy. Use a development calibration to give gates comparable alarm/intervention budgets, then freeze settings for unseen trajectories. Match token exposure, initialization, optimizer/RNG state and data order within seed. Do not tune each test trajectory until intervention counts look favorable.

Start with three independent new initializations, explicitly as an initial confirmation rather than a powered population study. A complete trajectory is eligible whether or not it produces a spike. Record ordinary-training interventions, misses, false alarms, event/severity sensitivity, lead to onset separately from lead to peak, intervention duration, cumulative LR reduction, clipping frequency, validation and wall time. A policy that suppresses a spike but loses equal-budget progress has a tradeoff, not an unconditional win.

Input clipping and actual-step clipping are distinct methods under Adam. A sign gate based on whether the current Adam step is first-order uphill would miss all 28 audited rise updates in the three selected examples. Reducing LR also changes AdamW shrinkage unless that factor is explicitly controlled. Neither raw clipping nor the proposed gate guarantees future loss decrease.

Do not launch this entire policy grid before the existing-data comparison and optional action pilot justify it. If geometry fails to improve on norms, retain that result and finish an interpretive paper with interventions as discussion.

## Cost and realistic-training bridge

Reuse the complete Pythia/MiniPile runs for the first practical section. Their raw gradient, momentum, adaptive direction and actual displacement must remain distinct. LR affects both movement and decoupled decay; old hardware differences prevent a clean timing comparison across every cohort.

For “cheap,” benchmark capture disabled versus only previous-gradient cosine/Q, then separately add cumulative retention and other streams. Synchronize GPU timings, include warmup and repeated measurement blocks, report full-step time and peak allocation, and retain unchanged-update checks. A stored FP32 vector costs 4p bytes: about 56/122/282 MB for the exact existing 14M/31M/70M parameter counts, or 4 GB for one billion parameters. These are theoretical buffer sizes, not measured peaks or runtime results.

If a new realistic AdamW confirmation is necessary, prefer one existing moderate Pythia configuration with independent seeds, standard scheduled/clipped training, pinned MiniPile inputs and complete validation over a single much larger model. Save the required cross-step products and immutable fork states from the start. Choose its exact token budget and runtime only in a concrete approval packet.

## Closing checklist

- [ ] Authors choose the strongest claim the paper will actually test: interpretation, warning, or intervention utility.
- [ ] Exact secant equivalence and overlap with Bai/Mishkin/SPAM are positioned accurately.
- [ ] Existing HVP, schedule and forecast evidence is assessed before commissioning replacements.
- [ ] Every empirical claim has a source, comparison boundary, denominator and independent replication unit.
- [ ] Current spike anatomy and historical episode forecasts remain separate.
- [ ] Required component and actual-step comparisons are completed or the claims narrowed accordingly.
- [ ] Minimal diagnostic overhead is measured before making a measured low-cost claim.
- [ ] Any selected new run has explicit authorization, reproducible inputs and a bounded launch packet.
- [ ] Main-text figures earn their space; removed material remains in research records rather than a bloated appendix.
- [ ] Original TNNLS objections and current venue/submission conditions are checked against the final version.
- [ ] Author and advisor review the compact manuscript, then freeze the evidence set for submission.
