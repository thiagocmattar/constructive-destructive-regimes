> Evidence snapshot; original scientific content retained. Relative links were rebased.
> Source: [004-timing-severity-audit.md](../../../../../../analyses/055-2026-11-05-adam-directional-amplification/observations/004-timing-severity-audit.md).

# Observation004: timing, severity and remembered warnings

- **Request, 2026-11-08:** adversarial consolidation-plan review; audit every loss increase/decrease, then measure forward horizons and minimum step-increase magnitudes, comparing curvature and angle/norm conditions.
- **Result:** there is pre-update temporal structure, especially for larger ResNet/Transformer rises. Negative cosine covers more rises; adding amplification generally flags fewer states. Useful lead times differ. Neither the universal next-step rule nor an accurate universal alarm is established.
- **Source:** unchanged Run215 / Analysis036 histories, checked against their pinned hashes by the existing Analysis055 loader. One initialization/model, three LRs/model, 1,024 updates/arm. Objective, precision and data contracts remain in [PROTOCOL.md](../../../../../../analyses/055-2026-11-05-adam-directional-amplification/PROTOCOL.md).
- **Scope:** local scalar reductions only; no fitting, training, new run/hypothesis, cloud resource or manuscript edit. The paper's executive interpretation is in [PLAN.md](../../../../PLAN.md).

## Measurement contract

- State s has s completed updates. At s, rho compares raw gradients at s−1 and s; r is their norm ratio; Q=1−rho*r. These measurements precede update s→s+1 and describe the incoming gradient change.
- Conditions: D = rho<0; G = r>1; DG = D AND G; Q≥2. The manuscript's rho<−1/r is exactly Q>2. All 9,207 defined states have Q≠2, so the strict/non-strict classifications agree. Q≥2 already implies DG; three flags are not independent evidence.
- One-step targets: incoming L_s−L_(s−1), explicitly retrospective; outgoing L_(s+1)−L_s, genuinely future relative to the diagnostic. Common states 1..1023 give 9,207 pairs; state 0 lacks the incoming diagnostic and state 1024 lacks an outgoing update.
- Strict increases use ΔL>0. A 10^-5 relative sensitivity excludes negligible changes. The severity grid uses >0, then minimum relative increases 10^-5, .001, .01, .05, .10, .25, .50 and 1, normalized by loss immediately before the target update.
- Forward offsets h=1..16 distinguish: the individual increase at s+h; net change L_(s+h)−L_s; and the largest rise above L_s within h updates. Forward tables include minimum rises 0, 1% and 10%.
- All-available forward support is s=1..1024−h. A separate common cohort s=1..1008 removes endpoint-composition changes across h. The additional subset L_s≤L_(s−1) uses only observed information and checks warnings outside an already-rising step.
- Warning memory counts each target update once: at s, flag if the condition held at any state s−w+1..s, for w=1..16. All widths use s=16..1023. This is a history-only rule, not access to future gradients.
- Coverage/recall = TP/(TP+FN); precision = TP/(TP+FP); false-positive rate = FP/(FP+TN). Precision lift divides precision by unconditional target frequency in the identical cohort. Empty denominators stay undefined.
- These are exploratory grids on known trajectories, not preregistered independent confirmation. No winning horizon or magnitude cutoff is selected for a universal claim.

## Findings

- Across all nine arms, 2,233 outgoing steps increase and 6,974 decrease. Q≥2 covers 853 increases and flags 955 decreases. No exactly flat steps occur.
- At the three high LRs, Q≥2 covers 172/454, 21/95 and 81/152 strict next-step increases, with 136/15/66 flagged decreases (MLP/ResNet/Transformer).
- The magnitude dependence supports different roles for direction and amplification:

| Next-step rise ≥10% | Number of rises | D hits | G hits | DG hits | Q hits | Q flags | Q precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP, LR .001 | 168 | 149 | 110 | 105 | 82 | 308 | 26.62% |
| ResNet, LR .003 | 12 | 12 | 11 | 11 | 7 | 36 | 19.44% |
| Transformer, LR .003 | 26 | 26 | 22 | 22 | 22 | 147 | 14.97% |

- Coverage does not increase monotonically everywhere: all three MLP ≥50% rises escape the current Q threshold. Their small denominator is shown in red on the figure.
- On common forward support, Transformer Q coverage of ≥10% individual rises is 22/26, 25/26, 26/26 at h=1,2,3. At h=3 it has 121 false positives: precision 26/147=17.69%, versus base frequency 26/1008=2.58%. Coverage later declines; h=3 is not a validated optimum.
- ResNet Q covers 7/12 at h=1 and 8/12 at h=2; negative cosine covers all 12 at h=1..4. MLP negative cosine covers 166/167 at h=2, whereas Q covers 78/167. Lead-time effects are not shared identically across architectures.
- For non-rising decision states, D covers 169/191, 29/33 and 64/65 cases of a ≥10% rise above current loss within eight updates. Precision is 43.44%/13.62%/16.75%, versus base frequency 33.99%/3.61%/7.59%. Q covers only 4/191, 2/33 and 20/65 in those same cohorts; broad regime information differs from a strict amplification alarm.
- Remembering Q over 16 past states covers all 168/12/26 ≥10% target steps, but flags 688/248/488 non-target steps. Precision is 19.63%/4.62%/5.06%, versus base frequency 16.67%/1.19%/2.58%. This supports a temporal antecedent, not a high-precision automatic intervention rule.
- Rising precision with a longer forward window must be compared with the rising base frequency. Conversely, rising recall with longer warning memory must be compared with the increased alarm population.

## Figure and reductions

- [Six-page vector figure](../support/adam_forward_and_severity.pdf), generated by [07_plot_forward_and_severity.py](../../../../../../analyses/055-2026-11-05-adam-directional-amplification/07_plot_forward_and_severity.py). Every page shows all three preselected high-LR cases and all four conditions; complete nine-arm tables remain available below.
- **Page 1 caption:** coverage and precision versus minimum outgoing step increase. All 1,023 eligible states/model; positive-target counts above curves, red when below 10. Gray is unconditional target frequency. Zero-target recall is undefined, not zero.
- **Page 2 caption:** a ≥10% increase at each future update h=1..16; identical s=1..1008 support and current-state flags throughout.
- **Page 3 caption:** net ≥10% rise relative to decision loss at each horizon, using the same support.
- **Page 4 caption:** any ≥10% rise above decision loss within each horizon. Positive counts grow with window length by construction.
- **Page 5 caption:** the same window outcome restricted to L_s≤L_(s−1). Alarm counts disclose sparse Q support, especially five MLP states.
- **Page 6 caption:** unique next-step ≥10% increases covered by a condition observed in the last w states. All widths use s=16..1023; increased memory improves coverage and increases non-target flags.
- [One-step summary](../data/one-step-summary.csv), [counterexamples](../data/one-step-counterexamples.json), [manifest](../../../../../../analyses/055-2026-11-05-adam-directional-amplification/derived/one_step_audit/manifest.json): generator [04_one_step_audit.py](../../../../../../analyses/055-2026-11-05-adam-directional-amplification/04_one_step_audit.py).
- [Severity](../data/severity.csv), [horizons](../data/horizons.csv), [manifest](../../../../../../analyses/055-2026-11-05-adam-directional-amplification/derived/forward_and_severity/manifest.json): generator [05_forward_and_severity.py](../../../../../../analyses/055-2026-11-05-adam-directional-amplification/05_forward_and_severity.py).
- [Warning memory](../data/warning-memory.csv), [manifest](../../../../../../analyses/055-2026-11-05-adam-directional-amplification/derived/forward_and_severity/memory_manifest.json): generator [06_warning_memory.py](../../../../../../analyses/055-2026-11-05-adam-directional-amplification/06_warning_memory.py).

## Verification and scope

- Source hashes and original state/update/gradient alignment pass the existing loader's checks.
- Independent scalar raw-file reconstruction verifies 36,828 one-step classifications. Exact rational checks distinguish incoming/outgoing timing and verify the GD/BB identity.
- All h=1 forward targets agree with the one-step audit. Common-support window positives are nested; angle/norm and Q predicates agree at every defined state.
- The plot generator independently checks 972 forward summary rows from raw records at h=1,3,16, covering 865,395 decision classifications across all nine arms, targets, thresholds and subsets. It uses the angle/norm inequality independently of stored Q.
- Direct slices verify 580,608 warning-memory flags; all nested warning sets and history boundaries pass.
- Input/output hashes and numerical checks are in [verification.json](../data/audit-verification.json). All six final pages were rendered and visually reviewed: legends, titles, counts, curve limits and footers are readable without overlap.
- States/windows overlap and are temporally dependent; one initialization/model does not yield independent-seed confidence intervals. The largest-rise counterexamples are illustrations of exhaustive counts, not a selected evaluation set. No causal mechanism, semantic forgetting, generalization or calibrated forecast follows from these counts alone.

Reproduce from the repository root, in order: scripts 04, 05, 06, then 07. Old Analysis055 artifacts and source trajectories are unchanged.
