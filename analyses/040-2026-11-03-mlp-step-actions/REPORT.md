# Local MLP pilot: half LR wins this selected fork

Single-seed exploratory result,3 November2026. All eight approved arms completed128 updates from the exact authenticated state657. No A7, extra seed, LR search or event reselection. The owned RTX5070Ti Laptop GPU completed the full worker in **64.06seconds**, including qualification/prefix/setup/checkpoints/evaluations; zero cloud spending.

**A1 half LR and A4 proposal clipping meet the prespecified pilot screen.** A1 has the strongest training result; this does not establish an online trigger or generalization benefit.

## Focal view and scientific reading

[Three-method focal figure (PDF)](figures/focal_three_arm.pdf): unchanged Adam,
half learning rate and Adam proposal clipping. This is a **descriptive selection
after the exploratory screen, not independent confirmation**. The aligned panels
retain all 129 states without smoothing, with common scales within each row.
The original eight-arm figure and every method in the tables remain available;
the severe raw-numerator and transported-secant failures are outside this focal view.

Every update uses the same complete 5,000-image objective and fixed physical
chunks; the checked replay was exact. The jagged traces here reflect deterministic
full-objective oscillations, rather than random minibatch-sampling noise. Half LR
is strongest at this fork. Proposal clipping reduces excursion size but leaves
or replaces events. Quarter LR (A2) also removes most excursions, yet its final
loss is 3.19% worse than A0 and misses the predeclared 1% final-loss margin. The A8
transported-secant failure remains negative evidence for that rule at this fork.

This is one seed, one selected state and a 128-update horizon. There is no
established incremental benefit from diagnostic-based timing beyond the simple
step-size controls, and no generalization claim. A clearer figure adds no new
observations and does not resolve that scientific uncertainty.

## Fixed-reference training comparison

B=L_A0(689)=0.013039679266512394. Peak excess is max(0,L/B-1); area is the discrete sum of positive excess on the unchanged97-state window689..785. Mean loss covers exactly128 post-fork states658..785. Target=0.95B=0.012387695303186773. Time-to-target is first observed attainment from fork657, including initial checkpoint/observation overhead; nonattainment is censored.

| Arm | Peak excess | Excess area | Mean loss | Final loss | Target updates / seconds | Clip or fallback | Runtime s | Screen |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |
| A0 | 3.176299 | 46.621156 | 0.01842484 | 0.01162407 | 76 / 2.689 | none | 5.05 | control |
| A1 | 0.000000 | 0.000000 | 0.01215099 | 0.01064666 | 51 / 1.633 | none | 4.73 | passes |
| A2 | 0.024061 | 0.231566 | 0.01290706 | 0.01199505 | 98 / 2.839 | none | 4.54 | does not pass |
| A3 | 1.932850 | 34.946353 | 0.01697318 | 0.01720669 | 63 / 2.694 | raw clip88/128 (68.75%) | 6.53 | does not pass |
| A4 | 1.538168 | 16.890751 | 0.01509753 | 0.01109911 | 53 / 1.792 | proposal clip55/128 (42.97%) | 4.89 | passes |
| A5 | 30.998188 | 2706.459400 | 0.88282813 | 0.36648791 | >128 / censored | none | 6.32 | does not pass |
| A6 | 1.779660 | 40.466055 | 0.01877117 | 0.01863936 | 106 / 5.135 | none | 7.14 | does not pass |
| A8 | 100.143708 | 436.800524 | 0.05771944 | 0.02026736 | >128 / censored | fallback5/128 (3.91%) | 4.97 | does not pass |

The screen requires >=50% reductions in both excess measures, no more than1% worse mean/final loss, complete coverage and finite execution. It is a development ranking rule, not statistical confirmation. Runtime includes source/data setup, logging, diagnostics and saving; A0 also saves three extra replay/event states, so these are charged observed costs rather than an isolated optimizer-speed benchmark.

## What the controlled comparison teaches

- The untreated event is reproduced at exactly689/703/729 with peak/pre-rise4.1762988431. All786 losses through state785 match historical Run215 **bitwise**; contemporary replay also matches weights, moments, RNG and retained vectors. Thus the intervention comparisons address this historical selected event on the same machine.
- A1 half LR removes all positive fixed-window excess, lowers mean training loss34.05% and final loss8.41%, and reaches the target in51 rather than76 updates. A simple smaller-step control is effective here; no diagnostic-conditioned scheduling benefit has been established.
- A4 proposal clipping reduces peak excess51.57% and area63.77%, with18.06% lower mean and4.52% lower final loss. It passes the screen but produces later/replacement excursions, including a peak at741. The result supports a conditional motion-cap benefit, not elimination of all spikes.
- A2 quarter LR nearly removes excess but ends3.19% worse than A0 and reaches the target later (98 updates). The stronger dose fails the1% final-loss margin; suppressing excursions alone does not select useful progress.
- A3 raw-gradient clipping acts88 times. It reduces peak excess39.15% and area25.04%, but final loss is48.03% worse; its larger remaining excursion peaks at734 and remains unrecovered at785. This setting is not kept as a successful pilot treatment.
- A6 faster second-moment adaptation reduces peak excess43.97% but area only13.20%; mean loss is1.88% worse and final loss60.35% worse. Correct historical bias correction does not rescue this treatment at the selected fork.
- A5 raw numerator causes an early loss16.8173 at state659 and remains unrecovered through785. A8 transported secants cause a new peak1.31888 at725 and also remain unrecovered. Both miss the target. A8 uses mean alpha0.3619 times base with5 fallbacks,2 lower-floor and5 upper-cap clamps; a cap on transported curvature steps is insufficient here. Direction and magnitude change together for A5, and no isolated mechanism is identified.
- H028 is supported only in the restricted sense that two predeclared interventions improve this selected single-seed training tradeoff. A5/A6/A8 at these settings are negative results; the pilot supplies no population robustness, online prediction, stability guarantee or generalization claim.

## Delayed/replacement and censored excursions

The unchanged loss-only Analysis036 detector is applied to each arm joined to the entire shared prefix. Events retain their own fixed pre-rise references. The count below includes >10% events with a peak in the continuation; a later event starts after the original A0 recovery729. No cross-arm event identity is assumed. Exact small/large events and cross-fork flags are retained in [excursions.csv](excursions.csv).

| Arm | >10% events | Right-censored | Later starts >729 | Largest own-reference ratio | Full-window maximum loss @state |
| --- | ---: | ---: | ---: | ---: | --- |
| A0 | 7 | 1 | 3 | 4.1763 | 0.054458 @703 |
| A1 | 0 | 0 | 0 | none | 0.014676 @657 |
| A2 | 0 | 0 | 0 | none | 0.014676 @657 |
| A3 | 6 | 1 | 1 | 3.5039 | 0.038243 @734 |
| A4 | 7 | 1 | 4 | 2.9893 | 0.033097 @741 |
| A5 | 1 | 1 | 0 | 1145.9136 | 16.817277 @659 |
| A6 | 7 | 1 | 4 | 2.7334 | 0.036246 @720 |
| A8 | 2 | 1 | 0 | 103.6977 | 1.318882 @725 |

A0 still has later excursions, including745/757/780, and a censored783/785 event. A4 has a later740/741/742 excursion and a757/767 event censored at785. A3 has731/734, A5 has657/659, A6 has775/782 and A8 has718/725 excursions that remain unrecovered at785. These tails are observations, not failed arms or claimed recoveries. The fixed target window starts689, so it excludes A5's much larger early peak; the full-window audit above exposes it.

## Evaluation and compute accounting

Every arm has128 accepted updates and129 complete training-gradient observations; no rejected trials or extra candidate forwards. Main work is1681 updates /8,405,000 training-image update exposures. Including the nine unapplied main boundaries and16 qualification gradient calls gives1706 complete forward/backward evaluations /8,530,000 image evaluations. Qualification adds12 optimizer updates (8 initial save/resume,4 same-fork replay), separately charged.

Nine complete held-out evaluations cover90,000 images: one fork test (loss0.72189137) and eight endpoints. The held-out split does not select thresholds or methods. Results below are descriptive, without a generalization claim.

| Arm | Endpoint test loss (10k images) | Test evaluation ms | GPU peak allocated MiB |
| --- | ---: | ---: | ---: |
| A0 | 0.74461313 | 3.399 | 315.11 |
| A1 | 0.74925405 | 3.072 | 315.11 |
| A2 | 0.73826058 | 3.302 | 315.11 |
| A3 | 0.75065990 | 3.049 | 324.63 |
| A4 | 0.75220178 | 3.207 | 315.11 |
| A5 | 0.49466550 | 3.516 | 324.67 |
| A6 | 0.75084956 | 3.166 | 324.63 |
| A8 | 0.70660961 | 3.274 | 315.11 |

All held-out evaluations total0.02951seconds on the warmed device. The common prefix took13.14seconds including setup/saving; initial real-device save/resume qualification took4.95seconds. Exact phase and event times are preserved in raw records. Local retention is3,099,787,287bytes (2.887GiB), including immutable model/optimizer/RNG/history forks and endpoints, separate rolling recovery and failure-capable status records. No cloud resources were created or billed.

## Figure, sources and direct checks

[Aligned loss and diagnostics, all eight arms (PDF)](figures/local_pilot.pdf). Each129-point curve is unsmoothed. Panels show full training loss, raw-gradient norm/cosine, outgoing alpha, preceding actual increment norm and completed-step q; no alignment sign is labeled a stability bound. The PDF was rendered and visually reviewed after fixing title/legend placement.

Raw attempt: `runs/218-2026-11-03-mlp-adam-step-actions/artifacts/attempts/001-20261103T172300Z-local-pilot`. Source implementation is92954bd; the executed launcher and all input identities are frozen inside that attempt. Original config flags remain frozen preparation metadata; the explicit later approval is recorded in authorization_and_plan.json and supersedes the prior no-launch state.

[comparison.csv](comparison.csv), [aligned_diagnostics.csv](aligned_diagnostics.csv), [results.json](results.json), [raw inventory](raw_inventory.csv) and [direct verification](empirical_verification.json) retain exact values and hashes. Direct checks cover all eight authenticated fork starts, source-aligned outcomes,6192 plotted scalars, complete counts, full held-out coverage and unchanged parent bytes. No150-test suite rerun was used as an execution receipt.

No failed/stopped arms were omitted. A7 remains deferred. Stop here for scientific review.
