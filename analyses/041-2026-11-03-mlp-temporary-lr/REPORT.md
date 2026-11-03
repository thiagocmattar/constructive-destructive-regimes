# Temporary reduction using $\eta\widehat\kappa_t$: negative selected-state result

All three conditions completed 367 canonical updates from the authenticated state 657 to 1024. The frozen rule halves LR only when the available $\eta\widehat\kappa_t\geq1.8$; otherwise it immediately selects base LR. No threshold search, smoothing, persistence, changed moments or additional variants were used. This is a single-seed exploratory comparison.

**Always-half LR is strongest here. The temporary rule uses fewer reduced-LR updates but does not preserve progress.** Its final loss is 8.80 times always-half and 6.21 times unchanged Adam.

## Fixed-reference outcome

B=untreated loss at state 689=0.013039679266512394. Peak excess=max(0,L/B-1) and excess area=sum of positive excess over the fixed 336-state window689..1024. Mean covers exactly the 367 post-update losses at 658..1024. The largest whole-continuation loss includes 657..1024.

| Condition | Peak excess | Excess area | Mean loss | Final loss | Whole-window maximum @state | Half-LR updates | LR switches within branch | Runner seconds |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| Unchanged Adam | 3.176299 | 80.061606 | 0.01472254 | 0.00952230 | 0.054458 @703 | 0/367 (0.00%) | 0 | 9.546 |
| Always-half LR | 0.000000 | 0.000000 | 0.00979396 | 0.00672606 | 0.014676 @657 | 367/367 (100.00%) | 0 | 8.041 |
| Temporary reduction | 98.463380 | 1277.099957 | 0.06381674 | 0.05916404 | 1.296971 @950 | 18/367 (4.90%) | 36 | 8.283 |

Switches count changes between consecutive outgoing choices. Always-half has one additional entry change from the inherited base LR; the other conditions have none. Runner cost includes diagnostics/checkpoint/status I/O. Unchanged cost includes the failed segment and recovery/replayed work, so it is not an isolated optimizer-speed comparison.

## Findings against the controls

- Controls are qualified: unchanged and always-half match all 129 overlapping Run218 boundary loss/norm observations exactly. Unchanged also matches all 368 Run215 observations through 1024 exactly. All conditions inherit identical weights, optimizer moments/age, RNG, previous raw gradient and actual increment.
- The $\eta\widehat\kappa_t$ rule reduces LR on 18/367 updates (4.90%), with 18 isolated one-update reductions and 36 LR switches. LR returns to base after every reduction and remains at base at the endpoint. No undefined cases occur. This is intermittent feedback, not an almost-always-half regime.
- The temporary rule has 6.52 times the mean loss and 8.80 times the final loss of always-half. Its fixed-reference area is 1277.10 versus 80.06 unchanged and 0 always-half. Fewer reduced-LR updates therefore do not translate into competitive progress in this test.
- The conditional trajectory shifts the excursions: 660/668/743,778/788/874, then 938/950 with no recovery by 1024. The last peak reaches loss 1.29697 and an own-pre-rise ratio 154.89. Keeping the full horizon exposes both replacement events and the censored tail.
- The zoom was fixed at 673..745 before results. There are no temporary half-LR updates inside it: the first reductions occur at 659,661,664,666; the next at 774. The zoom shows the tail of an earlier replacement excursion. It is not recentered around that trajectory's peak or presented as a successful intervention during the original rise.
- Always-half has no >10% excursions and no positive excess above the shared reference in 689..1024. It attains 0.95B after 51 updates, versus 76 unchanged and 102 temporary. Simple persistent reduction remains the stronger training control in this selected case.
- A Windows status-file replacement error stops the first unchanged attempt after 243 updates. Recovery resumes the last valid state 881 checkpoint and reproduces 19 overlapping observations exactly. The failed status/traceback is retained; only bounded I/O retry handling is added. No scientific input or rule changes.
- The current $\eta\widehat\kappa_t$ heuristic is negative evidence at this selected seed/state. Under Adam, $\eta\widehat\kappa_t$ remains the observed normalized projected raw-gradient change, not a stability boundary or a forecast rescaled by LR. One case cannot establish generalization or condemn every possible rule; an equally frequent alternative timing control and independent replication would be needed to claim special timing value.

## Required views and decision clock

[Fixed zoom 673..745](figures/fixed_zoom.pdf) and [full continuation 657..1024](figures/full_continuation.pdf) each show three aligned panels: unsmoothed loss, $\eta\widehat\kappa_t$ with threshold 1.8, and actual LR/base LR as a post-step function. Loss is linear in the fixed zoom and logarithmic in the full view. Both PDFs were rendered and visually inspected.

At parameter state s, manuscript index t=s+1. The available raw pair directly gives $\eta\widehat\kappa_t$ before the outgoing update. Its LR choice governs s to s+1; the resulting loss is observed at the next state. Shading covers exactly the temporary half-LR intervals [s,s+1) in every panel. There are no such intervals within the fixed zoom. The final LR segment closes at 1024 for drawing only; no outgoing decision is applied at 1024.

Reduced states: `659,661,664,666,774,777,780,783,786,791,931,934,937,940,943,946,948,950`. Each interval has length one. The direct raw-pair expression is not multiplied by an LR ratio. All 1,101 canonical decisions satisfy the frozen rule and availability clock.

## Excursions and held-out description

The existing fixed-pre-rise detector is applied after joining each continuation to the shared prefix. Counts below retain events with >10% own-reference increase and a peak in the continuation, including unrecovered events. Later starts are after the original recovery 729.

| Condition | >10% events | Censored | Later starts | Largest own-reference ratio | Endpoint test loss (10k images) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unchanged Adam | 17 | 1 | 13 | 4.17630 | 0.77706603 |
| Always-half LR | 0 | 0 | 0 | none | 0.78286463 |
| Temporary reduction | 4 | 1 | 2 | 154.89229 | 0.66036537 |

All three endpoint evaluations cover the complete 10,000-image test split. These values are descriptive and do not select the rule or support a generalization claim. The temporary rule's lower held-out loss does not overturn its training-progress failure under the declared objective. Exact event endpoints, including all small excursions, are in [excursions.csv](excursions.csv).

## Execution, recovery and cost

Canonical main work is 1,101 updates /5,505,000 nominal image-update exposures. Qualification adds 16 updates /21 full-gradient calls. The interrupted control replays 19 updates. Actual totals are 1,136 updates and 1,144 full forward/backward objective evaluations /5,720,000 image evaluations; held-out work is another 30,000 image forwards. There is no retrained prefix or dense all-step vector archive. Start/end states, rolling recovery, raw scalar histories and the failed traceback remain local.

Active worker time is **37.26seconds** (30.51 initial process +6.75 recovery). First-worker start to completed recovery spans **218.34seconds**, including the manual recovery gap, within the approved600-second bound. Initial unchanged attempt fails atstate 900 after 243 updates; last periodic checkpoint is 881. The canonical control combines first-attempt states below 881 with recovery states881..1024. Original partial manifests are preserved rather than relabeled successful.

Endpoint timing for always-half and temporary is 3.183ms and 3.088ms. The control recovery records 102.510ms for its evaluation-plus-subsequent-audit interval; exclusive control evaluation time was not separately captured. This interval is charged as observed overhead, not compared as inference speed. Maximum measured allocated GPU memory is 310.59MiB. Zero cloud resources or spending.

Raw artifacts occupy 1,539,827,884bytes locally. [Raw inventory](raw_inventory.csv) hashes all files. The parent remains `359097d026b45f8f0b8a33665aca4ad0382e52835745ceaeccf027ed29539ba4`.

## Sources and checks

[comparison.csv](comparison.csv), [aligned_values.csv](aligned_values.csv), [results.json](results.json) and [verification.json](verification.json). Reproduce the reduction with `python analyses/041-2026-11-03-mlp-temporary-lr/01_report.py` (no training).

Four focused policy checks passed before launch; the full 150-test suite was not rerun. Real-device qualification compares existing control traces and exact save/resume state. Direct checks verify source clocks, all 1,104 canonical boundary values and1,101 choices, shared start states, scalar outcome sums and raw-file hashes. The retry is infrastructure-only and is recorded separately.

No additional condition or threshold was launched. Run217/218 scientific inputs and the manuscript were not edited. Stop for scientific review.
