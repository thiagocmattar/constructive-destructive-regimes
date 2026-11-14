# Matched intervention experiment — decision plan

- **Status, 14 November 2026:** prepared for review; no experiment or cloud worker launched for this plan. Continue Analysis 065; allocate new run IDs only when configurations are frozen for execution.
- **Recommendation:** initialization 0, **12 conditions per model / 36 continuations / 11,904 post-fork updates**. Spark handles MLP; one RTX 4090 handles ResNet; three RTX 5090s handle Transformer. Estimated execution and verified retrieval: **6–9 hours, approximately $25–37**, with a proposed **$45 / 10-hour** execution limit.
- **Scientific question:** at the same warm state and update budget, how do changes to LR, Adam's second-moment dynamics, raw-gradient magnitude, and local representation displacement affect **loss excursions, learning progress, and functional oscillation**?
- **Contribution supported by this design:** a matched, mechanistically instrumented comparison of these four intervention families. Spike suppression alone is insufficient; retain training/held-out tradeoffs and unfavorable settings. One initialization supports these controlled cases, not an across-initialization superiority claim.

## 1. One common setup

| Dataset / model | Untreated fork → last state | Treated updates | Original LR | Full training objective per update | Fixed representation/calibration probe |
|---|---:|---:|---:|---|---|
| CIFAR-10 / MLP | **512 → 800** | **288** | .001 | 5,000 fixed images; half squared output error, summed over 10 coordinates and averaged over examples | All 5,000 images; 5 stage endpoints |
| CIFAR-10 / ResNet32 | **256 → 576** | **320** | .003 | 50,000 images; cross-entropy; original fixed groups of 1,000 for batch normalization | First intact 1,000-image normalization group; 6 stage endpoints |
| WikiText-2 / 2-layer Transformer | **544 → 928** | **384** | .003 | 2,088,600 supervised tokens; token-weighted NLL; physical groups of 64 sequences | Same fixed 64 sequences / 2,240 supervised positions; 6 stage endpoints |

- **Clock:** `Training step s` = completed optimizer updates. Fork `f` means first changed update **f → f+1**; stop after reaching `e`, with no extra update to `e+1`. Evaluate loss at all states `f,…,e`; primary post-fork means exclude the common state `f`.
- **Matched within each model:** initialization-0 tensors, complete fork parameters/moments/clock/RNG, objective and ordering, precision, physical groups, probes, measurement schedule, runtime and final state. Architectures retain their different scientifically established windows.
- **Native baseline:** Adam β₁=.9, β₂=.999, ε=1e−8; no weight decay, gradient clipping or moment reset. FP32 training/storage, FP64 diagnostic reductions; TF32 off; deterministic kernels; unfused/non-foreach Adam; Transformer dropout zero; original ResNet normalization semantics.
- **Build the prefix once per model:** replay the pinned initialization through `f` on the chosen runtime; retain the original input/config hashes and full fork. Capture local scores on the **eight preceding untreated transitions, f−8 → … → f**. Quantile budgets are frozen before any treated update. This corrects the previous cap's eight-step delay without borrowing future treatment observations.
- **Calibration:** one threshold per targeted module, pooling units/channels and those eight transitions; q25/q50/q75 use the same pool and recorded quantile interpolation. Scores use current inputs and native stored Adam proposals. Retain scores/budgets and the two preceding representation states for incoming-change diagnostics.
- **Runtime controls:** one qualified hardware/software stack per model. Verify identical fork gradient and short parameter/optimizer/RNG replay on every worker before sharing its baseline. If a worker differs, replace/qualify it; do not mix its treatment with another runtime's control.
- **Historical reuse:** only after fork/runtime/measurement equivalence. A replay matching a few scalar losses does not establish full tensor identity. Prefix reconstruction may create a qualified native trajectory that differs from the historical plot; label it as such. Keep fixed seeds/windows even if an anticipated excursion is absent.

## 2. Exact intervention matrix

| Intervention type | Settings per model | Change at the fork and thereafter | Current evidence | New treatment cells | Goal / interpretation |
|---|---|---|---|---:|---|
| **Unchanged Adam** | 1 | No changes | Historical baselines exist; qualify one common control per model | 3 matched controls budgeted | Reference for every metric and intervention |
| **Immediate LR reduction** | **0.75×, 0.5×, 0.1×** original LR | Change native LR before the first treated Adam step; hold constant | 0.5× in all three models | **6** | Test mild, medium and strong global damping; distinguish fewer spikes from slower learning |
| **β₂ reduction** | **.95, .9, .5** | Replace .999; retain m, v and global step; new β₂ enters recurrence **and native bias correction** | **All 9 settings already exist** | **0** | Test second-moment adaptation; this intervention also changes the native correction transient |
| **Global gradient clipping** | **L2 thresholds .5, 1.0** | Clip the full accumulated **raw gradient before Adam**; keep LR and betas native | MLP and Transformer: both; ResNet: 1.0 | **1: ResNet .5** | Test optimizer-input damping; report activation and actual step response, since Adam can offset gradient rescaling |
| **Distributed displacement cap** | **q25, q50, q75** | Scale local native parameter proposals; moments advance normally; fixed past-calibrated budgets | Historical q50 all models; wider MLP grid uses a different fork/horizon | **9 at the common forks** | Test selective direct affine-movement control; q50 remains the preselected main representation-inspired intervention |

- **Accounting:** 1 + 3 + 3 + 2 + 3 = **12 conditions/model**. There are **16 genuinely missing treatment cells** and 17 historical treatment settings. Budget all 36 continuations because common functional histories, checkpoints and runtime controls are not available for every historical setting. Qualified reuse reduces cost; it is not assumed in the estimate.
- **LR semantics:** 0.75×/0.5×/0.1× mean **25%/50%/90% reductions**. Actual LRs: MLP .00075/.0005/.0001; ResNet and Transformer .00225/.0015/.0003. No cosine/ramp schedule in this matrix.
- **Cap scope:** both MLP hidden affines; all 31 ResNet convolutions; two Transformer QKV and four feed-forward affines. Other parameters update natively. Scale each weight row and corresponding bias by `min(1, budget / local_RMS)`; zero score gives factor one, zero budget disables that module's cap. Verify the stored FP32 bound. This controls the **direct local affine term**, not total propagated network motion.
- **Fairness:** settings have equal starts/horizons, not equal effective strength or tuning budgets. Show every setting; do not select a family winner from its most favorable metric or tune thresholds on held-out results. No added intervention combinations, initializations or window search now.

## 3. Common measurement contract

| Readout | Timing / reduction | What it resolves |
|---|---|---|
| Training loss | Full native objective at every state `f,…,e`; unsmoothed records | Excursions and learning progress on an identical clock |
| Raw gradient and optimizer | Every update: raw norm, consecutive cosine ρ, existing projected-change diagnostic; optimizer-input norm, Adam proposal norm, actual applied norm/cosine, gradient dot actual increment | Keep raw-gradient interference distinct from clipping, adaptive directions and applied movement; no automatic Adam stability-bound claim |
| Functional motion | Every state on the same fixed probe: stage/output changes, cosine of consecutive changes, RMS; retain rolling history | Whether reduced loss spikes accompany reduced movement bursts, while reversals may persist |
| Intervention activity | LR and β₂; clipping factor/activity; cap thresholds, active unit fraction, scaling distribution, proposed/applied norm and maximum stored-bound residual | Whether a control acts, where, and how strongly; inactive clipping is not evidence of inferiority |
| Held-out performance | Full splits at **f, f+32, midpoint, e−32, e**; fork shared across arms | Training benefits versus held-out deterioration and endpoint phase sensitivity |
| Compute | Synchronized component timers, pass counts, peak allocated GPU memory; total wall time separately | Separate training, common diagnostics/evaluations, and controller overhead; no cross-GPU method-speed ranking |

- **Exact held-out states:** MLP **512/544/656/768/800**; ResNet **256/288/416/544/576**; Transformer **544/576/736/896/928**.
- **Full held-out coverage:** CIFAR-10 10,000 test images; WikiText-2 217,630 validation and 245,550 test tokens, token-weighted with the pinned masking/ordering. Test results are descriptive; no checkpoint/setting selection from them.
- **Probe context:** clone pre-in-place tensors; preserve mode, RNG and full ResNet normalization group. Reuse the existing 5/6/6 stage aggregation and output convention. MLP output changes cover the full objective; the other probes are subsets and must be labeled.
- **Retention:** full fork and calibration record; rolling recovery every 64 updates plus terminal state; raw gradient/proposal/applied-update audits at outgoing offsets 0, 32 and `e−f−1`; scalar and stage-metric streams throughout. No dense activation/gradient movie or new temporal SVD is needed for this comparison.

## 4. Endpoints fixed before execution

| Role | Metric | Definition / reporting rule |
|---|---|---|
| **Primary: spike burden** | Positive loss variation, PV | `sum(s=f+1…e) max(L_s−L_(s−1), 0)`; report absolute PV, PV/L_f and **100 × (1−PV_treatment/PV_baseline)**. Negative reduction means worse. |
| **Primary: learning progress** | Mean post-fork training loss | `mean(L_(f+1),…,L_e)` and percent change from unchanged Adam. The shared fork is excluded for every family. |
| Required endpoint context | Final and late loss | Absolute `L_e`, mean of the last 32 states, and their baseline-relative changes; report all, without choosing the favorable endpoint. |
| Required spike context | Worst excursion; event count/recovery | [Analysis036 loss-only reference/recovery rule](../../../../../analyses/036-2026-11-02-controlled-adam-diagnostics/02_extract_loss_spikes.py); max relative height; events exceeding 10%, 25%, 50%, 100%; duration and recovery censoring. Full-window maximum loss is also retained. |
| Required tradeoff | Held-out loss | Full split losses at the five fixed states; terminal percentage change prominently displayed. No generalization claim from training improvement. |
| Mechanism | Motion and reversing motion | Per stage: RMS distribution/max, fraction of negative consecutive-change cosines, total squared-motion energy, and squared-motion energy on reversing updates. These are descriptive motion statistics, not a causal propagation test. |
| Mechanism / cost | Activity and overhead | Clipped-update fraction / capped-unit fraction, applied-to-proposed norm ratios, controller time and memory, common measurement cost separately. |

- **Event rule:** on the unsmoothed loss history, start an excursion at the last unoccupied state before an increase; recover at the first subsequent loss no greater than that reference; take the earliest maximum before recovery. Multiple peaks before recovery form one excursion. Carry ongoing pre-fork events into the window with their original reference; flag left-truncation/right-censoring. Compare window-wide burdens and distributions, not assumed one-to-one event identities across divergent treatments.
- **Zero denominators:** report the absolute result and `N/A` for an undefined relative effect; do not silently replace it with zero. Numerical failure is a reported outcome with its step and partial coverage, not a discarded run or a fabricated full-window mean.
- **Inference:** all settings are visible; no seed confidence intervals or tests treating training steps as independent replicates. q50's local bound, loss response, and actual propagated motion are separate observations. For MLP, retain the exact squared-loss output-change decomposition when interpreting overshoot; do not transfer that identity to cross-entropy.

## 5. Paper-ready outputs

| Output | Proposed layout | Intended message / completeness |
|---|---|---|
| **Loss comparison** | One vector PDF, **4 family rows × 3 model columns**; gray unchanged Adam in every panel; all requested settings; same loss scale/window within each model; fork marked | Which controls suppress excursions, and at what learning cost; raw loss curves, no smoothing-dependent spike decisions |
| **Tradeoff comparison** | Three model panels; x = mean-loss change, y = PV reduction; one point per setting, family shape/color, values labeled; unchanged Adam at the origin | Show the full response curve and tradeoffs; no winner selected only by peak suppression |
| **Main outcome table** | Model blocks, all 12 conditions each; columns: setting, PV reduction, mean loss change, final loss, last-32 loss, terminal held-out change | Compare interventions consistently. Export a single **36-row CSV** and editable TeX; complete event/activity/cost metrics in the appendix table |
| **Functional mechanism** | Updated q50 baseline-versus-treatment stage heatmaps across the three models, at the common forks; loss + paired cosine/RMS; existing 5/6/6 stages | Large movement bursts can diminish while direction reversals persist. Use one baseline stage RMS maximum for both arms; cosine scale [−1,1] |
| **Supporting ablation** | Same-style functional diagnostics for every setting, activation/strength curves, held-out trajectories and overhead table | q25/q75 remain ablations of the same cap method; demonstrate activity, strength and architecture dependence without adding new methods |

- **Display contract:** `Training step s`; endpoint placement for incoming changes; full fixed windows rather than treatment-selected peaks; consistent family colors, scales and legible final-size labels. Put any gray-band definition in captions. One initialization, no error bands implying replication.
- **Delivery:** vector PDFs, TeX tables, CSV/JSON reductions, captions and local observations with source hashes, exact reduction/coverage/caveats and generating scripts. Keep the existing figures as historical evidence until replacements are verified; update the selected asset links only then.
- **Preparation estimate:** approximately **2–4 hours** to adapt existing continuation/capture code and validate the common reducer, then **1–2 hours** for final figure/table assembly and visual QA. These are engineering estimates, separate from the GPU ETC below; preparations can reuse existing code in Analysis065.

## 6. ETC, hardware and cost

- **Measured anchors:** MLP on Spark ~.014 s/update unchanged and .058 capped; ResNet on RTX4090 ~3.5 unchanged and 3.83 capped, versus ~15.15 unchanged/clipped on Spark; Transformer on RTX5090 ~7.81 unchanged and 7.96 capped. RTX4090 Transformer runs span ~9.58–16.03 s/update across hosts/runtime versions.
- **Weakest estimate:** Transformer on Spark has only a **three-update mechanics measurement** (~47 s/update unchanged, ~61 capped), not a sustained continuation. Its multi-day ETC is a broad extrapolation; remeasure sustained throughput before committing a long Spark-only run.
- **Cost basis, live 14 November:** RTX4090 **$0.89/h secure / $0.34/h community**; RTX5090 **$1.39/h secure / $0.69/h community**. Both listed available with **low stock**; fleet availability is not reserved. Spark was reachable, idle and had ~3.3 TB free. [Official GPU pricing](https://www.runpod.io/pricing); [storage pricing](https://docs.runpod.io/pods/pricing).
- **Included:** full untreated prefixes from initialization, every continuation, terminal diagnostic, 10–20% work allowance for evaluations/audits, 0.5–1 hour staging/qualification/retrieval per cloud worker, temporary storage. Prefix construction is on the critical path; whole branches, not fractions of them, are assigned to GPUs. **Excluded:** capacity queues, code/figure preparation, taxes and owned-Spark electricity. Community estimates require qualified throughput on the actual host.

| Option for initialization 0 | Hardware distribution | Execution + verified retrieval | Cloud cost including temporary storage | Proposed execution limit |
|---|---|---:|---:|---|
| **Recommended balance** | Spark: MLP; **1×4090: ResNet; 3×5090: Transformer** | **6–9 h** | **~$25–37 secure**; ~$12–18 community if qualified | **$45 total / 10 h** |
| Faster, more capacity needed | Spark: MLP; **2×4090: ResNet; 6×5090: Transformer** | **4–6 h** | **~$27–42 secure**; ~$13–20 community if qualified | **$50 total / 8 h** |
| 5090 unavailable | Spark: MLP; **2×4090: ResNet; 4×4090: Transformer** | **6–11 h** | **~$22–38 secure**; ~$9–15 community if qualified | **$45 total / 12 h** |
| No cloud spending | Spark sequentially: all models | **~4–6 days** | **$0 cloud** | Re-estimate after sustained Transformer calibration |

- **Why this split:** MLP takes minutes of device work; the whole staged MLP task is budgeted at ~17–34 minutes. ResNet takes ~5–7 hours on one 4090 with prefix/overhead. Transformer is ~14–21 aggregate 5090-hours including worker overhead, spread across three workers for ~6–9 hours elapsed. The 24/32 GB devices exceed recorded ~5.4/4.1 GB peak allocations; initial provisioning should also provide at least 32 GB host RAM.
- **Execution order:** implement and mechanically validate locally/Spark; start one leader per model; build/verify the fork and calibration; acquire additional workers near readiness; verify replay; dispatch all fixed arms. No event-selection gate or outcome-based retuning.
- **Storage/monitoring:** 30 GB container + 30 GB temporary volume per Pod; plan up to ~30 GB new retained evidence on Spark for the current matrix, with compact tables/manifests/PDFs local. Record step/tokens, loss, throughput, memory, refreshed ETC and estimated aggregate spend every 2–5 minutes. Include retries in the total limit; reserve retrieval/cleanup time. Checkpoint at the limit rather than silently expanding it.
- **Closeout:** immutable input manifest, component-timing report, update/bound/evaluation checks, no missing arms hidden from tables; verify file sizes and SHA-256 on retained storage before terminating task-owned Pods. Leave no cloud volume unless explicitly retained; billing is provisional until account reconciliation. Existing unrelated resources are outside this task.

## 7. Deferred work and its separate budget

| Future addition | Exact size | Spark only | Recommended fleet, secure | Faster fleet, secure |
|---|---|---:|---:|---:|
| **Initializations 1 and 2 for the entire current matrix** | **72 additional continuations**: 24/model, including 2 new controls/model; total study becomes 108 | **~8–12 extra days**, $0 cloud | **10–14 h; ~$47–67** | **6–9 h; ~$49–73** |
| **Cap quantile × LR, three initializations** | **45 capped + 9 same-LR uncapped controls per model = 54/model; 162 total** | **~21–30 days**, $0 cloud | **22–30 h; ~$100–145** | **11–16 h; ~$102–150** |

- **Extra initialization contract:** same absolute windows and all 12 conditions, with a native untreated prefix for each model/initialization. Initial tensor files for 0/1/2 are present; their presence does not establish qualified trained prefixes. Report absent excursions without choosing a new window. No additional seeds execute now.
- **Deferred cap-grid assumption:** use the existing exploratory **LR multipliers 1×/2×/4×** and **q25/q50/q75/q90/q99** solely to budget the requested 3×3×5 design. The grid is deferred; its final scientific settings can be revisited before execution. For an LR-isolation question, keep each seed's cap budgets calibrated once at its original untreated LR, and compare every cap with its **same nominal LR uncapped control**, plus the original-LR Adam reference.
- **Per-model gross cap-grid cost/compute:** MLP on Spark **~0.5–1 hour, $0 cloud**; ResNet on RTX4090 **~21–29 aggregate device-hours, $19–27** (Spark ~98–141 h); Transformer on RTX5090 **~57–84 aggregate device-hours, $80–119** (Spark ~388–565 h). Fleet parallelism determines the total elapsed time above; GPU-hours are summed, not wall hours.
- **Reuse credits, not assumed:** current initialization-0 matrix can fill 4 cells/model of that future cap grid (baseline + three caps at 1×), leaving **150**; completing all three initializations first can fill 12/model, leaving **126**. Exact fork/measurement/calibration equivalence is required. Historical MLP 448→1152 runs remain supporting evidence, not cells of the new 512→800 benchmark.
- **Other TODOs, outside these budgets:** independent intervention times, probe/calibration-size sensitivity, alternative β₂ correction/reset conventions, ramped LR schedules, and combinations of intervention families. None is required to complete the current single-initialization comparison.

## 8. Acceptance and provenance

- **Ready to write V-C when:** all 36 conditions have a verified outcome or explicit failure; fork/calibration/timing and full held-out coverage agree; loss/motion/activity/cost reductions reconcile with source records; figures show all settings and tradeoffs; claims are restricted to the three initialization-0 cases. This does not require the deferred grids.
- **Verification before launch:** cap scores/calibration use only past transitions; no `e+1` update; unchanged arm matches the native continuation; β₂ and clipping occur at their declared optimizer locations; hooks/evaluations preserve RNG and normalization; checkpoint-resume and cross-worker forks reproduce; outcome reductions handle zero denominators/censored events. Mechanics tests do not become paper evidence.
- **Evidence consulted:** [conventional outcomes](tables/conventional-outcomes.md), [distributed-cap observations](observations/65-008-cross-architecture-interventions.md), [cap geometry](observations/65-010-local-displacement-cap-geometry.md), and [existing quantile/LR evidence](observations/65-013-distributed-cap-quantile-lr.md). Existing results are not changed by this plan.
- **Reproducible estimates:** [calculator](../../../work/intervention_design_20261114/estimate.py), [computed estimates](../../../work/intervention_design_20261114/estimates.json), [live price receipt](../../../work/intervention_design_20261114/hardware_quote.json), [Spark snapshot](../../../work/intervention_design_20261114/spark_status.json), [source sizes/hashes](../../../work/intervention_design_20261114/source_manifest.json). The calculator checks source hashes, 36/72/162 condition counts and 11,904 current post-fork updates. These are audit records; this document is the self-contained decision view.
