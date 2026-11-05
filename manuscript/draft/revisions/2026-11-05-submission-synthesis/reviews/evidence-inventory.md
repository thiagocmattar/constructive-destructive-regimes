# Evidence inventory for submission closure

Prepared 5 November 2026 from existing repository evidence. This is an independent empirical synthesis, not a new experiment or a claim of established literature novelty. Repository paths below identify the owning records. The companion `../evidence_selection.json` pins selected source files by SHA-256, records exact source fields and values, and supplies the methods for a compact closing section.

## 1. The strongest supported empirical argument

The evidence supports a diagnostic paper about **what temporal gradient interference measures, how it is organized, and how its interpretation changes with sampling, optimizer transformations and observation horizon**. A useful diagnostic need not predict spikes or define a successful controller. Here the practical value is distinguishing mechanisms that otherwise look alike in a loss curve or an adjacent-gradient cosine, then evaluating their consequences against the objective.

The positive contribution is the combined measurement account: distinguish cancellation within consecutive pairs from coherence of the residual motion; distinguish sampled gradient disagreement from motion through the objective; distinguish gradient response from optimizer memory and actual steps; and check interventions against complete loss responses. Existing experiments demonstrate why each distinction matters. They do not establish that reversal is inherently harmful, that suppressing it improves learning, or that a new optimizer is needed.

The original constructive/destructive vocabulary can remain if it names the sign of a specified interaction. It should not silently rename good/bad optimization, convergence/divergence, or retention/forgetting. The current results actively rule out those automatic equivalences.

## 2. Measurement definitions that must remain consistent

| Object | Meaning and exact limit |
| --- | --- |
| Raw gradient `g_t` | Gradient of the declared sampled or complete objective at the current state, before clipping or optimizer transformation. Whole trainable parameter scope; count tied parameters once. |
| Adjacent cosine `rho_t` | Angle between successive raw gradients along the actual parameter path. Its negative sign identifies opposing vectors, not an uphill optimizer step. |
| `A=sum ||g_t||`, `B=||sum g_t||`, `R=B/A` | Gradient-path length, resultant norm and normalized resultant. Under constant-rate plain SGD these correspond to nominal parameter-path geometry. Under Adam/AdamW raw-gradient accumulation is not actual parameter displacement. |
| `E=sum ||g_t||^2`, `B^2-E` | Energy and exact all-pair cross interaction. A small `R` does not imply a negative all-pair cross term. `sqrt(E)/A` is an observed-length zero-cross-term reference, not an empirically fitted random walk. |
| `C=H/A`, `D=B/H`, `R=CD` | `H` sums norms of consecutive two-gradient sums. `C` locates within-pair cancellation and `D` locates turning among residual pair displacements. Both are geometry, not amounts of retained learning. |
| Projected gradient change | `z=1-rho*||g_new||/||g_old||`. Under Adam this is an observed response, not automatically curvature or a stability threshold. Threshold one is algebraically equivalent to negative cosine. |
| Moments, adaptive directions and increments | Moments use the clipped optimizer input when clipping is active. Adaptive directions exclude LR/decay unless explicitly included. Actual increments include applied LR, decay and arithmetic rounding. |
| Loss excursion | Fixed pre-rise reference; peak before first return to that level; recovery and right censoring explicit. Event counts, maximum amplitude, mean loss, late loss and endpoint loss answer different questions. |

Sources: `research/DEFINITIONS.md`, `research/METRICS.md`, Analyses028/036/056/057 README and observation contracts. None of these batch-gradient records identifies per-sample energy and intra-batch interaction separately.

## 3. GD and SGD: strongest main-text evidence

### 3.1 Consecutive reversal and accumulated progress coexist

`analyses/024-2026-10-17-eos-gd-lr-calibration/` contains nine complete 3,000-update full-GD trajectories: three rates per CIFAR-10/MLP, CIFAR-10/ResNet-32 and WikiText-2/Transformer, with one matched initialization per case. `analyses/056-2026-11-05-gd-paired-displacement/observations/001-paired-displacement.md` reanalyzes the entire cohort, both pair offsets, all 26,991 defined adjacent pairs and complete endpoint losses.

All nine trajectories have **positive all-pair interaction**, although seven have negative cumulative adjacent interaction. At the highest tested rates:

| Case/rate | Within-pair `C` | Between-pair `D` | `R` | Zero-cross reference | Final training loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP / 0.04 | 0.11386 | 0.22544 | 0.02567 | 0.01993 | 0.153869 |
| ResNet / 0.1 | 0.15579 | 0.23991 | 0.03737 | 0.02019 | 0.489838 |
| Transformer / 0.2 | 0.08592 | 0.63002 | 0.05413 | 0.01838 | 6.069747 |

Each largest-rate trajectory has its case's lowest final training loss. The factorization localizes cancellation without claiming that its residuals decrease loss at each pair: in the last 1,000 updates, only 246/500, 248/500 and 386/500 primary pairs decrease loss, while all three total changes are negative. Pair-offset changes are below 0.004 for both factors in these highest-rate runs.

The lag evidence is complementary, not a replacement for displacement: Analysis035/056's nine backward profiles have negative odd-lag and positive even-lag means over lags 1–32. Only the Transformer has those signs at every individual lag across all three endpoints. The longer-lag envelopes can contract and rebound. The profiles are separately recomputed continuations with recorded FP32 drift, not exact extensions of the original scalar paths.

**Recommended claim:** strong local cancellation can be organized into nonzero residual motion, positive all-pair interaction and net loss reduction. **Do not claim:** reversal causes better optimization, the pair decomposition explains all loss change, asymptotic periodicity, or a universally stable regime.

### 3.2 Sampling changes both the observation and the path

`analyses/037-2026-11-02-sgd-batch-extension/REPORT.md` integrates 546 retained 200-update continuations across three initialization blocks, eight repetitions per stochastic condition, two starting states and three model/dataset cases. At trained states, the smallest batches weaken mean reversal from full-GD values -0.991/-0.967/-0.993 to -0.384/-0.612/-0.258, but all corresponding mean retention values remain below 0.05.

The clearest human-readable counterexample is within one Transformer configuration, batch eight: initial/trained-state mean cosine is almost identical (-0.259496/-0.258445), while `R_200` differs (0.109896/0.041632) and mean complete-objective loss changes have opposite signs (-3.505137/+0.064266). That loss-change sign holds in every initialization-block mean.

`analyses/027-2026-10-30-minibatch-noise/REPORT.md` additionally has full-gradient references on the same MLP SGD states. At initialization with batch32, sampled mean cosine is -0.141, complete-gradient cosine on those same SGD states is +0.353, and separate full-GD cosine is +0.994. Frozen controls and moving paths must not be conflated.

The retained HVP check is useful validation if space permits: at batch32, median absolute error between the raw projected-change proxy and state-matched directional HVP is 0.7467; including the explicitly measured noise term reduces it to 0.0030 over 288 probe pairs. That correction requires full gradients and is not available from two minibatch gradients alone.

**Limits:** equal update count implies unequal data exposure; three cases are not three datasets; selected rates and starting states; runtime/seed confounding in some small ResNet batches; block ranges are not confidence intervals. Retain these compactly in methods rather than repeating them in every conclusion.

## 4. Adam: event anatomy without a forecasting claim

`analyses/036-2026-11-02-controlled-adam-diagnostics/` supplies nine fixed-rate, 1,024-update full-objective Adam paths from matched untrained weights, one seed per model. There is no schedule, clipping or decay, with Adam `(0.9,0.999)`. This deliberately isolates deterministic trajectory behavior from minibatch sampling.

There are 812 candidate excursions: 806 recovered and six censored. Strict recovered thresholds of 5/10/20% select 206/129/74 events. Of the 127 above-10% events with defined pre-rise cosine, 117 (92.1%) have negative cosine; every one of the 129 has a larger gradient norm at its loss peak than at its pre-rise state. Only 21/127 (16.5%) already exceed projected change two before the first rise. Recovered-event counts are dependent observations, not 129 initialization replications.

`analyses/057-2026-11-05-adam-response-memory/observations/001-response-memory.md` provides the decisive vector distinction: all **794 outgoing updates in the recovered rise intervals have negative `g dot actual_increment`**. Recovery has 1,374/1,376 such steps. Raw reversal is therefore not evidence that Adam points uphill relative to the current gradient. A locally descending step can have an adverse finite-step loss response. The rise intervals contain all outgoing steps from baseline through peak; do not imply every one of their individual loss differences is positive.

Analysis055's complete-history controls delimit specificity. Lowest-rate ResNet/Transformer have 726/953 and 764/953 eligible negative-cosine states, and 186/953 and 215/953 states with projected response above two, with zero qualifying 10% increases in the next eight updates. This fixed-horizon outcome differs from the recovered-event detector. It is enough to reject an unqualified alarm; it is not an event-duration-matched causal comparison.

The severity fits are **secondary evidence about limitations**, not a lead contribution. Adding cosine to log norm increases in-sample fit but worsens held-rate error in all seven folds. The sign of its coefficient changes across trajectories. Peak norm is retrospectively observed, and 84/127 cosine-defined events belong to two MLP trajectories. Put the full fit comparison in an appendix or compact discussion; avoid spending main-text pages explaining an unvalidated severity law that the paper does not need.

The response/memory identities are analytically verified but not empirically calibrated against neural HVPs. They can remain a qualified bridge or appendix result. They cannot be sold as measured Adam spike causation. The pending experiment plan answers a stronger question; it is not a prerequisite for submitting the narrower diagnostic paper.

## 5. Pythia: existing scale and intervention evidence

### 5.1 Replicated separation of local cancellation and quality

Analysis003's `observations/003-replicated-conclusions.md` audits nine scratch Pythia-14M/31M/70M MiniPile runs, three paired initialization/data-order seeds, 712 updates each. In every seed block, larger models have lower complete validation loss but more frequent negative adjacent raw-gradient pairs and greater unconditional negative intensity. Conditional magnitude given negativity decreases. For seed1234, negative fractions are 49.23/56.96/67.65%, while validation losses are 5.182392/4.611768/4.063201.

All nine full-prefix raw all-pair interactions are positive. Lag-one truncation misses 97.90–98.57% of squared cumulative raw-gradient norm. This is a direct empirical rejection of treating the omitted temporal terms as negligible in that setting. Architecture/initialization scales co-vary with size, order/init seeds co-vary across blocks, and hardware differs between seed blocks. This is recipe-specific replication, not a dimensional scaling law or actual AdamW displacement identity.

### 5.2 Matched schedule changes do not make alignment an objective

Analysis004/028 branches exact update356 states at sizes14M/70M, seed1235, through update712. Original decay, held midpoint LR and accelerated decay share weights, moments, future data and full validation coverage. Holding LR increases cancellation and reaches the lowest validation loss; accelerated decay restores alignment earlier and reaches the highest loss. The exact values are in the proposed table below.

Analysis030 adds three-seed, 18-branch evidence at matched decoupled shrinkage. Twice LR wins final full-validation loss in all six Pythia14M/70M starting states despite early negative raw cosine. Fixed-rate and loss-only selectors already have zero regret. Analysis031's apparent 16-update retention-prediction gains (37.0/37.2%) do not persist across observation windows or the subsequent-benefit target. These are useful negative outcomes; a scheduling narrative would oversell them.

### 5.3 A component intervention is already available

Analysis025 supplies 27 Pythia runs and three seed blocks. Eighteen 50-update branches remove coordinate scaling while preserving each native parent's adaptive-step norm schedule. Every branch worsens fixed-probe loss, at both checkpoints175/225. At the225 fork, geometric mean penalties are 26.84%, 57.90%, 106.49% for14M/31M/70M.

The same-state substitution audit distinguishes linear prediction from finite response: at70M/update200, the unpreconditioned direction predicts a 5.010% linear decrease versus native1.199%, yet its nonlinear remainder is141.298%, yielding a136.288% increase; native achieves a0.083% decrease. This is exact measured residual accounting, not an HVP curvature measurement.

Keep this compact: it demonstrates why vector transformation and finite-step outcome matter, without turning this paper into a size-mediated preconditioning study. The norm-matched intervention is not a retuned momentum baseline, its probe is same-corpus held-out material, and neither the size gap nor Adam spike formation is causally explained in full.

## 6. Existing full-batch interventions and negative outcomes

| Analysis | What it adds | Editorial use |
| --- | --- | --- |
| 039 | Verified driver, observer and replay foundation; no neural experiment | Provenance only |
| 040 | Selected MLP657 fork: half LR and proposal clipping pass the training screen; quarter LR suppresses excess but loses endpoint progress; raw numerator and transported secants fail | Optional appendix; no proposed new optimizer |
| 041 | One-step threshold rule reduces LR18/367 times but mean loss is6.52 times always-half; new/later excursions replace the selected one | Negative control against naive feedback claims |
| 042 | Hold+cosine modestly improves training loss over always-half (2.26% mean/4.30% final), but365/367 updates remain below base and held-out loss does not improve | Development result, not controller validation |
| 043 | Same warm1024 state,36 branches: model-dependent LR effects; inactive double-median clipping; decay shrinks parameter norms without task-loss improvement | Strong compact closing evidence |
| 044 | Earlier wider windows: immediate/cosine half mean changes MLP -25.04/-26.30%, ResNet +27.20/+0.62%, Transformer -9.66/-10.68%; residual spikes remain | Timing dependence; optional compact paragraph |
| 045–047 | MLP LR restoration renews spikes; gradual restoration can delay but enlarge them;0.75 floor leaves large excursions while0.5 is stronger | Selected development path, not independent replications |
| 048–049 | Replots existing MLP paths; no new training | Do not count as new evidence |
| 050 | Second selected decay reduces first peak48.77% and post650 mean27.66% vs held0.75 | Development timing chosen after inspection |
| 051 | MLP raw clip1 reduces first peak but raises full-window mean48.01% and maximum3.82x; endpoint alone improves19.26% | Strong demonstration that clipping activation and response horizon matter |
| 052 | MLP cap0.5 lowers mean31.03% vs cap1, but remains2.08% above unchanged | Threshold dependence; no optimum |
| 053 | Cap1 under cosine half never activates; all scientific values/endpoint states equal cosine alone | Zero incremental treatment, not evidence against active clipping |
| 054 | ResNet cap1 reduces mean28.3%/maximum52.2%, but late/endpoint loss rises; event count4→26 | Strong complementary tradeoff |

Sources are the corresponding `analyses/NNN-.../REPORT.md`, `FINDINGS.md` or README, with machine-readable comparisons and owning-run verification. Single selected seeds/windows and reused parents limit independence. The dozens of branches are not dozens of independent repetitions of a universal intervention effect.

## 7. Proposed compact closing table and methods

Use six **within-study comparisons**, with a labeled outcome for every row. Do not pool their percentages or present them as an optimizer leaderboard.

| Comparison | Diagnostic or intervention | Objective outcome |
| --- | --- | --- |
| Pythia14M, original/held/faster decay | Negative intensity0.0681/0.3620/0.0222 | Final full-validation loss5.1820/5.0674/5.2176 |
| Pythia70M, original/held/faster decay | Negative intensity0.1591/0.3201/0.0981 | Final full-validation loss4.0559/3.9846/4.1042 |
| Pythia14/31/70M, remove coordinate scaling | Native-parent adaptive-step norm matched; three seeds |50-update fixed-probe loss penalty26.84/57.90/106.49% at fork225 |
| Transformer, half LR from1024 | Mean raw cosine -0.299→+0.589 |64-update mean/final training loss +3.234/+4.855% |
| MLP, raw clip1 from512 |125/288 updates clipped |Mean training loss +48.01%; maximum3.82x; endpoint -19.26% |
| ResNet, raw clip1 from256 |318/320 updates clipped |Mean/maximum training loss -28.32/-52.15%; last32 mean +78.03% |

The JSON contains full-precision values, exact selectors and method capsules. Minimum textual methods:

- Pythia schedules: scratch MiniPile models; seed1235; shared state356;356 remaining updates;1024 length2048 sequences/update with microbatch4 and final tail; AdamW(0.9,0.95), raw clip1, decay0.1; full693,329-target validation. LR also scales decay.
- Coordinate-scaling comparison: three seeds, two50-update forks per size; same future batches and moment update; branch momentum normalized to each native parent's adaptive-step length, preserving ordinary decay on branch weights. The fixed128-sequence probe has262,016 targets and is disjoint from the union of training prefixes. Report geometric means of paired loss ratios across seeds, not arithmetic means of individual percentages.
- Full-objective interventions: inherited Adam(0.9,0.999) state, fixed native rates0.001MLP/0.003ResNet/Transformer; one seed; same warm start, objective and hardware for each treatment/control. Complete objectives are5000-image half-squared error,50000-image cross entropy and2,088,600-target NLL. Clipping acts globally before both moments; geometry remains raw. Use fresh same-Spark controls where historical cross-GPU replay did not qualify. Intervals and reduction weights appear in the table/caption.

## 8. Minimal evidence architecture and stopping rule

1. Establish the precise geometric quantity and exact displacement accounting.
2. Show GD reversal, the `C/D` factorization, positive all-pair interaction and observed loss change together.
3. Show SGD's measurement/path distinction with the three-case batch comparison and one concrete same-cosine/opposite-loss example.
4. Show Adam excursion anatomy and the all-rise-step descent check, with ordinary low-rate behavior preventing a universal threshold interpretation.
5. Close with the compact matched interventions/Pythia evidence, demonstrating how the distinctions discipline interpretation and future intervention design.

Move full severity regressions, exploratory prediction scans, repeated MLP schedule variants and detailed optimizer-memory discrepancy bounds to supporting material as space requires. Preserve negative outcomes and declare selection, but do not let them replace the paper's central scientific question. Activation sparsity and orthogonalized updates can be one clearly speculative future-work sentence; neither is a contribution or required baseline here.

No new training is required to make these bounded claims. Stronger claims about universal spike prediction, a calibrated Adam curvature approximation, cross-seed Adam anatomy, a mechanism causing all observed spikes, or a novel controller would require the new plan. Choosing the diagnostic scope closes those obligations explicitly rather than treating them as an endless backlog.

## 9. Independent verification performed for this synthesis

The selection script reread the retained JSON/CSV reductions, recomputed Analysis025 paired log ratios and seed geometric means, and independently confirmed:9/9 positive GD all-pair interactions;7/9 negative adjacent interactions;794/794 locally descending Adam rise updates;7/7 worsening held-rate errors after adding cosine to log norm;18/18 positive preconditioning-ablation penalties. These checks agree with the owning observation records.

The JSON pins selected results and methods files by SHA-256. This is a check of retained analysis outputs, not a claim to have rerun all raw-file inventories or neural experiments during this revision. The earlier raw transfer, source/config identity and numerical/vector checks remain in the owning analyses/runs. No main manuscript, registry, scientific input, raw artifact or commit was changed by this evidence-review task.
