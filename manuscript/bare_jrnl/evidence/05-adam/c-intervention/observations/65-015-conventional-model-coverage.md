> Evidence snapshot; original scientific content retained. Relative links were rebased.
> Source: [015-conventional-model-coverage.md](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/015-conventional-model-coverage.md).

# Conventional intervention coverage: H050 / Runs 289-291

- **Status:** completed and verified. All 10 continuations, 3,584 updates, 3,594 full-gradient boundaries and 20 first/last update audits pass; all seven task pods are deleted after full retention verification.
- **User request:** complete Transformer global-gradient clipping and ResNet/Transformer beta2 reductions in the selected V-C figures. Continue Analysis 065.
- **Question:** how do the same late beta2 changes and absolute clipping thresholds affect the three existing Adam cases?
- **Scope:** one initialization per model and the original retrospective fixed windows; no new representation-inspired method, seed sweep, timing sweep or manuscript edit.

## Fixed comparison

| Run | Objective | Fork / terminal state | Treatment arms | Control |
|---|---|---|---|---|
| 289 | ResNet-32, all 50,000 CIFAR-10 images; fixed intact 1,000-image BN groups; cross entropy | 256 / 576 | beta2 .95, .9, .5 | Fresh unchanged native Adam |
| 290 | Two-layer WikiText-2 Transformer; all 2,088,600 training tokens; physical batch 64; dropout 0 | 544 / 928 | beta2 .95, .9, .5 | Fresh unchanged native Adam |
| 291 | Same Transformer objective and fork | 544 / 928 | Global raw-gradient L2 caps .5 and 1 | Run 290 control, after exact cross-worker replay |

- **Matched:** original Run 223 model/optimizer/RNG/previous-gradient tensors; initialization 0; LR .003; beta1 .9; epsilon 1e-8; zero weight decay; deterministic execution; FP32 parameters/gradients/moments, FP64 scalar reductions; native unfused/non-foreach Adam.
- **Reused references:** MLP uses the original 5,000-image half-squared-error objective, LR .001 and states 512-800 (Runs 223/228/229/232). Existing ResNet clipping at 1 uses Run 231's own Spark control at states 256-576. These sources are hash-checked against Analyses 058/059; the new ResNet beta2 control is not substituted for the historical clipping control.
- **Beta2 intervention:** preserve both moments and the global optimizer step; set beta2 before the outgoing update. The new value controls both the second-moment recurrence and native bias correction `1 - beta2**step`. No reset, rescaling, or history-product correction. Thus this is the complete native schedule, not an isolated time-constant manipulation.
- **Clipping intervention:** compute the full-objective raw gradient, then multiply each gradient tensor by `min(1, cap / raw_global_norm)` before Adam updates its moments. No epsilon in the clipping factor. Raw gradient histories, norms, rho and secants remain unclipped.
- **Clock:** boundary state s is measured before outgoing update t=s+1. Rows run fork through terminal state inclusive; outgoing updates stop at terminal state minus 1. There are 320 ResNet or 384 Transformer updates per arm.
- **Runtime:** ResNet uses PyTorch 2.11/CUDA 13 on RTX 4090; Transformer uses PyTorch 2.11/CUDA 12.8 on RTX 4090. The latter runtime made adequate parallel capacity available.
- **Historical portability:** every initial tensor/state matches the original checkpoint. ResNet's recomputed fork gradient is exact. Transformer's relative gradient L2 discrepancy from the historical runtime is 1.367764647187215e-6; all six new workers agree exactly with each other. Historical Transformer responses are not substituted for the new control.

## Verification and retention

- Seven preflights independently check retained moments/clock, clipping placement, FP64 Adam recurrence and actual increments.
- Every worker checks native instrumentation and exact two updates versus one plus resumed one.
- All six Transformer workers have identical hardware/runtime records, initial fingerprints, eight-update scalar histories and full model/optimizer/gradient-history/RNG fingerprints: [control gate](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/conventional_control_gate.json).
- Full first/last update audits permit independent reconstruction of moment recurrences, bias correction and parameter increments. Boundary/update counts and clipping bounds are checked separately.
- Full terminal held-out coverage: ResNet 10,000 test images in 10 intact BN groups; Transformer 217,630 validation tokens/6,220 sequences and 245,550 test tokens/7,020 sequences. No truncation or equal weighting of unequal token batches.
- Complete raw forks, recovery/final states, audits and scalar histories are retained on owned Spark; compact outputs are verified locally. Every downloaded artifact has size/SHA-256 checks before its task pod is terminated.
- [Execution plan](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/conventional_execution_plan.json): maximum USD 20 including unsuccessful infrastructure attempts; scientific inputs remain frozen. The five slower hosts received 20 minutes of additional lease/retrieval margin while the worker's 6500-second bound stayed fixed.
- **Closeout:** [verified retention and cost](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/conventional_closeout.json). All 155 files rehashed on Spark: 12.371 GB logical inventory / 10.650 GB allocated disk, with final/recovery hard links preserved. Provisional compute USD11.36; conservative total including the reserved storage allowance USD15.53, below USD20. No task-owned Runpod storage remains; temporary retrieval keys are removed.
- **Transfer optimization:** immutable forks/first audits were transferred and verified during the last updates. Final retrieval skipped only files whose complete size/SHA-256 already matched, then reverified the entire inventory. Compact scalar data arrived first to overlap figure generation with raw retention. No scientific input or outcome was changed.

## Outcomes and interpretation

- **Full paired table:** [14 treatment outcomes](../tables/conventional-outcomes.md); [absolute per-arm values](../data/conventional_summary.csv); [curves](../data/conventional_curves.csv); [source hashes and checks](../data/conventional_results.json).
- **Transformer beta2 .95/.9/.5:** mean training loss changes -13.15%/-18.47%/-11.82%; maximum changes -54.46%/-61.57%/-64.79%; test changes +24.64%/+33.12%/+27.77%. Training improvement does not imply a held-out benefit.
- **Transformer clipping 1/.5:** mean changes -17.74%/-19.02%; maximum -71.13% for both; test +14.84%/+11.30%. Clipping acts on 243/384 and 349/384 updates, respectively (63.28%/90.89%).
- **ResNet beta2 .95/.9/.5:** mean changes +28.46%/-3.53%/+6.28%; last-32 changes +3636.05%/+981.27%/+506.38%; test -32.14%/+2.46%/+11.85%. The large relative tails reflect the control's small late loss; absolute values remain in the CSV.
- **Reused MLP:** all tested beta2 and clipping settings increase mean training loss, while reducing terminal test loss. **Reused ResNet clipping:** mean -28.32%, tail +78.03%, test +8.32%. Preserve these outcomes rather than treating conventional controls as uniformly ineffective.
- **New-control context:** ResNet has four >10% excursions, two at least 2x; Transformer has five and three, respectively. Each retains a 2x peak within 32 states of the historical target. The existing Spark ResNet clipping control has a shifted peak; its own paired baseline is retained.
- **Bounded contribution:** the same late interventions have architecture- and endpoint-dependent effects in these fixed cases. The results distinguish gradient/optimizer control from the separately specified functional cap; they do not establish a universal stabilization rule, a generalization benefit or a ranking against q50.

## Reduction, captions and display

- **Primary:** mean training loss over fork+1 through terminal state, paired with unchanged Adam on the same runtime.
- **Also report:** last 32 mean, final and maximum loss; clipping activity; full terminal held-out losses. Negative and inactive outcomes stay in the table.
- **Descriptive control check:** use Analysis036's loss-only excursion definition on the complete historical prefix plus each measured control continuation, retaining unrecovered events. Report post-fork peaks above 10%, at least 2x rises, and whether a 2x peak remains within 32 states of the original target (MLP703/ResNet422/Transformer794). This portability description was added during analysis; it is not an H050 acceptance criterion, and no outcome is excluded or retuned.
- **Beta figure:** three architecture columns; loss, raw-gradient secant, rho, raw norm and beta2 schedule rows. Four curves per column: unchanged and the three reductions. Existing MLP data remain frozen.
- **Clipping figure:** three architecture columns; loss, raw-gradient secant, rho, raw norm/optimizer-input norm and gradient-scale rows. ResNet threshold .5 remains explicitly missing. Existing MLP/ResNet data remain frozen.
- **Display contract:** no smoothing; log loss/norm and linear secant/rho/policy axes; independent scales by model; shaded historical prefixes. A secant value of 2 is a raw-gradient diagnostic reference, not a validated Adam stability boundary.
- **Generator:** [49_plot_conventional.py](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/49_plot_conventional.py); exported curves and per-arm summaries retain source hashes.
- **Independent export check:** [53_consolidate_conventional.py](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/53_consolidate_conventional.py) recomputes window/last-32 means with `math.fsum` from the CSV, verifies paired changes, maxima, terminal losses and clipping activity, then updates the selected assets. Source inventories and frozen configuration hashes are checked before plotting.
- **Checked display coverage:** 20 series / 6,644 fork-to-terminal boundary rows in the CSV; 384 shaded historical-prefix rows displayed once per model/family. Terminal policy points are omitted because no outgoing update exists. Both one-page PDFs were inspected after rendering; the raw/optimizer-input distinction, state labels, legends, clipping gap and footnotes are legible.
- **Figure 15 caption:** Late native Adam second-moment reductions across CIFAR-10 MLP, CIFAR-10 ResNet-32 and WikiText-2 Transformer. Each column compares unchanged Adam with beta2 .95/.9/.5 from the retained fork. Rows show full training loss, raw-gradient secant response, consecutive raw-gradient cosine, raw norm and beta2. Moments and optimizer clock are retained; native bias correction uses the new beta2. Shading denotes historical prefixes. One initialization; no smoothing; raw secants are not Adam stability bounds.
- **Figure 16 caption:** Global gradient clipping before Adam moments across the same tasks. Thresholds .5/1 are available for MLP and Transformer, and 1 for ResNet. Rows show loss, raw secant, raw cosine, raw/optimizer-input norms and applied gradient scale. Dashed norm curves show the actual optimizer inputs. Every treatment has its matched unchanged control; ResNet .5 remains unrun. Fixed absolute thresholds are objective-scale dependent; no smoothing or cross-family ranking.
- **Limits:** fixed absolute thresholds depend on loss scale; single seeds and retrospective windows do not establish universal stabilization, efficiency, generalization or a ranking against q50 caps. Cap and conventional forks differ.
- **Missing ablations:** [the executive TODO](../TODO_ABLATIONS.md) distinguishes model coverage, seeds, thresholds, timing and optional interactions.
