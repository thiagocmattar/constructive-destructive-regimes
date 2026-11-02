# Analysis036: controlled full-batch Adam diagnostics

Run215 completed all nine1,024-update arms. Scientific verification passed for
9,216 updates,9,225 boundary losses, all four vector streams and full held-out
split evaluations. All task cloud workers/disks are deleted; combined calibration
and main compute/storage estimate isUSD13.89 provisional. The complete figures
are linked in the main-study section below.

## Calibration at128 updates

Nine arms completed 128 full-objective updates each (1,152 accepted updates).
All use the same frozen Analysis024 seed0 weights per case and fresh Adam.
No seed sweep, Hessian/HVP, LR intervention or schedule; no validation/test passes.

| Case | LR | Final loss | Tail loss rises | Tail raw cosine | Tail moment cosine | Tail adaptive cosine | Loss label |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| mlp | 0.0001 | 0.304545 | 0.0% | 0.9995 | 0.9999 | 0.9999 | smooth_progress |
| mlp | 0.0003 | 0.263448 | 0.0% | 0.9933 | 0.9997 | 0.9997 | smooth_progress |
| mlp | 0.001 | 0.222867 | 0.0% | 0.9828 | 0.9996 | 0.9992 | smooth_progress |
| resnet | 0.0003 | 2.068170 | 0.0% | 1.0000 | 1.0000 | 0.9989 | smooth_progress |
| resnet | 0.001 | 1.315642 | 1.6% | -0.1318 | 0.9760 | 0.9187 | smooth_progress |
| resnet | 0.003 | 0.853716 | 17.5% | -0.5116 | 0.5106 | 0.6342 | unclassified |
| transformer | 0.0003 | 5.981771 | 0.0% | 0.9988 | 0.9998 | 0.9995 | smooth_progress |
| transformer | 0.001 | 4.593133 | 0.0% | -0.6602 | 0.9273 | 0.9959 | smooth_progress |
| transformer | 0.003 | 2.943088 | 3.2% | -0.4132 | 0.7559 | 0.9754 | smooth_progress |

All three MLP tails are smooth; its largest rate has a startup loss spike,
not sustained tail oscillation. Continuation evidence narrowed the grid but
did not transfer a from-initialization regime boundary. ResNet and Transformer
already show raw-gradient reversals while adaptive directions remain more
coherent. Interpret loss and geometry separately; negative raw cosine does
not establish harmful training or Hessian curvature.

The last64 boundary losses define labels (63 adjacent changes); tail stream
medians use the last64 accepted updates. Those windows differ by one boundary.
Labels and all pointwise values are preserved in fast_calibration_results.json
and fast_calibration_plot_values.json. Missing requested regimes are reported
in regime_candidates; no rate was changed or automatically added.

## Scientific coverage and next stage

The three fixed objectives are first5k CIFAR10 MLP images, all50k CIFAR10
ResNet images with50 fixed BN groups1000, and all2,088,600 prepared WikiText2
causal targets per update. Gradients cover states0..128; four complete prefixes
cover exactly128 applied updates. Stored m and bias-corrected mhat norms remain
distinct. Actual displacement is measured directly from parameters.

fast_calibration_results.json also records first pointwise and trailing64-mean
attainment of every frozen GD loss milestone. Unattained targets are censored
at128. For example, Transformer Adam at3e-3 first passes Run126's best
tested Transformer GD3000-state loss (6.069747) at update23; its trailing64
mean passes at state63 (the first complete window). ResNet Adam at3e-3
passes Run124's GD3000 loss (0.967229) at update107, while its trailing64
mean has not passed by128. These comparisons support shorter loss-matching
horizons in these cases. Matching an earlier GD loss is not convergence, matched location,
generalization, or a rate-independent optimizer benefit.

At calibration closeout the main study was provisional: up to three fixed LR regimes per case,
initially1,024 updates each (at most9 arms /9,216 updates). Calibration does
not establish convergence or authorize automatic main runs. Any new LR or
horizon requires a new frozen scientific run ID; the one-initialization and
no-Hessian constraints persist.

## Verification and execution

Eleven focused diagnostic tests pass. All manifests and artifact bytes/SHA256
are verified. Independent CPU checks reconstruct final moments/adaptive
directions, q at boundaries1/128, all prefix lengths/energies, parameter
telescoping and frozen initial/final tensor hashes. Cross-host initial gradient
qualification is recorded per arm; FP64 CPU/GPU norm reductions are compared
with relative1e-12 and absolute1e-10 tolerances. Smoke results are excluded.

MLP ran on the owned RTX5070Ti laptop. All ResNet/Transformer rates ran on
RTX5090 class at the live catalog rate USD1.19/GPU-hour, pinned PyTorch2.11/cu130
image, TF32 disabled. Different host CPU/runtime throughput prevents an
unqualified rate-by-rate wall-time comparison.

Two containers never became ready and received no scientific input; their
unchanged arms moved to warm matched workers. A Windows SSH newline issue
and isolated-package setup were corrected before scientific execution.
All preparation/retry/transfer time counts toward the same envelope.

Task compute plus temporary-storage estimate: USD2.26, provisional.
Every retrieved file was verified before deleting its worker and disks.
All five task pods are deleted; retained task cloud storage0GB. Unrelated
stopped resources were preserved. See execution_receipt.json and per-worker
retrieval receipts. The independent USD20 /90-minute guard did not trigger.

## Figures

07_reduce_plot_calibration.py generates both PDFs and exact plotted arrays.
See observations/001-fast-calibration.md for captions, reduction, coverage,
uncertainty and caveats. No manuscript was changed.

The requested eight-row [smoothed Adam composite](figures/adam_diagnostics_smoothed.pdf)
uses manuscript notation,16-step trailing means, separate gradient/momentum
norm rows, and R_t only for Adam updates. See
[Observation002](observations/002-smoothed-adam-composite.md) for exact timing,
coverage and reduction, and smoothed_composite_verification.json for checks.
This adds a third reviewed PDF without new training or cloud execution.

## Main-study completed-run preview

The following six-arm snapshot preceded completion of Run215. The interim
[preview](figures/adam_diagnostics_completed_preview.pdf) shows three completed
MLP rates, ResNet LR0.0003/0.003 and Transformer LR0.0003 through1,024 updates.
Following the user revision, training loss and Adam-update R_t are unsmoothed;
the other six diagnostic rows retain16-step trailing means. One ResNet rate and
two Transformer rates are pending. All48 curves /48,594 values and the rendered
PDF pass checks. [Observation003](observations/003-completed-production-preview.md)
records sources, reduction, coverage and limits; no new scientific claim is added.

## Complete1,024-update main study

All nine approved task/LR arms completed from the same frozen initialization
per case, with fresh Adam and constant LR. No extra seeds, Hessians/HVPs,
clipping, decay, schedule or LR interventions were introduced. The complete
[diagnostic-row-smoothed figure](figures/adam_diagnostics_full.pdf) and
[fully unsmoothed figure](figures/adam_diagnostics_full_unsmoothed.pdf) retain
the requested eight-row layout, raw training loss and only actual-Adam-update
R_t. The six diagnostic rows use complete 50-step trailing arithmetic means,
matching manuscript Analysis024. Both include all nine arms; the earlier preview
files retain their historical 16-step windows.

| Case | LR | Final training loss | Tail64 rho raw | Tail64 rho momentum | Tail64 rho Adam | Final Adam R_t |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mlp | 0.0001 | 0.0236636 | -0.9694 | -0.4092 | -0.0787 | 0.5609 |
| mlp | 0.0003 | 0.0119957 | -0.9292 | -0.7199 | -0.5422 | 0.3753 |
| mlp | 0.001 | 0.0095223 | -0.9506 | -0.8889 | -0.7936 | 0.2774 |
| resnet | 0.0003 | 0.526588 | -0.9089 | -0.5153 | -0.0818 | 0.3769 |
| resnet | 0.001 | 0.066282 | 0.1693 | 0.9049 | 0.9771 | 0.2779 |
| resnet | 0.003 | 0.000831704 | 0.9952 | 1.0000 | 1.0000 | 0.2492 |
| transformer | 0.0003 | 2.35697 | -0.7946 | -0.5981 | 0.9392 | 0.5402 |
| transformer | 0.001 | 0.988006 | -0.3306 | 0.0838 | 0.9598 | 0.4591 |
| transformer | 0.003 | 0.616266 | -0.0749 | 0.4148 | 0.9560 | 0.3152 |

Tail values are medians over the last64 applied updates, without plot smoothing.
Held-out losses and exact GD-loss milestone attainment remain in Run215/results.json.
This is a single-initialization descriptive comparison; different qualified GPU
models limit exclusive LR attribution. Negative raw cosine is not a Hessian,
convergence, stability or generalization result.

Both full figures pass source, numerical and visual checks:72 curves each,
71,055 values with six diagnostic rows smoothed and73,701 values fully raw.
Loss and R_t are identical between versions. See
[Observation005](observations/005-complete-production-diagnostics.md) for sources,
reduction, complete split coverage, uncertainty, hardware and reproduction.

Run215 main cost isUSD11.62365 and calibration cost isUSD2.26282; totalUSD13.88647,
provisional. All10 main task Pods and the earlier5 calibration Pods/disks are
deleted. Fresh account inventory confirms no Analysis036 worker remains and0GB
retained task cloud storage. Per-worker hash/scientific acceptance receipts,
finalization.json and completion.json are local. No manuscript was edited.

## Highest-LR Adam versus full-batch SGD figure

The [five-row comparison](figures/adam_vs_sgd_highest_lr.pdf) pairs Adam's
largest tested rates (MLP 0.001; ResNet/Transformer 0.003) with Analysis024's
largest plain-GD rates (MLP 0.04; ResNet 0.1; Transformer 0.2). It displays the
common first 1,024 updates from matching frozen initial tensors. Rows are log
training loss, raw-gradient eta*kappa_hat, raw-gradient rho_t, log gradient norm
and R_t. Loss and R_t are raw; the three other rows use 50-step trailing means.
All 30 curves / 29,832 points match the prior verified figure arrays exactly.
[Observation006](observations/006-highest-lr-adam-sgd-comparison.md) records sources,
reduction, retention-vector semantics, caption, coverage and limitations.
No additional training or cloud cost is incurred.

## Unsmoothed Transformer peak zoom

The [nine-panel zoom](figures/adam_transformer_peak_zoom.pdf) isolates the visible
Transformer Adam LR 0.003 spike near the user's approximate step 768. Its
prominent loss peak is at state 794 (loss 2.145566); the crop starts at 784
(loss 0.652293) and ends at the first post-peak loss below that reference,
state 852 (loss 0.651872). All eight original unsmoothed statistics are retained,
with a new panel for L_t-L_(t-1) over each adjacent pair inside the crop.
All 552 original points and 68 loss differences pass source, index, numerical
and visual checks. [Observation007](observations/007-transformer-peak-zoom.md)
records selection, caption, reduction, clock alignment, coverage and limitations.
This is a retrospective single-spike view, without new training or cloud cost.

## Repeated peaks and forward-blocked temporal prediction

The [three-column raw micro figure](figures/adam_peak_micro_three_models.pdf)
shows the largest completed relative loss episode at each highest-LR arm:
MLP peak 703, ResNet peak 422 and Transformer peak 794. Raw gradient, momentum
and adaptive diagnostics use their input parameter-state clock; actual R_t
continues to use the state after the accumulated updates. All 1,408 original
values and 173 loss differences pass exact source and index checks.

The [full temporal analysis](PEAK_TEMPORAL_ANALYSIS.md) extracts 129 loss-only
episodes across nine arms, including 45 large episodes and two censored
recoveries. Fresh crossings above 2 precede 13/17, 3/3, 7/7 large highest-LR episodes
for MLP/ResNet/Transformer under the fixed 1-8-state warning rule. On 4,534 later
time-block states, adding continuous projected change to loss history raises
average precision 0.260 to 0.454; a simple threshold>2 instead has 21% precision.
The lowest-LR ResNet/Transformer negative controls cross 2 frequently without
qualifying spikes. Gradient growth is algebraically required by the threshold,
so it is not independent corroboration. These are exploratory temporal results,
not causal identification or an Adam stability criterion.

See [Observation008](observations/008-peak-timing-and-blocked-validation.md),
the frozen analysis protocol, complete source/prediction tables and independent
verification. A proposal describes untouched-arm validation and separately
authorized optimizer-state forks with LR held fixed. No new training, seed,
Hessian/HVP, LR intervention or cloud cost accompanies this analysis.

## Manuscript integration after author review

[Observation009](observations/009-manuscript-raw-gradient-warnings.md) records
the two compact manuscript views and forward-test table. The six-comment
[Adam revision](../../manuscript/draft/revisions/2026-11-02-adam-feedback/README.md)
centers raw-gradient information about impending loss excursions beyond recent
loss history, despite Adam transforming gradients before parameter updates.
All 66 plotted series /53,777 values and 12 table scalars are independently
checked against existing evidence. Experimental scope, false alarms and causal
limits remain explicit. No new training, scientific inputs or cloud work.

## Evidence package before another manuscript edit

[Observation010](observations/010-paper-evidence-package.md) and the
[paper package](paper_evidence/README.md) prepare four focused figures, an
expanded recurrence/specificity census, prediction tables and candidate prose.
The overview and micro view now share four rows including raw-gradient rho;
the overview uses trailing 8 means plus raw projected-change context. All
micro/event observations remain unsmoothed. Raw source values are unchanged.

Additional local checks retain incremental AP improvement with all 16 recent
loss changes (0.258 to 0.447), and show raw cosine alone is informative (0.423).
Delayed-feature controls support timing specificity in aggregate, with exceptions
retained. Probability-error comparisons to a constant reference limit any
calibrated-warning claim. All 225 curves /67,530 values and 30 independent refits
pass verification. The active manuscript and its figure copies are unchanged;
prior analysis copies are archived. Interventions remain a deferred TODO.

## Simplified author-selected presentation

[Observation011](observations/011-simplified-presentation.md) supersedes the
presentation proposal above with a three-row overview, unchanged spike examples
and a descriptive recurrence table. The overview has 16-observation smoothing,
no raw overlays or relative norms, projected-change limits [-0.1,2.2], and
ordinary numeric loss ticks. Precision-recall and onset-overlay figures are
withdrawn from the proposed paper. Earlier results remain preserved; the
candidate text now makes an observational timing/recurrence claim. All 27 curves
and 27,369 plotted values pass source/display checks. Manuscript unchanged.
