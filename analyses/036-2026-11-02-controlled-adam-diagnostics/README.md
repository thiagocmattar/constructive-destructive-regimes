# Analysis036: controlled Adam diagnostics from initialization

The user confirmed the same frozen initial weights as Analysis024. This extends
Analysis029's Adam diagnostics to full-objective CIFAR-10/MLP,
CIFAR-10/ResNet-32 and WikiText-2/Transformer training. There is one fixed
initialization per case, no seed sweep and no Hessian/HVP measurements. Every
trajectory uses one constant LR and fresh Adam moments.

The user requested calibration first and a reduced search informed by
Analysis034's continuation evidence. [FAST_CALIBRATION.md](FAST_CALIBRATION.md)
specifies nine arms of 128 updates: MLP rates 1e-4/3e-4/1e-3, and
ResNet/Transformer rates 3e-4/1e-3/3e-3. This replaces the unexecuted 21-arm
proposal. The user subsequently approved the calibration and full training: Run215 executes
nine fresh 1,024-update arms (9,216 updates total). All nine arms are complete and
verified. All task Runpod workers and disks are deleted; retained cloud storage
is0GB. Combined main/calibration estimate isUSD13.89, provisional. See
[Run215](../../runs/215-2026-11-02-controlled-full-batch-adam/README.md).

Calibration completed on November 2, 2026: all nine arms /1,152 updates are verified. MLP uses the owned RTX5070Ti
laptop across all its rates; ResNet and Transformer use matched RTX5090-class
Runpod workers across their rates. All nine arms preserve the same scientific
config and pinned inputs. See [REPORT.md](REPORT.md), [results](fast_calibration_results.json) and
[execution receipt](execution_receipt.json). All task pods/disks are deleted;
USD2.26 estimated Runpod cost is provisional. Training-only
calibration skips validation/test passes. Labels describe this short horizon;
they do not establish stability or convergence.

[PROTOCOL.md](PROTOCOL.md) defines the comparison, measurement semantics,
interpretation limits and later study. [Run213](../../runs/213-2026-11-02-eos-adam-from-init-calibration/README.md)
owns scientific code, config and immutable input manifest. Analysis035/Run212 is
separate work. Raw-gradient projected change q remains a gradient-change
diagnostic under Adam; it does not identify Hessian curvature or a stability
threshold.

Eleven focused diagnostics tests and a two-update full-objective MLP mechanics
check pass. [verification.json](verification.json) records preparation checks;
the smoke is excluded from LR evidence. The input verifier hashes data, initial
weights and scientific sources. Executed scientific inputs must not be rebuilt
or changed in place.

The earlier [parallel Runpod estimate](RUNPOD_ESTIMATE.md) is retained as the
historical estimate for the original screening plus main-study proposal. The
reduced calibration has its own USD20 / 90-minute maximum, inside the earlier
USD50 suggested study envelope, with independent guard and verified retrieval.

The two reviewed PDFs are [calibration](figures/fast_calibration.pdf) and
[stream comparison](figures/fast_calibration_streams.pdf). Eight loss labels are
smooth-progress and one is unclassified; none meets the declared sustained
loss-oscillation label. ResNet/Transformer show raw-gradient reversal despite
coherent adaptive directions. MLP remains smooth across the tested grid.
The [observation](observations/001-fast-calibration.md) records figure sources,
reduction, coverage and interpretation limits.

The requested eight-row [smoothed Adam composite](figures/adam_diagnostics_smoothed.pdf)
uses manuscript notation,16-step trailing means, separate gradient/momentum
norm rows, and R_t only for Adam updates. See
[Observation002](observations/002-smoothed-adam-composite.md) for exact timing,
coverage and reduction, and smoothed_composite_verification.json for checks.
This adds a third reviewed PDF without new training or cloud execution.

The [completed production preview](figures/adam_diagnostics_completed_preview.pdf)
uses six verified1,024-update histories in the approved eight-row, three-column
layout. Training loss and Adam R_t are unsmoothed; the other six diagnostic rows
use16-step trailing means. Coverage is3/3 MLP,2/3 ResNet and1/3 Transformer. See
[Observation003](observations/003-completed-production-preview.md),
completed_preview_manifest.json and completed_preview_verification.json.

The separate [fully unsmoothed alternative](figures/adam_diagnostics_completed_unsmoothed.pdf)
uses the exact same six-condition snapshot, with no averaging in any row.
All48 curves /49,134 values and indices match the original observations exactly.
See [Observation004](observations/004-fully-unsmoothed-alternative.md) and
completed_unsmoothed_verification.json. Both preview versions are retained.

The complete nine-arm figures are [diagnostic rows smoothed](figures/adam_diagnostics_full.pdf)
and [fully unsmoothed](figures/adam_diagnostics_full_unsmoothed.pdf). Both keep
training loss and Adam-update R_t unsmoothed, with all three LRs per column.
The six diagnostic rows now use complete 50-step trailing arithmetic means,
matching manuscript Analysis024; the earlier calibration and six-arm previews
retain their historical 16-step windows.
See [Observation005](observations/005-complete-production-diagnostics.md) and
full_diagnostics_verification.json: all72 curves in each version and all source
identities pass checks. The earlier six-arm previews above remain as snapshots.

The [highest-LR Adam versus SGD comparison](figures/adam_vs_sgd_highest_lr.pdf)
selects each optimizer's largest tested LR per task and displays their common
first 1,024 updates in five rows: log loss, raw-gradient projected change,
raw-gradient cosine, gradient norm and R_t. Loss/R_t are unsmoothed; other
rows use 50-step trailing means. All 30 curves match the prior verified figures.
See [Observation006](observations/006-highest-lr-adam-sgd-comparison.md) and
highest_lr_comparison_manifest.json. Full-batch SGD is Analysis024's plain GD.

The [Transformer loss-peak zoom](figures/adam_transformer_peak_zoom.pdf) shows
Adam LR 0.003 at steps 784-852: ten steps before the prominent peak at 794 through
the first loss below L_784. It retains all eight original unsmoothed statistics
and adds step-by-step loss differences. All 552 original points match the full
figure exactly; the added 68 differences are direct adjacent-loss subtractions.
See [Observation007](observations/007-transformer-peak-zoom.md) and
transformer_peak_zoom_manifest.json for the selection rule and clock conventions.

The [three-model peak micro view](figures/adam_peak_micro_three_models.pdf) aligns
all diagnostics to the parameter state where they become available and retains
the actual-update clock for R_t. Its 27 panels are fully unsmoothed. The
[temporal analysis](PEAK_TEMPORAL_ANALYSIS.md) covers all nine arms, 129 loss-only
episodes and five forward test blocks. Projected change adds predictive value
over loss history, while the standalone threshold has many false alarms.
[Observation008](observations/008-peak-timing-and-blocked-validation.md) records
sources, reductions, coverage and limitations; peak_temporal_verification.json
contains independent checks. The proposal includes untouched-arm validation and
matched optimizer-state forks at unchanged LR; no new training is launched.
