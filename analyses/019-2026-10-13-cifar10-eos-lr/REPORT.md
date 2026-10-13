# Analysis019 results: full-batch CIFAR-10 GD/Adam LR sweep

All eight runs completed 20,000 updates on the owned DGX Spark. Training and diagnostics took 52.03 minutes in total; the observed queue timestamps are in [queue_state.json](queue_state.json). Training ETC is zero. Peak CUDA allocation/reservation across the cohort was 340.1/374.0 MiB.

| Run | Method | LR | Final train loss | Train accuracy | Test loss | Test accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 059 | GD | 0.004 | 0.09687642 | 98.94% | 0.5587034 | 32.99% |
| 060 | GD | 0.00666667 | 0.05650662 | 99.72% | 0.6534718 | 30.21% |
| 061 | GD | 0.01 | 0.03204213 | 99.96% | 0.6842399 | 29.39% |
| 062 | GD | 0.02 | 0.01945231 | 100.00% | 0.708012 | 29.00% |
| 063 | ADAM | 5e-05 | 0.0007658233 | 100.00% | 0.8471853 | 26.07% |
| 064 | ADAM | 0.0001 | 0.0008775688 | 100.00% | 0.8220989 | 26.38% |
| 065 | ADAM | 0.0002 | 0.0007075287 | 100.00% | 0.7615954 | 28.02% |
| 066 | ADAM | 0.0004 | 0.0006593872 | 100.00% | 0.7092397 | 28.35% |

These are final-state evaluations on all 5,000 training and 10,000 test images. Loss is half the summed squared output error to one-hot labels, averaged over images. The loss figure includes all pre-update training states plus the separately evaluated final state. Test performance is reported descriptively; no separate validation/tuning split or seed ensemble was used.

| Run | Raw negative pairs | Late raw cosine | Late mhat cosine | Raw final B/A | mhat final B/A | Final eta*lambda1 | Final eta*lambda1(P) | Late eta-kappa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 059 | 0.00% | 1.00000 | n/a | 0.760229 | n/a | 1.67674 | n/a | 4.34581e-05 |
| 060 | 52.67% | -0.99894 | n/a | 0.0249768 | n/a | 2.07624 | n/a | 1.99876 |
| 061 | 77.18% | -0.99874 | n/a | 0.0148272 | n/a | 2.07017 | n/a | 1.99847 |
| 062 | 93.92% | -0.99855 | n/a | 0.0105121 | n/a | 2.07734 | n/a | 1.99885 |
| 063 | 94.41% | -0.77231 | -0.76542 | 0.00655222 | 0.108143 | 0.0330723 | 36.8898 | 1.83288 |
| 064 | 96.00% | -0.81857 | -0.80809 | 0.00317808 | 0.0565549 | 0.0379111 | 37.8122 | 1.85443 |
| 065 | 97.62% | -0.84038 | -0.82376 | 0.00279562 | 0.0501679 | 0.0242918 | 39.2431 | 1.87122 |
| 066 | 98.12% | -0.86574 | -0.84818 | 0.00303002 | 0.0559629 | 0.0170393 | 39.0251 | 1.88684 |

Late cosine means cover updates 19,001-20,000. Negative-pair fractions cover every 19,999 defined pair. A=sum of vector norms and B=norm of the vector sum use the complete prefix; no temporal truncation is applied. Momentum in this table is bias-corrected mhat. Stored m is also retained and separately plotted. Their adjacent cosines agree under positive rescaling, but their early norms and cumulative geometry differ.

The final recorded raw gradient belongs to theta[19,999], while final train/test evaluations and the last Hessian belong to theta[20,000]. Late eta-kappa uses the last 1,000 defined gradient pairs. These operational state indices are preserved; the tables do not pretend that the last pre-update gradient was recomputed at the final checkpoint.

## Descriptive findings

The lowest final training loss within the preselected GD grid occurs at LR 0.02: 0.01945231, with test accuracy 29.00%. This is a finite-horizon single-initialization comparison, not an optimized or general method ranking.

The lowest final training loss within the preselected ADAM grid occurs at LR 0.0004: 0.0006593872, with test accuracy 28.35%. This is a finite-horizon single-initialization comparison, not an optimized or general method ranking.

The three larger GD rates finish with eta*lambda1 in 2.070-2.077; their late raw cosine means are all approximately -0.999. LR 0.004 instead retains a late cosine near+1 and finishes below the spectral reference. The onset moves earlier with larger LR in this cohort.

Adam's final eta*lambda1(P) spans 36.89-39.24, near the frozen-preconditioner reference 38. Its late bias-corrected momentum cosine is negative at every rate (-0.848 to -0.765). Ordinary eta*lambda1 is only 0.0170-0.0379, while late eta-kappa remains near 1.8-1.9. The gradient proxy therefore does not supply an ordinary-Hessian curvature estimate on these Adam trajectories.

| GD LR | First negative raw pair (update) | First sampled eta*lambda1 >2 (state) | Late sampled eta*lambda1 mean | Last 1,000 eta-kappa mean |
| ---: | ---: | ---: | ---: | ---: |
| 0.004 | None | None | 1.62896 | 4.34581e-05 |
| 0.00666667 | 9467 | 9200 | 2.06118 | 1.99876 |
| 0.01 | 4565 | 4400 | 2.07184 | 1.99847 |
| 0.02 | 1156 | 1000 | 2.0766 | 1.99885 |

The Hessian is sampled every 100 updates, while gradient pairs are measured every update. These event indices have different resolution and do not establish a causal ordering. GD 2/eta is the standard local quadratic reference. For Adam, the plotted frozen-preconditioner reference is 38/eta at beta1=0.9, using P=(1-beta1^t)(sqrt(vhat)+epsilon) as in the released adaptive-EOS implementation. The moving preconditioner and nonlinear objective prevent treating that line as a universal stability guarantee.

The gradient-derived quantity eta*kappa=1-rho_t*||g_t||/||g_(t-1)|| is retained from the previous analyses. There is no injected gradient noise here, so its explicit noise term is zero. On Adam it remains a trajectory proxy: the actual step is adaptive and differs from -eta*g. Adjacent sign, parameter norm, B/A or a low training loss alone does not certify stability, convergence or generalization.

## Exact raw-gradient interaction

| Run | Energy E | All-pair B squared minus E | Lag-one interaction | Beyond-lag-one interaction | Absolute lag-one omission / B squared |
| --- | ---: | ---: | ---: | ---: | ---: |
| 059 | 126.5472 | 981530.5 | 246.7958 | 981283.7 | 0.99962 |
| 060 | 89810.29 | 493212.2 | -179071.7 | 672283.9 | 1.1531 |
| 061 | 109002.4 | 249751.8 | -217300.9 | 467052.6 | 1.3019 |
| 062 | 76307.95 | 71959.85 | -151700.9 | 223660.7 | 1.5085 |
| 063 | 8050.696 | -2950.201 | -13554.03 | 10603.83 | 2.079 |
| 064 | 20871.74 | -17623.98 | -35411.84 | 17787.86 | 5.477 |
| 065 | 18615.48 | -16542.79 | -31359.64 | 14816.85 | 7.1486 |
| 066 | 7355.876 | -6509.916 | -12093.6 | 5583.687 | 6.6004 |

These are temporal sums of batch-mean gradients, not per-example gradient energy or intra-batch alignment. The last column measures the full-prefix squared-norm error from omitting all interactions beyond lag one, relative to the true B squared. It can exceed1. Strongly negative adjacent cosines do not determine the sign of exact all-pair interaction.

## Evidence and checks

[Verification](verification.json) checks all 160,000 update rows, all 160,000 complete test evaluations, 201 ordinary spectra per run and 200 additional preconditioned spectra per Adam run: 4,816 eigenpairs in total. Source/config/input identities, initial raw norms/loss/spectra, scalar prefix identities, cosine and ratio bounds, mhat scaling, kappa arithmetic, final optimizer vectors and exact displacement telescoping pass. An independent direct affine/tanh forward computation with 512-image chunks reproduces every final train/test result.

[Figure manifest](figure_manifest.json) checks every plotted line's actual arrays and 1,194,732 complete trailing means. Loss remains unsmoothed; the norm grid retains faint original data beneath a 10-update raw mean; cosine and kappa panels use 100 complete-pair means. All 20,000 updates are included, without temporal thinning.

[Retrieval](retrieval.json) verifies 120 retained files (1.333 GB uncompressed) and 0.597 GB transfer archives by size/SHA256. Full scalar histories, prepared inputs, recovery/final states and archives remain locally and on Spark outside Git; compact manifests, spectra, results and figures are tracked. No rented cloud resources or billing. Dense intermediate vector histories were intentionally disabled: scalar identities can be audited, but arbitrary long-lag vector profiles and per-image gradient energy cannot be reconstructed.

The [final Spark check](compute_closeout.json) records 0% GPU utilization and 45 C, with 1.930 GB in the task directory including transfer archives. Prepared inputs from Run058 remain in their separate frozen directory. No unrelated process or storage was changed.

## Current GD figure

The requested six-panel GD-only view is [cifar10_gd_diagnostics.pdf](figures/cifar10_gd_diagnostics.pdf).
Loss, cosine and eta-kappa are unsmoothed on linear Y; both pair metrics show
all 19,999 defined pairs per run. Curvature is eta times lambda_max directly; raw B/A is linear
and raw norms are unsmoothed on log Y. See
[Observation002](observations/002-gd-six-panel.md),
[numeric checks](gd_figure_manifest.json) and
[visual verification](gd_visual_verification.json).

## Current Adam figures

The Adam counterpart has separate [full-horizon](figures/cifar10_adam_diagnostics.pdf)
and [first-2,000-update](figures/cifar10_adam_diagnostics_early.pdf) views.
Each uses ten panels in the requested order, including momentum adjacent cosine,
momentum B/A, momentum norm and raw-gradient/momentum norm ratio. Momentum is
bias-corrected mhat. The two norm panels use log Y; panel (j) also uses log Y
in the early view. The norm ratio is raw/mhat and unsmoothed. Loss is unsmoothed
and linear; both cosines and eta-kappa use
complete 100-pair trailing means. Curvature remains eta times the ordinary
lambda_max, without a GD threshold line. The two views share Y scales except (j) and
preserve the same full-prefix/window statistics. No training results changed.
See [Observation003](observations/003-adam-ten-panel.md),
[numeric checks](adam_figure_manifest.json) and
[visual verification](adam_visual_verification.json).

## Archived detailed figures

- [cifar10_eos_comparison.pdf](figures/archive/cifar10_eos_comparison.pdf): Matched GD/Adam learning-rate comparison: unsmoothed loss, full train/test accuracy,100-pair raw/update or mhat cosine, ordinary top 2 eigenvalues, optimizer-specific spectral reference, raw B/A, and 100-pair eta-kappa. No shared LR values or universal stability inference.
- [cifar10_gd_geometry.pdf](figures/archive/cifar10_gd_geometry.pdf): GD full-horizon scalar geometry with unsmoothed loss,100-pair cosine means, log vector norms and cumulative A/B, linear B/A. Adam separates stored m, bias-corrected mhat, adaptive d and actual displacement; GD has no momentum.
- [cifar10_adam_geometry.pdf](figures/archive/cifar10_adam_geometry.pdf): ADAM full-horizon scalar geometry with unsmoothed loss,100-pair cosine means, log vector norms and cumulative A/B, linear B/A. Adam separates stored m, bias-corrected mhat, adaptive d and actual displacement; GD has no momentum.
- [loss_and_norms_by_learning_rate.pdf](figures/archive/loss_and_norms_by_learning_rate.pdf): Unsmoothed loss followed by one panel per run, common log norm Y limits, faint original raw-gradient norms,10-update raw trailing means, and fill to unsmoothed mhat on Adam or actual-update norm/LR on GD.

The [observation](observations/001-learning-rate-sweep.md) records sources, reductions, captions, coverage and interpretation limits.
