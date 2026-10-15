# First-stage GD overview across the three EOS cases

Status: source, transfer, scalar, plotted-array and visual checks passed.
Generator: [10_plot_gd_overview.py](../10_plot_gd_overview.py).
Publication figure: [seed 0 GD overview](../figures/eos_gd_overview_seed 0.pdf).
The reviewed report places it after Table 1, on the following page.

## Sources and comparison

Use complete seed 0 trajectories: Run 061 (CIFAR-10/tanh MLP, LR0.01),
Run 071 (CIFAR-10/ResNet-32, LR0.2), and Run 074
(WikiText-2/Transformer, LR0.1). The first 500 updates of the existing 20k Run 061
are reused; its raw evidence is unchanged. The other two runs each completed
500 updates. These are three task/model cases over two datasets. There is
**one learning rate and one initialization per column**. Additional LR runs
remain held, and the other Transformer seeds are not included as partial curves.

All updates are full-batch plain GD with no momentum, decay, clipping or schedule.
The full objectives, preparation, precision and historical source revisions are
defined in [the protocol](../PROTOCOL.md) and Run 067 inputs. ResNet accumulation
preserves its 50 fixed 1,000-image ghost groups. The Transformer uses the complete
prepared 2,088,600-target-token objective, not its Hessian subset for training.

The source rows and inventory digests are in
[gd_overview_seed 0_results.json](../gd_overview_seed 0_results.json); plotted
arrays, generator/loader/source hashes and review are in
[gd_overview_seed 0_manifest.json](../gd_overview_seed 0_manifest.json).
All 73 retained source/production files were checked against their size/SHA-256
transfer inventories. New Run 071/074 receipts are in their owning run folders.
Raw evidence/checkpoints remain ignored locally and retained on Spark.

## Reduction, coverage and caption

The 3x3 grid has columns MLP, ResNet-32 and Transformer; rows are full training
loss, eta times the largest algebraic Hessian eigenvalue, and

    q_t = eta*kappa_hat_t = 1 - rho_t*||g_t||/||g_(t-1)||.

Loss covers parameter states 0-500, with the final-state objective taken from the
complete summary. The 499 adjacent gradient pairs per case cover gradient
indices 2-500 and are plotted at ending parameter states 1-499. The undefined
first pair is omitted; no denominator epsilon is added. Spectra are measured at
states 0,100,...,500. No temporal smoothing, seed averaging, interpolation of
missing observations, or extrapolation beyond 500 updates is performed.
Connecting the six spectral markers supplies only a visual guide.

Hessians use all 5,000 MLP images, the first five fixed ResNet ghost groups
(5,000 images), and the first 2,500 Transformer sequences. The latter two are
subset Hessians. All plotted eigenpairs have independent relative residual
<=0.005. Run 061 originally requested two leading eigenvalues; only its largest
is shown, matching the target of the newer single-eigenvalue measurements.

Dashed references mark eta*lambda_max=2 (the GD quadratic reference) and q=1
(zero adjacent inner product when gradient norms are nonzero). The complete
caption is [gd-overview-seed 0.tex](../gd-overview-seed 0.tex). Axes are linear;
colors identify task/model cases and legends state the actual learning rates.

## Observed values and interpretation limits

| Case | Initial loss | Loss at state 500 | Maximum sampled eta*lambda_max | First negative adjacent-gradient index |
| --- | ---: | ---: | ---: | ---: |
| MLP | 0.596781 | 0.367796 | 0.786679 | None within 500 gradients |
| ResNet-32 | 2.309567 | 1.405459 | 3.426332 | 109 |
| Transformer | 10.729183 | 7.070935 | 2.141334 | 456 |

Negative adjacent pairs constitute 0%,78.16% and 9.02% of the 499 defined pairs,
respectively. Their signs agree exactly with q>1. The plotted losses decrease
over the horizon, while the ResNet objective fluctuates. Its final-state loss
is higher than its pre-final-update value; these are different parameter states.
The Transformer reaches the reversal boundary late in this first stage; the
MLP has not reached it at this learning rate within 500 updates.

These are descriptive results without seed uncertainty. They do not establish
a universal regime, a learning-rate effect within a model, asymptotic behavior,
generalization or global stability. q measures a finite-step projected gradient
change; identifying it with local Hessian directional curvature neglects a
Taylor remainder. ReLU nonsmoothness and subset/full-objective differences also
limit comparison with the maximum curvature. No equivalence between q and
eta*lambda_max is claimed.

Verification independently recomputes all 1,497 q values from cosine/norms,
checks reversal signs, all 18 spectral states/residuals,501 loss states per case,
and all 3,018 plotted values against their source arrays. The standalone Poppler
render and all 14 current report pages were reviewed; the report builds in two
passes without overfull boxes, undefined references or pending reruns. Source
insertion preserves the surrounding manuscript bytes. No new training or GPU
work was launched for this figure.
