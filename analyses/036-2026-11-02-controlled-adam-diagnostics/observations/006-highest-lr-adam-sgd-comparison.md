# Observation006: highest-LR Adam and full-batch SGD comparison

## Sources and comparison

The user requested a new figure pairing the highest tested Adam LR with the
highest tested SGD LR for each controlled task. Full-batch SGD here is the plain
GD reference from Analysis024, without momentum. Select rates from each fixed
grid, independently of the attained losses or diagnostic outcomes:

| Case | SGD LR / source run | Adam LR / source run |
| --- | --- | --- |
| CIFAR-10 / tanh MLP | 0.04 / Run122 | 0.001 / Run215 |
| CIFAR-10 / ResNet-32 | 0.1 / Run120 | 0.003 / Run215 |
| WikiText-2 / Transformer | 0.2 / Run126 | 0.003 / Run215 |

The model, complete training objective, prepared data and frozen initialization
match within each task. Initial tensor hashes are explicitly checked across
optimizers. Objectives cover the same first 5,000 CIFAR-10 images for MLP, all
50,000 images with fixed ghost-BN groups for ResNet, and all 2,088,600 prepared
causal targets per update for Transformer. Loss is averaged over images or
supervised targets. No held-out results are plotted.

The common first 1,024 updates are displayed. Adam's complete recorded horizon
is 1,024; SGD has 3,000 available, and its later history is retained but excluded
from this comparison. Selection and cropping do not launch training or change
the executed inputs. There is one fixed initialization per task, no seed sweep,
new Hessian measurement, LR intervention or schedule.

## Caption and reduction

`figures/adam_vs_sgd_highest_lr.pdf`: highest tested constant learning rates for
full-batch SGD (blue) and Adam (orange), from identical frozen initial weights.
Columns show CIFAR-10/MLP, CIFAR-10/ResNet-32 and WikiText-2/Transformer. Rows
show training loss on a log scale; raw-gradient eta*kappa_hat; consecutive
raw-gradient rho_t; raw-gradient norm on a log scale; and cumulative R_t.
Each legend identifies the optimizer and its selected LR.

Loss and R_t are unsmoothed. The three diagnostic rows use complete 50-observation
trailing arithmetic means, aligned to the rightmost observation, matching the
manuscript Analysis024 policy and the current full Adam figure. There is no
padding, partial-window averaging, binning or thinning. Norms are averaged in
linear space before log display. The projected-change expression
1-rho_t*norm(g_t)/norm(g_(t-1)) is computed pointwise before smoothing.

Loss covers parameter states 0 through 1,024. Vector rows refer to applied
updates 1 through 1,024, with the raw gradient evaluated at state t-1 used for
update t. Adjacent quantities are initially undefined and omitted, so their
first complete smoothed window ends at update 51. The norm's first complete
window ends at update 50. The unapplied terminal Adam gradient is excluded.

Adam R_t uses the measured actual parameter increments. SGD retains Analysis024's
raw-gradient prefix ratio. Under its constant LR and plain GD recipe this equals
the corresponding update-retention ratio in exact arithmetic, since the common
LR and sign cancel. It does not replace that historical ratio with a newly
measured FP32 displacement statistic. Neither prefix is reset or smoothed.
The dashed reference at 2 in the projected-change row follows the earlier
visual convention; it is not an identified Adam stability boundary.

## Verification, coverage and uncertainty

`13_plot_highest_lr_comparison.py` generates the figure and
`highest_lr_comparison_manifest.json`. Sources include Analysis024's cohort,
verified results, original scalar histories and hash-pinned figure arrays, plus
Run215's verified scalar histories and full-figure manifest. The new manifest
pins every used source, the generator and output, selected rates, initial tensor
hashes, source arrays, plotted arrays and step indices.

All 30 curves across 15 panels match the corresponding prior figure arrays
exactly over the displayed horizon. There are 29,832 plotted points. Independent
math.fsum window means agree within 8.89e-16 absolute error. Source hashes,
highest-rate selection, initial tensor identities, projected-change identities,
positive log inputs and exact Matplotlib artist arrays pass checks. The one-page
PDF passes text-boundary checks and Poppler visual review; labels and legends
are legible without overlap or clipping. The PNG render remains ignored scratch.

One trajectory per task and optimizer supplies no across-initialization
uncertainty estimate or confidence band. Optimizer and LR vary together, and
the rates are selected as grid maxima, not claimed optimal rates. Qualified
hardware and FP32 reduction differences remain. The comparison does not isolate
an optimizer-only causal effect. Under Adam the projected raw-gradient change
does not identify Hessian curvature; alignment or R_t alone does not establish
loss decrease, convergence, stability or generalization. No new scientific
claim, cloud execution or spending accompanies this figure.

Reproduce:

    python analyses/036-2026-11-02-controlled-adam-diagnostics/13_plot_highest_lr_comparison.py
    pdftoppm -singlefile -scale-to 1800 -png analyses/036-2026-11-02-controlled-adam-diagnostics/figures/adam_vs_sgd_highest_lr.pdf analyses/036-2026-11-02-controlled-adam-diagnostics/scratch/adam_vs_sgd_highest_lr
    python analyses/036-2026-11-02-controlled-adam-diagnostics/13_plot_highest_lr_comparison.py --accept-reviewed
