# Completed learning rates across the three GD cases

At the user's October18 request, the existing
[publication PDF](../figures/eos_mlp_learning_rates_3k.pdf) is expanded to
three columns and four rows. Its original filename is retained. A live Spark
check at16:10UTC confirmed Run125 active; Run126 has not started. The figure
uses seven complete, retrieved and verified conditions, with no partial runs.

## Sources and coverage

| Column | Dataset / model | LRs | Source runs |
| --- | --- | --- | --- |
| 1 | CIFAR-10 first5k / tanh200-200 MLP | 0.004,0.02,0.04 | 059,062 (3k prefixes),122 |
| 2 | CIFAR-10 all50k / ResNet-32 | 0.025,0.05,0.1 | 123,124,120 |
| 3 | WikiText-2 / two-layer Transformer | 0.05 | 121 |

All trajectories cover3,000 constant-LR full-GD updates, with the same frozen
seed0 initialization within each case. Architectures, objectives and training
data differ across columns. Plain GD has no momentum, decay, clipping,
schedule, augmentation or dropout. Training uses FP32 with TF32 disabled;
diagnostic reductions use FP64. The ResNet objective averages the same50
fixed1k ghost-BN groups; the language loss retains its complete prepared
token weighting and2,088,600 supervised target tokens per update.

Generator: `06_plot_mlp.py`. The PDF, manifest, summary and ignored arrays
retain the original `mlp_learning_rates` naming for compatibility. The summary's
`scope`, `rates_by_task` and `completed_conditions` describe the expanded
coverage; they no longer mean MLP-only. Source/config/inventory hashes appear
in `mlp_learning_rates_results.json`; output/curve hashes and visual acceptance
appear in `mlp_learning_rates_figure_manifest.json`.

All190 source-inventory files pass byte-size/SHA256 checks. Each frozen initial
tensor and config is checked, and every included source hash/final loss matches
the independent `03_results.py` reduction. The qualified Analysis020 loader
and Analysis021 retrieval verifier are reused. All12 previously plotted MLP
curve hashes are unchanged; the additional16 curves represent the other cases.
No executed training input, Spark admission or manuscript is changed.

Loss spans states0..3000, using the actual endpoint; legacy MLP endpoints
come from their next preupdate row. Norms span states0..2999 and are finite
and positive, with no clipping/flooring for log display. The first adjacent
pair is undefined; the2,999 defined q values per case end at states1..2999.
No extra gradient is inferred at state3000. Hessians are sampled at states
0,100,...,3000: full5k MLP objective, first five fixed ResNet ghost groups
(5k images), or the first2,500 Transformer sequences. These scopes are labeled
above the corresponding Hessian panels. All217 largest algebraic eigenpairs
pass relative residual<=0.005. Legacy MLP runs requested two eigenvalues;
the other runs requested one. Compare the largest in both, with this solver
difference disclosed.

In total:21,000 update rows/norms,20,993 adjacent pairs,217 spectral states,
and63,217 exact plotted points across28 colored data curves and12 panels.
Every artist array matches its source; every q identity and its exact sign
boundary is checked. No temporal smoothing, seed averaging or LR rescaling
of another trajectory is applied. The first three rows have linear Y; only
the norm row has log Y. Axes share iteration within each model; Y limits
remain independent.

## Caption

Completed full-batch GD trajectories at constant learning rates, seed0,
through3,000 updates. Columns: CIFAR-10/tanh MLP (three rates), CIFAR-10/ResNet-32
(three rates), WikiText-2/Transformer (one complete rate). Rows: unsmoothed
training loss; eta times the largest algebraic Hessian eigenvalue, sampled
every100 updates; unsmoothed gradient-derived directional change
q=eta*kappa_hat=1-rho*norm(g_t)/norm(g_previous); and unsmoothed raw-gradient
norms on log Y. Within each column, blue/orange/green identify increasing
planned LR; the legends give the actual rates. Hessian measurements use the
full MLP objective but declared subsets for the larger models. Dashed lines
mark eta*lambda=2 (GD quadratic reference) and q=1 (exact adjacent-gradient
reversal boundary). Neither is a global-stability guarantee. Connecting
spectral markers does not provide unsampled curvature measurements.

## Observations, uncertainty and limits

The MLP0.004 and Transformer0.05 trajectories remain positively adjacent-aligned
in this window. Higher MLP rates develop late negative alignment and increased
raw norms; MLP0.04 also has an initial negative-cosine transient. The first
negative gradient update is not necessarily a persistent regime onset; see
the historical MLP observation and `mlp_transition_check.json` for the window
criterion. All three ResNet conditions develop negative adjacent alignment.
Their first negative updates are1,121/494/231 at LRs0.025/0.05/0.1, with final
losses1.466540/0.967229/0.489838. These are descriptive results at one matched
initialization, with no ensemble uncertainty estimate or robustness claim
across initializations. Reused prefixes are existing evidence, not new reruns.

The q/cosine relation is algebraic. q is a finite-step projected gradient
change; interpreting it as local directional Hessian curvature requires a
Taylor approximation. ResNet/Transformer subset sharpness is not the exact
full-objective maximum curvature. Alignment/norms alone do not establish loss
decrease, generalization, asymptotic behavior or global stability. The two
unfinished Transformer rates are excluded rather than extrapolated.

The current Poppler render is reviewed for all12 panels, three model headings,
seven LR legends, full/subset spectral labels, reference lines, norm log axes
and margins. One-page PDF text and output hashes also pass verification.
