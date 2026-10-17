# MLP: three selected GD learning rates through3k

The reviewed [four-panel PDF](../figures/eos_mlp_learning_rates_3k.pdf) overlays
the three rates selected for Analysis024:0.004,0.02 and0.04. This is the MLP
column's calibration preview, not a completed nine-condition figure.

## Sources and reduction

Runs059/062: first3,000 updates of the immutable20k CIFAR-10 tanh200-200 MLP
GD trajectories at0.004/0.02. Run122: complete new3,000-update GD trajectory
at0.04. All use the same frozen seed0 tensors, first5,000 training images,
half-squared one-hot objective, physical batches of1,000, FP32 training and
FP64 diagnostics. No momentum, decay, clipping, schedule, augmentation or dropout.

Generator: `06_plot_mlp.py`; qualified Analysis020 array loader and Analysis021
inventory verifier. All62 source-inventory files pass size/SHA256 checks; each
config and frozen initial tensor is verified. Source/config/inventory hashes
and scalar summaries appear in `mlp_learning_rates_results.json`. The plot
values match the independent Analysis024 reduction. The executed scientific
packet and Spark queue are unchanged; this is local postprocessing only.

Loss covers states0..3000, including the actual endpoint (next-row loss for
legacy prefixes). Hessians cover states0,100,...,3000 and the complete5k MLP
objective; all93 eigenpairs pass the declared relative residual<=0.005.
Only the largest eigenvalue is plotted. Legacy runs requested two eigenvalues;
Run122 requests one, with the documented solver difference. Norms cover all
9,000 gradient states0..2999; no zero is clipped or replaced for log display.
The8,997 defined adjacent pairs end at states1..2999; no additional gradient
is inferred at state3000. Every plotted q identity and its sign boundary pass.

There is no temporal smoothing or seed aggregation. Training loss, scaled
sharpness and q use linear Y; raw-gradient norms use log Y. All12 colored
artist arrays match their recorded sources exactly:27,093 plotted points.
The common LR legend applies to all four panels. Poppler rendering was checked
for typography, spacing, legend, axes, reference lines and unclipped labels;
the one-page PDF text and output hashes also pass verification.

## Caption

Full-batch GD on the first5,000 CIFAR-10 images with a tanh200-200 MLP and one
matched initialization, through3,000 updates. Learning rates0.004,0.02,0.04
are blue, orange and green. Panels: (a) unsmoothed full-objective training loss;
(b) eta times the largest algebraic full-objective Hessian eigenvalue, sampled
every100 updates; (c) unsmoothed gradient-derived directional change
q=eta*kappa_hat=1-rho*norm(g_t)/norm(g_previous); (d) unsmoothed raw-gradient
norms on log Y. Dashed lines mark eta*lambda=2 (GD quadratic reference) and
q=1 (exact adjacent-gradient reversal boundary), not global stability guarantees.
Lines between spectral markers do not provide unsampled Hessians. A local
curvature interpretation of q requires a Taylor approximation.

## Observations and limits

| LR | Final training loss | Last1k mean cosine | Last1k mean q | Final B/A |
| ---: | ---: | ---: | ---: | ---: |
| 0.004 | 0.336508 | 0.999999957 | 0.0000437457 | 0.794144 |
| 0.02 | 0.211443 | -0.991301 | 1.992392 | 0.0487741 |
| 0.04 | 0.153869 | -0.979903 | 1.981980 | 0.0256697 |

LR0.004 has no adjacent reversal in this window. LR0.02 first reverses at
gradient update1,156, with a transient return toward positive alignment before
the later negative phase. LR0.04 has an initial negative-cosine transient
starting at update2, returns to positive alignment, then enters a later
negative phase. Its first100-pair window with at least80% negative pairs ends
at323; LR0.02 reaches that same criterion at1,345. See
`05_check_mlp_transitions.py` and `mlp_transition_check.json` for the descriptive
window criterion and indexing. These are not first-onset timestamps.

The two higher rates show increased raw-gradient norms alongside late negative
alignment while loss decreases overall. This is a conditional LR comparison
at one initialization; there is no ensemble uncertainty or initialization-
robustness claim. Reused prefixes are existing evidence, not independent reruns.
The q/cosine correspondence is algebraic, not independent corroboration or
a proof of loss decrease. No asymptotic or global-stability conclusion follows.
