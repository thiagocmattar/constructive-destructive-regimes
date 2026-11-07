# LR halving: immediate spikes, delayed spikes, and the strict no-peak endpoint

## Conclusion

The halving sequence stops at **LR 0.000625 (I8)**: every loss transition from
state 500 through 800 decreases. The largest adjacent difference is -2.446e-6,
so this result does not rely on the numerical tolerance allowing a small rise.

**LR 0.00125 (I7) already eliminates visible spikes**, leaving only a single
1.701e-6 increase at 500 -> 501 (0.0060%), followed by decreasing loss. It ends
at a lower loss than I8. Conversely, **0.0025 (I6) has a delayed peak at 707**;
an inspection restricted to the immediate response would miss this failure.
?No peak? must specify both the size criterion and the observation horizon.

These are finite-window, one-fork outcomes. The search does not estimate a
universal stable LR, prove that future spikes are impossible, or select the
best rate for learning. In particular, the selected smoothest trajectory is
not the one with the lowest final loss.

## Complete intervention and adaptive rule

All new arms freeze first-linear weights and biases from state 448, then set
Adam's LR at state 500. They branch independently from the **same complete frozen
state 500** as I2/I3, with identical model tensors, optimizer state, RNG and
previous/current gradients. They do not continue sequentially from each other's
endpoints. First changed update: 500 -> 501; captured final state: 801.

After I4 at 0.01, the user requested halving until no peak. Before launching I5,
we recorded the conservative rule in [halving_plan.json](../halving_plan.json):
all losses at states 500-800 finite, no adjacent increase above absolute 1e-8,
and no loss above loss500 + 1e-8. On failure, halve the rate and retry from the
same fork. All earlier failed trials are retained. I7 fails this strict rule
because of its tiny first-step bump; I8 is the first pass in the requested
halving sequence. The user subsequently confirmed stopping.

Match the original seed-0 tanh MLP 3072-200-200-10, all 5000 CIFAR-10 examples/order,
mean per-example half squared error, physical batch 1000, FP32/TF32 off, Adam
betas (0.9,0.999), epsilon 1e-8, and no clipping or decay. There is no optimizer
reset or unfreeze. Native Adam moments/clocks advance; first-layer parameters
are restored before the next forward, giving exact zero applied increments.
Full raw gradients still include derivatives for those frozen parameters.

## Results against unchanged Adam

The unchanged baseline has maximum loss **0.0550626 at 522**, mean loss **0.0210322**
and endpoint loss **0.0302502** over 500-800. Its 162 upward transitions reflect
its ongoing oscillations. Every intervention is plotted against this same
no-intervention baseline; comparisons among interventions are supplementary.

| Intervention | LR from 500 | Maximum loss (state) | Loss at 800 | Upward steps >1e-8 |
|---|---:|---:|---:|---:|
| I4 | 0.01 | 12.504344 (505) | 0.034916366 | 58/300 |
| I5 | 0.005 | 2.5857999 (505) | 0.028191306 | 59/300 |
| I6 | 0.0025 | 0.072447798 (707) | 0.026651038 | 63/300 |
| I7 | 0.00125 | 0.028337052 (501) | 0.026909852 | 1/300 |
| I8 | 0.000625 | 0.028335351 (500) | 0.027487822 | 0/300 |

For reference, I3 at 0.02 peaks at 107.099 (504), while I2 at 0.002 peaks at
0.0283517 (501), only a small transient. I1, the freeze alone at 0.001, has no
upward loss transitions after 500. Halving is therefore a search along the
requested sequence, not a claim that 0.000625 is the largest smooth rate.

I4 and I5 markedly reduce I3's overshoot but still create large excursions.
I6 initially looks quiet, then develops the delayed peak. I7 removes that
behavior, and I8 additionally removes the tiny initial bump. I8's mean loss is
0.0278753 versus baseline 0.0210322, despite its lower endpoint; suppressing
oscillation does not uniformly improve optimization progress.

## Figure and diagnostic coverage

[Figure 01](../figures/01-frozen-first-linear-double-lr-cifar10-mlp.pdf) now retains
all eight interventions, with at most four intervention columns per page group.
Each column overlays unchanged Adam. Pages 1/2 cover I1-I4; pages 3/4 cover I5-I8.
The overview pages retain loss, eta kappa hat, full/later-layer gradient cosine
and norms, output-change cosine/RMS, and applied LR over 448-800. Companion pages
retain all five learned representations over 500-550. Axes have matching scales
across columns **and page groups**; log/symmetric-log scales are labeled.

The loss is the full 5000-example training objective. Representation reductions
retain all 5000 sample rows and all 200 hidden or 10 output coordinates. They use
outgoing differences delta_t = A_(t+1)-A_t, cosine with delta_(t-1), and coordinate
RMS. Raw gradients are not differenced. First-layer zero motion has undefined
cosine. Full gradients include all 656810 parameters; later-layer metrics use
the same 42210-parameter subset in baseline and interventions. The projected-
gradient diagnostic is not an Adam curvature or stability threshold.

All-layer rolling diagnostics cover every outgoing update 500-800; dense
representations/gradients cover states 499-551. Thus the late I6 loss excursion
is visible in the full-window overview, while the layerwise figure's 500-550
window does **not** depict its late representation dynamics. No new SVD is fitted.

## Sources, verification, compute and limitations

Runs 247-251 own the new configurations and immutable native evidence.
[01_reduce.py](../01_reduce.py) checks inventories and independently reduces
all dense representation cosine/RMS and raw/later-gradient statistics.
[02_figures.py](../02_figures.py) generates the figure and checks baseline copies.
[03_halving.py](../03_halving.py) prepares each requested rate and records the
remote stopping assessment; [04_verify_halving.py](../04_verify_halving.py)
independently verifies that decision against downloaded boundaries and compares
all full fork states exactly. [halving_summary.json](../halving_summary.json),
[comparison.json](../comparison.json), and [metrics.csv](../metrics.csv) retain
numerical results and source hashes. Previous baseline/I1-I3 arrays and summaries
are unchanged.

Each run passes exact parent qualification through 501, LR-only start-state
checks, unchanged moments/clocks, exact first-layer constancy through 801,
finite parameters, and independent full-data reductions. All 67 input identities
per run and 110 retrieved files across five runs are verified. One initial Run247
collection connection failed before archive creation; collection was retried
successfully without rerunning or changing the scientific experiment.

The five new workers took **40.62 seconds total** (1,505 intervention updates),
peak allocated GPU memory 456.7 MB each. All workers exited. Owned Spark retains
**10.034 GB** for these five run directories, including archives, with verified
local copies. No cloud resources or charges. The final PDF is rendered and
reviewed on all four pages. These exploratory results concern training on one
seed and fork; there is no validation/generalization test or global LR threshold.
