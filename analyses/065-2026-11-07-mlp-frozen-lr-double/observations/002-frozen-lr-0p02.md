# I3: a 10x increase over Intervention 2 causes severe overshoot

## Conclusion

With first-linear weights/biases still frozen, **Adam LR 0.02 from state 500**
causes a large loss excursion: 0.028335 at 500, 1.33964 at 502, and **107.099 at
504**. Loss subsequently recovers to **0.308132 at 800**, still 10.19x unchanged
Adam and 11.71x I2. All values remain finite. This rate increase does not improve
training over the measured horizon.

The first layer is exactly stationary. Therefore freezing it does not protect
the remaining trainable layers against arbitrarily large learning rates.
This does not negate suppression of the original oscillation at LR 0.001/0.002,
nor locate a unique initiating mechanism. It demonstrates that later-layer
updates can themselves produce a large excursion under the stronger intervention.

## Complete intervention and matched comparison

I3 inherits I1's first-layer freeze from state 448 and sets LR to 0.02 at state
500, **10x I2's 0.002 and 20x baseline's 0.001**. It branches from the same
qualified frozen state 500 as I2; it does not continue from I2's final state.
First changed update: 500 -> 501; final state: 801, supplying outgoing diagnostics
through 800. No reset, unfreeze or additional change to Adam, data, seed, objective,
precision, batch reduction, or measurement coverage. Full settings and definitions
are in [observation 001](001-frozen-lr-double.md) and
[Run 246](../../../runs/246-2026-11-07-mlp-frozen-lr-0p02/README.md).

The primary comparator is always **unchanged Adam**. The same verified baseline
is reproduced in every figure column; I1/I2 provide supplementary dose comparisons.
All 5000 images, 200 coordinates per hidden representation, and 10 raw output
coordinates are retained, with no averaging across samples before motion metrics.

## Results

| Quantity | Unchanged baseline | I1: freeze | I2: freeze + 0.002 | I3: freeze + 0.02 |
|---|---:|---:|---:|---:|
| Mean loss, 500-550 | 0.0289606 | 0.0281841 | 0.0280714 | 8.75062 |
| Maximum loss, 500-550 | 0.0550626 | 0.0283354 | 0.0283517 | 107.099 |
| Loss at 550 | 0.0273520 | 0.0280555 | 0.0278545 | 0.749408 |
| Mean loss, 500-800 | 0.0210322 | 0.0276774 | 0.0272261 | 1.79231 |
| Loss at 800 | 0.0302502 | 0.0271296 | 0.0263062 | 0.308132 |
| Maximum output RMS, 500-550 | 0.155872 | 0.00140750 | 0.00358001 | 6.38532 |
| Negative output cosines, 500-550 | 49/51 | 7/51 | 36/51 | 5/51 |

I3's initial motion grows enormously: maximum coordinate RMS is 2.42807 for
second-linear preactivations, 0.918516 for the second tanh representation, and
6.38532 for outputs. First-linear and first-tanh motion stays exactly zero.
Only 2/51 second-hidden changes and 5/51 output changes have negative cosine
over 500-550. Much subsequent recovery moves forward in representation space.
**A low reversal count does not imply a good trajectory:** amplitude and loss
are essential, and the large initial excursion dominates this window.

This tests one requested rate at one deterministic fork; there is no continuous
LR search or threshold estimate, and no validation/generalization assessment.
Recovery after the excursion must not be mistaken for improvement over baseline.

## Figure, source evidence and verification

[Figure 01](../figures/01-frozen-first-linear-double-lr-cifar10-mlp.pdf) now has
three intervention columns, each overlaid with unchanged Adam. Page 1 retains
the eight diagnostics, including eta kappa hat and full/later-layer gradient
norms, over 448-800. Common log scales for loss, norms, output RMS and LR show
the large dynamic range; the projected-gradient diagnostic uses a symmetric-log
scale linear within +/-1. Cosines remain linear on [-1,1]. Page 2 retains all
five learned representations over 500-550, with matched per-layer RMS scales
linear within +/-0.0001 and logarithmic above. Zero motion remains visible;
its undefined cosine is explicitly labeled. No new SVD is fitted.

[01_reduce.py](../01_reduce.py) verifies sources and independently reduces dense
full-data representations and gradients. [02_figures.py](../02_figures.py)
generates the PDF and checks that every column carries identical baseline values.
[comparison.json](../comparison.json) pins raw inventory, configuration,
qualification, retrieved boundaries/representations and prior evidence;
[metrics.csv](../metrics.csv) records all declared windows. At state 500, the
outgoing displacement already uses the new LR even though the state loss and
gradient still match I1/I2.

Run 246 exactly qualifies frozen-parent states/gradients 448-501 and all five
representations at 499-501. The starting full state differs only in LR; moments
and clocks are unchanged. The first actual later-layer increment scales 20x
relative to the LR-0.001 parent within 1.49e-7 absolute FP32 rounding error.
First weights/biases remain exact through 801; all other parameter tensors
change, clocks reach 801, and parameters remain finite. Independent dense
cosine/RMS and gradient reductions pass; all 67 source and 22 transfer identities
pass; both final PDF pages are rendered and reviewed.

Worker runtime 7.96 s, including 6.64 s for 301 intervention updates; peak GPU
allocation 456.7 MB. Worker exited. Owned Spark retains 1.911 GB for Run 246,
including the archive, with verified local copies. No cloud resources or charges.
