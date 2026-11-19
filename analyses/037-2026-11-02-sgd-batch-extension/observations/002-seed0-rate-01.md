# Figure 4 · Batch sampling from two GD states at one initialization

The comparison asks how sampling changes adjacent-gradient alignment when
full GD is constructive in the initial window and destructive in the
window starting after 2,000 updates. We fixed initialization 0 for each
model. The MLP uses a learning rate of 0.02 and ResNet-32 uses 0.1.
For the Transformer, the archived full-GD windows at 0.05, 0.1 and 0.2
guided rate selection before the new sampled continuations. At 0.1,
the 199 initial-window cosines were all positive and the 199 trained-window
cosines were all negative. This rate selection conditions the comparison
on the regimes of interest.

## Sources and reduction

- [Analysis 027](../../027-2026-10-30-minibatch-noise/results.json) supplies
  the 66 seed-0 MLP continuations. [Analysis 037](../results.json)
  supplies the 50 seed-0 ResNet continuations. Both use the matched GD
  states from their original measurements.
- [Run 298](../../../runs/298-2026-11-19-transformer-sgd-seed0/README.md)
  uses the prepared WikiText-2 corpus and the seed-0 initialization and
  2,000-update checkpoint from [Run 125](../../../runs/125-2026-10-17-eos-transformer-gd-lr-0.1-3k/README.md).
  Its two full-GD controls use the same 512-sequence physical accumulation
  as its 64 sampled continuations. Run 125 used physical batches of 64.
- Each continuation contributes its mean of 199 adjacent raw-gradient
  cosines over 200 applied updates. Every stochastic point averages eight
  such means. The shaded range is their minimum to maximum, conditional
  on the starting state. Each hollow full-GD marker represents one
  deterministic continuation. ResNet group counts are multiplied by 1,000
  for the image axis. Different batch sizes entail different data exposure.

[Figure generator](../26_seed0_figure.py) · [Plotted cells](../seed0_cells.csv) ·
[Publication figure](../figures/sgd_seed0_batch_size.pdf).

## Findings

| Transformer sequences | From initialization | After 2,000 GD updates |
|---:|---:|---:|
| 8 | −0.089 | −0.261 |
| 32 | −0.003 | −0.389 |
| 128 | +0.212 | −0.561 |
| 512 | +0.536 | −0.752 |
| 59,680 (full GD) | +0.999 | −0.973 |

All eight initial batch-8 Transformer trajectories had a negative mean
cosine (range −0.108 to −0.078). At batch 32 the range crossed zero.
From the trained state, mean alignment remained negative at every tested
size and weakened as the batch shrank. The smallest batches reduced the
absolute mean cosine in both windows for all three models. The sampled
trajectories explore different parameters over the 200 updates. These
curves describe alignment conditional on one initialization and cannot
estimate variation over initializations or an effect on generalization.

The [independent verification](../../../runs/298-2026-11-19-transformer-sgd-seed0/verified.json)
checks all 13,134 defined Transformer cosines for coverage, recomputes
the trajectory means, reproduces the sampling indices, and checks
endpoint displacement and selected vector audits. Figure 4 contains
28 points summarizing 182 continuations and 36,218 within-window cosines.
