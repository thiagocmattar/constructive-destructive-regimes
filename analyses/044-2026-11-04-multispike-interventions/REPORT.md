# Multi-spike intervention results

Single seed, fixed full-objective windows. Historical controls are reused only where recorded qualification passes; otherwise comparisons use the fresh same-worker control. No historical intermediate-tensor identity is claimed.

| Case | Policy | Mean loss change | Final loss change | >10% excursions | Censored |
|---|---|---:|---:|---:|---:|
| mlp | unchanged | +0.00% | +0.00% | 16 | 1 |
| mlp | fixed_half | -25.04% | -64.54% | 0 | 0 |
| mlp | cosine32_half | -26.30% | -65.26% | 1 | 0 |
| resnet | unchanged | +0.00% | +0.00% | 4 | 0 |
| resnet | fixed_half | +27.20% | +712.54% | 4 | 0 |
| resnet | cosine32_half | +0.62% | +254.25% | 3 | 0 |
| transformer | unchanged | +0.00% | +0.00% | 5 | 1 |
| transformer | fixed_half | -9.66% | -25.12% | 2 | 0 |
| transformer | cosine32_half | -10.68% | -25.57% | 2 | 0 |

All three figures show complete fixed windows, raw diagnostics, solid unsmoothed curves and independent Y ranges. Smaller oscillations alone do not establish better training progress. A finite window can leave delayed events censored.

Verified 8896 source-derived schedule/diagnostic checks and retained 47456 plotted values. Every plotted scalar was checked against its source; vector PDFs were rendered for final visual review.

Final visual review passed for all three vector PDFs. See FINDINGS.md for nine conclusions, censoring caveats and cost/cleanup verification.
