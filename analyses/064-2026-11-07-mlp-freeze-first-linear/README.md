# Analysis 064: freezing the first MLP projection

H044 / Run 244. Compare the qualified retained Adam control with a continuation
from state 448 in which only first-linear weights and biases remain fixed.
All other learning-rate, optimizer, data and precision choices are matched.
Primary interval 450-500; extended follow-up 448-800. Raw gradients are not
differenced, and representation probes retain all 5000 examples.

The experiment asks whether suppressing first-projection motion suppresses the
observed downstream oscillation and loss excursion, or whether later layers
continue oscillating with a stationary first representation. A beneficial
intervention would not establish a unique first-layer origin: gradients and
Adam state are coupled, and this block contains 93.6% of model parameters.

## Completed result

Freezing strongly suppresses downstream oscillation. Over450-500, output reversal
falls from 51/51 to 4/51, maximum output RMS falls 87.38%, and loss at 474 falls 46.43%.
First-layer weights/biases and their representations remain exactly fixed;
later layers continue updating with unchanged Adam settings. The longer448-800
mean training loss is 22.27% higher, so suppressing spikes does not uniformly
improve optimization progress. No unique initiating mechanism is established.

- [Observation, interpretation and caveats](observations/001-freeze-first-linear.md)
- [Two-page comparison figure](figures/01-freeze-first-linear-cifar10-mlp.pdf)
- [Numerical results and provenance](comparison.json), [window metrics](metrics.csv)
- [Independent run verification](../../runs/244-2026-11-07-mlp-freeze-first-linear/verification.json)

Reproduce local analysis with `01_reduce.py`, then `02_figures.py`, using the
verified retained Run 244 evidence. All 5000 examples and every declared feature
coordinate are retained. Full-state control qualification, zero-update/zero-motion
checks, source/transfer hashes and numerical reductions pass; both PDF pages
were rendered and reviewed. Worker exited, about 2.067GB retained on owned Spark,
no cloud charges. No new architecture or optimizer intervention beyond the freeze.
