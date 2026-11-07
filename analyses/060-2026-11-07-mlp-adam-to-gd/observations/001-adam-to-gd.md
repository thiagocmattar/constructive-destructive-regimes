# Plain GD after the warm Adam fork

Source: Run233 attempt01, owned Spark GB10, plus retained Run223 unchanged
same-Spark control. Generator `../01_report.py`; full reductions, event lists
and source/PDF hashes in `../results.json`, plotted values in
`../plotted_values.csv`. Independent checks: Run233 `03_verify.py` and
`verification_final.json`; source/cache/checkpoint manifest pins41 files.

## Design and coverage

The user requested switching Adam to GD at high LR0.02, following the beta2
experiment. This is a separate branch from original untreated Run223 state512,
not a continuation of any beta treatment. Same MLP weights, seed0, fixed5000-image
CIFAR-10 full MSE objective, original ordering, physical1000 partitions and RNG.
Use native SGD with momentum0, no decay or clipping: ordinary full-batch GD.
FP32 gradients/weights, FP64 diagnostics. Adam moments are not used by GD.

Counterfactual is continued Adam(.9,.999),LR0.001 through800. This comparison
jointly changes optimizer and LR; it cannot identify an optimizer-only effect.
Exact fork-gradient and two-step Adam reference replay pass. GD two consecutive
updates equal one update followed by a saved-state resume, for model/optimizer,
gradient history and RNG. No full baseline or prefix rerun;288 scientific GD
updates,2 Adam qualification updates and4 GD qualification updates.

States512..800 have full training loss and gradient diagnostics; the first
adjacent pair at512 includes the preceding Adam gradient. Outgoing updates
512..799 use LR0.02. Complete10000-image test evaluation occurs at800.
Mean reduction uses all288 resulting states513..800; tail is769..800. No
smoothing, aggregation across seeds, truncation or event retuning.

## Results

| Outcome | Continued Adam | Switch to GD | Relative change |
| --- | ---: | ---: | ---: |
| Mean training loss513..800 | 0.0208042 | 0.0212831 | +2.30% |
| Last32-state mean | 0.0134585 | 0.0209599 | +55.74% |
| Endpoint training loss | 0.0302502 | 0.0209284 | -30.82% |
| Maximum training loss513..800 | 0.0550626 | 0.0220705 | -59.92% |
| >10% excursions with post-fork peaks | 16 | 0 | descriptive counts |
| Training-loss-increasing updates | 161 | 0 | all288 updates |
| Endpoint test loss | 0.776960 | 0.705448 | -9.20% |

GD decreases the observed training loss on all288 updates and removes the
selected-window spikes, while progressing slowly near0.021. Its mean is
slightly above Adam and its tail mean substantially above Adam. The endpoint
comparison favors GD partly because Adam ends inside an excursion. The lower
single endpoint test loss is a measured secondary outcome, not replicated
generalization evidence. Smoother raw diagnostics do not establish faster
optimization, convergence or a universally preferable optimizer.

Excursions use the established fixed pre-rise reference on complete0..800
histories, selecting peak>512 and peak/pre-rise>1.1. Adam's16 events comprise
15 recovered and1 censored; GD has none. No uncertainty bands: one selected
development seed/state, dependent time points and inherited historical control.

## Figure caption and provenance

CIFAR-10/MLP: continue Adam at LR0.001 (gray) or switch to plain full-batch GD
at LR0.02 (blue) from identical warm state512. Rows show full training loss
(log scale), projected gradient change, adjacent raw-gradient cosine, raw-gradient
norm and actual applied LR. Gray shading covers shared pre-fork context448..512;
all curves are unsmoothed through800. LR intervals are closed at800 only for
display. The projected-change expression is the same in both streams; its
interpretation depends on the optimizer and is not a universal stability test.

All3528 plotted scalar values match source records. The single-page7.16 x7.4-inch
PDF uses embedded fonts/vector curves; full-size render inspected, with legible
labels, legend and full LR range. Scratch PNG remains ignored under
`tmp/pdfs/analysis060/`.

## Verification and execution

All39 artifact files and the archive are size/SHA-256 verified. Exact control
and GD resume checks pass independently on CPU. First actual GD update matches
theta-0.02*g within7.20e-9 maximum coordinate rounding. All288 gradients have
negative dot product with the actual increment. Because small FP32 parameter
increments are rounded, gradient/actual-step cosine is not literally-1:
maximum-0.999997582, largest relative vector residual0.002199. The initial
overly strict cosine assertion was replaced with norm and dot-product error
bounds from the measured rounding residual; no scientific input was changed.
Largest absolute residual norm4.42e-7. Full endpoint gradient/checkpoint and
empty SGD optimizer state are verified. No momentum or Adam state remains in
the GD update rule. An inherited scalar-conversion warning is retained in
the log; it does not alter the qualified update path.

Worker3.73s including qualification; main branch2.78s. Peak CUDA allocation
0.458GB; zero cloud resources or charge. Worker exited. First-step, recovery,
final checkpoints and raw logs retained locally and on Spark; remote run folder
occupies147,365,888 bytes including its transfer archive. Unrelated processes
and project files remain unchanged.
