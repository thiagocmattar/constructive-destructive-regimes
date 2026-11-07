# GD learning-rate extension from the same Adam state

Historical comparison of0.02/0.04/0.08. Its figure, generator and results are
preserved in [this snapshot](../archive/2026-11-07-rates-0p02-0p04-0p08/README.md).
The current figure follows [Observation003](003-intermediate-gd-rate.md) with
columns0.02/0.03/0.04. All results below remain unchanged; relative output
references below describe the archived comparison.

User request7 November2026: run LR0.04, then0.08 on Spark and regenerate the
existing Analysis060 PDF with one column per LR. Run234 owns both new branches;
Run233's0.02 and Run223's Adam control remain unchanged. Generator:
`../01_report.py`; numerical reductions and SHA provenance: `../results.json`;
every plotted scalar: `../plotted_values.csv`. The original figure and outputs
are preserved in `../archive/2026-11-07-single-rate/`.

## Comparison and coverage

All branches load the original Run223 untreated state512, independently of each
other and of beta2 treatments. Same656810-parameter MLP, seed0, pinned first5000
CIFAR-10 training images, full MSE objective, physical1000-image partitions,
weights/RNG and FP32 arithmetic; diagnostics reduce whole-model vectors in FP64.
GD is native SGD with zero momentum and weight decay, no clipping or adaptive
scaling. Continued Adam uses(.9,.999),LR0.001. Exact fork-gradient and two-update
Adam replay qualify control reuse; each new GD rate passes2 versus1+1 exact
resume for model, optimizer, RNG and gradient history before its scientific run.

LR0.04 completes all288 outgoing updates512..799 and289 boundaries512..800.
LR0.08 completes35 outgoing updates512..546; finite loss/gradient diagnostics
cover512..546. The next evaluation, at547, returns infinite loss and the worker
stops. Its last periodic progress event was544; the terminal status establishes
547 as the actual failed boundary. Replaying the three recorded updates from
recovery544 exactly reproduces loss/norms544..546 and infinite loss547. Parameters
and gradients at547 remain finite; this is FP32 objective overflow. The update
at547 is never applied. No post-failure continuation or replacement run.

Primary full-window mean uses resulting states513..800, tail mean769..800,
with final/max losses and fixed pre-rise >10% excursions. The full10000-image
test objective at800 is secondary. Only complete branches get these summaries.
LR0.08 has no full-window mean/tail/endpoint/test result. Its finite-prefix event
count is censored at546 and must not be compared as equal-duration evidence.

## Results

| Outcome | Adam0.001 | GD0.02 | GD0.04 | GD0.08 |
| --- | ---: | ---: | ---: | --- |
| Mean training loss513..800 | 0.0208042 | 0.0212831 | 0.102867 | unavailable |
| Last32-state mean | 0.0134585 | 0.0209599 | 0.0447293 | unavailable |
| Training loss at800 | 0.0302502 | 0.0209284 | 0.0588509 | unavailable |
| Maximum training loss513..800 | 0.0550626 | 0.0220705 | 1.61503 | unavailable |
| >10% excursions | 16 (1 censored) | 0 | 1 (censored) | 1 (shorter prefix, censored) |
| Loss-increasing finite transitions | 161/288 | 0/288 | 96/288 | 30/34 |
| Full test loss at800 | 0.776960 | 0.705448 | 0.684493 | unavailable |

LR0.04 raises mean/tail loss394.45%/232.35% versus retained Adam; endpoint
loss94.55% and maximum2833.08% higher. It has a large initial excursion and
later oscillations, without returning below the fixed pre-rise reference by800.
Thus the event detector reports one long censored excursion, not a separate
event for every local peak. LR0.08 reaches6.0501139e34 at546 before overflow.

All applied increments still have negative raw-gradient dot product, including
the explosive branch. This verifies local descent direction, not observed loss
decrease at finite LR. Lower endpoint test loss for0.04 is a single-seed
secondary observation and does not establish generalization. No confidence
intervals: one selected seed/checkpoint/window with dependent time points.
Within-GD rate comparisons keep optimizer fixed; comparisons against Adam
jointly change LR and optimizer. No universal stability threshold is identified.

## Figure caption

CIFAR-10/MLP: continue Adam at LR0.001 (gray) or switch from the same warm
state512 to plain full-batch GD (blue), with columns LR0.02,0.04,0.08. Rows:
full training loss, projected gradient change
<g_(s-1),g_(s-1)-g_s>/||g_(s-1)||^2, adjacent raw-gradient cosine,
raw-gradient norm and applied outgoing LR. Shared context448..512 is shaded.
Loss and raw norm use log scales;0.02/0.04 share y ranges, while0.08 has separate
ranges to preserve its divergence. The red dotted line marks loss overflow547;
finite diagnostics end546 and the last recorded LR interval closes at547.
Complete LR intervals close at800. No smoothing or post-failure extrapolation.
The projected-change expression is descriptive across optimizers, not a
universal stability test; the adjacent pair at512 includes Adam prehistory.

## Verification and retained evidence

Run234 `03_verify.py` passes all42 pinned input identities and68 artifact
size/SHA checks, exact control replay and both GD resumes. The first updates
match theta-LR*g to maximum coordinate rounding below1.00e-8. Every increment
obeys norm/dot bounds using its measured FP32 rounding residual. For0.08 the
absolute residual grows with exploding vectors, so verification uses a
scale-aware FP32 bound, not the small absolute tolerance from the0.02 case.
These post-run verification changes do not alter executed inputs or evidence.
`04_audit_failure.py` and `failure_audit.json` authenticate the overflow from
the retained recovery checkpoint. Executed shared helper/config/worker are
frozen in the input manifest and source archive; original Run233 is unchanged.

All9315 plotted scalars match source rows, and all3528 old values plus original
summary/event details match the archived single-rate result exactly. Archive
generator/PDF/CSV hashes are checked. One7.16 x7.1-inch vector PDF with embedded
fonts passes text-bound, line-array and visual-render checks. Scratch PNG is
ignored under `tmp/pdfs/analysis060/`.

Run234 worker4.159s:0.04 scientific branch2.623s,0.08 branch0.352s, plus10
qualification updates. Three extra replay updates audit the failure without
changing the original artifacts. Peak CUDA allocation457,576,960 bytes;
end-of-branch throughput about111 and last0.08 progress about99 updates/s.
Worker exited. All68 raw files (205,710,082 logical bytes) and123,726,679-byte
archive verified locally; Spark retains261,300,224 physical bytes in Run234,
including its archive. First-step/recovery/final files retained where reached;
no final800 checkpoint exists for0.08. Owned Spark, zero cloud cost or cleanup
of unrelated resources. No manuscript edits or publication.
