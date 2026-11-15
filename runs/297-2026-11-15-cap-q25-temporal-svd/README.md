# Q25 versus unchanged Adam: temporal representation SVD

Completed: all three model cases, both arms, 384 production updates and 390
full-objective/probe states. [Observation017](../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/017-q25-representation-svd.md)
contains the interpretation and figure captions; [closeout.json](closeout.json)
records integrity, runtime and retention.

Full-output rank95 changes from 6 to 10 (MLP), 26 to 19 (ResNet), and 15 to 32
(Transformer). Squared-motion ratios are 0.01664, 0.77229 and 0.09635. ResNet
increases near the input but decreases deeper in the network; higher rank95 is
not a universal consequence of the cap. Concentration metrics also disagree at
some sites, so the report preserves participation ranks and leading-mode shares.

All 114 compact checks, nine direct SVD/projection checks and 81 original-source
dependency checks pass. Four PDFs / ten pages are rendered and reviewed.
Capture durations were 9.51 seconds, 35.62 minutes and 78.22 minutes. Production
evidence accounts for 67.204 GB; the complete retained task directory, including
qualification captures and compact archives, is 70.434 GB on owned Spark.
All task workers finished successfully. No cloud resources or charges; no
manuscript/consolidation edits. MLP reproduces its historical trajectory exactly;
ResNet and Transformer are fresh matched Spark continuations with measured
historical drift.

## Executed protocol

H053 / ongoing Analysis065. User authorized analysis on owned Spark, MLP first,
then ResNet and Transformer; no manuscript/consolidation edits or cloud usage.

Retained Runs293-295 have full fork/audit/recovery states and scalar histories,
but no complete dense representation histories. Reconstruct the first 64
post-fork updates for unchanged Adam and distributed Q25 with unchanged training
objectives, optimizer state, calibration, precision, probe, and update indexing.
Both arms run on the same Spark runtime. Compare original scalar histories and
the retained state at fork+32; label deviations explicitly rather than asserting
cross-hardware exact reproduction. Fork and first-step qualification precede
production. These are finite matched continuations, not a new hyperparameter
search or an extension of the original training horizon.

Capture states f through f+64. Incoming changes f+1 through f+64 form a
64-by-P matrix per measured site, with no centering or normalization. P contains
every sample/position/feature coordinate in the original probe: all 5000 MLP
images, one intact ResNet 1000-image normalization group, or 64 Transformer
sequences / 2240 positions. Include all original 5/6/6 stage streams plus
correct-label log probabilities for cross-entropy models. Do not average features
or samples before SVD. Same-state repeat probes must be exact and leave RNG and
model buffers unchanged.

Use exact FP64 temporal Grams and cross-arm Grams. Report squared singular-value
shares, ranks95/99, participation ratio, entropy effective rank, top1/top5/top10,
absolute squared motion and coordinate RMS, adjacent reversal and displacement
retention. Compare first/second 32-update windows and the complete 64-update
window; equal lengths within each comparison avoid rank-ceiling artifacts.
Evaluate Q25 movement inside/outside the unchanged baseline's fixed right
singular subspace covering 95% of its energy. More effective modes can reflect
attenuated dominant modes rather than new directions; low-rank motion is not
low representational capacity or less learning.

Retain FP32 activation histories, full starting/ending states, scalar trajectories,
source hashes, temporal and cross Grams, and independent reduction checks on
Spark. Download compact verified artifacts and figures locally. Numerical tests
compare Gram results with direct SVD, and validate cross-subspace projections.
Budget: owned machine, zero cloud cost; maximum 3 hours per architecture,
bounded worker with progress events and recovery every 16 updates. Calibration
provides the estimate; the upper bound is not an expected duration. Storage
estimate was below 100 GB; the measured closeout above is within that estimate.

This predeclared early window is not the historical late ResNet spike window.
Results are local to one initialization and these post-fork intervals.
