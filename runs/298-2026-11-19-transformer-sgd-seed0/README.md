# Run 298 · Transformer batch-size continuations

Repeat the Transformer arm of Analysis 037 at learning rate 0.1, with the
archived seed-0 initialization and the 2,000-update GD state from Run 125.
This learning rate was selected after inspecting the historical full-GD
cosines in the two 200-update windows. The stochastic outcomes are new.
Use four sampled sequence counts (8, 32, 128, 512), eight repetitions per
count, and one 59,680-sequence full control at each start. Keep the Run 216
physical accumulation, sampling procedure, geometry, endpoint losses and
200-update horizon. Each trajectory supplies 199 within-window cosines.
Reuse the seed-0 MLP and ResNet arms from Analysis 037.

Run `00_prepare.py` at the repository root to verify the historical inputs,
package both starts and build the worker archive. On an extracted worker
run `01_run.py --deadline <unix-time>`. It executes the Run 214 measurement
code with this run's config and writes restartable outputs under artifacts.

All 66 continuations completed. Independent verification of 13,200 updates,
13,134 within-window cosines and 548 inventory entries (8.370 GB) passed.
Run `02_verify.py` after retrieval and the Analysis 037 figure generator
after verification. The [observation](../../analyses/037-2026-11-02-sgd-batch-extension/observations/002-seed0-rate-01.md)
records outcomes and interpretation.

One Secure Cloud RTX 5090, PyTorch 2.8.0, CUDA 12.8, NumPy 2.5.0 and SciPy
1.18.1. The pod was allocated at 11:16:02 UTC and deleted after verified
retrieval at 12:14:02 UTC on 19 November 2026. The 58-minute lease at
USD 1.39/hour implies USD 1.34 GPU rental before temporary disk charges.
Runpod had posted no billing record for the pod at closeout. No network
volume was created. Raw evidence is retained locally.
