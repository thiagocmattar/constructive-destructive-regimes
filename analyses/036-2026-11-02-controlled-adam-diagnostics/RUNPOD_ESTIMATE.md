# Parallel Runpod estimate, November 2, 2026

Historical estimate for the original 21 x256 screening proposal. The user's
later calibration-first instruction reduced this to9 x128; see
[FAST_CALIBRATION.md](FAST_CALIBRATION.md) and the eventual calibration-closeout.json
for current scope and realized execution. Main-study timing remains provisional.

Plan for approximately **4--8.5 hours end to end and USD 18--36 expected**,
with a **USD 50 suggested aggregate ceiling** for calibration plus the proposed
1,024-update main study. Calibration alone: about **1--2.5 hours, USD 8--16**,
with a suggested USD 20 ceiling. These are planning scenarios, not guaranteed
completion bounds or launch authorization. No billable resource was created.

| Phase | Work | Parallel wall time | Expected cost including temporary disk |
| --- | --- | --- | --- |
| Calibration | 21 arms, 256 full updates each | 1.2--2.4h | USD 7.6--15.4 |
| Main study, additional | 9 arms, 1,024 full updates each | 2.7--5.9h | USD 10.2--20.5 |
| Both phases | 14,592 full updates, plus terminal/evaluation passes | 3.9--8.3h | USD 17.8--35.8 |

Use up to seven independent single-GPU workers. Each screens the three cases
at one LR, starting each arm from the same initial tensors and fresh Adam state.
After all screening outcomes are available, select the declared regime rates.
Then run three Transformer and three ResNet conditions concurrently; execute
the short MLP arms on ResNet workers. The seventh worker can be released after
calibration. Retrieve, verify and delete each completed worker promptly. Holding
the whole fleet idle until the longest arm ends would increase the cost.

## Live price and capacity

The authenticated, read-only catalog query through runpodctl 3.0.1 lists Secure
on-demand RTX 4090 at **USD 0.89/GPU-hour**, available with **Low** stock in
US-NC-1 in the initial read. RTX 5090 is USD 1.19/h; its availability fluctuated
from unavailable to Low stock during this estimate. The catalog provides no
guarantee that seven workers can be obtained. With fewer workers, queues
increase wall time; total GPU work is similar, while setup and idle costs vary.
The quote uses Secure 4090 pricing throughout rather than an assumed cheaper
Community offer. The installed CLI matches the latest v3.0.1 release at this read.

Sources: [Runpod GPU pricing](https://www.runpod.io/pricing) and
[Pod compute/storage billing](https://docs.runpod.io/pods/pricing).
The exact catalog timestamp and returned records are in
[runpod_estimate.json](runpod_estimate.json). Storage arithmetic assumes
100GB running disk per worker at USD 0.10/GB/month, prorated with a 30-day month.
No network volume or ongoing cloud storage is included; all task disks are
deleted after verified retrieval. Final billing can differ from this estimate.

## Timing evidence and assumptions

The 4090 Analysis034 arm records give approximately 0.04--0.06s/update for
MLP, 2.54--2.66s for ResNet, and 6.30--6.33s for Transformer. These include
the former capture and arm checkpoint work. The earlier Transformer used
chunk512, whereas the new scientific config preserves chunk64; simply
multiplying 6.3s by the proposed updates would not account for this difference.

A full-objective mechanics profile on the owned RTX5070Ti Laptop measures
15.86/16.05s for two post-warmup Transformer updates at chunk64 with the actual
four-stream capture. It covers all 59,680 sequences / 2,088,600 targets per
update. This is a small local timing sample, not a Runpod performance result,
sustained thermal benchmark or LR-regime observation. A separate local chunk512
attempt pressed against VRAM capacity and was stopped; its timing is excluded
from any transfer ratio. No scientific config or learning-rate schedule changed.

Use explicit new-workload planning assumptions of 0.05--0.10s for MLP,
2.8--3.5s for ResNet and **8--16s for Transformer** on the proposed 4090s.
The Transformer range is a conservative extrapolation from two different
hardware/partition references; it is not an empirical interval measured on the
target cloud setup. Recheck it with a short production-shaped cloud timing
inside any later authorized envelope before committing the fleet to the study.

The estimate adds 10--20% per-arm work for evaluation/checkpoint overhead,
15--30min setup per initial worker, 5--10min main-phase configuration per reused
worker, and 6--15min per-worker retrieval/verification allowance. Parallel
critical-path time and aggregate GPU-hours are calculated separately. Shared
download bandwidth, installation failures or slow storage can exceed these
allowances. The USD 50 suggestion leaves approximately USD 14 above the upper
planning scenario for such contingencies; an aggregate guard would still stop
work before exceeding an authorized cap.

The scope is one seed, the current seven-rate grid, and three proposed regimes
per case through 1,024 production updates. It assumes the grid spans the desired
regimes; missing regimes require a revised plan. Extra seeds, HVP audits, grid
expansion, or production beyond 1,024 updates are outside this quote.

Reproduce with 04_estimate_runpod.py after a mechanics profile using Run213's
00_profile_timing.py. The calculator performs read-only live catalog calls and
records source hashes, timing samples, assumptions, scenario arithmetic and
capacity caveats in runpod_estimate.json. Raw mechanics outputs stay ignored in
the run's scratch folder. No pods, templates, endpoints or volumes are created
or changed by either script.
