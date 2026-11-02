# Reduced Adam calibration: one initialization, no Hessians

The user requested calibration first, as fast as practical, using Analysis034
to reduce the search. This supersedes the unexecuted 21-arm proposal.

| Case | Constant Adam learning rates | Updates per run | Runs |
| --- | --- | ---: | ---: |
| CIFAR-10 / tanh200-200 MLP, first5k images | 1e-4, 3e-4, 1e-3 | 128 | 3 |
| CIFAR-10 / ResNet-32, all50k images | 3e-4, 1e-3, 3e-3 | 128 | 3 |
| WikiText-2 / two-layer Transformer, all2,088,600 prepared targets | 3e-4, 1e-3, 3e-3 | 128 | 3 |

There are nine independent runs and at most1,152 optimizer updates: 57% fewer
conditions and 79% fewer updates than the original21 x256 proposal. Each arm
starts at the exact same Analysis024 seed0 frozen weights for its case, with
fresh zero Adam moments. No extra seeds, Hessian/HVP calculations, clipping,
weight decay, warmup, LR schedules or within-run LR changes. Adam betas are
(0.9,0.999), epsilon1e-8. Full batch fixes batch sampling, while conclusions
remain conditional on the chosen initialization.

## Why these rates and 128 updates

Analysis034 selected3e-4 for MLP; its1e-3 candidate violated the continuation
safeguard at some primary starts. Keep1e-4 as the lower reference and1e-3 as a
stress case. ResNet/Transformer improved through the upper tested1e-3; retain
that reference, compare3e-4 and probe3e-3 above the old grid. Drop1e-5/3e-5 and
the initial1e-2 stress tests. This is prior-informed grid selection, not proof
that the same regimes transfer from trained states to initialization.

The first64 states can be startup transients. A128-update pilot gives a final
64-state window beyond that interval, at half the original duration. Report raw
losses, all pointwise geometry and the declared training-loss labels. A128-step
label is provisional; absence of a desired regime is a calibration outcome,
not permission to silently change LR or extend the experiment. Any unresolved
bracket is reported before a longer run or new rate is proposed.

## Measurements and next stage

Capture every full-objective gradient and all four streams: raw gradients,
stored first moments, LR-free adaptive directions and actual parameter steps.
Keep norms, consecutive cosines, exact prefix A/B/B-over-A and energy/cross-term
reductions, incoming-gradient/prior-moment alignment, bias-corrected moment
norms, projected gradient change q and actual-motion/rounding audits. The
terminal gradient is measured but unapplied. No dense history is retained.

Calibration is training-only: no validation/test passes or Hessian work.
Recover at64 and the final128, retaining model, Adam, RNG and diagnostic state.
The final64 training losses supply the provisional smooth/oscillatory labels;
cosine does not define those labels. Preserve failures and loss-guard stops.

After the results are reviewed, the main study can use up to three fixed-rate
regimes per case, initially1,024 updates each. Rates and the suitable horizon
are decided from calibration; no main run launches automatically. The same
single initialization and no-Hessian constraints apply. Full-prefix reuse
requires qualified continuation of saved Adam/diagnostic state.

## Execution bounds

Up to nine independent workers, preferably one per arm. Keep hardware class
matched across rates within a case. If fewer matching GPUs exist, distribute
that case's three arms over its available matching workers. Live quoted
Secure RTX PRO5000 price: USD0.96/h; fallback GPU price <=USD1.19/h. The earlier
4090/5090 catalog entries were unavailable at this preflight. Actual placements
and times are recorded by the controller, not inferred from old resource IDs.

Qualified estimate30--75min, maximum90min including retrieval; aggregate
calibration cap USD20 inside the prior USD50 study quote. An independent local
cost guard requests a boundary stop at estimated USD14 or15min before deadline;
at USD18.5 or deadline it stops task compute and preserves task volumes if
verified recovery is not yet complete. Native CLI3.0.1 has no create-time expiry
flag in its live help, so this separate guard is required. Both controller and
guard are detached hidden processes. Work on each pod is also detached.

Each worker has60GB container disk and20GB pod volume; no network volume is
created. Official PyTorch2.11/cu130 image is pinned by registry digest in the
lease. Only SSH port22 is exposed. Input archive and every scientific file are
hash checked. Monitor progress read-only, capture final byte/SHA inventories,
download and verify each archive and file before deleting its worker and disk.
Pre-existing account pods/volumes are untouched. Report final retained storage
and provisional billing. Preparation, setup, partial failures and retries count
toward the same envelope.

## Realized execution

All9 arms completed and pass cohort verification. Actual placements were three
active RTX5090 workers plus two never-ready containers (deleted before scientific
input delivery); MLP ran on the owned laptop. Unchanged missing arms were placed
on warm matching workers. Setup, training, retrieval and cleanup took about42min;
Runpod compute/storage estimate USD2.26, provisional. All task cloud storage is
deleted and4.323GB of raw evidence is retained locally.
See [REPORT.md](REPORT.md) and [execution_receipt.json](execution_receipt.json).
No sustained loss-oscillation bracket was found; raw-gradient reversal was found
in ResNet and Transformer. No main runs or automatic extra rates were launched.
