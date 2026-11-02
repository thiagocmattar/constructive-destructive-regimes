# Analysis035: backward anchored raw-gradient profiles

User-defined statistic: hold one endpoint gradient fixed and compare it with
earlier gradients, rho_t(k) = cos(g_t, g_(t-k)), k=1..256. This is distinct
from Analysis033/034's forward anchor cos(g_s,g_(s+k)) and from a moving adjacent
pair cos(g_(u-1),g_u). No averaging over endpoints is requested.

The user chose later Adam states as anchors and requested separate GD and Adam
figures in a new analysis folder. Use the same largest-source-GD starts at
s=500/2000/3000, then set the endpoint t=s+256, i.e. 756/2256/3256 for both
optimizers. The first row therefore means an endpoint at 756, not at 500.
MLP: source Run122/GD LR 0.04, Adam LR 0.0003; ResNet-32: Run120/0.1/0.001;
Transformer: Run126/0.2/0.001. No new LR calibration or source-state selection.

Run212 recomputes 256-update full-objective continuations from each identical
retained start with GD or cold Adam, temporarily storing all 257 FP32 gradients. New
endpoint dot products cannot be recovered from the former scalar profiles.
Every retrospective cosine uses exact FP64 products, independently checked
with CPU NumPy. No dimension compression, smoothing or parity split. Keep
all inputs, scalar losses/products, every gradient hash, six full audit vectors
per arm and terminal optimizer/RNG recovery locally. Dense histories are deleted
after exact backward products and audit vectors are saved, to accelerate the
requested run/retrieval. Figures use the previous 3x3 structure, common axes and one
connected curve per panel. Preserve all earlier analyses unchanged.

All nine source states are directly saved and primary. Data, ordering, losses,
parameter scope and precision match the source contracts. Transformer chunk 512
must qualify against chunk 64 and the saved gradient within 1e-5; other starts
must also pass saved-gradient qualification. Changed hardware/chunking can
change later FP32 trajectories; matched starts are not a bitwise replay claim.
The completed historical scalar comparison reports substantial later differences
in some cases. These figures describe fresh continuations from identical starts;
see `historical_comparison.json` and the report for their measured discrepancies.
One seed per model and dependent forks, without uncertainty bands. GD and Adam
have different learning rates and unmatched step norms. No stability,
generalization or optimizer-superiority conclusion follows from cosine alone.

Execution: prior Runpod/funding/cleanup authorization applies to the requested
measurements. Initial live preflight: RTX 5090 USD 1.19/h, A100 USD 1.79/h;
RTX 4090 USD 0.89/h initially unavailable, with capacity returning during launch.
Up to 13 GPUs; qualified ETC 45–90 min including retrieval, max 2 h lease.
USD 15 aggregate ceiling, graceful compute stop 12.5, hard compute stop 14
to reserve storage/retrieval margin. Each Pod has 60 GB container disk and
35 GB volume; no network volumes created. Dense gradient history uses up to
14.22 GB temporary disk per Transformer worker; only compact results,
six vectors per arm and recovery are retrieved (a few GB). Detached workers and local cost
guard; bounded read-only progress polls; per-worker byte/SHA verification
before deletion. Preserve the account's existing two unrelated volumes.

Runtime installation uses direct official downloads with the source package
versions pinned in `runtime_constraints.txt`; the abandoned environment-bundle
upload and placement qualification failures are retained in execution records.

Completed: [report](REPORT.md), [GD figure](figures/gd_backward_lag256.pdf), [Adam figure](figures/adam_backward_lag256.pdf). All task cloud deleted; compute USD9.26, provisional. [Run212](../../runs/212-2026-11-02-eos-backward-anchors/README.md).

## Manuscript-only revision, 2026-11-02

The [author-feedback observation](observations/002-manuscript-view.md) records the new compact publication
view and its unchanged source evidence. Reproduce the current manuscript
through `manuscript/draft/revisions/2026-11-02-feedback/`; earlier paper
validators refer to their historical snapshots. No training or cloud work.
