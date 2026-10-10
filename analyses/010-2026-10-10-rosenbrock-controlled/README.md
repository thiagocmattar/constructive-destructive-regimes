# Analysis010: controlled Rosenbrock optimization

Complete, 2026-10-10. Runs028-031 each executed all 10,000 deterministic CPU
updates; every update, vector history, prefix statistic and artifact hash passed
verification. Eight publication PDFs and the offline step explorer passed visual
review. No cloud execution or new billing. The user requested a controlled
Rosenbrock analogue of loss navigation with the Analysis009 collector metrics.

Open [the step explorer](rosenbrock-explorer.html) in a current browser. Select a
condition and raw gradient, stored momentum, adaptive direction, or actual
displacement. Use the slider, update field, previous/next buttons, or a time
plot to inspect any update. Enable **Nearby 100 updates** and **Zoom selected
transition** to see the late oscillations in both loss and contour motion.
Everything is embedded; a server/network connection is unnecessary for use.

[Comparison PDF](figures/controlled_comparison.pdf) ·
[Squared geometry](figures/squared_geometry.pdf) ·
[Exact temporal interaction](figures/exact_temporal_interaction.pdf) ·
[Paired norm ratios](figures/norm_ratios.pdf)

## Matched comparison and result

f(x,y)=(1-x)^2+100(y-x^2)^2, known minimum (1,1), shared start (-1.2,1),
initial loss 24.2. LR=0.01 constant, beta2=0.95, epsilon=1e-8, exact analytic
gradients without clipping, float64, both coordinates measured/decayed.
Vary first-moment memory beta1 and decoupled decay in a 2×2 design. Full
[pre-execution contract](METHOD.md), [cohort](cohort.json) and frozen per-run
configs retain the explicit design. There is no data order, batch, validation
set, stochastic noise, or experiment seed.

| Run | beta1 | Decay | Final objective | Final point | Final gradient norm |
| --- | ---: | ---: | ---: | --- | ---: |
| [028](../../runs/028-2026-10-10-rosenbrock-adam/README.md) | 0.9 | 0 | 3.45215e-6 | (1.00001486, 0.99984392) | 0.083117 |
| [029](../../runs/029-2026-10-10-rosenbrock-no-momentum/README.md) | 0 | 0 | 0.0224507 | (0.98014500, 0.97553565) | 6.57188 |
| [030](../../runs/030-2026-10-10-rosenbrock-adamw/README.md) | 0.9 | 0.1 | 0.000973108 | (0.96982082, 0.93976296) | 0.292225 |
| [031](../../runs/031-2026-10-10-rosenbrock-no-momentum-decay/README.md) | 0 | 0.1 | 0.0826439 | (0.73739621, 0.55545067) | 4.61278 |

At this LR, horizon and start, first-moment memory yields lower final objective
at both decay settings. Decay yields higher final objective at both momentum
settings. These are matched intervention outcomes for this setup, not a
general optimizer ranking or a claim that alignment mediates them.

Run028 transiently meets f<=1e-8 and gradient norm<=1e-4 at update 7,678,
with its lowest observed loss 2.21497e-9. None of the four meets both thresholds
throughout the final 100 updates. Its last 1,000 loss values range from
2.21917e-9 to 1.42761e-4: the final low loss is a point within continuing
oscillations, not sustained stationarity. Constant LR and beta2=0.95 are part
of the frozen pilot, not optimized hyperparameters.

Raw adjacent cosine is negative in 80.64% of Run028's defined pairs;
the corresponding stored-momentum frequency is 80.36%. Its late momentum
median is approximately -1, whereas Run030's is +0.6321 despite a higher objective
floor. Raw Run028 full-prefix interaction I=B²-E is positive 2.20656e6 even
though most adjacent pairs are negative. This objective makes the distinction
between local vector cancellation, long-horizon accumulation, and objective
improvement directly inspectable. A/B/ratio/norm sign or size alone does not
establish convergence, stability, information destruction, or learning quality.

## Evidence and reproduction

- [Observation001: trajectories, loss and alignment](observations/001-controlled-results.md)
- [Observation002: cumulative geometry and collector ratios](observations/002-cumulative-geometry.md)
- [Observation003: exact temporal interaction](observations/003-temporal-interaction.md)
- [Observation004: step explorer](observations/004-step-explorer.md)
- [Scientific verification](verification.json), [browser verification](ui_verification.json),
  [PDF hashes](figure_manifest.json), [explorer provenance](explorer_manifest.json)

Scientific implementation lives in Run028's 01_run.py and is shared by the four
explicit configs. 00_prepare.py creates configs once; executed inputs cannot be
overwritten. Run the implementation with a run folder as its sole argument
only before its attempt exists. 01_verify.py verifies the preserved attempts
and replays in memory without replacing raw histories. 02_plot.py generates all
eight PDFs; 04_build_explorer.py embeds the complete histories using
03_explorer_template.html. Both read verified source hashes. 05_ui_check.mjs
checks 16 condition/stream combinations, selected-step changes, undefined cases,
desktop/mobile layouts and light/dark rendering with no uncaught exceptions.
For that optional check, serve this analysis directory on 127.0.0.1:8765 using
`python -m http.server 8765 --bind 127.0.0.1 --directory analyses/010-2026-10-10-rosenbrock-controlled`
from the repository root; the standalone explorer itself works offline.

Exact replay is identical for every stored array. SciPy, autograd and finite
differences validate the analytic objective/gradient. Independent PyTorch AdamW,
fed the captured gradients, agrees at every update with maximum point absolute
error <=1.45e-15. Exact prefix sums, momentum recurrence identity, zero-vector
handling, displacement telescoping, summary reconstruction, file sizes and
SHA-256 all pass. The optimizer reference test holds gradients fixed only for
verification; the four experiments each recompute gradients along their own path.

17,133,508 bytes of ignored raw NPZ histories remain locally under the four
owning attempts; compact summaries/inventories/manifests/final states are
versioned. Scratch renders and isolated browser profiles are ignored. Original
Pythia experiments, external resources and manuscript assets were preserved.
The cohort executed in about 1.2 aggregate trajectory/save seconds; independent
checks and figures added seconds. One deterministic start/2D objective/LR/beta2
does not supply seed uncertainty or establish an LLM or asymptotic result.
