# Analysis012: Rosenbrock AdamW across learning rates

Completed and verified, 2026-10-10. **[Open the single comparison figure](figures/rosenbrock_learning_rates.pdf).**
Same layout and diagnostics as Analysis011: a large full-domain contour and
trajectory panel above eight metric panels, all three LRs overlaid, no zoom
or thinning. Panels (c,d) show 100-step trailing means of adjacent cosines;
panels (e,f,g) use linear Y axes. Loss retains its log Y axis, and ratios are B/A.
The first complete cosine window appears at update 101; other series are unsmoothed.

AdamW with momentum, **weight decay 0.1**, betas (0.9,0.95), epsilon 1e-8,
CPU float64, exact Rosenbrock f(x,y)=(1-x)^2+100(y-x^2)^2, start (-1.2,1),
zero initial moments and no clipping. Learning rates are constant; weight
decay is decoupled coordinate shrinkage, not a learning-rate schedule. Every
condition runs exactly 2,000 updates. Drop 1e-4 and use the geometric midpoint
sqrt(1e-3*1e-2)=0.0031622776601683794 between the retained endpoints.

| LR | Run | Final objective | Final point | Final gradient norm |
| --- | --- | ---: | --- | ---: |
| 1e-2 | [035](../../runs/035-2026-10-10-rosenbrock-adamw-lr-1e-2/README.md) | 0.00104246819 | (0.969238783, 0.940404714) | 0.483408 |
| 3.162277660e-3 | [036](../../runs/036-2026-10-10-rosenbrock-adamw-lr-midpoint/README.md) | 0.000115598949 | (0.989646729, 0.979690626) | 0.147387 |
| 1e-3 | [037](../../runs/037-2026-10-10-rosenbrock-adamw-lr-1e-3/README.md) | 0.234111865 | (0.516148946, 0.266394818) | 0.964627 |

The midpoint has the lowest final objective among these three decay conditions.
At 1e-2 and the midpoint, raw adjacent pairs frequently reverse while stored
momentum is mostly or entirely positively aligned. Neither condition settles
at the measured objective's optimum; all three fail the joint loss/gradient
stationarity test. The low LR is still progressing with no sampled loss increases.
These are finite-horizon outcomes, not a universal LR ranking or convergence proof.

The two endpoint LRs have matched no-decay controls already executed in
[Analysis011](../011-2026-10-10-rosenbrock-lr/README.md). Their initial point,
numerical implementation, optimizer settings and 2,000-update horizon match;
only the decay coefficient changes. The midpoint has no no-decay control.

| LR | Final loss, decay 0 | Final loss, decay 0.1 | Negative momentum pairs, decay 0 | Negative momentum pairs, decay 0.1 |
| --- | ---: | ---: | ---: | ---: |
| 1e-2 | 0.000141348339 | 0.00104246819 | 65.3327% | 0.4002% |
| 1e-3 | 0.325412631 | 0.234111865 | 0% | 0% |

At 1e-2, decay gives higher final loss despite much more positive adjacent
momentum alignment. At 1e-3, decay gives lower final loss within this horizon.
The response therefore depends on LR in this setup. LR also multiplies the
shrinkage step. Changing decay changes the path and subsequent gradients;
alignment itself is not independently manipulated or shown to mediate the outcome.

[Method](METHOD.md) · [Observation and caption](observations/001-learning-rate-comparison.md) ·
[Exact reductions and endpoint comparisons](results.json) · [Scientific verification](verification.json) ·
[Figure provenance](figure_manifest.json) · [PDF verification](figure_verification.json).

Reproduction from the repository root: `01_verify.py` replays preserved attempts
in memory and reuses the independent Analysis010 verifier; `02_plot.py` calls
the shared nine-panel plotting module; `03_record.py` writes reductions/run
records and renders the PDF for visual review. The immutable prior scientific
and plotting sources are preserved. `00_prepare.py` refuses existing runs;
do not reexecute a completed attempt or modify its frozen inputs.

All 6,000 independent PyTorch fixed-gradient updates agree within 5.56e-16
point error; analytic objective/gradient, exact replay, geometry, decay
displacement, denominator cases and file sizes/hashes pass. Run035 matches
every saved array in the first 2,000 updates of Run030. The endpoint controls'
config/source/history hashes are checked read-only. The 16-by-23-inch,
single-page PDF passed integrity checks, direct-window rolling-mean checks and visual review, with the same
landscape domain and endpoint colors as Analysis011.

Complete raw histories total 2,906,275 bytes and remain ignored locally;
compact configs, summaries, final states, inventories, manifests and receipts
are versioned. Trajectory/save execution totals about 0.176 seconds; checks
and plots add seconds. No cloud resources, transfers or billing. One
deterministic function/start/beta2 and fixed horizon supply no sampling-error
estimate, general stability, asymptotic convergence or LLM conclusion.
