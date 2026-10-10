# Analysis011: Rosenbrock momentum at three learning rates

Completed and verified, 2026-10-10. **[Open the single comparison figure](figures/rosenbrock_learning_rates.pdf).**
It has one large full-domain contour/trajectory panel followed by the eight
requested diagnostics in a four-row, two-column grid. All three learning rates
are overlaid. The 16-by-23-inch vector PDF has explicit trajectory, start,
optimum and final-point legends, labeled objective levels and a separate
contour color scale. There is no zoom panel, smoothing or thinning. Adjacent
cosines use individual points so rapid alternation does not fill the panel.

One optimizer setting: Adam with momentum, zero weight decay, betas (0.9,0.95),
epsilon 1e-8, exact Rosenbrock f(x,y)=(1-x)^2+100(y-x^2)^2, start (-1.2,1),
constant LR, CPU float64. Exactly 2,000 updates per LR; only LR varies.
The user's “single run” request is one method repeated at three LRs; repository
provenance assigns three run IDs because LR/horizon are frozen scientific inputs.
Prior executed Runs028-031 and Analysis010 outputs are preserved.

| Learning rate | Run | Final objective | Final point | Final gradient norm |
| --- | --- | ---: | --- | ---: |
| 1e-2 | [032](../../runs/032-2026-10-10-rosenbrock-lr-1e-2/README.md) | 0.000141348 | (1.000349215, 0.999510164) | 0.532236 |
| 1e-3 | [033](../../runs/033-2026-10-10-rosenbrock-lr-1e-3/README.md) | 0.325413 | (0.429550564, 0.184486846) | 1.136300 |
| 1e-4 | [034](../../runs/034-2026-10-10-rosenbrock-lr-1e-4/README.md) | 4.138634 | (-1.031981630, 1.074827294) | 1.968242 |

LR 1e-2 reaches the optimum's neighborhood quickly but keeps oscillating:
its minimum sampled loss is 2.74037e-9 at update 757, and its final gradient
norm remains 0.532236. Raw/momentum adjacent cosines are negative in 66.73% /
65.33% of defined pairs. The smaller LRs have positive adjacent cosines and
monotonically decreasing sampled loss in this horizon, while traversing much
less of the valley. None meets the declared joint loss/gradient stationarity
test throughout the final 100 updates. This compares finite-horizon progress;
it does not establish asymptotic convergence or a general LR/optimizer ranking.

Compared with the directly matched previous Run028, the 1e-2 optimizer,
initialization, betas, epsilon, precision and trajectory arithmetic are unchanged.
Only the stopping budget changes from 10,000 to 2,000 updates; all saved arrays
are identical over the common prefix. The previous experiment also included
no-momentum/decay controls; this one holds momentum and zero decay fixed and
adds LR 1e-3/1e-4. A new stopping budget cannot change earlier steps here.

The apparent difference in oscillation is presentation: [the previous plot](../010-2026-10-10-rosenbrock-controlled/02_plot.py)
used a trailing 100-update mean for adjacent cosines, whereas this figure shows
every defined pair. Previous loss was also unsmoothed, but shown in a smaller
panel over 10,000 updates. Oscillation persists in the original late history:
[Run028's last 1,000 losses](../../runs/028-2026-10-10-rosenbrock-adam/artifacts/attempts/001-local/summary.json)
range from 2.21917e-9 to 1.42761e-4. Its final loss 3.45215e-6 is one point
within that continuing oscillation. Smaller LRs have not reached the same
neighborhood in this fixed budget, so their positive cosine does not establish
that they would settle without oscillation if run longer.

The bottom panels are loss, raw cosine, momentum cosine, raw/momentum norms,
raw A and B, momentum A and B, raw **B/A**, and momentum **B/A**, in that order.
A is the sum of vector norms; B is the norm of the vector sum. B/A follows
the user's correction before handoff, matching Analysis010's ratio. Stored
momentum is the uncorrected first moment; the full adaptive/displacement
histories remain available but are outside this simplified figure.

[Method and operational definitions](METHOD.md) ·
[Observation and caption](observations/001-learning-rate-comparison.md) ·
[Exact reductions](results.json) · [Scientific verification](verification.json) ·
[Figure provenance](figure_manifest.json) · [PDF verification](figure_verification.json).

From the repository root, `01_verify.py` replays the existing attempts in memory
and checks them without replacing raw evidence. It reuses Analysis010's unchanged
independent verifier, with the newly shared numerical module as the target.
`02_plot.py` reads verified histories and generates the single PDF; `03_record.py`
records reductions, run observations and PDF integrity, then requires visual
review of its rendered scratch page. All 6,000 independent PyTorch fixed-gradient
optimizer updates agree within 5.56e-16 point error. Objective/gradient checks,
all-stream geometry, exact replay, B/A zero-denominator cases and sizes/SHA-256
pass. Run032 matches every array in the first 2,000-update prefix of Run028.
The original numerical source remains unchanged; it was promoted to
[src/controlled/rosenbrock.py](../../src/controlled/rosenbrock.py) for reuse.

All 6,003 positions and 6,000 transitions are retained in three ignored NPZ
files totaling 2,846,401 bytes. Compact configs, manifests, summaries, final
states, inventories and verification receipts are versioned. Trajectory/save
time totals about 0.185 seconds; checks and figures add seconds. No cloud
resources, transfers or billing. No stochastic samples, dataset, validation
split or experiment seeds exist; one deterministic start/function/beta2 and
fixed update budget provide no seed uncertainty, stability or LLM conclusion.
