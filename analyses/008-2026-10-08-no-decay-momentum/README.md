# Analysis 008: warmup-only LR scale and Adam first-moment geometry

Complete: all three runs finished 712 updates, passed full verification, and
were retrieved with every output size/SHA-256 verified. Run023 finished at
2026-10-09 12:59:25 UTC (09:59 Brasilia time). Its two PDFs and the automatic
joint figure passed visual review, as did the five updated full-cohort figures.
The [final three-LR results](complete-three-lr/README.md) preserve the interim
figures. Final validation loss is 4.763734 / 6.232207 / 6.636489 at LR
1e-3 / 1e-4 / 1e-2; the baseline LR performs best in this one-seed comparison.
The full queue launched at 2026-10-08 21:58:41 UTC. User explicitly authorized three full
Pythia-14M passes on the owned DGX Spark and requested Run002's comparison with
additional stored Adam momentum measurements. Runs 021/022/023 use constant
post-warmup LR 1e-3/1e-4/1e-2. This is H007's exploratory comparison; no direction
of effect is asserted in advance.

All conditions load Run002's identical frozen initial model, seed 1234 data order,
the exact pinned MiniPile cache, and the same FP16/FP32 AdamW recipe, clipping,
batch 1024 / microbatch 4 / sequence length 2048, tokenizer, architecture and complete coverage.
Only LR scale and replacement of cosine decay with constant plateau vary.
There are 712 updates, 728,375 sequences, 1,491,711,416 input tokens and 1,490,983,041
targets per pass. Both validation passes cover all 693,668 input tokens / 693,329
targets including the tail. No pretrained model is imported.

At optimizer update t (starting at 1), the schedule is
`LR(t) = peak*(t-1)/7.12` through update 8, then exactly peak from update 9.
The zero-LR first update still advances Adam moments and all diagnostic streams.
Adam betas = (0.9, 0.95), epsilon = 1e-8, decay = 0.1 and bias/LayerNorm decay exclusions
remain unchanged. Changing LR also changes AdamW's LR-scaled decay displacement.

## Measurement contract

Streams are the whole-model raw unscaled token-mean gradient g before clipping,
stored uncorrected first moment m after its accepted Adam update, and the existing
bias-corrected adaptive direction d after updating actual FP32 moments:

```text
m_t = 0.9*m_(t-1) + 0.1*g_t_clipped
d_t = (m_t/(1-0.9**t)) / (sqrt(v_t/(1-0.95**t)) + 1e-8)
A_t(x) = sum_(u<=t) ||x_u||_2
B_t(x) = ||sum_(u<=t) x_u||_2
R_t(x) = B_t(x)/A_t(x)
rho_t(x) = dot(x_(t-1),x_t)/(||x_(t-1)||*||x_t||)
```

For x in {g, m, d}, record individual norm, A, B, B/A and adjacent cosine every accepted
update. Also record d/g, m/g and d/m norm ratios, token-weighted train loss, applied
LR, loss scaling/overflow events, diagnostic timings, throughput and peak memory.
Momentum is `exp_avg`, not m-hat and not an actual parameter displacement. LR,
weight decay, second-moment division and descent sign do not enter m.
All unique trainable parameters use the same fixed named order. Previous g/m are
FP32; previous d is FP64. Prefix buffers and reductions are FP64. First cosine and
zero-denominator statistics are null with reasons; exact cancellation has B/A=0.
Accepted updates only; overflow retries repeat the same batch without advancing
the measurement clocks. Nonfinite loss or exhausted retries fail visibly.

No per-sample decomposition, stability proxy, generalization claim or causal
benefit of alignment is assumed. One matched seed identifies within-seed
intervention differences, not cross-seed uncertainty. Historical Run002's decay
curve uses different hardware/CUDA kernels and is descriptive context only.

## Execution and retained evidence

Three sequential detached jobs on the single GB10, approximately 15–18 hours total;
each has an 8-hour timeout, a 24-hour aggregate training bound. No cloud GPU rental.
New retained storage allowance is 40 GB including the shared 6 GB input and checkpoints.
Initial/midpoint/final and every 100-update recovery states retain model, optimizer,
scaler, RNG, data position and all three streaming states. Source/config/input
identities are frozen before execution; scientific changes require a new run.
Monitor bounded read-only snapshots every 60 seconds, with loss/coverage/throughput/ETC,
memory, overflow and stale-event warnings. The queue continues to other conditions
after a visible failure, without changing settings.

Local retrieval verifies every selected file's size/SHA-256. Retain the Spark
and its data; do not terminate or reconfigure the user's machine. Calibration
states are separate and never reused for production. Full-pass verification
checks coverage, schedule, all streaming clocks and final geometry reconstructed
from recovery buffers/optimizer state.

Deliverables: original 3x2 `raw_adaptive_comparison.pdf` for each condition,
`momentum_evolution.pdf` with all added metrics for each condition, and a joint
three-stream LR comparison. Main figures display updates 1–711; final partial
update 712 is retained in numerical results. Each figure has local source hashes,
reduction/caption/coverage/uncertainty and a generating script/observation.

## Launch verification and monitoring

Thirteen focused tests passed: the clipped-gradient moment recurrence, exact
warmup clock/plateau, streaming geometry and recovery, unchanged parameter updates
with capture enabled, and compatibility with the existing adaptive/scale and
continuation contracts. All three independent 3-update calibrations passed
[verification](calibration-acceptance.json), including reconstruction of all three
streams and exact equality of previous m with Adam's stored `exp_avg`.
Post-warmup calibration times were 25.147 / 25.193 / 25.197 seconds per update,
with 5.608 GiB peak allocated CUDA memory and no overflow retries. These are
mechanics/timing checks, not full-pass scientific results.

The initial tensor hash is identical to Run002:
`0e9f77f965f45df79b5dc696256e011dc08ab2d34fb07c53fd4fed9be6b8b6bc`.
[Source inventory](source_inventory.json) and [packet identity](packet_identity.json)
freeze executed scientific bytes. `cohort.json` retains its prepared value as
part of that freeze; runtime status lives separately in remote `queue_state.json`.
Remote task root is
`~/experiments/constructive-destructive-regimes/no-decay-2026-10-08`;
detached queue PID at launch was 49984.

The hidden local controller (launch PID 33180) polls, retrieves each terminal
attempt, verifies every file's size and SHA-256, generates seven final PDFs, and
renders them for later visual review. Latest snapshots are in the ignored
`latest-status.json`; controller/log files are also ignored. At update 9, Run021
held LR = 0.001, took 25.119 seconds, and recorded all momentum fields with no
overflow retry. All per-run figures and the automatic joint comparison are now
complete and visually reviewed; the final revised layouts are linked above.

At the 09:24 UTC progress check, Run023 held LR = 0.01 with zero total overflow
retries and complete momentum records. The last 30 updates averaged 25.304 seconds
and 82,880 input tokens/second. Remaining training time was approximately 3 hours
35 minutes (finish near 10:00 Brasília time). The GPU was 65°C, 94% utilized,
about 36 W, with no active thermal or power slowdown; 109 GiB system memory
and 3.5 TiB disk space remained available. Runs021/022 each took about 5 hours.

Training survives SSH or local computer disconnection. Automatic monitoring and
retrieval require the local computer to remain awake and connected; remote
outputs remain preserved if it disconnects. To resume the local controller,
run `python analyses/008-2026-10-08-no-decay-momentum/06_watch_retrieve.py` only
after confirming its previous process has stopped. Never relaunch the existing
remote queue. Remote checkpoints and shared inputs are retained; no cloud
resources are created or billed.

The 10:45 UTC read-only SSH check found Run023 at 392/712 updates and
822,083,584 input tokens. The last 30 updates averaged 25.323 seconds and
82,816 input tokens/second; remaining training was about 2 hours 15 minutes,
with finish near 10:00 Brasilia time, followed by final validation/saving,
verification and retrieval. Latest loss was 5.860; the last-30 token-weighted
training loss was 5.937. GPU temperature was 66 C, with no active thermal/power
slowdown, zero overflow retries, 5.608 GiB peak CUDA allocation, approximately
110 GiB available system memory, no swap use and 3.40 TiB free disk. The queue
and local monitoring/retrieval controller remained active.

At the common update 392, the partial Run023 raw/momentum B values were
57.990/48.200 (momentum 16.88% lower), compared with 65.219/63.499 for Run021
and 241.232/236.993 for Run022. Raw norms exceeded the clipping threshold at
197/392 Run023 updates versus 14/392 and 11/392 in the other runs. Therefore
the near-overlap of raw and momentum B in Runs021/022 does not extend as closely
to the large LR at this prefix; momentum uses clipped gradients. Run023's first
negative unsmoothed raw pair occurred at update 12. These are provisional
matched-prefix measurements, not final validation or full-pass conclusions.
Read-only snapshots/reductions are retained in ignored analysis scratch;
completed Run023 artifacts now supply the final immutable source evidence.

## Completed two-run interim analysis

The user's requested [Run021/022 analysis](interim-021-022/README.md) includes
one 2x3 figure with training loss, raw cosine, momentum/adaptive cosine,
overlaid raw/momentum A and B, all three B/A streams, and raw/momentum norms.
Two supplements show exact sparse checkpoint lag profiles and pairwise cosine
matrices at common updates 100–700. Complete short-lag histories were not
recorded; these coarse profiles are labeled as sampled checkpoint evidence.
All three new PDFs passed visual review. Numerical results include all 712
updates; the main figure omits the final partial update, as in the source figures.

Run021 reaches final validation loss 4.763734 versus 6.232207 for Run022,
despite more negative raw pairs (84.53% versus 26.30%). Momentum stays positively
aligned in both. Run022 has higher final raw/momentum B/A, while Run021 has
higher adaptive B/A. These describe one matched seed and do not establish
alignment as a performance, stability or convergence predictor. The full
Run023 comparison is available separately. This interim analysis is separate from
the automatic final-cohort outputs and preserves all executed inputs.

The revised [simplified versions](interim-021-022/README.md) separate a 3x2
comparison across LRs from a 2x2 within-LR geometry comparison. All requested
panels are present and both PDFs passed visual review. The generator supports
adding the third LR as overlaid curves in the first figure and a third column
in the second; that extension is now complete and verified locally.

A further [focused 2x2 figure](interim-021-022/figures/loss_alignment_norms.pdf)
compares loss, raw adjacent cosine and per-LR raw/momentum norms. The first
negative unsmoothed raw pairs occur at updates 54 and 385; shading marks time
after these events, including later positive pairs. The norm gap is already
present at the preceding updates, and does not suddenly widen at the first
negative pair. Local windows show increased raw-norm fluctuations with smoother
momentum. [Observation003](interim-021-022/observations/003-loss-alignment-norms.md)
records the event data and limits; no causal onset or gradient-noise variance is
inferred. The latest PDF passed visual review; all six interim PDFs are reviewed.

The completed three-LR extension preserves the two-run interim outputs. It
adds Run023 curves to all six combined metric panels, a third geometry column
(A/B above, g/m/d B/A below), and a focused view with two shared loss/cosine
panels above three per-LR norm panels. First negative raw pairs are marked at
54/385/12, with shading after each event rather than a median-based regime.
Geometry first-row panels use independent linear y-ranges to emphasize shape
within each LR; B/A panels share a linear scale. Focused norm panels also
declare individual linear scales.
Final data retain all 712 records while main curves show full-sized updates 1-711.
Two temporal supplements extend the exact sparse checkpoint comparisons to all
three LRs. [Observation001](complete-three-lr/observations/001-three-lr-results.md)
records captions, reductions, uncertainty and interpretation limits.

Run023's final raw/momentum A is 1946.971/88.342 and B is 87.308/57.698:
momentum A is 95.46% lower and B is 33.91% lower. Raw norms exceed the clipping
threshold at 517/712 updates, compared with 63 and 74 in the other conditions.
The ideal normalized-EMA reconstruction of cumulative clipped B is about
57.896 (ignoring FP32 recurrence rounding), close to momentum B. Thus raw and
momentum B cannot be compared as if clipping were absent. All adjacent momentum
and adaptive cosines remain positive; some sparse long-lag momentum cosines
are negative in Run023. Alignment at one horizon does not guarantee another.

Training ETC is zero. Each pass took about five hours; the sequential queue took
15 hours 00 minutes 44 seconds. The final live check at 13:03 UTC found the GPU
idle at 38 C, 0% utilization and 4.88 W. All three runs had zero overflow retries.
The local controller finished automatically after verified retrieval and plotting;
no job or machine was terminated. All 144 retrieved run artifacts (about 21.62 GB)
and recovery states remain locally and on Spark, plus the shared input cache on
Spark. No cloud rental was created or billed, and no new experiment was launched.
