# Single-seed LR intervention and raw-gradient cancellation

## Question and comparison

Does late LR decay contribute to recovery of adjacent raw-gradient alignment?
Runs 011-016 branch the immutable update 356 states of Runs 005/007, respectively
Pythia14M/70M, seed 1235. Within each size, weights, Adam moments and clock,
loss scaler, RNG states, cumulative diagnostics, data position and future batches
are matched. Original decay, held midpoint LR and accelerated decay vary.
This is a test of late annealing conditional on a trained prefix, not a
constant-LR experiment from initialization. Extra seeds are deferred.

The held LR is 0.0005179103055. Accelerated decay advances the original clock
twice as fast after 356. The original and accelerated rules reach floor 0.0001
at optimizer updates 575 and 466; their effects on the next raw gradient begin
one update later. Raw cosine357 is common before the changed parameter update;
the primary affected interval is 358-712,355 pairs per branch.

## Sources, measurement and verification

Use the accepted `artifacts/attempt-01/full-pass/` in each of Runs 011-016,
the inherited1-356 prefix in Runs 005/007, and their archived continuations for
replay sensitivity. Exact paths, event/summary/acceptance hashes and archive
identities are in [results.json](../results.json). Run-local configs and frozen
source inventories pin model, tokenizer, dataset, precision, data order and
checkpoint bytes. [cohort.json](../cohort.json) maps all six conditions.

Gradients are whole-model, unique trainable-parameter, unscaled accumulated
batch gradients before clipping and AdamW. Vectors are FP32; cosine/norm and
cumulative reductions are FP64. These are raw-gradient diagnostics, not actual
AdamW updates. The unchanged recipe uses1024 packed sequences per full update,
microbatch 4, FP16 dynamic scaling, clip norm 1, and AdamW with LR-scaled decay.
Each continuation covers 363831 sequences,745125304 input tokens and 744761473
targets without repetition, including the final partial update. Prefix plus
continuation covers every training token. All five validation points
(356,445,534,623,712) include all 500 documents,693668 inputs/693329 targets and
the packed tail. Validation and aggregate training loss are target-token weighted.

Native SSH/SCP transfers were checked by exact archive size/SHA and strict
per-file inventories. Frozen source/parent identities, terminal raw/adaptive
cumulative states, Adam clocks and complete coverage were independently checked
on CPU. All six new batch-index hashes match at every update. CUDA mechanics,
including the first batch/scalar diagnostics and new original controls, are
recorded in run-local prelaunch receipts. Numerical replay is close, not bitwise;
the qualification revision and retained repeat are disclosed in the analysis
[README](../README.md). No executed scientific input was changed.

## Reduction and results

Generating script: [05_analyze.py](../05_analyze.py). Define
`F=mean(rho<0)`, `D=mean(max(-rho,0))`, and
`M=mean(-rho | rho<0)`, so `D=F*M`. Statistics below use unsmoothed cosines,
equal pair weights, and pairs 358-712. The final partial batch is retained here;
its exclusion is a recorded sensitivity reduction and preserves all four
within-size contrast directions. Full reductions, fixed phase windows and
prefix-inclusive terminal B/A are in results.json; figure data are in
[curve_data.csv](../curve_data.csv).

| Size | Schedule | F negative | D | M conditional | Median cosine | Final validation loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 14M | Original |31.83% |0.068078 |0.213873 |0.194055 |5.182014 |
| 14M | Held |96.90% |0.362040 |0.373616 |-0.358151 |5.067429 |
| 14M | Accelerated |11.27% |0.022167 |0.196731 |0.330621 |5.217633 |
| 70M | Original |68.17% |0.159054 |0.233323 |-0.108564 |4.055890 |
| 70M | Held |92.96% |0.320120 |0.344371 |-0.311242 |3.984569 |
| 70M | Accelerated |51.55% |0.098094 |0.190292 |-0.006496 |4.104163 |

Accelerated decay lowers F,D,M and raises median cosine at both sizes, while
held LR moves every one of those statistics in the opposite direction.
Relative to original decay, accelerated D decreases by 0.045911/0.060960 at
14M/70M; held D increases by 0.293962/0.161066. Yet held LR achieves the lowest
training and validation loss at both sizes. Its validation improvements are
0.114586/0.071321; accelerated decay worsens validation by 0.035619/0.048273.
Thus more negative raw alignment coexists with better loss in these directly
matched continuations. This does not show that cancellation causes the improvement.

Recovery is the first31 consecutive strictly positive centered means, using
full31-pair windows wholly within358-711. It is retrospective and nonpermanent.
Original/accelerated centers are467/402 for14M and 538/437 for70M: accelerated
recovery is 65/101 updates earlier. Neither held branch meets the criterion.
Both decaying70M branches later return to negative smoothed alignment; both 14M
decaying branches remain positive over the remaining eligible centers.
Windows15/61 retain earlier accelerated recovery, no held recovery, and 70M
reversals. Faster annealing contributes to recovery; it does not ensure lasting
positive alignment or formal convergence.

New original controls differ from archived parents in D by-0.000555/-0.002812
and terminal validation loss by-0.00000117/-0.002537 at 14M/70M. Schedule
contrasts exceed these observed aggregate replay differences. This is a
sensitivity comparison, not a statistical noise bound.

## Figure captions and coverage

1. **[14M loss and raw cosine](../figures/loss_and_raw_cosine_14m.pdf).**
   Left: training next-token loss; right: adjacent raw-gradient cosine. Original
   decay is blue/solid, held midpoint LR orange/dashed, accelerated decay
   green/dash-dot, shared parent gray. Seed 1235, global display300-711,
   centered equal-update 31 means with complete windows. The final partial
   update 712 is omitted from curves but included in terminal statistics.
   Gray parent means end at 341 so their windows contain no future branch data;
   branch traces start at 357 and initially overlap the shared prefix in their
   centered windows. The vertical line marks the update 356 fork, not a
   mathematical convergence boundary. Holding LR maintains negative alignment
   while reaching lower loss; faster decay promotes positive alignment while
   reaching higher loss. Loss-axis bounds use the displayed late interval.
2. **[70M loss and raw cosine](../figures/loss_and_raw_cosine_70m.pdf).**
   Same sources, reductions, legend, fork and coverage at 70M. Cosine limits
   match the 14M figure; loss limits reflect this size's displayed values.
   Held LR has persistent negative alignment and lower loss. Earlier positive
   alignment under accelerated decay reverses later, despite the fixed floor LR.
3. **[Schedule and validation](../figures/schedule_and_validation.pdf).**
   Left: LR used for parameter updates 357-712, in units of0.001. The raw
   gradient is evaluated before that parameter update; cosine response follows
   the preceding update. Right: complete token-weighted validation at the five
   declared points. Colors/line styles denote schedule; circles/squares denote
   14M/70M. Held LR has the lowest final loss at both sizes despite the most
   negative raw alignment. All PDF pages were rendered, visually inspected,
   and checked for text within page bounds.

## Interpretation, uncertainty and manuscript scope

One seed supplies conditional intervention evidence. Gradient pairs are not
independent seed replicates; no pair-level CIs or p-values are reported.
LR changes both adaptive step scale and AdamW shrinkage. Positive scalar
rescaling alone cannot change a fixed pair's cosine; the effect arises through
the changed trajectory. Curvature, effective step size, data heterogeneity and
dimensionality mechanisms are not separately identified. GPU numerical
variation and the qualification changes remain part of the provenance.

Model-size ordering is schedule-dependent: F and D are larger at 70M under
original/accelerated decay but larger at 14M under held LR in this pilot. There
is no universal size/destructiveness law here. Analysis003's original-recipe
three-seed findings are preserved; its full-pass and this post-fork interval
are different estimands. The dimensionality/performance explanation remains untested.

Recommended restricted claim: **In these checkpoint-matched continuations,
the LR schedule changes adjacent raw-gradient cancellation, and increased
cancellation can coexist with lower validation loss.** Do not call negative
alignment learning failure or positive alignment stability/convergence.
Cross-seed validation is the next scientific check, not part of this launch.

No manuscript asset or working TeX was changed. All six workers and Pod volumes
were removed after artifact acceptance; unrelated resources were preserved.
Cloud elapsed 87.91 minutes; estimated USD 6.92, below the USD 15 cap. Billing is
provisional; the API snapshot has incomplete records, not the settled total.
See [cloud_closeout.json](../cloud_closeout.json) and
[billing_snapshot.json](../billing_snapshot.json).
