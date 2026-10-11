# Observation001: larger constant learning rates on matched Spark trajectories

Source runs: completed Run023 (LR0.01), newly authorized Runs053/054
(LR0.02/0.04). Frozen initialization, seed1234 token order, cache hashes,
architecture, optimizer/clipping/batch/precision/validation/retention match.
Only peak LR varies; LR-scaled AdamW decay displacement also changes.
Sources and generator hashes are pinned in source_inventory.json and
comparison_data.json; owning raw events and all recovery states are retained.

Caption: Pythia-14M higher-LR stress comparison on DGX Spark. The compact
4-panel figure puts unsmoothed training loss first and then one raw/momentum
norm panel per LR. The full12-panel figure adds raw/momentum/Adam adjacent
cosines, raw/momentum cumulative A and B, and all three B/A curves.
All completed trajectories show712 accepted updates, including zero-LR
update1 and the marked partial final batch at712 (311 versus1024 sequences).
Failed prefixes show only accepted updates and are labeled as failed.

Loss is the recorded supervised-token-mean cross-entropy, nats; no loss
smoothing. Norm panels share log Y, show raw originals at opacity0.20 and
a trailing10-update arithmetic mean above them, with fill (opacity0.14) to
unsmoothed momentum. Average norm units before log display; prefix windows
use available values. Nonpositive norms are counted and omitted from log
display without flooring. Fill denotes a magnitude gap, not uncertainty.
Adjacent cosines show faint originals and arithmetic means: raw50 updates,
momentum/Adam25. Available finite prefixes preserve the undefined first cosine.
Cumulative panels are unsmoothed and linear: solid A=sum||x|| and dashed
B=||sum x||, with B/A in[0,1]. No time thinning or loss-value filtering.

Raw g is whole-model accumulated/unscaled pre-clipping gradient; m is actual
uncorrected FP32 Adam exp_avg based on clipped g; d is actual-moment
bias-corrected/preconditioned direction excluding LR, decay and subtraction
rounding. All reductions/prefixes are FP64. g/m/d are different streams and
none is automatically the realized parameter displacement.

Every completed pass covers all1,491,711,416 input tokens and1,490,983,041
supervised targets. Both validation passes cover339 sequences/693,329 targets
and the short tail. Summary last100 loss is token weighted over full-sized
updates only; endpoint/negative-pair statistics declare their coverage.
No cross-seed uncertainty is estimated. Clipping/EMA, one seed/order/start,
finite budget and LR-scaled decay limit interpretation. Sign or norm growth
alone establishes neither stability nor generalization or a causal noise effect.

Generator: [03_plot.py](../03_plot.py); reduction: [04_record_results.py](../04_record_results.py).
[Data, source hashes and smoothing](../comparison_data.json),
[verification and pending visual review](../figure_verification.json),
[recorded outcomes](../RESULTS.md), [full-precision table](../summary.csv).
