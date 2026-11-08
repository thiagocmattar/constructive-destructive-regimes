# Cap-quantile ablation with a shared loss axis

[Figure 14](../figures/14-distributed-cap-quantile-ablation-mlp.pdf) presents the
existing H049 MLP results as three columns: Adam LR 0.001, 0.002 and 0.004.
The X axis lists cap quantiles q25/q50/q75/q90/q99. Blue markers are connected
to show the ablation trend; a separate black diamond is uncapped Adam at the
column's LR. All columns share identical logarithmic Y limits (0.0105-0.34).
The original uncapped Adam at LR 0.001 is the dashed reference in every column,
with its across-seed standard deviation shown as a gray band.

**Reduction and uncertainty.** For each initialization, first average the full
training-objective loss over all 705 states from 448 through 1152, inclusive.
Then compute the arithmetic mean and sample standard deviation (ddof=1) across
the three initializations. Error bars are mean +/- one sample SD of those three
time averages. They are neither temporal variation nor a confidence interval.
Each objective measurement covers the same 5,000 CIFAR-10 training examples.
All 18 conditions and all three seeds are included; the initial untreated state
is included in the time averages, as in the original sweep.

**Caption.** Distributed local displacement-cap quantile ablation for MLP/CIFAR-10.
Connected blue points show mean training loss for q25/q50/q75/q90/q99, and black
diamonds show uncapped Adam at the same LR. Columns use LR 0.001/0.002/0.004
with a shared logarithmic loss axis. Points and the baseline band show mean +/-
sample SD across three initializations, after averaging states 448-1152 within
each initialization. The dashed reference is original Adam at LR 0.001.

The cap settings use categorical spacing; connecting segments guide the eye
and do not estimate a continuous response curve. Cap budgets remain calibrated
at the original 1x LR. This figure summarizes the existing continuation experiment;
it adds no training, tuning or evidence of statistical significance. The
[full results and limitations](013-distributed-cap-quantile-lr.md), including
test-loss tradeoffs, remain applicable.

Sources: `cap_lr_curves.npz` and `cap_lr_summary.json` in this analysis folder.
`43_plot_cap_ablation.py` recomputes all 54 per-seed time averages, checks all 18
means and SDs against the saved summary, and checks plotted means and error-bar
endpoints. `cap_ablation_verification.json` pins source/script/PDF hashes and
records the reductions. The rendered page and text bounds were reviewed;
`cap_ablation_visual_review.json` records that check. Earlier figures are unchanged.

Reproduce locally:

```powershell
python analyses/065-2026-11-07-mlp-frozen-lr-double/43_plot_cap_ablation.py
```
