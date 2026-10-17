# Observation007: short-window differences of log training loss

The [log-loss-difference figure](../figures/training_size_overlay_log_loss_diff_n5_1_400.pdf)
uses the natural logarithm of training loss, retaining the previous N=5 complete
trailing mean. The user requested diff(log(loss)) to inspect relative changes.
It replaces only the upper-left panel of the 1-400 overlay. The three raw
diagnostic panels, solid model-size colours, axis ranges and log norm display
are unchanged. Runs017/018/019, seed 1235, are reused; no training is launched.

```text
delta_log_L[t] = ln(L[t]) - ln(L[t-1]) = ln(L[t]/L[t-1]),   t = 2..400
mean_delta_log_L[t] = mean(delta_log_L[t-4], ..., delta_log_L[t])
                    = ln(L[t]/L[t-5]) / 5,                 t = 6..400.
```

For a ratio r close to 1, ln(r) is approximately r-1, not r. The statistic is
therefore approximately fractional loss change per update, with zero indicating
no change and negative indicating falling loss. Exponentiating the smoothed
quantity gives the geometric mean loss ratio per update over the five intervals:
exp(mean_delta_log_L[t]) = (L[t]/L[t-5])^(1/5). It is not the arithmetic mean of
loss ratios. A value of -0.01 corresponds to exp(-0.01)-1, about -0.995% per update.

Log transforms are applied to the unsmoothed positive raw losses, then adjacent
differences are formed and averaged. No loss floor, denominator epsilon, absolute
value, sign reversal or extra smoothing is applied. All actual losses are finite
and positive. There are 395 complete means per model, first defined at update 6;
earlier points stay undefined. The other variables retain 400 norm values and 399
defined rho/q pairs per size. All x axes cover 1..400; all finite values are shown.

Sources, matched factors, batch coverage and scientific limits are those in
[observation005](005-unsmoothed-first400.md) and
[observation006](006-short-loss-difference.md). These losses evaluate changing
minibatches; relative changes do not isolate same-objective improvement, raw
gradient magnitude or an AdamW mechanism. One seed gives no seed uncertainty or
causal, curvature or stability conclusion.

**Caption.** Pythia-14M/31M/70M MiniPile dynamics over updates 1-400, seed 1235.
The upper-left panel shows the complete five-update trailing mean of adjacent
natural-log training-loss differences. Zero indicates no change; negative values
indicate falling loss. The first mean is at update 6. Other panels show unsmoothed
eta*kappa_hat gradient-change proxy, adjacent raw-gradient cosine rho and raw
gradient L2 norm. Blue/orange/green solid curves identify 14M/31M/70M. Only the norm
axis is logarithmic. Earlier raw-loss and linear-difference figures are preserved.

Generate with `14_plot_raw_first400.py --log-loss-diff`. The
[CSV](../training_size_overlay_log_loss_diff_n5_1_400_data.csv) retains raw loss and
diagnostics plus `log_loss_diff` and `log_loss_diff_mean`; missing values are blank.
The [manifest](../training_size_overlay_log_loss_diff_n5_1_400_manifest.json) records
source/output hashes, positive-loss checks, log-ratio/log1p/telescoping identities,
all 1,185 complete means, all 4,779 finite plotted values, CSV roundtrip, unchanged
diagnostic-array hashes and preservation of previous PDFs/data/manifests. Render
with Poppler to `scratch/training_size_overlay_log_loss_diff_n5_1_400`, inspect
the page, then record acceptance with `--log-loss-diff --accept-reviewed`.
No new run/analysis ID is used.
