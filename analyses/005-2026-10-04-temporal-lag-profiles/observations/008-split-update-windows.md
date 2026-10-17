# Observation008: four metric rows and two update ranges

The requested [4x2 figure](../figures/training_size_overlay_log_loss_diff_n5_split.pdf)
puts updates 1-100 in the left column and 100-400 in the right. Rows show the
N=5 mean log-loss change, eta*kappa_hat proxy, rho and raw gradient L2 norm.
Each panel overlays Pythia-14M/31M/70M (Runs017/018/019, seed 1235) with solid
blue/orange/green curves. Previous figures and their data remain unchanged.

Inputs are the verified CSV and manifest from
[observation007](007-log-loss-difference.md), preserving all its scientific
definitions, source trajectories, matching, coverage and limitations. There is
no new reduction: the script slices the already computed arrays by update.
The log-loss row remains mean(ln(L_t)-ln(L_(t-1))) over five complete intervals,
first defined at update 6. All other rows remain unsmoothed; their first adjacent
pair is undefined.

Both endpoints of each requested range are included. Update 100 appears in both
columns, giving 12 repeated finite points across four metrics and three sizes.
The smoothing is computed before slicing, so the right column's point at 100
retains its original window 95..100; it is not reset at the column boundary.
The figure contains 4,791 finite plotted points, representing 4,779 unique values.

Y limits vary independently by column so that the later range is readable;
numeric Y tick labels are retained on both columns. Units and transformations
match within every metric row. Only the norm row uses log Y. Dotted references
mark zero for loss change/rho and one for eta*kappa_hat. No values within either
range are averaged again, thinned or excluded. Negative loss change indicates
falling recorded minibatch loss, not a same-objective derivative or AdamW norm.

**Caption.** Pythia MiniPile dynamics in two update ranges, seed 1235. Columns
show updates 1-100 and 100-400, including the shared boundary 100. Rows show the
five-update trailing mean of adjacent natural-log loss changes, raw-gradient
eta*kappa_hat proxy, adjacent raw cosine rho and raw gradient L2 norm. Solid
blue/orange/green curves denote 14M/31M/70M. Y limits vary by column; the norm
row uses log Y. Smoothing spans the column boundary. Single-seed descriptive
evidence retains the AdamW/minibatch interpretation limits; this display change
adds no scientific finding or new training.

Generate with `15_plot_split_windows.py`. The existing
[source CSV](../training_size_overlay_log_loss_diff_n5_1_400_data.csv) supplies
the plot data. The [new manifest](../training_size_overlay_log_loss_diff_n5_split_manifest.json)
records source/output and preserved-file hashes, exact ranges, 24 plotted-array
checks, boundary duplication, all finite point counts and visual acceptance.
Render with Poppler to `scratch/training_size_overlay_log_loss_diff_n5_split`,
inspect the page, then run with `--accept-reviewed`. No new run/analysis ID is used.
