# Observation011: remove additional averaging of Adam-direction metrics

The user noted the optimizer's internal accumulation and requested removal of
plot smoothing for d, keeping it only for training-loss change. The
[revised figure](../figures/training_size_adam_direction_split_loss_only_n5_n21.pdf)
retains the 4x2 layout, solid model-size colours, independent column Y limits and
logarithmic norm axes. The earlier fully smoothed figure is preserved.

Sources, vector definitions, matched factors and precision are unchanged from
[observation010](010-adam-direction.md): saved adaptive-direction scalar histories
for Pythia-14M/31M/70M, Runs017/018/019, seed1235, with the matched MiniPile/AdamW
recipe. The generator rechecks accepted source hashes and inventory sizes. All
712 accepted updates are retained in the sources; the figure selects updates1-400,
which are full batches. The final partial batch at712 remains outside the figure.
No new training, diagnostic reconstruction or cloud execution is introduced.

The loss row still plots complete trailing means of ln(L_t)-ln(L_(t-1)): N=5 in
the 1-100 column and N=21 in the 100-400 column. Smoothing uses the full history
before selecting ranges. All 1,188 finite loss points and their undefined
placements exactly match the preceding figures.

The remaining rows plot the original per-update scalars without any moving mean:

```text
q_d[t] = 1 - rho_d[t] * ||d_t|| / ||d_(t-1)||
rho_d[t] = cosine(d_(t-1), d_t)
norm[t] = ||d_t||_2
```

The internal Adam moment accumulation, bias correction and preconditioning are
unchanged. Removing display smoothing does not remove optimizer memory. d still
excludes LR, decay and subtraction rounding; q_d remains a descriptive projected
direction change, not a measured-curvature estimate or actual displacement.

Norms now start at update1 and paired metrics at2; the first complete left loss
mean is at6. At update100 the two columns have identical direction values but
different loss windows. There are 4,791 finite plotted values: 1,188 smoothed loss
changes and 3,603 original direction scalars, including duplicated update100.
Zero-denominator handling and whole-model scope follow observation010; all norms
here are positive and undefined first pairs stay blank.

Sharper local fluctuations are now visible, especially for the 70M cosine and
q_d. This display revision provides no new seed replication, causal model-size
effect, curvature, stability or generalization conclusion. Whole-vector norms
remain dependent on dimensionality and Adam preconditioning. Finite-precision
source reductions and the loss row's smoothing blur remain; no error bars are
introduced. No manuscript claim is added.

**Caption.** Pythia MiniPile dynamics using Adam adaptive directions, seed1235.
Columns show updates1-100 and100-400. Only adjacent natural-log training-loss
change is smoothed, with complete trailing means N=5 and N=21, respectively.
Projected direction change q_d, adjacent cosine rho_d and direction L2 norm use
the original per-update values. Solid blue/orange/green curves identify14M/31M/70M.
Y limits vary by column and norms use log Y. Update100 appears in both columns.

Generate with `python 16_plot_adam_direction.py --loss-only-smoothing`.
The [CSV](../training_size_adam_direction_split_loss_only_n5_n21_data.csv) has1,203
rows with separate `loss_window` and `direction_window` fields; the latter is1.
The direction columns are deliberately not labelled as means. The
[manifest](../training_size_adam_direction_split_loss_only_n5_n21_manifest.json)
records all source/output and curve-array hashes, 1,188 independent loss-window
checks, 3,603 exact raw-direction comparisons, plotted-array checks, CSV roundtrip
and preserved-file hashes. Render and inspect the PDF, then record visual
acceptance with `--loss-only-smoothing --accept-reviewed`.
