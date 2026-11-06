# 02: MLP floors and abrupt return

Sources: Analysis047 and Analysis045 reports, raw-source manifests and plotted
tables; Runs223/224/226. Generator `../01_figures.py` selects two floor panels
and only the immediate-half return history. No new experiment is interpolated.

Caption: CIFAR-10/MLP learning-rate floors and restoration from matched warm
states. Left and middle: 32-update cosine reductions from base LR 0.001 to
0.75x and 0.5x, each compared with unchanged LR. Right: immediate half LR at
state 512 followed by an abrupt rise to 0.75x at 650, compared with holding
the same immediate-half branch. Rows show loss (log), projected gradient change,
raw-gradient norm and applied/base LR. Gray bands mark context before 512;
the dotted line at 650 marks restoration. No cosine initial reduction appears
in the right panel. The four-row layout omits rho; source data retain it.

Coverage: all states 448--800; first interventions at 512, cosine floors attained
at 544, restoration at 650. All full-objective MLP curves use the same Spark
context; response losses retain full 5,000-image coverage.

Result: cosine floors 0.75x and 0.5x lower mean loss over 513--800 by 17.24%
and 26.30%, respectively, versus unchanged. The higher floor retains large late
spikes. Abrupt restoration from the immediate-half history increases mean loss
over 651--800 by 57.41% versus its held-half control and restores large spikes.
The third comparison has a different control and response window from the first
two; effects must not be treated as one uniform ranking. One development seed,
selected timing, finite-window censoring and no independent confirmation.
