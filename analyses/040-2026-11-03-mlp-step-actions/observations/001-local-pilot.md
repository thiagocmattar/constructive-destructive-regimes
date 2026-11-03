# Observation001: approved local selected-fork pilot

Question: can a predeclared intervention reduce the known Adam excursion while
preserving later training progress against the same-state A0 control?

Sources: Run218 attempt001-20261103T172300Z-local-pilot, frozen source92954bd and
launcher03_local_pilot.py, unchanged Run215 historical MLP0.001 observations,
Analysis038/program definitions. Full5,000-image objective, seed0, no filtering
or smoothing;8 complete128-update branches from state657. No independent seed
uncertainty is available; repeated deterministic replay is mechanics only.

Reduction:02_report_local_pilot.py. Mean over states658..785; fixed untreated
B=L_A0(689), excess window689..785; first target attainment0.95B; existing fixed
pre-rise event detector including censored tails. CSV/PDF data are checked with
simple independent scalar sums and exact source-value/fork comparisons.

Caption: All eight seed0 trajectories after the authenticated state657 fork.
Loss/gradient/increment norm axes are logarithmic; completed-step q uses symlog.
The vertical line marks689, horizontal loss lines markB and0.95B. Outgoing alpha
uses the current staged action; preceding increments are already completed.
No sign or crossing is an asserted curvature/stability threshold.

Result: A1 half LR and A4 proposal clipping meet the training-loss pilot screen;
A2 fails final-loss margin; A3/A6 fail progress criteria; A5/A8 produce large new
excursions. Later/unrecovered events are retained. No generalization/online-policy
or population claim. See ../REPORT.md for all values, costs and interpretations.

Artifacts: ../figures/local_pilot.pdf, ../comparison.csv,
../aligned_diagnostics.csv, ../excursions.csv, ../raw_inventory.csv.
All786 untreated losses match historical values exactly; all8 fork states share
weights, moments/age, RNG and retained vectors. PDF rendered/visually reviewed.
