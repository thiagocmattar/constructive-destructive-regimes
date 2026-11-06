# Figure observations

All figures are generated or copied by `../01_figures.py`, verified independently
by `../02_verify.py`, and bound to sources in `../manifest.json`. No uncertainty
bands are estimated from these single-seed selected trajectories. Gray shading
in 01--06 identifies pre-intervention context, not uncertainty; Figure 00 retains
its different original event shading. Full-objective losses retain the owning
run's complete example/token weighting. Raw gradients are measured before Adam
transformation and before clipping. Norms are absolute unless stated otherwise.
No spike detector is refitted and no new event counts are estimated here.

- [00: High-LR overview](000-high-lr-overview.md)
- [01: Persistent half LR](001-persistent-half-lr.md)
- [02: Floors and abrupt return](002-floors-and-return.md)
- [03: Return styles](003-return-styles.md)
- [04: Two-stage decay](004-two-stage-decay.md)
- [05: Clipping](005-clipping.md)
- [06: Pythia schedules](006-pythia-schedules.md)

All new curves are unsmoothed, with log loss and linear diagnostic axes.
Axis limits are independent across panels. A dimensionless projected gradient
change computed from adjacent norms/cosine is labeled eta*kappa_hat to retain
the requested manuscript notation. Under Adam/AdamW it is not an identified
directional Hessian curvature or a universal stability threshold. Pythia also
changes the sampled objective at each update.

Full-objective state s records the loss/gradient at that parameter state and
the outgoing LR for s to s+1. Terminal states have no outgoing update; LR curves
extend the last applied interval only to close it. Pythia uses its original
update-index logging convention, documented separately in Observation006.

Independent read-back preserves all selected original plot values and raw
sources. Publication PDFs use embedded Times/STIX fonts and native vector
curves. Original figures, datasets, training states, logs and manuscript assets
are untouched. Per-figure observations give captions and the limits applicable
to each comparison; source findings supply the reported effects.
