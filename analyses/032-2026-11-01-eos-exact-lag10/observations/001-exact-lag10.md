# Observation001: exact short-lag profiles at retained GD states

Source: Run205's frozen config/input manifest, all three retrieved worker
inventories, 21 exact extracted checkpoint starts, and two qualified-or-flagged
MLP replay histories. The nine parent IDs are 059, 062, 120–126. Reduction:
`02_analyze.py`; independent complete-vector NumPy FP64 norms and dot products;
`03_record.py` generates the report and its displayed endpoints. Every curve
uses one anchored pair at each lag 1–10, with no projection, centering, averaging
or smoothing. All 297 vectors / 270 cosines pass verification. Primary coverage
is 25 windows / 250 values; the remaining two windows
are preserved as unqualified historical reconstructions.

Primary caption: Exact whole-model gradient cosine with the gradient at the
selected starting state held as reference, along ten constant-rate full-GD
updates. Rows identify anchors 500, 2000, 3000; columns identify MLP, ResNet-32 and
Transformer; colors identify the three original learning rates per model.
MLP LR 0.02 at 2000/3000 is omitted after historical replay fails qualification.
The final row is an extension of the Analysis024 horizon, with complete
training-objective coverage, including token-weighted Transformer reduction.
There is one dependent vector pair per point and one initialization per model;
no error bars or statistical population claims are supported.

Companion caption: The same full-vector measurements, adding the two failed
historical reconstructions with dashed lines and explicit legend labels.
Their numerical cosines are exact for the reconstructed vectors; those vectors
are not qualified as the original Run062 states. Small qualifying scalar
errors do not establish exact recovery of the other reconstructed starts.

Among primary profiles with negative lag 1, 17/17
have positive lag 2, 17/17 have positive
lag 10, and 17/17 are positive at
all even lags. These are descriptive counts over selected states, not
independent trials or a stability criterion. The source values and displayed
arrays are in `results.json`, `lag_profiles.csv` and `figure_manifest.json`.
Publication PDFs are under `figures/`; ignored Poppler renders are under
`scratch/`. Visual-review evidence is recorded separately after inspection.
