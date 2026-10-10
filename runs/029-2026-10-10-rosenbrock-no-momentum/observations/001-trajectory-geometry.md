# Observation001: Run029 trajectory and vector geometry

Sources: artifacts/attempts/001-local/history.npz and summary.json; identities
and sizes in inventory.json, scientific/source/config coverage in verification.json
and manifest.json. Generator: Analysis010/02_plot.py. Output:
figures/trajectory_geometry.pdf, with its source/output hashes in Analysis010's
figure_manifest.json. The PDF passed visual review.

Coverage: all 10,000 accepted updates, both coordinates; position 0 and all
post-update objective values. The objective is exact 2D Rosenbrock at one fixed
start. Raw g is pre-update/unclipped; m is uncorrected first moment; d excludes
LR/decay/descent sign; u is the actual displacement. All stored values are
float64. No sampled/skipped/failed update or excluded final point.

Reduction/caption: six panels show analytic contours and the full trajectory
(start square, minimum star, final circle), log post-update objective,
adjacent cosines, log individual norms, log cumulative A/B, and B/A. Cosines
use trailing 100-update means requiring 100 valid pairs, first mean at 101.
Other curves are unsmoothed. Full contours use500×500 analytic mesh.
A=sum ||z||, B=||sum z||. First cosine is undefined; no epsilon replacement
or inferred optimizer displacement from g. Stream color and labels distinguish
g/m/d/u; A solid and B dashed. Independent stream units remain labeled.

Result: final objective 0.02245074667498919, final point(0.9801449971, 0.9755356549). Run029 never meets the declared joint convergence criterion. Removing first-moment memory retains adaptive second moments; this control is not SGD. Its raw and stored-momentum vectors are identical.
Full exact endpoint/phase statistics are in summary.json, with the cross-run
comparison and limitations in Analysis010/observations/001-controlled-results.md.

Uncertainty/caveats: one deterministic start/2D objective/LR/beta2, no seed
confidence intervals. Neither adjacent cosine nor cumulative ratio alone
establishes objective benefit, formal stability, information destruction,
generalization or asymptotic convergence. Decay0 is Adam as AdamW's zero-decay
case; beta1=0 removes first-moment memory but keeps adaptive second moments.
