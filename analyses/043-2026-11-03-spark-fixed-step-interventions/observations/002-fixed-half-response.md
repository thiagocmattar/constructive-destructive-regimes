# Observation002: first complete three-model fixed-half response

Partial suite milestone, not a final intervention ranking. At snapshot
20261104T005942Z,7/30 primary branches are complete and no branch has failed.
The unchanged and fixed-half conditions are complete for all three cases;
other interventions remain in progress or queued. Run222 science is unchanged.

Source: verified snapshot in Run222/retrieval.json, with payload under
artifacts/snapshots/20261104T005942Z/payload. The transfer archive SHA256 is
34f97280f312ecb9c1e0ad2aa51c7cd50a8f32a27cd96da54e24a0599f00cc21;
all103 captured files pass individual byte-size/SHA256 checks. Historical
before values come from the byte-pinned Run221 parent-history JSON files.
Generating script: ../01_report.py. Source-aligned values and reductions are
../aligned_values.csv and ../comparison.csv; ../results.json records coverage.

Figure: ../figures/fixed_half.pdf. Three model/dataset columns compare unchanged
Adam (gray) with fixed half LR (blue), after the same saved warm state1024.
Rows show task loss, raw consecutive-gradient rho, raw gradient norm,
$\eta\widehat\kappa_t$ and applied LR/base. Solid unsmoothed lines and independent
panel Y ranges retain all observations. Historical960..1023 is the common
before period; Spark1024..1088 is the matched after period. The intervention
and hardware boundary is marked at relative update0. Gradients and norms use
the same full-objective reductions and partitions as the frozen protocol.

All2,910 plotted scalars match selected source arrays exactly. Independent
norm/cosine identities, chosen LR, unchanged fork boundaries and clip-factor
formulas pass direct checks. The endpoint LR value only closes the final
applied interval; no update is applied at1088. Raw rho is not an optimizer
alignment, and the projected statistic is not an identified Adam curvature.

The vector PDF was rendered and visually inspected: all15 panels, legends,
independent ticks, raw rho and the hardware boundary are legible. No label
collision or curve clipping was observed. Creation/modification timestamps
are omitted from PDF metadata so unchanged completed curves do not churn across
snapshot reports. Two complete reductions produce byte-identical PDF SHA256
daf7823972efd9e3385b7fc9c85a5f8239865b861ccb577d07b8c0ef2309194f.

In this fixed-half comparison, mean/final post-update task loss changes versus
same-Spark unchanged are -46.824%/-38.256% for MLP, +4.838%/+9.696% for ResNet,
and +3.234%/+4.855% for Transformer. Mean uses64 losses at1025..1088; final is
1088. These provisional three-case responses show that smaller LR does not
uniformly improve the attained task loss in this window. They do not rank the
remaining unfinished methods or establish generalization, optimality or a
universal scheduling rule. Full precision values remain in the CSV/JSON.

The original Run221 ResNet cross-GPU gradient gate failure0.87% remains
preserved. This analysis uses exact matched same-Spark starts and verified
within-Spark replay; historical before values are context, not a claim of
historical bitwise continuation. One selected initialization/state, finite
64-update horizon and differing LR exposure limit scope. Full-suite findings
and final raw-checkpoint retrieval are pending; no manuscript edit or push.
