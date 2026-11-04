# Observation003: complete fixed-quarter response across three cases

At verified snapshot20261104T015956Z,11/30 primary branches are complete with
no failed branch. Fixed-quarter LR is now complete across all three cases;
cosine tests have begun, while the broader suite remains unfinished.
This observation records one partial response, not a final method ranking.

Source: Run222 scalar snapshot under
artifacts/snapshots/20261104T015956Z/payload, selected by retrieval.json.
Transfer archive SHA256:
270067dc1a8cd9d1c26be77baf2d68c000137526f54603ee32962d055c23bb34.
All123 captured files pass byte-size/SHA256 checks. Historical960..1023 values
come from the pinned Run221 parent histories. Generating script: ../01_report.py;
source-aligned table ../aligned_values.csv; exact reductions ../comparison.csv
and ../results.json. No scientific input or runtime rule changed.

Figure: ../figures/fixed_quarter.pdf; SHA256
17bc231b0bf01c8a6dc274a905651961e41b9fc6af32a43b8e5af764fb7179b1.
It compares unchanged Adam with fixed0.25x LR after the same warm state1024.
Three columns identify MLP/CIFAR-10, ResNet-32/CIFAR-10 and Transformer/WikiText-2.
Rows show loss, raw rho, raw gradient norm, $\eta\widehat\kappa_t$ and applied
LR/base. All lines are solid and unsmoothed; each panel has an independent Y
range retaining both complete curves. The shared historical before period and
same-Spark after period are explicitly separated at relative update0.

Coverage:64 common historical before observations and65 Spark boundary values
per case, plus64 applied updates. The unchanged prefix is drawn once; the
intervention begins at0. All2,910 scalars for this figure match source arrays;
with the preserved fixed-half figure the total is5,820. Independent norm/cosine,
LR, clip-factor and identical starting-boundary checks pass. Terminal LR only
closes the last applied interval. Both PDFs render as one page; the new quarter
figure was visually inspected for legible legends/ticks, raw rho, independent
ranges and no clipped lines or label collisions. The fixed-half PDF remains
byte-identical to the previously reviewed artifact.

Mean/final unregularized training loss changes against the matched unchanged
continuation are -45.766%/-35.567% (MLP), +7.436%/+15.114% (ResNet), and
+5.922%/+9.099% (Transformer). Mean covers1025..1088; final is1088. These are
conditional64-update responses. A positive raw cosine or smaller projected
change does not by itself rank attained task loss. No claim is made about
unfinished clipping/decay/schedules or longer horizons.

Run221's ResNet cross-GPU gradient gate failure remains retained. Same-Spark
weights/moments/raw-history/RNG replay is exact, and all treatment starts are
matched to the fresh control. The historical hardware handover, one selected
seed/state, finite horizon and unequal LR exposure limit interpretation. Final
suite findings and complete checkpoint retrieval remain pending. No manuscript
change, cloud expense or push.
