# Observation006: complete 32-step cosine response to one tenth LR

Verified snapshot 20261104T042939Z contains 19/30 completed primary branches
and no failures. All three 32-step cosine floors now have complete responses
across the three cases. Clipping and decay conditions remain unfinished.

Source: Run222 artifacts/snapshots/20261104T042939Z/payload, selected by
retrieval.json. All 163 captured files pass byte-size/SHA256 verification.
Archive SHA256:
90b9c05fff46b2b69b6015b540ce388585ce8b45c1665d86e199d93ceb2418ab.
Historical states 960..1023 come from pinned Run221 parent histories.
Generating script: ../01_report.py; exact reductions: ../comparison.csv and
../results.json; aligned source values: ../aligned_values.csv.

Figure: ../figures/cosine32_floor0p1.pdf; SHA256
16ae72217209d7a0daf2d53eed5005b0065e56ace0f56d626e421fc95feb66d8.
Caption: unchanged Adam versus cosine LR reduction from the same warm state
1024, reaching 0.1x after 32 updates and holding thereafter. Columns identify
MLP/CIFAR-10, ResNet-32/CIFAR-10 and Transformer/WikiText-2. Rows show loss,
raw rho, raw gradient norm, $\eta\widehat\kappa_t$ and applied LR/base.
The solid, unsmoothed curves retain both complete trajectories within independent
panel ranges. Relative update zero marks intervention onset and the hardware
handover to Spark for all arms.

Coverage: 64 historical before observations, 65 Spark boundaries and 64 applied
updates per case. All 2,910 scalars in this figure match source arrays; 14,550
pass across the five available figures. Independent schedule, clip-factor,
norm/cosine identity and identical-start checks pass. The new vector PDF was
rendered and visually inspected for legible legends/ticks, raw rho, norms in the
third row, independent ranges and no clipped curves or label collisions.
The four earlier reviewed PDFs remain byte-identical. No smoothing or extra
terminal update is introduced.

Mean/final unregularized training loss changes against the fresh Spark unchanged
control are -45.519%/-36.015% (MLP), +4.887%/+13.312% (ResNet), and
+3.492%/+8.179% (Transformer). Mean covers states 1025..1088; final is 1088.
For ResNet and Transformer, lower cosine floors give higher final training loss
within this common 64-update horizon. This describes unequal LR exposure;
smaller or smoother raw diagnostics do not themselves imply faster loss decrease.
No final ranking of clipping/decay or optional schedules is established.

All responses are conditional on one selected seed and warm state. No uncertainty
over seeds, long-horizon convergence or held-out gain is established. Run221's
ResNet cross-GPU gradient gate failure remains preserved. Run222 exact within-Spark
replay supports matched current controls without claiming historical bitwise replay.
Frozen science is unchanged. Full checkpoint retrieval and final 5-10 factual
findings remain pending. No manuscript change, cloud expense or push.
