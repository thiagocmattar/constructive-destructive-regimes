> Evidence snapshot; original scientific content retained. Relative links were rebased.
> Source: [007-clipping-median-response.md](../../../../../../analyses/043-2026-11-03-spark-fixed-step-interventions/observations/007-clipping-median-response.md).

# Observation007: complete historical-median clipping response

Verified snapshot 20261104T052943Z contains 22/30 completed primary branches
and no failures. Global full-gradient clipping at the historical median is
complete across all three cases; the wider suite remains unfinished.

Source: Run222 artifacts/snapshots/20261104T052943Z/payload, selected by
retrieval.json. All 178 captured files pass byte-size/SHA256 verification.
Archive SHA256:
6c7ed9422c5343ceb92a1d3bc1a7ed531539c39bae555796bbcd828741ad1b0d.
Historical states 960..1023 and threshold medians use pinned Run221 parent
histories. Generating script: ../01_report.py; exact reductions:
../comparison.csv and ../results.json; aligned values: ../aligned_values.csv.

Figure: ../figures/clip_median_x1.pdf; SHA256
65fd66561aa866464839de119095a9267eeb403b0c73ed6aab711d5a76bab1f6.
Caption: unchanged Adam versus global raw-gradient norm clipping after warm
state 1024, at the fixed median of historical states 960..1023. The LR stays at
its original highest tested value; both Adam moments consume the clipped input,
while plotted rho and gradient norms use the original raw task gradients.
Thresholds are 0.754178587823245 (MLP), 0.0018601596289618749 (ResNet) and
0.31280553094335695 (Transformer), fixed before treatment outcomes.

Columns identify MLP/CIFAR-10, ResNet-32/CIFAR-10 and Transformer/WikiText-2.
Rows show loss, raw rho, raw gradient norm, $\eta\widehat\kappa_t$ and applied
LR/base. Solid unsmoothed curves retain complete trajectories with independent
panel ranges. Relative update zero marks intervention onset and the hardware
handover to Spark for all arms. Coverage is 64 historical observations, 65 Spark
boundaries and 64 applied updates per case. All 2,910 plotted scalars match source
arrays; 17,460 pass across six figures. Independent LR, clip-factor, norm/cosine
identity and identical-start checks pass. The vector PDF was rendered and visually
inspected: legible labels, rho, norms in row three, independent ranges, no clipped
curves or label collisions. The five earlier PDFs remain byte-identical.

Clipping activates on 15/64 MLP, 0/64 ResNet and 3/64 Transformer updates.
Mean/final unregularized training loss changes against the fresh Spark unchanged
control are -16.585%/+21.956% (MLP), 0%/0% (ResNet), and -0.058%/-0.178%
(Transformer). Mean covers states 1025..1088; final is 1088. The MLP mean and
endpoint therefore give different assessments over this oscillating window.
The inactive ResNet condition exactly reproduces all 65 control boundaries for
loss, raw rho/norm, projected change, parameter norm and LR. This is evidence
about the fixed threshold's inactivity, not evidence for active clipping benefit.

These are conditional short responses from one selected seed and warm state;
no uncertainty over seeds, long-term convergence or held-out gain is established.
Run221's ResNet cross-GPU gradient gate failure is retained. Exact within-Spark
replay and identical treatment starts do not imply historical bitwise replay.
Frozen science is unchanged. Full checkpoint retrieval and final findings remain
pending. No manuscript change, cloud expense or push.
