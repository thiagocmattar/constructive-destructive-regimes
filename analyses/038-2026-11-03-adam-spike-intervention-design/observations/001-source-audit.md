# Source audit for the intervention design

The author requested an intervention design around the three examples in
Analysis036's adam_manuscript_spikes.pdf, allowing earlier context. This
observation records existing evidence used to place the proposed forks; it
contains no intervention outcomes.

## Sources and reduction

The [audit script](../01_audit_sources.py) independently segments all 1,025 loss
states in each selected Run215 trajectory, recovers the largest recovered event
at its highest tested rate, and checks its a/p/c boundaries and ratio against
Analysis036's event table. It checks every loss/gradient-state join and source
SHA-256 against the prior manifest. It verifies the sizes and SHA-256 hashes of
all 12 retained tensor files in the three selected artifact folders, including
their complete final Adam recovery states. The [source review](../source_review.json)
records generator/input identities and the full covered paths.

The formal events are MLP 689/703/729, ResNet-32 415/422/482 and Transformer
787/794/855. Corresponding peak/pre-rise loss ratios are 4.1762988431,
9.5876975867 and 3.3140280203. These are fixed-reference excursions; the plot's
shaded +10% crossings are not their formal onsets. The MLP plotted window starts
after its formal onset. The original PDF was rendered with Poppler and inspected.

The [case table](../cases.csv) places prospective main forks at a-32 and computes
candidate clipping medians strictly over [a-96,a-33]. The
[extended context](../extended_context.csv) retains 225 raw scalar rows per case,
[a-96,a+128], without smoothing. It includes loss, raw-gradient projected change,
raw/momentum cosines, gradient/adaptive-direction cosine, raw/proposal norms,
and gradient dot the subsequently applied increment. The latter is an outcome
of the outgoing update and must not be used as a pre-update predictor.

## Findings relevant to design

All 14/7/7 updates at states a through p-1 have a negative raw-gradient dot
actual applied increment. The adaptive direction has positive gradient cosine
at every one of those states. These observations are consistent with
finite-step overshoot despite a locally descending direction; they do not
identify curvature or a causal source of the spikes. A sign-only uphill-step
guard would not activate on those 28 untreated rise updates.

In the pre-fork histories, negative adjacent raw cosine occurs at 61/64 MLP,
26/64 ResNet and 15/64 Transformer states. Adam's prospective direction has
positive gradient cosine throughout all three histories. These counts alone do
not estimate alarm precision or define harmful dynamics.

The candidate gradient caps are 0.6265010/1.3080322/0.3019403; the candidate
whole-Adam-proposal caps are 0.03731127/0.14692690/1.54104582. These norms have
different meanings and are not comparable across objectives. Their historical
rise exceedance counts are 6/14, 5/7, 7/7 for gradients and 5/14, 4/7, 3/7 for
proposals. Intervention paths would diverge, so those counts are not predicted
clipping rates.

The trainer's save routine overwrote recovery.pt and then renamed it final.pt.
The actual retained folders contain final.pt and raw audit gradients at 0, 1 and
1024, without intermediate recovery snapshots. New prefix reconstruction and
an untreated replay qualification are necessary to study earlier forks.

## Coverage and interpretation

Three fixed-rate arms, 3,075 loss observations and 3,072 optimizer records are
checked; 675 rows are exported as context. One seed0 initialization per model;
events selected retrospectively for large amplitude. There are no independent
seed uncertainty intervals. Historical hardware differs across models, although
new arms within each model are proposed to share hardware and a common prefix.

These are descriptive source checks, not causal intervention results or a
validated forecasting rule. The full proposed comparison, safeguards, outcomes,
replication and execution boundaries are in the [protocol](../PROTOCOL.md).
