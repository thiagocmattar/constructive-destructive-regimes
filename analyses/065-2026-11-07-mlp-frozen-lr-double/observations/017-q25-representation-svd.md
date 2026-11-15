# Q25 and the effective dimension of representation movement

H053 / Run297 / continuation of Analysis065. Completed on owned Spark, following
the analysis requested on 15 November 2026. The manuscript/consolidation was
not edited.

Q25 changes the spectrum of representation movement, but does not uniformly
increase its effective dimension. Rank95 rises throughout the measured MLP and
Transformer sites; ResNet increases near the input and decreases deeper in the
network. Total squared movement falls at every measured site in the complete
64-update windows. Counts, leading-mode concentration and movement into the
baseline's complementary subspace need not change in the same direction.

| Full output, 64 changes | Rank95: Adam to Q25 | Participation rank: Adam to Q25 | Q25 / Adam squared movement |
|---|---:|---:|---:|
| MLP | 6 to 10 | 3.60 to 4.20 | 0.01664 |
| ResNet | 26 to 19 | 11.27 to 9.74 | 0.77229 |
| Transformer | 15 to 32 | 5.35 to 9.50 | 0.09635 |

Artifacts: [measurements](../svd_q25/metrics.csv),
[paired comparisons](../svd_q25/comparisons.csv),
[MLP figures](../svd_q25/mlp-q25-svd.pdf),
[ResNet figures](../svd_q25/resnet-q25-svd.pdf),
[Transformer figures](../svd_q25/transformer-q25-svd.pdf),
[one-page comparison](../svd_q25/comparison-q25-svd.pdf), and
[measurement contract](../svd_q25/README.md).

## Question and comparison

Does the distributed Q25 displacement cap make representation movement less
concentrated in a few directions than unchanged Adam? The intervention is the
complete original Q25 policy, with no additional learning-rate change. Both arms
start with identical parameters, optimizer moments, clock and random states.
The source is the authenticated fork from Runs293/294/295. Their saved scalar
metrics do not determine the representation SVD, so Run297 captures new dense
histories over the first 64 post-fork updates on owned Spark.

MLP uses all 5,000 training/reference images; ResNet trains on all 50,000 images
with its original physical groups of 1,000 and probes one intact group;
Transformer trains on all 59,680 sequences / 2,088,600 target positions and
probes the original 64 sequences / 2,240 positions. The complete original
feature coordinates are retained. There is one initialization per architecture.
Windows contain incoming changes 513-576, 257-320 and 545-608, respectively.
The first and last 32 changes are prespecified sensitivity windows. The ResNet
window precedes its historical late loss excursion; no claim about that spike
is made from this early window.

## Reduction and interpretation

For a site with P sample/position/feature coordinates, form D with 64 rows of
consecutive representation differences. Activations are stored in native FP32;
subtraction and accumulation use FP64. Do not center the changes, normalize
rows, average samples, or average features. The eigenvalues of D D-transpose
are squared singular values, and their sum is total squared movement. Report
rank95/99, participation and entropy effective ranks, top spectral shares,
absolute energy, coordinate RMS, adjacent reversal and displacement retention.
Window length and coordinate coverage are identical within every paired test.

Cross-arm Grams give Q25 energy inside and outside the unchanged Adam right
singular subspace that covers 95% of baseline energy. This is a comparison in
the same sample-feature space across interventions, not cross-layer temporal
overlap. Baseline residual energy outside its own selected subspace is reported
alongside Q25 residual energy. A larger outside fraction need not mean larger
absolute outside energy. It also does not establish directions absent from the
full baseline span. More effective modes do not imply more learning, greater
network capacity or better generalization.

## MLP result

The complete 64-update window has rank95 8 to 23 at the first affine site,
8 to 22 at the first tanh, 6 to 15 at the second affine and tanh, and 6 to 10
at the output. All five sites also have higher rank95 in each 32-update window.
Output squared movement falls to 0.016642 times baseline, a 98.34% reduction.
The output participation rank rises from 3.60 to 4.20, while its leading-mode
share rises from 39.46% to 42.01%. Thus a higher rank95 does not imply that every
concentration statistic decreases.
Uniformly scaling every change would preserve the normalized spectrum, so the
observed MLP effect is not merely uniform rescaling of the baseline trajectory.
This analysis does not isolate the policy's advantage over matched-amplitude
learning-rate controls; those arms are outside this requested SVD comparison.

At the output, 45.94% of Q25 energy lies outside the fixed baseline95 subspace,
but its absolute outside energy is 149.98 versus the baseline residual 725.20.
Every MLP site's absolute outside energy is smaller under Q25. The evidence
supports substantially damped movement with a different relative spectrum;
it does not show greater aggregate movement outside the selected baseline95
subspace.

## ResNet result

The effect changes across the network. In the 64-update window, rank95 rises
from 11 to 16 at conv1 and 18 to 23 at group1, but falls from 27 to 26 at group2,
29 to 23 at group3, 25 to 19 at pooling, and 26 to 19 at full-output log
probabilities. The target-log-probability readout also falls, from 27 to 23.
Output participation rank falls from 11.27 to 9.74. The output rank95 reduction
appears in both halves: 16 to 15 and 17 to 13. This does not support a universal
increase in effective dimension under Q25.

Total squared movement falls at all sites: the Q25/baseline ratio ranges from
0.1043 at conv1 to 0.7723 at the full output. In contrast to the MLP, the full
output's absolute movement outside the baseline95 subspace rises from 24,626
to 68,299, even as its total movement and rank95 decline. The outside fraction
is 17.05% of Q25 movement. All sites except conv1 have greater absolute outside
energy in the complete window (group1's increase is small, about 1.8%).
Thus fewer effective modes can coexist with greater movement outside the
baseline's dominant subspace. Directional change, spectral concentration and
total amplitude are separate measurements.

As context, mean full-training loss over the 64 post-update states changes from
0.37424 to 0.34733, while endpoint loss changes from 0.26611 to 0.27032.
The lower mean loss therefore coexists with lower output rank95; it is not an
endpoint improvement, nor evidence that increasing dimensionality is the
mechanism by which the intervention affects training.

## Transformer result

All seven sites have higher rank95 in the complete window and in each half.
The complete-window counts are 20 to 38 at embedding, 19 to 41 at attention1,
17 to 41 at block1, 16 to 33 at attention2, 17 to 35 at block2, 15 to 32 at the
full output, and 13 to 24 at target log probability. Output participation rank
rises from 5.35 to 9.50. However, embedding participation rank falls from 11.79
to 8.38 and target-log-probability participation falls from 5.85 to 5.50; their
leading-mode shares increase. Higher rank95 therefore does not mean uniformly
flatter spectra according to every concentration measure.

Full-output squared movement falls by 90.36%. The reduction is strongly
window-dependent: Q25/baseline energy is 0.01852 in the first half and 0.90435
in the second. Output rank95 rises from 9 to 21 and from 17 to 20 in those
halves. The full-window result is not a claim of uniform damping across time.

At the full output, 84.44% of Q25 energy lies outside the fixed baseline95
subspace. Absolute outside energy rises from 4.606e8 to 7.535e8, about 1.64x,
while total energy falls. Six of seven sites have greater absolute outside
energy; the target-log-probability readout is the exception. This differs from
the MLP's lower absolute outside energy at every site. Neither comparison tests
whether a direction is absent from the baseline's full span.

Mean full-training loss over the 64 post-update states falls from 0.95863 to
0.79393, and endpoint loss falls from 0.80386 to 0.75260. Nevertheless, adjacent
full-output reversals rise from 23.81% to 76.19%; all 31 adjacent pairs in the
second capped half reverse. Smoother loss and lower total motion thus coexist
with more frequent representation reversals in this window. A larger mode count
does not establish increased model capacity or identify what was learned.

## Verification and provenance

Run297 retains starting/ending states, recovery checkpoints, full activation
histories, scalar trajectories, source hashes and exact temporal/cross Grams on
owned Spark. Same-state probes repeat exactly and preserve random state and
model buffers. Both arms share an identical fork. Data/source identities and
historical first-gradient/update comparisons are recorded. The MLP continuation
matches all historical losses and retained gradient/parameter checks exactly.
ResNet/Transformer originated on different GPU models; their comparisons with
historical trajectories must be reported separately from the matched new pair.
Maximum historical loss differences (control/Q25) are 0.01212/0.02403 for
ResNet and 0.01934/0.00319 for Transformer. At fork+32, relative gradient
differences are 4.33%/10.69% and 20.77%/18.73%, respectively. The largest
relative parameter difference at that audit is 1.39% (Transformer control).
These results are fresh matched continuations, not exact reconstructions of the
historical non-MLP trajectories. Full qualification records are retained in
`svd_q25/provenance.json`.
All 81 source-dependency comparisons against the three original run manifests
pass; the manifests themselves match the local retained originals. The audit
and its checksum/size retrieval receipt are retained in `svd_q25/`.

The reducer independently checks diagonals and adjacent products against CPU
calculations and retained RMS values. A separate direct SVD checks MLP full
output, ResNet full-output log probabilities, and Transformer target log
probabilities, with explicit right-basis projections checking the cross-Gram
formulas. All nine direct checks and 114 compact site/window/arm checks pass.
Compact downloads are verified by archive and individual-file SHA-256 plus file
size. See `svd_q25/verification.json`, case retrieval receipts and source manifests.
The figures and tables are generated by `71_report_q25_svd.py`; complete source
histories are retained rather than reconstructed from plotted values.

All four PDFs / ten pages pass rendered visual review. The
[compute closeout](../../../runs/297-2026-11-15-cap-q25-temporal-svd/closeout.json)
records successful worker exits, unchanged scientific inputs, capture durations
of 9.51 seconds / 35.62 minutes / 78.22 minutes, and 70.434 GB retained on owned
Spark (67.204 GB of production evidence plus qualification captures and other
task files). No cloud resources were created and cloud spend is zero. Raw
histories and recovery states remain available; no manuscript files were edited.

## Figure captions and scope

`comparison-q25-svd.pdf` places the three cases in separate rows. Left panels
show rank95 at every site; right panels show the full-output cumulative spectrum
and total squared-motion ratio. Each row compares the matched 64-change windows
within that model; differences in datasets and coordinate sets prevent treating
the rows as a controlled architecture-only comparison.

Each `<case>-q25-svd.pdf` has three pages:

1. Unchanged Adam and Q25 over the complete 64-change window: rank95 by site,
   Q25-to-baseline squared-motion ratio, cumulative output spectrum, absolute
   output spectrum, outside-baseline95 energy normalized by baseline total
   energy, and full-training loss. Points in the residual panel share a common
   baseline-energy denominator within each site.
2. Cumulative squared singular-value shares at every measured site, with the
   95% threshold and both effective mode counts. Cross-entropy models include
   the auxiliary target-log-probability readout alongside the full output.
3. Rank95 for the first 32, last 32 and all 64 incoming changes. Rank ceilings
   differ across panel lengths but are matched between interventions in each
   panel. These windows are descriptive sensitivity checks, not independent
   experimental replicates.

No uncertainty interval is estimated from one initialization; updates, features
and examples are not treated as independent training replicates. Results concern
these fixed probes and early post-intervention windows. No inference of learning
capacity, generalization or causal origin of loss spikes follows from the SVD.
