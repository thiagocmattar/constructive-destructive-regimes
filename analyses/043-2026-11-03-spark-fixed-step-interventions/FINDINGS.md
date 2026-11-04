# Findings: fixed-step Spark interventions

All 36 branches completed: ten primary conditions and two admitted optional
cosine durations, each across MLP/CIFAR-10, ResNet-32/CIFAR-10 and
Transformer/WikiText-2. Every branch starts from the exact same warm state 1024
within its case and performs 64 full-objective updates to state 1088. Effects below
compare with fresh unchanged Spark controls, using the highest tested LR: 0.001
for MLP and 0.003 for ResNet/Transformer. Mean loss covers states 1025..1088;
final loss is at 1088. Negative percentages indicate lower unregularized training
loss. These are controlled responses at one selected seed and state.

- **LR reductions help the MLP training response.** Across the seven fixed/cosine
  LR treatments, mean loss is 44.907-46.824% lower and final loss is 35.567-39.306%
  lower than unchanged. Fixed half has the lowest mean loss; cosine32 to 0.5x has
  the lowest final loss among the frozen grid. This is a finite-grid observation,
  not an optimum beyond the tested conditions.
- **The same LR changes slow ResNet's training progress.** Mean loss is
  2.022-7.436% higher and final loss is 7.046-15.114% higher. The unchanged control
  already decreases throughout this post-fork window. This selected state does
  not provide a ResNet loss spike for an intervention to remove.
- **Smoother Transformer gradient diagnostics do not imply faster task-loss
  decrease.** Fixed half changes mean raw rho from -0.299 to +0.589 and the
  standard deviation of $\eta\widehat\kappa_t$ from 0.565 to 0.106, while mean/final
  loss becomes 3.234%/4.855% higher. All seven LR treatments give higher mean and
  final training loss here, despite weakening the late raw-gradient excursions.
- **Cosine duration changes exposure as well as transition shape.** At floor
  0.25, durations 8/32/64 have mean applied LR ratios 0.302734/0.443359/0.630859.
  The slower decay has smaller training-loss penalties for ResNet/Transformer,
  while MLP mean and endpoint rankings differ. These arms do not isolate a
  smoothing mechanism at matched cumulative LR. Cosine64 approaches its floor
  over the observed updates; its exact floor would be applied on the next,
  unexecuted update. No extra update or floor-hold period is claimed.
- **Median clipping has mixed responses and low duty.** It clips 15/64 MLP,
  0/64 ResNet and 3/64 Transformer updates. MLP mean loss improves 16.585%, but
  its final loss is 21.956% higher; an oscillating trajectory makes a single
  endpoint an incomplete summary. Transformer mean/final changes are only
  -0.058%/-0.178%; ResNet is unchanged because clipping never activates.
- **Double-median clipping is an inactive condition.** It clips no update in
  any case and exactly reproduces control weights, Adam moments, previous raw
  gradient and RNG at the endpoint, as well as the scalar trajectories. This
  closes the fixed-threshold test without showing that active clipping is
  generally ineffective.
- **Decay reduces parameter norms without improving these training-loss
  responses.** Both decoupled decay values give higher mean and final task loss
  in all three cases. At 0.1, final losses are 14.659%/12.872%/1.159% higher for
  MLP/ResNet/Transformer. The companion figures distinguish the decay increment,
  actual parameter increment and raw gradient norm; their magnitudes are not
  interchangeable diagnostics.
- **Better adjacent raw alignment does not consistently rank attained training
  loss in this response window.** Effects depend on the
  model and current regime. No branch fails or becomes nonfinite, but 64 updates
  from one selected state establish neither asymptotic convergence nor a general
  stability boundary, cross-seed uncertainty or held-out benefit. Raw rho and
  $\eta\widehat\kappa_t$ describe gradient history under Adam, not a proven Adam
  curvature or stability threshold.

## Complete training-loss comparison

Entries show mean / final percentage change against the contemporary unchanged
Spark control. Optional durations were admitted solely by the frozen time rule.

| Intervention | MLP / CIFAR-10 | ResNet-32 / CIFAR-10 | Transformer / WikiText-2 |
| --- | ---: | ---: | ---: |
| unchanged | +0.000% / +0.000% | +0.000% / +0.000% | +0.000% / +0.000% |
| fixed_half | -46.824% / -38.256% | +4.838% / +9.696% | +3.234% / +4.855% |
| fixed_quarter | -45.766% / -35.567% | +7.436% / +15.114% | +5.922% / +9.099% |
| cosine32_floor0p5 | -45.876% / -39.306% | +2.641% / +7.078% | +1.582% / +3.449% |
| cosine32_floor0p25 | -45.722% / -37.338% | +4.031% / +10.911% | +2.687% / +6.129% |
| cosine32_floor0p1 | -45.519% / -36.015% | +4.887% / +13.312% | +3.492% / +8.179% |
| clip_median_x1 | -16.585% / +21.956% | +0.000% / +0.000% | -0.058% / -0.178% |
| clip_median_x2 | +0.000% / +0.000% | +0.000% / +0.000% | +0.000% / +0.000% |
| adamw_0p01 | +0.182% / +1.869% | +0.615% / +1.228% | +0.025% / +0.015% |
| adamw_0p1 | +1.902% / +14.659% | +6.354% / +12.872% | +0.656% / +1.159% |
| cosine8_floor0p25 | -46.666% / -36.025% | +6.362% / +13.944% | +4.862% / +8.338% |
| cosine64_floor0p25 | -44.907% / -39.218% | +2.022% / +7.046% | +1.133% / +3.477% |

## Qualification, evidence and scope

Run221's original ResNet cross-GPU gradient qualification failed at relative error
0.0087187473 (0.87%) against the frozen 0.001 gate, despite exact weights/moments
and close loss. The failed qualification remains preserved. Run222 instead passes
exact within-Spark direct-versus-resumed weights, moments, previous raw gradient
and RNG in every case, with exact common starts. Historical states 960..1023 are
context from prior hardware; all after curves use Spark, and the hardware boundary
is marked. Historical bitwise replay is not claimed.

The final archive is 3,351,642,623 bytes, SHA256
b24f34a121e658c5c8da03ca0d051819a1c6ee0c8b97322bed2043b15d22c0d1.
All 347 transferred files are verified, including all 337 files in the worker's
terminal inventory. The final checks validate 31 frozen inputs, 19 executed
source copies and 45 endpoint checkpoints. Run222 records 2,304 main updates,
12 qualification updates and 2,361 full-gradient calls. Separate Run221 mechanics
used nine updates and three zero-update warm-parent gradients. Full validation/
test endpoints are retained descriptively in comparison.csv; no selection was
made using them. The frozen source and every failure remain inspectable.

The worker ran 39,620.6 seconds (11 h 00 min 20.6 s), finishing at 06:52:26 BRT,
before the 07:45 compute cutoff. Its process has exited; the unrelated suspended
Spark queue remains untouched. Owned Spark: USD 0 cloud spending. Retained storage
is approximately 10.66 GB for the final local payload plus archive and 7.31 GB
allocated in the isolated Spark task directory. Storage is retained for recovery;
no cloud resource or machine shutdown is involved. The overnight monitor is
stopped after verified completion. No manuscript edit or push.

## Figures and reproduction

[Report and all 13 vector figures](REPORT.md), [comparison](comparison.csv),
[aligned values](aligned_values.csv), [final verification](verification.json),
[storage accounting](storage.json), and [source observations](observations/INDEX.md).
Every main figure uses three model columns and compares unchanged with one
intervention using solid unsmoothed lines, independent Y ranges, raw rho, raw
gradient norms in the third row, and an explicit before/after hardware boundary.
The two decay companions show post-only parameter and increment norms because
historical parameter norms were not recorded. All 34,326 plotted scalars are
checked against source arrays; all 13 PDFs have been rendered and visually reviewed.

From the repository root:

    python analyses/043-2026-11-03-spark-fixed-step-interventions/01_report.py
    python analyses/043-2026-11-03-spark-fixed-step-interventions/02_verify_final.py

## Literature motivation

Global gradient norm clipping follows the intervention family studied by
[Pascanu, Mikolov and Bengio (2013)](https://proceedings.mlr.press/v28/pascanu13.html).
Decoupled parameter decay follows the AdamW distinction between decay and an L2
term inside Adam's gradient/moments in
[Loshchilov and Hutter (2019)](https://arxiv.org/abs/1711.05101).
The smooth half-cosine schedule is adapted from the cosine schedule in
[Loshchilov and Hutter's SGDR (2017)](https://arxiv.org/abs/1608.03983);
this experiment uses a single decay without restarts. These sources motivate
separate controls; they do not establish their assumptions or empirical benefits
for this selected three-case response window.
