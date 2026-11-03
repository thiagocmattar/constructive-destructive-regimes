# Eight-update cooldown and cosine recovery

The combined hold-and-cosine policy improves mean/final training loss over the
one-step rule and, slightly, over always-half LR in this selected case. The abrupt
hold still produces repeated excursions. Repeated triggers keep the cosine
policy below base LR for365/367 updates; this is not evidence that its timing
beats a matched fixed-rate or open-loop schedule.

[Paired three-column figure](figures/paired_three_column.pdf) compares unchanged
Adam with half LR, eight-update cooldown, and cooldown plus cosine recovery in
separate columns. All trajectories are solid; each panel uses its own Y range
while retaining both full curves. Rows are loss, raw adjacent-gradient $\rho_t$,
gradient norm, $\eta\widehat\kappa_t$, and actual LR. The [paired fixed zoom](figures/paired_three_column_zoom.pdf)
uses the same predeclared673..745 window. [Observation002](observations/002-paired-comparison.md)
records definitions and13,230 exact plotted-value checks. The one-step series
remains omitted from the primary figure. Earlier four-arm and full five-arm
overlays remain available; no measurements or training changed.

## Frozen comparison

Run220 / Analysis042 continues H028. User approval: `Go on` after the explicit
8-update hold /8-update cosine-recovery proposal. Development rung: same
CIFAR-10 full-objective MLP, seed0,5,000 training images, physical chunks1,000,
tanh3072-200-200-10, half squared error summed over outputs and mean over images.
All five arms start from the authenticated state657 weights, both Adam moments,
global age, RNG and raw/actual observer histories;367 updates end at1024.
Adam base LR0.001, half0.0005, betas(0.9,0.999), epsilon1e-8, no decay/clipping;
FP32 parameters/gradients/moments, FP64 products, TF32 off, foreach/fused false.

The available pre-update $\eta\widehat\kappa_t\geq1.8$ predicate is unchanged.
At state s, manuscript t=s+1; the raw-pair statistic governs s to s+1, without
LR rescaling. It observes projected gradient change under Adam; it is not an
identified curvature boundary or safe outgoing step certificate.

- Unchanged, always-half and one-step rules reproduce Run219 controls.
- Hold: every eligible update resets an8-update half-LR hold, including itself;
  after8 consecutive held updates without another trigger, return to base LR.
- Hold plus cosine: the same hold, then
  LR(k)=0.0005+0.00025(1-cos(pi*k/8)), k=1..8. A trigger at any stage restarts
  the hold. Consecutive eligible updates also restart it; this is not a rising-edge
  detector. No compounding or changes to moment recurrences.
- A finite undefined diagnostic continues an active schedule or uses base when
  inactive. Nonfinite gradients/statistics fail; no undefined decisions occurred.

N=8 and recovery=8 were fixed before this run, following the earlier development
result. This is one selected initialization/state, not independent confirmation
or a search for optimal durations.

## Outcomes

| Policy | Mean training loss | Final training loss | Peak excess | Excess area | Below base LR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unchanged Adam | 0.01472254 | 0.00952230 | 3.176299 | 80.061606 | 0/367 (0.00%) |
| Always-half LR | 0.00979396 | 0.00672606 | 0.000000 | 0.000000 | 367/367 (100.00%) |
| One-step reduction | 0.06381674 | 0.05916404 | 98.463380 | 1277.099957 | 18/367 (4.90%) |
| 8-update hold | 0.01177662 | 0.00694463 | 1.334186 | 26.055143 | 337/367 (91.83%) |
| 8-update hold + 8-update cosine recovery | 0.00957221 | 0.00643690 | 0.000000 | 0.000000 | 365/367 (99.46%) |

Mean covers367 post-update losses, states658..1024. Peak excess and area use
max(L/B-1,0) over states689..1024 inclusive, with fixed untreated
B=0.013039679266512394. Area is the sum over336 recorded states. The full
continuation also retains events before689 and later replacement events.
[Exact comparison](comparison.csv), [all1,840 source-aligned boundaries](aligned_values.csv),
and [all detected excursions](excursions.csv) retain values beyond this table.

## Research iteration: eight findings

- All three original policies reproduce every368 Run219 boundary loss, raw norm
  and available statistic exactly. Their endpoint weights/moments/RNG/previous
  raw gradient/actual increment also match. Every arm has the same saved start.
- The first return to base is a useful local comparison: all conditional arms
  share the trajectory throughstate660 (loss0.0139453749). For660 to661, one-step
  restores0.001 and reaches0.0216740761; both new policies keep0.0005 and reach
  0.0155140851. Keeping the next update smaller limits this rise; it does not
  make that update loss-decreasing or identify the mechanism of every later spike.
- The8-update hold greatly improves the one-step negative result, but remains
  20.24% worse than always-half in mean training loss and3.25% worse in final
  loss. Its largest full-view loss is0.0304370 at718. It has33 excursions of at
  least10% over their own pre-rise loss, versus0 for always-half; all recover by1024.
- The cosine version improves mean/final loss over the abrupt hold and is2.26%/
  4.30% lower than always-half, respectively. It reaches95% of the original
  pre-rise reference after46 updates, versus51 for always-half and66 for the hold.
  Its fixed-reference excess after689 is0; the two early >=10% own-reference
  excursions remain recorded (largest ratio1.11249), with no censored event.
- The abrupt hold spends337/367 updates at half LR, with77 triggers and55
  retriggers. The cosine version has59 triggers/58 retriggers:276 half-LR
  updates,89 ramp updates and only the initial2 base-LR updates. It never finishes
  a recovery to full LR after the first reduction. Mean LR/base is0.54087 for
  the hold and0.53993 for the cosine policy; similar averages coexist with
  different trajectories, but average-rate and timing effects remain unseparated.
- This weakens the immediate-return design and keeps smooth recovery alive as
  a development hypothesis. Raw reversals/threshold crossings still occur on
  the calmer cosine path; a crossing alone does not establish a damaging update.
  Holding LR low helps here, without validating the diagnostic as a predictor.
- Endpoint test losses are descriptive only: unchanged0.7770660, always-half
  0.7828646, one-step0.6603654, hold0.7785980, cosine0.7856769, each over all10,000
  images. The cosine training gain is not a held-out improvement. No tuning or
  selection claim is made from these endpoints.
- Four new schedule/native-arithmetic contracts plus parametrized cases total
  16 focused checks; four frozen-rule checks also pass (20 total,3.67s).
  Both new policies pass24 direct versus12+12 resumed equality for weights,
  moments, RNG, policy and complete numerical observer history. The original
  audit compared UTC metadata and stopped; its failure is retained, and only
  UTC fields are excluded by the infrastructure correction. No scientific inputs
  changed, no extra variant/seed was launched, and no manuscript was edited.

## Verification, cost and limits

The frozen26-input manifest and authenticated parent hash pass. An independent
last-trigger-age calculation verifies all1,835 outgoing LR choices and clocks;
all1,840 boundary statistics agree with the alternative norm/cosine expression.
All15,876 plotted scalars match source arrays exactly. Four PDFs were rendered;
the clean figures have no orange strokes or one-step legend entry. Common scales,
unsmoothed observations and the predeclared zoom are preserved. LR is a step
function over [s,s+1); the terminal plotting value only closes the last interval,
and the source table records no applied decision at1024.

Canonical main work:1,835 updates /9,175,000 image-update exposures. Qualification:
108 updates /117 gradient calls. Total actual work:1,943 updates and1,957 complete
forward/backward calls /9,785,000 image evaluations, plus50,000 held-out image
loss evaluations. No prefix retraining or cloud resource. Active workers57.75s;
first-to-final attempt span155.32s, including the audit correction; under the600s
bound. Raw evidence2,487,747,183 bytes (2.317GiB); peak allocated GPU memory
325,459,456 bytes. Zero rental/cloud spending.

[Verification](verification.json), [inventory](raw_inventory.csv), and
[results/cost record](results.json) retain the checks and exact quantities.
A fixed LR near0.00054 and a matched open-loop schedule are missing controls
for incremental timing value. Different total LR exposure, a selected development
state, one seed, finite horizon and the retrigger semantics limit inference.
No general stability, generalization, optimal-duration or LLM-transfer claim.
