> Evidence snapshot; original scientific content retained. Relative links were rebased.
> Source: [001-beta2-reduction.md](../../../../../../analyses/059-2026-11-07-mlp-beta2-reduction/observations/001-beta2-reduction.md).

# Beta2 reductions from the same warm MLP state

Source: Run232 attempt01 on owned Spark GB10; retained Run223 same-Spark control.
Generator: `../01_report.py`; `../results.json` records source hashes, full event
lists, reductions and PDF hash; `../plotted_values.csv` stores every plotted
value. Independent checkpoint/recurrence audit: Run232 `03_verify.py` and
`verification_final.json`. Complete original input/cache/checkpoint identities
are pinned in Run232 `input_manifest.json` (41 authenticated inputs).

## Comparison and coverage

Each treatment starts at the exact untreated state512 and changes beta2 from
0.999 to0.95/0.9/0.5. All run288 updates through800. Constant LR0.001, beta1=.9,
epsilon1e-8; no clipping or decay. Preserve Adam m/v/step and RNG. One seed0,
fixed5000-image full MSE objective and1000-image physical partitions, native
FP32 model/gradient/moments with FP64 diagnostic reductions. All10000 test
examples are evaluated at800. No dataset/model changes, smoothing or resampling.

Exact fork-gradient and two-update reference checks pass for weights, Adam
state, raw-gradient history and RNG. A second two-update unchanged branch
using the added observer also matches the retained reference exactly. No full
control or prefix rerun. Training trajectories do not include those qualification
updates; every treatment reloads the original checkpoint independently.

The native bias correction becomes 1-beta2_new^513 on the first outgoing update:
approximately1 for every new beta, versus0.40145685685590815 under unchanged Adam.
Thus the treatment changes both second-moment adaptation and its normalization.
No product-of-past-betas correction or moment rescaling is used. This is not
an isolated test of memory length. First-step tensors independently satisfy
the new-beta recurrence and native direction formula within FP32 rounding.

## Results

Mean uses all resulting parameter states513..800 equally. Tail uses769..800.
Event counts use the established fixed pre-rise reference on complete0..800
histories, selecting peaks after512 with peak/pre-rise ratio>1.1. Censored
events count separately; do not treat lack of recovery as a removed spike.

| beta2 | Mean training loss | Change vs unchanged | Last32 mean change | Final loss | Maximum loss | >10% events (recovered + censored) | Endpoint test loss |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 0.999 control | 0.0208042 | reference | reference | 0.0302502 | 0.0550626 | 15 + 1 | 0.776960 |
| 0.95 | 0.0242951 | +16.78% | +33.36% | 0.0191725 | 0.142619 | 12 + 1 | 0.754255 |
| 0.9 | 0.0245821 | +18.16% | +43.51% | 0.0158850 | 0.0739020 | 12 + 1 | 0.746844 |
| 0.5 | 0.0472288 | +127.02% | +163.88% | 0.0347593 | 0.0763622 | 0 + 1 | 0.691013 |

None improves mean or tail training loss in this selected window. Beta0.95
has the largest immediate maximum;0.5 produces a prolonged unrecovered
excursion even though its raw cosine is less negative and its event count is
smaller. Beta0.95/0.9 end below the control's current excursion, so endpoint
loss alone reverses the window-mean ranking. Test loss is lower for all three
at the one observed endpoint; no across-seed generalization claim follows.

## Figure caption

Three fixed beta2 interventions for CIFAR-10/MLP, each compared with unchanged
beta2=0.999. Columns set0.95,0.9,0.5 at state512; LR stays0.001. Rows match
Analysis058 Figure04: full training loss (log), projected gradient change,
adjacent raw-gradient cosine, raw-gradient norm, then beta2 in place of LR.
Gray shading marks the common pre-intervention context448..512. All curves
are unsmoothed with independent diagnostic scales; beta2 uses a common range.
The terminal state's last applied interval is closed at800 only for display.

Coverage: all353 boundary values448..800 per series and352 applied settings
448..799; shared control repeated in three columns. All10,584 plotted scalars
match records. The7.16 x6.35-inch PDF has embedded fonts and vector curves;
full-size render inspected with clear titles, legends, labels and no clipping.
Scratch render remains ignored under `tmp/pdfs/analysis059/`.

## Verification, uncertainty and limitations

41 transferred artifacts and archive size/SHA-256 checks pass; all864 treatment
updates and867 boundaries are complete. First-step recurrence residuals are
consistent with FP32 rounding; maximum parameter-step audit error1.19e-8.
The observer's sum-of-squares rather than dot-product denominator changes the
fork projected-change scalar by2.22e-16; loss, raw gradient, cosine, weights,
optimizer and replay match exactly. The initial overly strict scalar-equality
audit was corrected to an explicit1e-13 reduction tolerance; training inputs
and captured evidence were unchanged.

Worker time12.15s including4 qualification updates; treatment durations
3.59/3.83/3.77s. Peak allocated CUDA memory0.422GB. Zero cloud cost. Recovery,
first-step and terminal states, raw logs and input identities retained locally
and on owned Spark. No unrelated process was changed.

These are dependent trajectories from one selected development state/seed.
No confidence bands or independent replications; no causal mediation, universal
stability, convergence or optimizer-generalization conclusion. The bias-correction
transition is part of the tested intervention and limits a memory-only interpretation.
