# Intermediate GD rate0.03 and the requested three-column figure

User requested LR0.03 after viewing0.02/0.04/0.08, and requested columns
0.02,0.03,0.04 in the existing PDF. Run235 performs the additional branch;
Runs233/234 remain unchanged. This is exploratory selection after earlier
results, not an independent confirmatory experiment.

## Sources, comparison and coverage

New source: Run235 `artifacts/spark/payload/worker/gd_lr0p03/`, with source
identities in `input_manifest.json`, integrity checks in `verification_final.json`
and generator `../01_report.py`. Reused sources: Run233 GD0.02, Run234 GD0.04,
Run223 prefix/Adam control. Exact paths and hashes are in `../results.json`;
all plotted values are in `../plotted_values.csv`. Historical0.08 evidence is
preserved in results and `../archive/2026-11-07-rates-0p02-0p04-0p08/`.

Same original untreated Adam state512, seed0,656810-parameter MLP, pinned5000
CIFAR-10 images, full MSE, physical1000 batches, FP32 vectors/FP64 diagnostics,
RNG, data order and horizon800. GD uses native SGD, zero momentum/decay, no
clipping/scaling. LR is the only changed factor among GD arms. Comparison with
Adam(.9,.999),LR0.001 changes optimizer and LR jointly. The exact fork-gradient
and two-update Adam reference replay pass; GD2 versus1+1 resume is exact.

All288 scientific updates and289 boundaries512..800 are present. Prefix448..512
provides context. Mean loss uses513..800; tail uses769..800; endpoint test loss
uses the complete10000-image test set. Excursions use the unchanged fixed
pre-rise detector on full histories0..800, selecting peak>512 and ratio>1.1.
The raw adjacent pair at512 includes the preceding Adam gradient. No smoothing,
new seeds, uncertainty bands or universal stability/convergence interpretation.

## Results

| Outcome | Adam0.001 | GD0.02 | GD0.03 | GD0.04 |
| --- | ---: | ---: | ---: | ---: |
| Mean training loss513..800 | 0.0208042 | 0.0212831 | 0.0439842 | 0.102867 |
| Last32-state mean | 0.0134585 | 0.0209599 | 0.0391844 | 0.0447293 |
| Training loss at800 | 0.0302502 | 0.0209284 | 0.0409154 | 0.0588509 |
| Maximum training loss513..800 | 0.0550626 | 0.0220705 | 0.302456 | 1.61503 |
| >10% excursions | 16 (1 censored) | 0 | 1 (censored) | 1 (censored) |
| Loss-increasing updates | 161 | 0 | 125 | 96 |
| Full test loss at800 | 0.776960 | 0.705448 | 0.708491 | 0.684493 |

GD0.03 shows an initial excursion and later oscillations. Mean/tail loss is
111.42%/191.15% above Adam; maximum449.29% higher. Its peak is lower than0.04
but higher than0.02. Both0.03 and0.04 remain above their fixed pre-rise reference
through800, so the detector calls each one censored excursion despite multiple
local peaks. Lower test loss than Adam is an unreplicated secondary endpoint.
All actual steps have negative gradient dot product; this says they follow a
local descent direction, not that every finite update decreases the loss.

## Figure caption

CIFAR-10/MLP: continued Adam at LR0.001 (gray) and independent switches to
plain full-batch GD from state512 (blue). Columns show GD LR0.02,0.03,0.04.
Rows show full training loss, projected change
<g_(s-1),g_(s-1)-g_s>/||g_(s-1)||^2, adjacent raw-gradient cosine, raw-gradient
norm and outgoing applied LR. Shared pre-fork context448..512 is shaded.
Loss and norm use log scales; each row has identical ranges across columns.
All data remain unsmoothed through800, with the final LR interval closed at800
for display. The projected-change row retains the original definition; its
interpretation depends on optimizer dynamics and is not a stability criterion.

## Verification and execution

All42 Run235 source/cache/checkpoint identities and39 retrieved artifacts plus
archive pass size/SHA checks. CPU mechanics and native Spark replay/resume pass.
First-step coordinate rounding is at most7.411e-9; maximum step residual norm
4.416e-7. Norm/dot identities hold within recorded rounding bounds and all288
actual increments have negative raw-gradient dot product. Full endpoint
gradient, optimizer state and complete test coverage are verified.

All10,584 plotted scalars match source rows. The generator checks all3528
original plotted values and both previous sets of numerical summaries, plus
archived generator/PDF/CSV hashes. The7.16 x7.1-inch vector PDF has embedded
fonts, validated text bounds and a visually inspected render. Scratch image
remains ignored under `tmp/pdfs/analysis060/`.

Run235 main treatment2.855s, whole worker3.838s including6 qualification updates.
Final measured throughput101.7 updates/s, peak CUDA allocation457,576,960 bytes.
Worker exited. Retain first-step/recovery/final checkpoints and39 files with
121,402,198 logical bytes, plus70,374,848-byte transfer archive. Owned Spark,
zero cloud cost. `completion.json` records physical remote storage. No unrelated
processes or manuscript files changed. Original executed sources remain frozen.
