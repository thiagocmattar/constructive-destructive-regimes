# Observation001: cooldown, cosine recovery and clean four-arm figures

Source: Run220 frozen01_cooldown.py/config.json/26-input manifest, all raw
attempts under artifacts/attempts/, and infrastructure-only02_continue_after_audit.py.
Run218 authenticated parent SHA256
359097d026b45f8f0b8a33665aca4ad0382e52835745ceaeccf027ed29539ba4.
Run219/Analysis041 supply exact-replay references; the original event detector
and Run218 prefix supply excursion definitions. Raw sources are byte/hash listed
in raw_inventory.csv. Generating script: ../01_report.py; outputs ../comparison.csv,
../aligned_values.csv, ../excursions.csv and ../figures/*.pdf.

Caption: Five matched local full-objective MLP continuations from seed0/state657
compare unchanged Adam, always-half LR, the one-step threshold rule, an8-update
hold, and an8-update hold plus8-update cosine recovery. Every eligible available
$\eta\widehat\kappa_t\geq1.8$ resets the hold. Rows show unsmoothed training loss,
$\eta\widehat\kappa_t$, raw-gradient norm and applied LR/base; gradient norms use
log scales, as does loss in the full view. Dotted vertical markers show the
historical689/703/729 event; the horizontal diagnostic line is1.8. At the author's
request, focal_four_arm.pdf and focal_four_arm_zoom.pdf omit the orange one-step
series and use the same remaining observations and window. The complete five-arm
comparison is preserved in full_continuation.pdf and fixed_zoom.pdf.

Coverage: states657..1024,368 observations per arm and367 outgoing decisions;
fixed zoom673..745 was declared before this run. All15,876 plotted scalars are
asserted identical to source arrays. The last LR value only closes [1023,1024);
no terminal action is applied. No smoothing, clipping, decimation, alignment to
new own peaks, interpolation of loss/gradients, or re-training for figure edits.
The clean PDFs were independently checked for absence of orange strokes and the
one-step legend entry, rendered with PyMuPDF and visually reviewed for legibility.

Reduction: mean loss over658..1024; final loss at1024. Fixed-reference normalized
positive excess max(L/B-1,0), B=untreated loss689, summarized by maximum and
sum689..1024 inclusive. Original own-reference excursion detector retains pre-rise,
peak, recovery/censoring; >=10% event counts distinguish changes in event number
from peak/area. Scalar formulas have independent Python-sum checks. An independent
most-recent-trigger-age rule checks all1,835 LR decisions. Raw norm/cosine checks
all1,840 available statistics. All starts and three full historical controls match;
both new policy resumptions are exact after excluding only UTC audit metadata.

Interpretation: the cosine policy has2.26% lower mean and4.30% lower final training
loss than always-half in this selected case; abrupt cooldown is worse. Cosine
retriggers prevent any complete recovery to full LR after the first reduction,
so365/367 updates remain below base. This tests the combined rule, not an optimal
N or a general diagnostic. Training advantage is not held-out improvement.

Uncertainty/limits: one selected development seed/state, no replication interval.
No frequency/rate-exposure-matched fixed LR or open-loop schedule, no threshold
search, no curvature identification, no stability/generalization/LLM-transfer
claim. Original failed UTC audit and its correction are preserved; scientific
inputs and native moment recurrences never changed. Total active work57.75s,
zero cloud cost; all raw evidence retained locally. No manuscript edit.
