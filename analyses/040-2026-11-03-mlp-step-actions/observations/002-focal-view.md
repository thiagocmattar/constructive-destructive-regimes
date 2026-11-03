# Observation002: descriptive three-method focal view

Question: make the existing loss/cosine comparison readable while retaining its
real oscillations. This introduces no training, new observations or new test.

Sources: aligned_diagnostics.csv, from the completed Run218 local pilot.
Coverage: A0 unchanged Adam, A1 half learning rate, A4 Adam proposal clipping;
all 129 parameter states 657..785 for each method. These three were selected for
presentation after the exploratory screen, not as independent confirmation.
All eight methods and severe A5/A8 failures remain in the original report/table
and figures/local_pilot.pdf, which are unchanged.

Reduction:03_focal_comparison.py. Raw loss and raw_adjacent_cosine are plotted
unchanged, without smoothing/downsampling/normalization. Three columns share a
loss scale and a cosine scale; the gray vertical line marks the original pre-rise
state 689. All 774 plotted scalar values match the source CSV exactly. One-page
PDF rendered with PyMuPDF and visually reviewed; labels and scales are clear.

Caption: Descriptive three-method view of the existing seed0 local full-objective
MLP pilot. Loss above, adjacent raw-gradient cosine below; all 129 states retained,
common scales within each row. Selection follows exploratory screening and is
not independent confirmation. The fluctuations are full-objective dynamics under
the checked fixed numerical recipe, without random minibatch sampling.

Reading: half LR is strongest here; proposal clipping reduces excursion size
while leaving/replacing events. A2 also suppresses most excursions but misses
the final-loss margin. A8 is retained as negative evidence. One seed/state and
128 updates do not establish diagnostic-timing value or generalization; cosmetic
clarity does not reduce scientific uncertainty. See REPORT.md for context.

Artifact: figures/focal_three_arm.pdf. Original raw evidence is untouched.

Source CSV SHA-256: `ec86e7a3044cfabb849022809603733407367090c5356e7734835b820799e102`.
Figure SHA-256: `98f701065dc1a58fac21bf07139085d2bbe274349101d761ffb34be3605c5acc`.
