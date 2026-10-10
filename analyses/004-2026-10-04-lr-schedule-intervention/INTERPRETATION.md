# Interpretation fixed before completion

The first test asks whether late LR decay contributes to changes in adjacent
raw-gradient cancellation, conditional on the exact trained state at update 356.
It compares original decay, held midpoint LR and accelerated decay at 14M/70M,
seed 1235. Multiple seeds are explicitly deferred to validation. This experiment
does not test a dimensionality mechanism or whether a schedule from initialization
would produce the same result. It does not include an adaptive scheduler.

The primary affected pairs are358-712; pair357 is computed before the first
changed optimizer update. Every future batch and supervised token is matched.
Report unsmoothed negative frequency F, unconditional negative intensity D and
conditional negative magnitude M separately, with D=F*M. The final partial batch
is retained in primary statistics and excluded in a declared sensitivity reduction.
Use31-update centered means only for visualization and retrospective recovery;
also report15/61-window sensitivity and later reversals. No gradient pair is an
independent seed replicate, so there are no pair-level confidence intervals or
p-values. Complete validation and training loss are separate outcomes.

The anticipated directional pattern is higher D/more negative alignment under
held LR and lower D/earlier positive alignment under accelerated decay relative
to the new original-schedule control. Report each size and all exceptions. If
the interventions have different effects at 14M/70M, describe that contrast without
fitting a scaling law to two sizes. If they have little effect, the original
schedule/recovery association remains insufficient to identify decay as the cause.

Multiplying gradient vectors by positive learning rates leaves their cosine
unchanged at fixed parameter states. Any raw-gradient response here arises through
the changed training trajectory, including AdamW's LR-scaled shrinkage. This
experiment does not isolate preconditioning, weight decay or stochastic curvature.
Positive adjacent alignment does not by itself establish convergence, stability,
loss reduction or generalization. If validation loss and alignment move differently,
retain that result; do not equate less cancellation with better learning.

GPU resumption is numerically close rather than bitwise identical. Qualification
changes and repeated mechanics are disclosed in README.md and run-local receipts.
Use newly executed original controls for contrasts; quantify their differences
from archived parents across the full continuation. Source/config/cache hashes,
Adam/scaler/RNG/diagnostic clocks and token order are verified independently.
Do not describe small scalar agreement as equality of complete gradient vectors.

Only a replicated follow-up can establish whether the observed schedule effects
persist across initialization/data-order seeds. A useful restricted paper claim
would concern horizon-specific raw-gradient cancellation and its sensitivity to
training choices, supported by direct loss measurements. A dimension-mediated
performance explanation or broad stability framework requires separate evidence.
