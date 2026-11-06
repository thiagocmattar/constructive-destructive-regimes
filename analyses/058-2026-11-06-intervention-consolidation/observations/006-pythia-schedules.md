# 06: Pythia schedule interventions with recovered diagnostics

Sources: Analysis004 cohort, results and source hashes; Runs011--016 event logs
and resolved configurations; parent Runs005/007 event logs. Generator
`../01_figures.py` reconstructs each series from the first 356 parent updates
plus its 356-update continuation. Raw gradient norms and cosine were already
stored. No backward passes, checkpoints or new training are needed.

Reduction: at logged update t, q_t=1-rho_t*n_t/n_(t-1), where n is the raw,
unscaled, accumulated pre-clipping gradient norm. The manuscript-style label
eta*kappa_hat denotes this dimensionless projected gradient change. Under
stochastic AdamW, the pair compares sampled gradients along an adaptive path;
it is not a pure curvature measurement. A zero or missing denominator gives
undefined, without epsilon substitution. Only the initial pair is undefined
in these records. Applied LR is divided by the inherited update-356 LR.

Caption: MiniPile training with Pythia-14M (left) and Pythia-70M (right), seed
1235, branching from the update-356 checkpoint. Gray retains the original
decay, blue holds the midpoint LR, and green accelerates schedule decay.
Rows show sampled training loss (log), projected raw-gradient change,
consecutive raw cosine, raw gradient norm and applied/midpoint LR. Shading
marks inherited prefix updates 1--356. Raw per-update observations are shown
without the 31-update centered smoothing used in the original Analysis004
figures. All 712 updates, including the unequal-size final batch, are retained.

Clock and coverage: an update event reports the sampled loss/gradient before
that optimizer update, together with its applied LR. It is not a full-objective
loss evaluated after the update. The first new gradient is at update 357;
primary affected-pair contrasts in the owning analysis start at 358. Gradients
cover all trainable parameters after accumulation/unscaling and before clipping;
sampled losses retain token-mean reduction and the original supervised-token
counts. Norm/loss/LR have 712 points; cosine/q have 711. No validation curve is
substituted for the training curve. The raw tail is retained, and no equal-batch
assumption or aggregation is introduced by this figure.

Result: held LR retains stronger negative raw alignment but gives the lowest
terminal full-validation loss (14M 5.0674; 70M 3.9846), versus original decay
(5.1820; 4.0559) and accelerated decay (5.2176; 4.1042), as recorded in
Analysis004. Diagnostic recovery alone therefore does not rank validation
performance. LR changes also change LR-scaled AdamW shrinkage. One paired seed,
stochastic sampled objectives and the finite training horizon limit inference;
no causal mediation, universal threshold or scheduling benefit follows.
