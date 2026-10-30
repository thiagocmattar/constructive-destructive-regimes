# Measurement and paper design

Status: proposed, not executed. This contract accompanies the
[critical assessment](README.md).

## Matched comparison

Use the Analysis024 MLP objective: first 5,000 CIFAR-10 training images in the
pinned source order, existing coordinate normalization, tanh 200-200 network,
and half squared error summed across ten outputs and averaged across images.
Reuse verified preprocessing and initial tensors for seeds 0/1/2, recording
their hashes. TF32 off; FP32 parameters and gradients; FP64 scalar reductions
and cumulative vectors. Count all trainable coordinates once.

Keep constant LR 0.02, plain SGD, no momentum, decay, clipping, augmentation,
dropout or schedule. Batch size alone varies within each checkpoint/seed block.
Do not scale LR with batch size: that would change both factors in the primary
comparison. A later LR robustness test requires a separately specified design.

Prepare three reference GD paths to update 2,000 and retain states 0/1,000/2,000.
Seed 0 reconstructs the prefix of Run062, including its exact initial tensor
hash, objective and accumulation order. Compare its loss, gradient norms and
pairwise scalar history against the retained evidence with predeclared FP32
reproduction tolerances. Byte equality of nonlinear trajectories is not assumed
across hardware. If reconstruction does not qualify, label it a new reference
and preserve the discrepancy; never silently claim checkpoint reuse.
Seeds 1/2 are new reference trajectories at this LR.

Sample uniformly without replacement **within** a batch and independently
**across** optimizer updates. Within an initialization, checkpoint and sampling
seed, use common fresh permutation prefixes across batch sizes to pair the
interventions. Different sequence/fork seeds remain independent. Probe RNG is
separate and must not change training draws. This is not random reshuffling over
an epoch.

Use batch sizes 32, 128, 512, 2,048 and 5,000. At each of nine states:

- Frozen controls: 32 independent sequences, 128 draws each, per batch.
- Moving controls: eight independent 200-update forks per stochastic batch;
  one deterministic 200-update full-GD fork.

This is 45 frozen conditions, 1,440 sequences and 184,320 gradient observations,
including repeated deterministic full-batch values. It is 288 stochastic forks
and nine deterministic forks, or 59,400 continuation updates, plus 6,000 GD
reference updates if all three reference paths must be generated. Reuse a
verified deterministic prefix where possible; do not count it twice as evidence.
Do not confuse the large number of gradient observations with independent
initializations: there are only three initialization blocks.

The primary moving comparison matches optimizer updates and LR, not data
exposure. Record sampled examples (`b * updates`) and full diagnostic examples
separately. An exposure-matched view of the common available range is secondary
and has a different horizon. No claim about training efficiency or an optimal
batch size follows from the primary comparison.

## Frozen measurements

At each state, compute the full gradient `mu`, its squared norm `S`, and
`tr Sigma = (1/N) sum_i ||h_i-mu||^2` over all 5,000 per-example loss gradients.
The existing MLP outer-product norm identities can avoid storing a dense
per-example gradient bank; verify them against explicit gradients on a small
mechanics check. Measure covariance independently of the batch-draw noise used
to evaluate predictions. Retain coverage, numerical cancellation checks and
absolute error when the variance is very small.

Each draw records gradient norm, noise energy `||g-mu||^2`, adjacent dot/cosine,
noise projection `g_prev^T(xi-xi_prev)`, `q = 1-g_prev^T g/||g_prev||^2`,
`epsilon`, normalized accumulation and sequence identity. Assess:

| Equation | Prediction | Measured response | Error reporting |
| --- | --- | --- | --- |
| (18) | `V_b = (N-b)/(b(N-1)) tr Sigma` | Mean `||g-mu||^2` | Absolute and relative errors; full batch handled separately |
| (17) | `-V_b` | Mean `g_prev^T(xi-xi_prev)` | Signed residual and Monte Carlo interval |
| (19) | `S/(S+V_b)` | Mean adjacent cosine | Absolute prediction error, plus norm concentration |

For Eq. (19), record the coefficient of variation of `||g||` and of the adjacent
norm product. Compare approximation residuals with concentration rather than
declaring every mismatch a failed variance law. The expectation is of a ratio;
computing a ratio of sample expectations is not a replacement for the measured
mean cosine.

Use sequence-level bootstrap intervals for Monte Carlo uncertainty within each
fixed state. Adjacent pairs share a gradient, so do not treat all 127 pairs as
independent replicates. Show all three initialization summaries; with three
blocks, their range is descriptive and must not be labeled a population 95% CI.

## Moving measurements

Use local fork indices `k=1,...,200`, with `g_k` evaluated at `theta_{k-1}`.
Compute the complete gradient `mu_k` and full training objective at that same
preupdate state. Evaluate the final objective at `theta_200` too. Each recorded
pair must use consecutive optimizer gradients, never adjacent sparse log rows.

Record the following before every update, with independent evaluation that
cannot change the training state:

- Actual sampled gradient `g_k` and noise `xi_k = g_k-mu_k` through online
  sufficient statistics; raw norms, cosine and q for all 199 adjacent pairs.
- Complete-gradient cosine `cos(mu_{k-1}, mu_k)` along the same SGD path.
- Noise projection, `epsilon_k`, and realized noise/signal energy ratio.
  `||xi_k||^2/||mu_k||^2` is a realized statistic; it is not itself the conditional
  expectation `chi_b`. At HVP probe states, also compute full-population covariance
  trace for a conditional variance reference.
- `A=sum||g_k||`, `B=||sum g_k||`, energy, exact cumulative all-pair interaction,
  lag-one interaction and their difference. Derive all-pair interaction as
  `(B^2-sum||g_k||^2)/2`; no dense history is necessary for that identity.
- `R_K=B/A`, measured parameter displacement from the fork state, actual update
  path length and FP32 discrepancy from the nominal `eta*g_k` update. Reset only
  at the fork origin and label it explicitly. Preserve both actual-update and
  gradient-derived ratios if their numerical difference matters.
- Full-objective loss and sampled batch loss as separate streams. Retain finite
  loss increases and all nonfinite failures rather than replacing conditions.

At predecessor gradient indices `j=1,51,101,151`, compute the full-objective
directional HVP at `theta_{j-1}` along the actual `g_j`, for comparison with the
following pair (`k=j+1`). All 5,000 examples contribute. For that pair:

$$q_k+\varepsilon_k
=\frac{g_{k-1}^{\top}(\mu_{k-1}-\mu_k)}{\|g_{k-1}\|^2}.$$

This equality is bookkeeping, not independent validation. Compare its right
side against the separately measured
`eta * g_{k-1}^T H(theta_{k-2}) g_{k-1} / ||g_{k-1}||^2` and retain the finite-step
residual. The correction removes the explicit sampled-gradient difference; it
does not remove noise from the direction, visited state or nonlinear remainder.
This is full-objective directional curvature, not mini-batch Batch Sharpness
from the EoSS literature. Computing the latter would require its own per-batch
Hessians and expectation contract.

Use eight fork-level summaries to estimate sampling variability conditional on
each initial checkpoint. For cross-batch contrasts, preserve the paired sampling
seed in resampling. Report all initialization-block contrasts, not a pooled
confidence interval treating 288 forks as independent initializations.
Primary summaries: the 199-pair mean sampled cosine, mean full-gradient cosine,
final `R_200`, full-objective loss change, and the Eq. (17) residual averaged over
the fork. Each is a predeclared finite-window summary; no selecting favorable
windows after inspecting outcomes.

No smoothing enters those summaries. For visual trends, use a disclosed
20-pair trailing arithmetic mean for both cosine streams, requiring a complete
window. Retain raw curves in the appendix. Plot `R_K` and loss unsmoothed. A mean
curve across forks is a mean of diagnostics computed within each fork, never the
diagnostic of an averaged gradient or averaged parameter trajectory.

Zero gradient norms make the corresponding cosine, q or epsilon undefined:
record null and a reason, without adding an epsilon or replacing with zero.
Zero full-gradient norm makes `chi_b` undefined; keep absolute noise measurements.
Report missing counts and the population used by each summary. An actually zero
displacement with positive path length has a well-defined `R=0`.

## Figures

**Figure A, three columns by one row: calibration at matched frozen states.**
The panels show measured versus predicted noise energy, measured versus
predicted negative noise projection, and measured versus predicted cosine.
Use a solid neutral identity reference and label the cosine relation as an
approximation. Batch-size colors and checkpoint marker shapes identify the
interventions; individual initialization points and conditional Monte Carlo
intervals preserve uncertainty. Normalize the first two panels by fixed
state-specific `S+V_b` if needed to compare scales, applying the same independent
reference denominator to both axes. Full-batch endpoints remain explicit.
Do not fit a free slope to claim agreement with a parameter-free prediction.
An appendix view uses batch size directly on the x axis and shows norm
concentration and residuals.

Proposed caption:

> Sampling effects at fixed training states. Each point compares a theoretical
> prediction with a Monte Carlo estimate at a specified initialization,
> checkpoint and batch size. Full-objective per-example moments determine the
> variance predictions independently of the sampled sequences. Error bars
> represent sequence-level Monte Carlo uncertainty conditional on the state.
> The cosine prediction additionally assumes concentrated gradient norms.
> Parameters remain fixed throughout each sampling sequence.

**Figure B, three columns by three rows: motion from the same checkpoints.**
Columns are fork states 0/1,000/2,000. Rows show sampled-gradient cosine,
full-gradient cosine at the same SGD states, and normalized displacement from
the fork. Colors distinguish batch sizes, with solid curves throughout.
Separating the two cosine rows avoids an overcrowded nine-line comparison.
Average within initialization first, then give each initialization equal weight;
show their mean-curve range with that exact label. Put all individual and raw
curves in the appendix. Sampling intervals belong in endpoint summaries rather
than overlapping multiple types of bands in one panel.

Proposed caption:

> Consecutive alignment and displacement during 200-update continuations from
> matched GD checkpoints. Full-gradient cosines are evaluated on the same
> parameter states as the corresponding SGD gradients. Their trajectories can
> differ from the full-batch continuation. Cosines use a 20-pair trailing mean;
> normalized displacement uses the complete unsmoothed prefix from the fork.
> Curves average sampling repetitions within each initialization, then the
> three initialization blocks equally; shading gives the range of block means.
> Fixed update counts imply different sampled-example budgets across batches.

If the bands obscure all curves, split into isolated three-column rows, as in
Analysis024. Use vector PDFs, the existing IEEE widths/fonts, external panel
letters, no embedded footnotes and no dashed condition curves. A local
observation and manifest must record sources, reductions, captions, coverage,
uncertainty, script hashes and visual review for each generated figure.

**Appendix and table:** full-objective loss histories; q, epsilon and corrected
q against directional HVPs with residuals; finite-population scaling; raw cosine;
the existing three-model initialization control. A compact main-text table gives
paired batch effects on cosine, `R_200` and full-objective loss, with all
initialization values available. Do not hide loss deterioration behind favorable
geometric curves.

## Manuscript edits after evidence is available

Two prose adjustments to III-B are warranted independently of future outcomes:

1. After Eq. (17), clarify that its sign concerns the unnormalized noise
   projection, and need not be inherited by `E[epsilon_t]` after division by a
   correlated random squared norm.
2. Replace the sentence beginning “The resulting trajectory becomes increasingly
   driven...” with: “Under this local approximation, sampling fluctuations can
   dominate the directional agreement between gradient estimates. The cumulative
   displacement of a moving SGD trajectory additionally depends on its evolving
   full gradient and interactions across temporal lags.” This avoids deriving a
   trajectory claim directly from a frozen-point cosine calculation.

Suggested opening of the new empirical section, to use only after execution:

> We examine how mini-batch sampling changes the interpretation of the
> displacement diagnostics introduced in Section III. Our design separates
> repeated gradient estimation at fixed trained parameters from actual SGD
> continuations initialized at those same states. The former directly evaluates
> the sampling variance and cosine approximation; the latter measures their
> interaction with parameter motion. Complete-objective gradients evaluated
> along each continuation provide a reference for its realized sampling error.

Then present the matched design, accuracy and limits of Eqs. (17)--(19), moving
geometry and measured loss, followed by a short implication for large-batch
diagnostic studies. Result sentences must follow the measured directions and
errors; do not prewrite a monotone or universally near-zero conclusion.

## Feasibility and retention

Start with the MLP and keep the ResNet/Transformer expansion optional. ResNet
sampling must select whole fixed 1,000-image ghost groups to preserve the current
objective. Smaller BN groups would define a different experiment. For WikiText,
uniform sequence sampling needs per-sequence token-loss sums scaled by
`N / total_supervised_tokens`, followed by a mean across sampled sequences;
dividing by the random sampled token count generally creates a ratio estimator.
Full gradients must cover the complete prepared objective, not a spectral subset.

Calibrate training, full-gradient, covariance and HVP costs separately on the
chosen machine before stating an ETC. Prior run times do not establish the cost
of this diagnostic workload. Owned hardware is a natural option, but machine
selection, maximum duration/storage and any paid envelope are future execution
decisions. The held Analysis020/021 queues remain held.

Retain reference/fork initial states, final states, rolling recovery with RNG
and diagnostic accumulators, and predeclared HVP probe states plus batch indices.
Keep per-step sufficient statistics, per-sequence/per-fork summaries, failures,
precision/environment identities and all source/config/data hashes. Avoid dense
full-history storage unless a new lag-resolved question requires it. Verify
downloaded sizes and hashes before any worker cleanup. New scientific inputs
receive new run IDs at implementation; this proposal changes no executed run.
