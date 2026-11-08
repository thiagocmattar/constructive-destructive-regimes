> Evidence snapshot; original scientific content retained. Relative links were rebased.
> Source: [009-what-the-negative-controls-show.md](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/009-what-the-negative-controls-show.md).

# What the MLP negative controls establish

## Main point

**A diagnostic of oscillation is not automatically a useful control target.**
In these matched continuations, fewer reversals, smaller average output changes,
or a lower final loss can coexist with substantially worse transient and average
training loss. The useful question is whether an intervention preserves useful
learning while reducing harmful excursions over a sufficiently long window.

The policy-dissection experiments are exploratory continuations from the same
untreated seed-0 MLP state 448, with the unchanged full 5,000-example objective and Adam LR 0.001.
Runs 272 and 275-279 extend through 1152. Every comparison includes an unchanged
Adam branch; the controller never sees future-window measurements. The initial
selective screen is Run 269 through 800. The MLP uses tanh hidden activations and
a linear output layer; bounded hidden activations do not bound weight updates,
output changes, or loss.

## 1. Matching parameter norms does not reproduce neuron selection

Run 272 caps each hidden neuron's direct, frozen-input affine displacement RMS
at its module's prefix q50. Both hidden affine layers remain trainable; output
increments are unchanged. Mean loss 448-1152 falls 22.85% versus unchanged Adam,
although the last 32 mean rises 4.36%.

Run 275 instead keeps each hidden layer's own native Adam direction and scales
it to the corresponding selective branch's **actual layer norm at the same
update**. This retrospective counterfactual improves full-window mean only 3.43%
and worsens the last 32 mean 76.61%. Thus, in this comparison, the allocation
within layers matters beyond their scalar norm schedule. This does not establish
superiority for all budgets, seeds, or windows. The schedule is taken from
another branch and is not an online controller.

## 2. Immediate diagnostic-triggered damping can release too early

Run 269 tests a raw-gradient rule: damp when a block's lag-one cosine is negative
and its norm exceeds the preceding exponential moving average; otherwise apply
the full native increment. The block and global versions increase average loss
by 612.39% and 531.12%, respectively. The secant-based heuristic also worsens it.

The logged gates expose a failure mode: a small damped step can be followed
immediately by a full increment when the cosine turns positive, even while the
gradient norm is still very large. This is a description of the measured gate
sequence, not proof that release timing alone explains the entire failure.
Adam moments keep advancing under every policy, so the later trajectory and
direction are policy-dependent. Run 279 separately tests slow recovery of a
global factor, limiting increases to 1% per update.

## 3. Removing a reversing component can make training worse

Let the first affine layer's fixed input matrix be X. Its native candidate
parameter increment induces Z = X dW + db; its preceding actual increment
induces V. Run 276 computes beta = <Z,V> / ||V||² over all sample-neuron entries.
When beta is negative, it removes, halves, or caps that component by subtracting
the corresponding multiple of the preceding parameter increment. It leaves
the functional component orthogonal to V unchanged and leaves other layers'
increments native. This uses one preceding direction, which can change with
time; it is **not** a fixed-window or future-fitted SVD projection.

All three versions worsen average loss, despite greatly reducing output
reversals. Removing the component improves the last 32 mean 35.41% but raises
full-window mean 171.71% and maximum loss 1631.67%. Reporting only the endpoint
would give the wrong practical impression.

Functional orthogonality does not imply Euclidean parameter orthogonality:
the intervention can enlarge the parameter increment. Run 277 addresses this
confound by additionally bounding the modified first-layer increment's norm
by that layer's native Adam norm, with an explicit FP32 storage-rounding bound.
It still fails: remove/halve/cap increase average loss 277.21%/87.47%/141.43%.
This does not imply that reversals are intrinsically necessary. It shows that
these particular attempts to remove them are poor interventions here.

## 4. A bound on each step's motion does not bound the later loss trajectory

Run 278 measures actual complete-network probe changes after a candidate Adam
update, rather than only a local affine contribution. It scales the entire
native increment and backtracks until the output, or every tracked
representation, satisfies a fixed coordinate-RMS budget. Budgets use only the
eight untreated outgoing changes 440-447. All 5,000 MLP examples are included.
Thus the output check covers this entire training objective, not a subsample.

The output q50 rule raises mean loss 58.18%; the all-representation q50 rule
raises it 38.67%. Both pass the actual displacement bounds, yet develop larger
excursions. Their mean output RMS is smaller than baseline. A per-step distance
bound allows drift over successive steps and does not require movement toward
the targets or loss decrease. Smaller displacement alone is therefore not a
sufficient acceptance condition.

## 5. Exact output-space loss accounting

For the MLP, let F_t have N=5,000 rows (examples) and D=10 columns (outputs),
let Y contain one-hot targets, and let DeltaF_t=F_(t+1)-F_t. Its squared-loss
change has the exact decomposition

$$
L_{t+1}-L_t
=\underbrace{\frac{\langle F_t-Y,\Delta F_t\rangle}{N}}_{A_t}
+\underbrace{\frac{\|\Delta F_t\|_F^2}{2N}}_{Q_t}
=A_t+Q_t.
$$

The coordinate RMS already logged by the representation probe determines
Q_t=(D/2) RMS(DeltaF_t)^2=5 RMS(DeltaF_t)^2. Therefore the scalar histories
recover A_t from the loss change and Q_t, up to the small rounding error in
native FP32 loss reductions. Independent replay computes both terms directly
from the full output matrices and one-hot targets at three saved states per
arm. It also checks the loss reduction error.

Across outgoing updates 448-1152 (including the final transition to 1153), the
unchanged controls have 377/371/379 meaningful loss increases for seeds 0/1/2.
Every one has A_t<0 but Q_t>-A_t. Thus the actual functional displacement
initially points downhill, yet its quadratic cost outweighs that descent.
Their parameter-space directional derivatives g_t dot u_t are negative too.
This is direct evidence of finite-step overshoot in these squared-loss
trajectories, not an inference from cosine sign. A_t is based on the actual
nonlinear output change and need not equal the parameter-linear term g_t dot u_t.

This explains why a fixed movement budget is insufficient: it bounds Q_t but
does not ensure enough compensating negative A_t as the residual changes.
Run 284 supplies a deliberately loss-aware comparator. It tries dyadic fractions
of the native Adam increment and accepts the first strictly decreasing FP64
probe loss; after twelve unsuccessful positive fractions it uses a zero
increment while retaining the native moment/clock update. The MLP probe is its
entire training set. Mean loss falls 22.32%, last 32 mean falls 7.49%, and positive
loss variation is zero over 448-1152. Only 4 of 705 applied updates are zero; many output reversals remain. Monotonicity
is an acceptance condition here, not an independent discovery or a claim of a
new optimizer. Transferring this comparator with a smaller probe does not
guarantee monotonicity of the objective outside that probe.

The extra accounting is generated by `26_audit_mlp_loss_terms.py` and
`27_reduce_mlp_loss_terms.py`, with source/transfer identities and dense audit
records in `mlp_loss_terms*.json`. The increase threshold is 1e-8. Its terminal
transition is retained for outgoing diagnostics; the ordinary mean/PV tables
use loss states only through 1152 and therefore omit that last transition.

## 6. Smaller motion can still follow the wrong direction

The same loss accounting separates finite-step overshoot from an already
uphill actual displacement. For seed 0 over outgoing updates 448-1152:

| Rule | Mean quadratic term Q | Meaningful loss increases | Increases with A < 0 |
|---|---:|---:|---:|
| Unchanged Adam | 0.0230013 | 377 | 377 |
| Distributed neuron cap, q50 | 0.0124348 | 312 | 312 |
| Uniform half increments | 0.0049511 | 189 | 189 |
| Output budget, immediate recovery | 0.0124384 | 313 | 231 |
| Output budget, slow recovery | 0.0022556 | 262 | 224 |
| Probe-loss backtracking | 0.0000892 | 0 | ? |

The local neuron cap and immediate output budget have almost equal average Q,
but their mean training-loss changes are -22.85% and +58.18%, respectively.
Thus average squared functional motion alone does not rank these trajectories.
This is a descriptive comparison of their resulting paths, not a matched-motion
causal experiment. Both mean A and mean Q are also constrained by the telescoping
loss identity; they must not be treated as independent explanations of total
progress.

The immediate and slow-recovery budgets additionally create increases with
A >= 0: reducing the movement does not ensure that it points toward the targets.
For slow recovery, 38 of its 262 meaningful increases have A >= 0, and the same
38 have a nonnegative parameter directional derivative g_t dot u_t. In contrast,
all 377 increases of unchanged Adam have negative A and negative g_t dot u_t.
The optimizer's direction evolves with the damped trajectory while its moments
continue to advance. Consequently, an intervention can change the failure mode;
it is not merely shrinking a fixed sequence of baseline updates.

The loss-aware comparator explicitly tests the actual candidate's consequence
and avoids both kinds of increase on the full MLP probe. Its extension to a
smaller probe must still be judged against the full objective.

## What remains distinct

- Raw-gradient alignment is not alignment of Adam's actual increments. The
  unchanged MLP has g_t dot u_t <0 throughout this window, even on updates
  followed by loss increases. Finite-step effects matter.
- A local neuron cap bounds the direct parameter contribution with current
  inputs frozen. Upstream changes, nonlinearities, and normalization can change
  the total representation motion. Run 278 checks that total motion instead.
- Persistent negative representation cosine can accompany useful amplitude
  damping. A reduction in the number of negative cosines is not the objective.
- Mean, last 32 mean, maximum loss, and total positive loss variation answer
  different questions. No single endpoint summarizes the optimization path.

## A simple intuition for persistent reversals

For the scalar quadratic $L(x)=\lambda x^2/2$, gradient descent gives

$$
x_{t+1}=(1-\eta\lambda)x_t,
\qquad
\frac{L(x_{t+1})}{L(x_t)}=(1-\eta\lambda)^2.
$$

When $1<\eta\lambda<2$, consecutive gradients and consecutive changes in x
have cosine -1, yet their magnitudes and the loss decrease. For
$\eta\lambda>2$, the same cosine -1 accompanies growth. The sign alone cannot
distinguish these cases. This is an illustrative calculation, not a fitted
quadratic model or a stability guarantee for the Adam trajectories above.

## Evidence and reproduction

Sources are Runs 269, 272, 275-279, 282-284 under H048. Executed configs and input manifests
are immutable. Raw boundaries, actual/native updates, gates, prefix calibration,
three gradient/parameter audit states per arm, and recovery/end states are
retained under each run's ignored artifacts. Retrieval receipts record file
sizes and SHA 256 identities. `17_reduce_transfer.py` independently checks
FP64 gradient reductions, every gate's recurrence, stored FP32 increments,
calibration quantiles, and optimizer clocks. `18_verify_representations.py`
replays retained parameter triplets on the original hardware and checks probe
RMS/cosines with NumPy. Run 278's `02_audit_budget.py` independently replays
before/native/actual states and verifies the actual output/representation
budgets. These checks distinguish implementation correctness from efficacy.

Numerical results are in `transfer_comparison_run*.json` and
`transfer_metrics_run*.csv`; the primary screen is `transfer_comparison.json`.
The new Figure 08 includes the failed controls alongside unchanged Adam, rather
than omitting them. Its complete caption and coverage are in observation 008.
Most policy dissection uses one seed and one fork. The selected distributed cap
and slow-recovery rule were additionally repeated on seeds 1/2 (Runs 282/283),
without seed-specific tuning. This limited replication and finite window do not
support a universal optimizer claim or a powered uncertainty estimate.
