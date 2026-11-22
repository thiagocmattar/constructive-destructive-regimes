> Evidence snapshot; original scientific content retained. Relative links were rebased.
> Source: [010-local-displacement-cap-geometry.md](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/010-local-displacement-cap-geometry.md).

# What a local displacement cap controls

## TLDR

The rule limits the change that a proposed parameter update directly produces
at each affine output unit, measured on fixed probe inputs. It measures an
update in **representation units**, rather than by its parameter norm alone.
All layers remain trainable and Adam's gradients, moments, and clocks advance.

This is a local constraint. It does not bound the complete network's output
change, require loss decrease, or remove back-and-forth motion. The temporal
SVD analysis motivated looking at representation movement; the implemented
controller does **not** compute or use that SVD basis.

This note gives the algebra of the frozen H048 implementation. Empirical
results, coverage, and limitations belong to [observation 008](65-008-cross-architecture-interventions.md).

## Matrices and one proposed update

Consider one affine layer at completed optimizer update $t$. Let

- $H_t\in\mathbb R^{n\times d}$ contain its current input representations:
  rows are probe examples or token/spatial positions, columns are input features;
- $W_t\in\mathbb R^{m\times d}$ contain its weights: row $j$ belongs to output
  unit $j$;
- $b_t\in\mathbb R^m$ contain its biases;
- $Z_t=H_tW_t^\top+\mathbf1b_t^\top\in\mathbb R^{n\times m}$ be its affine output.

Native Adam proposes increments $U_t$ and $v_t$ to $W_t$ and $b_t$. These are
the differences between the actual stored FP32 parameters before and after
the native Adam step. They are not raw gradients or unscaled momentum.
Holding $H_t$ fixed, their direct output contribution is

$$
D_t=H_tU_t^\top+\mathbf1v_t^\top\in\mathbb R^{n\times m}.
$$

The score for unit $j$ is the RMS down column $j$:

$$
s_{t,j}=\sqrt{\frac1n\sum_{i=1}^{n}D_{t,ij}^2}.
$$

This is a per-unit RMS across rows, not a bound on every individual example's
response and not a norm of the sample-mean response.

H048 evaluates this affine product in FP64 from the captured FP32 inputs and
stored parameter increments. The score describes that direct affine map;
subtracting two complete FP32 layer forwards also includes forward-arithmetic
rounding, in addition to any change in their inputs.

Given the layer's positive threshold $\tau$, apply

$$
a_{t,j}=\begin{cases}
\min(1,\tau/s_{t,j}),&s_{t,j}>0,\\
1,&s_{t,j}=0,
\end{cases}
\qquad
\widetilde U_{t,j:}=a_{t,j}U_{t,j:},\quad
\widetilde v_{t,j}=a_{t,j}v_{t,j}.
$$

Thus the direct RMS becomes $a_{t,j}s_{t,j}\leq\tau$ in exact arithmetic.
The implementation checks a bound for final FP32 storage rounding. If the
calibrated threshold itself is zero, H048 disables that layer's cap rather
than freezing it. Untargeted parameter increments remain native.

The same factor scales the entire weight row and its bias. It preserves that
unit's direct output-change direction under the frozen input. Different factors
across units can change the combined update direction. This is not a rule that
zeros individual large gradient coordinates.

## Why parameter norm alone is different

Append a column of ones to include the bias:

$$
\widetilde H_t=[H_t,\mathbf1],\qquad
u_{t,j}=[U_{t,j:}^{\top},v_{t,j}]^{\top},\qquad
G_t=\frac1n\widetilde H_t^{\top}\widetilde H_t.
$$

Then

$$
s_{t,j}^2=u_{t,j}^{\top}G_tu_{t,j}.
$$

$G_t$ is an empirical **second-moment matrix**, not a centered covariance
matrix. It includes feature means and their interaction with the bias.
Two parameter updates with equal Euclidean norm can therefore have very
different scores: they can act along differently amplified feature directions,
or have different weight/bias cancellation on the probe.

The accepted update lies on the original proposed ray and inside the
quadratic region $u^{\top}G_tu\leq\tau^2$. This is an ellipsoid when $G_t$
is positive definite; it can be unbounded along unobserved directions when
$G_t$ is singular. The radial rescaling is not generally the closest Euclidean
projection onto that region. No inverse matrix is computed, and no claim of
natural-gradient optimization follows.

If we *describe* the current input using
$\widetilde H_t=Q_t\Sigma_tV_t^{\top}$, then

$$
s_{t,j}^2=\frac1n\sum_k\sigma_{t,k}^2
\bigl(v_{t,k}^{\top}u_{t,j}\bigr)^2.
$$

This identity explains sensitivity to input directions. The implementation
computes the direct product $D_t$ instead of this SVD. Moreover, this would be
an SVD of a **single state's input-representation matrix**, not the earlier
fixed-window SVD whose rows are flattened consecutive representation changes.
They answer different questions and their singular vectors must not be
identified with one another.

## Why the complete representation can still move more

Other layers also update, so the actual next input is
$H_{t+1}=H_t+\Delta H_t$. The affine output change is exactly

$$
\Delta Z_t
=\underbrace{H_t\widetilde U_t^{\top}
 +\mathbf1\widetilde v_t^{\top}}_{\text{direct contribution being capped}}
 +\underbrace{\Delta H_tW_t^{\top}
 +\Delta H_t\widetilde U_t^{\top}}_{\text{incoming motion and interaction}}.
$$

The last two terms are unconstrained by this layer's local cap. Subsequent
normalization, attention, nonlinearities, and output projections can further
change the relationship to the final output. This is why the experiments also
measure actual representation RMS, adjacent-change cosine, and the full loss.

In particular, the cap can reduce the amplitude of a reversing motion while
leaving its consecutive-change cosine negative. That behavior is compatible
with the rule and should not be counted as a controller failure by itself.

## Convolutions, attention, calibration, and cost

For a convolution, the corresponding rows of $H_t$ are receptive-field patches
over all probe examples and output spatial positions. The code performs the
convolution directly in chunks, without materializing this patch matrix.
For the Transformer, the score covers the direct QKV affine projection and
the feed-forward affine modules. A QKV output coordinate is not the complete
attention output, and capping it does not directly constrain attention weights.
The current 64-sequence probe has 2,240 valid positions and no padding. For
a future padded probe, its local-score input rows need an explicit valid-position
mask; the current affine-score routine flattens all captured input positions.

For each targeted module, H048 pools scores across all its output units and
eight preceding untreated updates, then fixes the chosen quantile as $\tau$.
The common distributed comparison uses q50. Inputs are recaptured at each
new state, but the threshold is not refitted using the future intervention
trajectory. All targeted scores use inputs captured before the native step.

The local rule needs a probe forward to capture inputs and a direct affine
product for each targeted module. H048 additionally recomputes actual scores
to verify the FP32 bound. It does not require a full Jacobian, an input Gram
matrix, a temporal Gram matrix, or an SVD. Convolutional FP64 products and
large captured tensors can still be expensive; use the measured timing and
memory records, rather than assuming this diagnostic implementation is a
production-efficient optimizer.

The fixed probes preserve sample/token/spatial structure and original
normalization groups. A cap calibrated on a small probe is not a worst-case
bound over the dataset. Transfer tests use each model's own calibration and
baseline, not a common numerical threshold or matched parameter fraction.

## Reproduction and scope

The implemented score and rescaling are in
[representation_policy.py](../../../../../../src/eos_tasks/representation_policy.py),
and native-step placement is in
[representation_continuation.py](../../../../../../src/eos_tasks/representation_continuation.py).
Runs 272/273/274 provide the distributed q50 comparison; Runs 282/283 repeat
the MLP rule on two additional initializations. The independent reducer
[17_reduce_transfer.py](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/17_reduce_transfer.py) checks thresholds, stored
increments, rounding bounds, and unchanged native optimizer advancement.

The equations above are algebraic identities under the stated affine model,
not a convergence guarantee or proof that a particular representation caused
a loss spike. In particular, successful transfer of this rule would support
controlling local functional movement; it would not establish that suppressing
the leading temporal SVD direction is the mechanism of improvement.
