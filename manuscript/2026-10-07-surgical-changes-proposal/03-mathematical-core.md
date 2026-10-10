# Mathematical core of the proposed revision

Start with plain mini-batch SGD and derive the diagnostics directly from its
gradients. Then treat Adam separately through its first moment and
preconditioned direction. The measurement contracts of completed runs remain
as recorded.

## Displacement under SGD

Let \(g_t\) be the mini-batch gradient evaluated at \(\theta_{t-1}\). For SGD
with constant learning rate \(\eta>0\), without momentum or weight decay,

\[
\theta_t=\theta_{t-1}-\eta g_t.
\]

After \(T\) updates, define the displacement from initialization as

\[
\Delta_T:=\theta_T-\theta_0=-\eta\sum_{t=1}^{T}g_t.
\]

Its norm \(\|\Delta_T\|\) measures the net distance traveled from
initialization. Taking its squared norm gives

\[
\boxed{
\|\Delta_T\|^2
=\eta^2\sum_{t=1}^{T}\|g_t\|^2
+2\eta^2\sum_{1\leq u<t\leq T}g_u^\top g_t.
}
\]

The first term adds the squared gradient magnitudes. The second measures
how gradients reinforce or oppose one another across training steps.
Studying displacement removes the initialization-origin cross term in the
draft's absolute parameter-norm expansion. Displacement still depends on
parameterization and does not by itself measure loss or stability.

## Gradient accumulation and adjacent alignment

Compare the sum of gradient norms with the norm of their sum:

\[
A_T=\sum_{t=1}^{T}\|g_t\|,\qquad
B_T=\left\|\sum_{t=1}^{T}g_t\right\|.
\]

For the constant-LR SGD case above, \(\eta A_T\) is the length of the discrete
parameter path and \(\eta B_T=\|\Delta_T\|\) is its endpoint displacement.
Thus

\[
\boxed{
\frac{\|\Delta_T\|}{\eta\sum_{t=1}^{T}\|g_t\|}
=\frac{B_T}{A_T}\in[0,1].
}
\]

The numerator is the net displacement; the denominator is the total length
of all SGD steps. This **normalized displacement** expresses how much of the
accumulated step length remains as net motion after the gradient directions
reinforce or cancel one another. Use the explicit displacement expression
in the argument; retain \(A_T\) and \(B_T\) to connect it to the measured curves.

Its relation to cumulative directional interference is exact:

\[
\left(\frac{\|\Delta_T\|}{\eta\sum_{t=1}^{T}\|g_t\|}\right)^2
=1-
\frac{2\sum_{1\leq u<t\leq T}
\left(\|g_u\|\|g_t\|-g_u^\top g_t\right)}
{\left(\sum_{t=1}^{T}\|g_t\|\right)^2}.
\]

The subtracted term measures the deficit relative to perfectly aligned
gradients with the same individual magnitudes. Each pair contributes
\(\|g_u\|\|g_t\|(1-\cos(g_u,g_t))\): opposing directions contribute more
than orthogonal directions, while perfectly aligned pairs contribute zero.
This includes every temporal separation, without a lag-one approximation.

A ratio of 1 means all nonzero gradients point in the same direction;
a ratio of 0 means their vector sum cancels exactly. The complement,
\(1-\|\Delta_T\|/(\eta\sum_t\|g_t\|)\), is the fraction of path length that
does not appear as endpoint displacement. It is a geometric measure,
not a fraction of wasted learning.

A ratio below 1 can arise from orthogonal or imperfectly aligned positive
pairs as well as negative pairs. Two orthogonal unit gradients give
normalized displacement \(1/\sqrt2\) with zero cross-step inner product.
A low ratio therefore does not establish negative total interaction or
poor optimization.

The local diagnostic compares consecutive mini-batch gradients:

\[
\rho_t=\frac{g_{t-1}^{\top}g_t}
{\|g_{t-1}\|\|g_t\|},\qquad t=2,\ldots,T.
\]

Negative cosine means that the two gradients oppose one another.

The first adjacent cosine is undefined. A zero norm makes its pair cosine
undefined; normalized displacement is undefined when \(\sum_t\|g_t\|=0\).
Store null with a reason and
report valid coverage. For a contiguous training window, use only the
gradients and pairs inside that window.

## Interaction at each temporal lag

Group the cross-step term by the separation \(k\) between gradients:

\[
I_k=\sum_{t=k+1}^{T}g_{t-k}^{\top}g_t,\qquad
\|\Delta_T\|^2
=\eta^2\sum_{t=1}^{T}\|g_t\|^2
+2\eta^2\sum_{k=1}^{T-1}I_k.
\]

This is exact for the stated SGD update. Adjacent cosine describes lag one;
the identity retains every lag.

The same decomposition can be written directly in terms of \(B_T\):

\[
B_T^2=\sum_{t=1}^{T}\|g_t\|^2+2\sum_{k=1}^{T-1}I_k.
\]

Therefore, the total cross-step interaction is

\[
\sum_{k=1}^{T-1}I_k
=\frac{1}{2}\left(B_T^2-\sum_{t=1}^{T}\|g_t\|^2\right).
\]

It is available from the cumulative gradient sum and the sum of squared
gradient norms, without storing every gradient or computing every pair.
Resolving each lag's contribution is the task for the compressed gradient
history.

These diagnostics answer different questions:

- \(\rho_t<0\): two consecutive gradients oppose one another.
- \(\sum_{k=1}^{T-1}I_k<0\): the sum of all cross-step interactions is negative
  over the declared horizon.
- \(\|\Delta_T\|/(\eta\sum_t\|g_t\|)<1\): the SGD path produces less net
  displacement than perfectly aligned gradients with the same magnitudes.

The interaction omitted by a lag-one approximation is
\(\sum_{k=2}^{T-1}I_k\). Its relative error in the squared cumulative gradient
norm is

\[
\frac{\left|B_T^2-\left(\sum_{t=1}^{T}\|g_t\|^2+2I_1\right)\right|}{B_T^2},
\]

when \(B_T>0\). If \(B_T=0\), report the absolute residual and leave the
relative error undefined. The lag-one truncated expression is not itself
guaranteed to be a valid squared norm.

Within a window of \(N\) gradients, reindexed locally from 1 to \(N\), the
mean lag-cosine profile is

\[
C_W(k)=\frac{1}{N-k}\sum_{t=k+1}^{N}
\frac{g_{t-k}^{\top}g_t}{\|g_{t-k}\|\|g_t\|}.
\]

Use the valid-pair denominator if some pairs have zero norms. This is an
uncentered alignment profile, not a stationary autocovariance estimate.
\(C_W(k)\) averages cosines equally, whereas \(I_k\) retains gradient
magnitudes. A small positive longer-lag background can dominate the total
interaction because of the number and magnitudes of those pairs.

With a learning-rate schedule, SGD instead gives
\(\Delta_T=-\sum_t\eta_tg_t\): the squared-norm expansion contains
\(\eta_t^2\|g_t\|^2\) and \(\eta_u\eta_t g_u^\top g_t\).
The constant-\(\eta\) path-length and displacement relations above must
then use the LR-weighted sums.

## Adam before and after preconditioning

Use the standard Adam equations, omitting gradient clipping from this
derivation. In the experiments, raw gradients are recorded after unscaling
and before clipping; moment diagnostics are read from the actual optimizer
state.

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^{\odot2},
\]
\[
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t},\qquad
d_t=\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
\]

The division is coordinatewise. The preconditioner uses the square root of
the bias-corrected second moment, not simply division by \(v_t\) or “\(v_2\).”
The completed recipe uses \(\beta_1=0.9,\beta_2=0.95\) and FP32 moments.
[Run 002](../../runs/002-2026-10-03-pythia14m-adamw-adaptive-geometry/README.md)
evaluates \(d_t\) in FP64 from the actual updated moments.

For Adam's first moment, calculate the same quantities explicitly as

\[
A_T(m)=\sum_{t=1}^{T}\|m_t\|,\qquad
B_T(m)=\left\|\sum_{t=1}^{T}m_t\right\|.
\]

Compare these and \(B_T(m)/A_T(m)\) with the raw-gradient \(A_T,B_T\) and
\(B_T/A_T\) defined above, and report the bias-corrected first moment and
adaptive direction in their own panels.
The new overlay should report:

| Vector | Comparison purpose |
| --- | --- |
| \(g_t\) | Preserve the user's raw-gradient reference |
| \(m_t\) | Measure the literal stored first moment requested in the discussion |
| \(\hat m_t\) | Separate averaging from the early bias-correction factor; recommended primary comparable-scale moment curve |
| \(d_t\) | Relate the new comparison to the existing adaptive-direction result |
| \(\theta_t-\theta_{t-1}\), if captured | Connect the geometry to realized parameter motion |

Bias correction preserves adjacent first-moment cosine while changing early
cumulative weighting. Coordinatewise preconditioning can change both angles
and scales.
Rescaling each full sequence by one common positive constant leaves \(B/A\)
unchanged; time-varying rescaling need not.

For ordinary AdamW, the ideal real-arithmetic update for parameter coordinate
\(j\) is

\[
\theta_{t,j}-\theta_{t-1,j}
=-\eta_t d_{t,j}-\eta_t\lambda_j\theta_{t-1,j},
\]

where \(\lambda_j=0\) for excluded parameter groups. Actual parameter
subtraction also reflects finite precision. The existing \(d_t\) stream excludes
LR, decay, and this rounding. A separate optimizer intervention would be
required to attribute learning outcomes to momentum; comparing vectors inside
one AdamW run describes transformations along that run.

The raw-gradient \(A_T,B_T\) and \(B_T/A_T\) remain well-defined during AdamW
training, but its parameter displacement follows the AdamW update above.
The SGD relation \(\|\Delta_T\|=\eta B_T\) does not apply to these raw-gradient
diagnostics under AdamW. When reporting moment diagnostics
inside a window, reset only their diagnostic sums; keep Adam's inherited
moment state.

## What temporal averaging predicts mathematically

Preserve the user's predictions \(A_T(m)<A_T\), \(B_T(m)>B_T\), and a smaller
relative gap, but test each separately. Here \(A_T,B_T\) are the raw-gradient
quantities. A smaller sum of norms does not imply a larger norm of the sum.

For a zero-initialized, uncorrected EMA with fixed \(0\leq\beta_1<1\),
real-arithmetic expansion gives

\[
m_t=(1-\beta_1)\sum_{s=1}^{t}\beta_1^{t-s}g_s,
\]
\[
A_T(m)\leq
\sum_{s=1}^{T}(1-\beta_1^{T-s+1})\|g_s\|
\leq A_T,
\]
\[
\sum_{t=1}^{T}m_t
=\sum_{t=1}^{T}g_t-\frac{\beta_1}{1-\beta_1}m_T.
\]

The first inequality explains attenuation of cumulative magnitudes. The
boundary term shows why \(B_T(m)\) need not exceed \(B_T\): its effect
depends on the terminal moment's direction. With nonzero initial moment,
replace \(m_T\) by \(m_T-m_0\) in the last identity. The attenuation bound
as written assumes \(m_0=0\) and does not automatically apply to \(\hat m_t\)
or to a middle window with inherited momentum.

For an alternating zero-mean component, an EMA attenuates the alternating
amplitude by \((1-\beta_1)/(1+\beta_1)\) after transients; for \(\beta_1=0.9\)
this is about 0.0526. A persistent component can survive while the alternating
component shrinks. This supplies intuition for \(A\) decreasing and \(B/A\)
increasing even if \(B\) stays similar. It is an illustration of a linear
filter, not proof of the mechanism of the measured network.

These elementary identities can explain the planned test, but are not
themselves a new optimization theorem.

## Basin crossing and learning-rate diagnosis

The proposed intuition is that an update crosses a local loss basin or
valley, causing the gradient to turn back toward its bottom. In a smooth
one-dimensional basin with a unique minimum, gradients with opposite signs
at the two endpoints bracket that minimum. Crossing it produces negative
adjacent cosine; repeated crossing produces oscillation.

For \(f(z)=\lambda z^2/2\), with \(\lambda>0\), deterministic gradient
descent makes this relation explicit:

\[
z_{t+1}=(1-\eta\lambda)z_t,\qquad
f(z_{t+1})=(1-\eta\lambda)^2f(z_t).
\]

The learning rate that reaches the minimum in one step is
\(\eta=1/\lambda\). Smaller steps remain on the same side and give cosine
\(+1\); larger steps cross the minimum and give cosine \(-1\).
At the minimum itself, the next gradient is zero and its cosine is undefined.

For example, with \(\lambda=1\) and starting position \(z=1\), learning
rates 0.9 and 1.5 lead to positions 0.1 and \(-0.5\), respectively.
Their gradient signs bracket the minimum between those positions and the
best learning rate between 0.9 and 1.5; here it is exactly 1.
Thus, paired positive- and negative-cosine responses can carry actionable
information about step size relative to the basin.

The further question is whether the oscillations contract or expand.
When \(1<\eta\lambda<2\), the updates cross the minimum repeatedly while
getting closer and reducing loss. When \(\eta\lambda>2\), they cross it
with growing amplitude. Negative cosine identifies crossing in this
example; gradient magnitudes and loss changes distinguish these outcomes.

In a multidimensional valley, an update can overshoot across a sharp
direction while making progress along a shallow direction. Repeated
cross-valley motion contributes to the gap between total step length and
net displacement. This connects adjacent cosine to the cumulative
interference diagnostic and motivates adjusting the learning rate to
reduce excessive oscillation while maintaining progress along the valley.
Positive alignment can indicate motion on the same side of the valley,
but its sign alone does not measure how close the step is to an optimum.

For mini-batch network training, basin crossing is a mechanism to test:
different batches, changing curvature, and optimizer transformations also
affect gradient alignment. The controlled toy study should expose valley
crossings, oscillation amplitude, loss, actual updates, and temporal
geometry. Matched learning-rate interventions should then test whether
the cosine response helps identify a better step size at a shared
checkpoint. A decrease in negative cosine is not itself the target;
the target is an improvement in optimization informed by the geometry.
