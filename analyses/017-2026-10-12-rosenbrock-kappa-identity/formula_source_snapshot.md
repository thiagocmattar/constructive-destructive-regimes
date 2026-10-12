# Problem formulation

Start with plain mini-batch SGD. Then treat Adam separately.

## 1. Displacement under SGD
Let \(g_t\) denote the mini-batch gradient evaluated at \(\theta_{t-1}\). Consider mini-batch stochastic gradient descent (SGD) with a constant learning rate \(\eta>0\), without momentum or weight decay:
\[
\theta_t=\theta_{t-1}-\eta g_t.
\]

After \(T\) updates, the parameter displacement from initialization is
\[
\Delta_T:=\theta_T-\theta_0
=-\eta\sum_{t=1}^{T}g_t.
\]

The norm \(\|\Delta_T\|\) measures the net Euclidean displacement between the initial and final parameters. Thus, the final displacement can be expressed as the cumulative sum of gradient updates over the optimization trajectory.
Expanding its squared norm yields
\[
\boxed{
\|\Delta_T\|^2
=\eta^2\sum_{t=1}^{T}\|g_t\|^2
+2\eta^2\sum_{1\leq u<t\leq T}g_u^\top g_t.
}
\]

The first term aggregates the squared magnitudes of the gradients, while the second captures their pairwise alignment across training steps. This decomposition separates the contributions of individual gradient magnitudes from their temporal interactions.
By considering displacement relative to initialization rather than the absolute parameter norm, the formulation also eliminates the explicit cross term involving the initialization vector \(\theta_0\).

We further decompose the pairwise interaction term into contributions from adjacent gradients (lag 1) and gradients separated by two or more optimization steps (lags \(\geq 2\)).
Define the lag-1 cosine similarity between consecutive gradients as
\[
\rho_t:=
\frac{g_{t-1}^{\top}g_t}
{\|g_{t-1}\|\|g_t\|},
\qquad t=2,\ldots,T,
\]

assuming nonzero gradient norms. The squared displacement can then be expressed as
\[
\boxed{
\begin{aligned}
\|\Delta_T\|^2
={}&\eta^2\sum_{t=1}^{T}\|g_t\|^2\\
&+2\eta^2\Bigg(
\underbrace{
\sum_{t=2}^{T}
\|g_{t-1}\|\|g_t\|\rho_t
}_{\text{Lag-1 interactions}}\\
&\qquad\quad+
\underbrace{
\sum_{t=3}^{T}\sum_{u=1}^{t-2}
g_u^\top g_t
}_{\text{Lag-}\geq 2\text{ interactions}}
\Bigg).
\end{aligned}
}
\]

The decomposition reveals that the magnitude of parameter displacement depends not only on gradient magnitudes but also on their directional alignment across optimization steps. In particular, persistently negative lag-1 cosine similarities (\(\rho_t<0\)) introduce negative contributions to the squared displacement through the interaction of consecutive gradients, reflecting partial cancellation between successive updates. Consequently, persistent gradient anti-alignment acts as a mechanism that suppresses net parameter displacement. However, the overall displacement also depends on longer-lag interactions, which may reinforce or offset this effect.



### Mini-batch Gradient Noise and Consecutive-Gradient Alignment

For mini-batch SGD, we explicitly incorporate stochastic gradient noise into the evolution of consecutive gradients and examine its contribution to their directional alignment.

Let \(\xi_t\) denote the stochastic gradient estimation error at step \(t\), such that

\[
g_t=\nabla\mathcal L(\theta_{t-1})+\xi_t,
\]

where \(\mathcal L\) denotes the full training objective.

Applying a first-order Taylor expansion of the full gradient around \(\theta_{t-2}\), and substituting the SGD update \(\theta_{t-1}=\theta_{t-2}-\eta g_{t-1}\), gives

\[
\boxed{
g_t\approx
(I-\eta H_{t-1})g_{t-1}
+(\xi_t-\xi_{t-1}),
}
\]

where \(H_{t-1}=\nabla^2\mathcal L(\theta_{t-2})\) denotes the Hessian of the full objective.

Taking the inner product with \(g_{t-1}\) and using the definition of lag-1 cosine similarity yields

\[
\boxed{
\rho_t\approx
\frac{
\underbrace{\|g_{t-1}\|^2}_{\text{Gradient magnitude}}
-
\underbrace{\eta g_{t-1}^{\top}H_{t-1}g_{t-1}}_{\text{Curvature contribution}}
+
\underbrace{g_{t-1}^{\top}(\xi_t-\xi_{t-1})}_{\text{Noise contribution}}
}{
\|g_{t-1}\|\|g_t\|
}.
}
\]

This expression characterizes the local alignment of consecutive stochastic gradients in terms of their magnitudes, the curvature of the objective along the preceding gradient direction, and the change in mini-batch gradient noise.


### Analysis of \(\rho_t\) sign

For nonzero gradient norms, the denominator is strictly positive. Therefore, to first order, the sign of \(\rho_t\) is determined by its numerator. Let's define

The directional curvature
\[
\kappa_t:=
\frac{g_{t-1}^{\top}H_{t-1}g_{t-1}}
{\|g_{t-1}\|^2}
\]

and the normalized noise contribution
\[
\varepsilon_t:=
\frac{g_{t-1}^{\top}(\xi_t-\xi_{t-1})}
{\|g_{t-1}\|^2}.
\]

Then
\[
\boxed{
\rho_t\approx
\frac{\|g_{t-1}\|}{\|g_t\|}
\left(1-\eta\kappa_t+\varepsilon_t\right).
}
\]

This makes the sign conditions explicit.

Positive alignment:  \(\rho_t>0 \quad \Leftrightarrow \quad 
\eta\kappa_t<1+\varepsilon_t
\quad \Leftrightarrow \quad
\eta < \frac{1+\varepsilon_t}{\kappa_t}
\)

Orthogonality:  \(\rho_t=0 
\quad \Leftrightarrow \quad 
\eta\kappa_t\approx1+\varepsilon_t
\quad \Leftrightarrow \quad
\eta \approx \frac{1+\varepsilon_t}{\kappa_t}
\)

Negative alignment: \(\rho_t<0 
\quad \Leftrightarrow \quad 
\eta\kappa_t>1+\varepsilon_t
\quad \Leftrightarrow \quad 
\eta > \frac{1+\varepsilon_t}{\kappa_t}\)

Consecutive gradients have an acute angle and positively reinforce their contributions to squared displacement. Consecutive gradients are orthogonal, and their lag-1 interaction contribution vanishes. Consecutive gradients have an obtuse angle and partially cancel their contributions to squared displacement. What the eta condition tells us? The curvature is smaller than noise? How this compare with the traditional \(1<\eta \lambda <2\)?


The quantity \(\kappa_t\) is the Rayleigh quotient of the Hessian along the preceding stochastic gradient direction, measuring the local directional curvature of the loss landscape. Equivalently, it is a weighted average of the Hessian eigenvalues, with weights determined by the gradient's projections onto the corresponding eigendirections. Consequently, \(\eta\kappa_t\) characterizes the first-order effect of local curvature on consecutive-gradient alignment, with larger positive values favoring curvature-induced gradient anti-alignment.







### Rwriting back to parameter displacement
Taking it back to the parameter displacement squared norm,

\[
\boxed{
\begin{aligned}
\|\Delta_T\|^2
\approx{}&
\underbrace{\eta^2\sum_{t=1}^{T}\|g_t\|^2}_{\text{Squared gradient magnitudes}}\\[4pt]
&+\underbrace{2\eta^2\sum_{t=2}^{T}\|g_{t-1}\|^2}_{\text{Lag-1 magnitude contribution}}\\[4pt]
&-\underbrace{2\eta^3\sum_{t=2}^{T}
g_{t-1}^\top H_{t-1}g_{t-1}}_{\text{Curvature contribution}}\\[4pt]
&+\underbrace{2\eta^2\sum_{t=2}^{T}
g_{t-1}^\top(\xi_t-\xi_{t-1})}_{\text{Stochastic noise contribution}}\\[4pt]
&+\underbrace{2\eta^2\sum_{t=3}^{T}\sum_{u=1}^{t-2}
g_u^\top g_t}_{\text{Longer-lag interactions}}.
\end{aligned}
}
\]