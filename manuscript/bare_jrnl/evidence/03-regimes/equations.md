- **Status:** consolidated notation for later integration; the manuscript is unchanged.
- **Editable source:** [equations.tex](equations.tex). See the section README for source records and claim boundaries.

# III. Interaction, curvature, sampling and loss
Let $f_s=\nabla L(\theta_s)$ and $g_s=f_s+\xi_s$. Plain SGD has
$u_s=\theta_{s+1}-\theta_s=-\eta g_s$ with constant $\eta>0$.
In full-batch GD, $\xi_s=0$. All norms below are Euclidean.

$$
\left\|\theta_T-\theta_0\right\|^2
=\eta^2\sum_{s=0}^{T-1}\|g_s\|^2
+2\eta^2\sum_{0\leq i<j<T}g_i^\top g_j .

$$
This is exact. Keeping only adjacent pairs is a truncation.
$$
R_T=\frac{\|\theta_T-\theta_0\|}{\eta\sum_{s=0}^{T-1}\|g_s\|}
=\frac{\|\sum_{s=0}^{T-1}g_s\|}{\sum_{s=0}^{T-1}\|g_s\|}\in[0,1].

$$
$R_T$ is undefined at zero total path length. It measures displacement,
not knowledge retention or efficiency.

$$
\rho_s=\frac{g_{s-1}^\top g_s}{\|g_{s-1}\|\|g_s\|},\qquad
r_s=\frac{\|g_s\|}{\|g_{s-1}\|},\qquad
Q_s=1-\frac{g_{s-1}^\top g_s}{\|g_{s-1}\|^2}=1-\rho_s r_s .

$$
The factorization needs both gradients nonzero. If only $g_s=0$, $Q_s=1$
but $\rho_s$ is undefined; if $g_{s-1}=0$, $Q_s$ is undefined.
$Q_s$ is available after backward at state $s$, describing the incoming step.

For a twice continuously differentiable fixed objective along the step, define
$H(\theta)=\nabla^2L(\theta)$ and
$\varepsilon_s=g_{s-1}^\top(\xi_s-\xi_{s-1})/\|g_{s-1}\|^2$. Then
$$
\widehat\kappa_s:=\frac{Q_s}{\eta}
=\int_0^1
\frac{g_{s-1}^\top H(\theta_{s-1}-a\eta g_{s-1})g_{s-1}}
{\|g_{s-1}\|^2}\,da-\frac{\varepsilon_s}{\eta}.

$$
With $H$ Lipschitz in operator norm and zero sampling noise,
$$
\left|\widehat\kappa_s-
\frac{g_{s-1}^\top H(\theta_{s-1})g_{s-1}}{\|g_{s-1}\|^2}\right|
\leq\frac{L_H\eta}{2}\|g_{s-1}\|.

$$
Thus point curvature is an approximation to the exact path average.
No extra Hessian-vector product or backward pass is required to compute $Q_s$.

For full-batch GD, put $v=\theta_s-\theta_{s-1}$ and $y=g_s-g_{s-1}$.
$$
\widehat\kappa_s=\frac{v^\top y}{v^\top v}
=\alpha_{\mathrm{BB1}}^{-1},\qquad
\alpha_{\mathrm{BB1}}=\frac{v^\top v}{v^\top y}.

$$
The reciprocal relation requires the corresponding quantities to be defined;
credit Barzilai--Borwein and the scalar-Hessian interpretation in Tan et al.

Fresh conditionally unbiased batches imply
$$
\mathbb E[g_{s-1}^\top(\xi_s-\xi_{s-1})]
=-\mathbb E\|\xi_{s-1}\|^2,\qquad
\mathbb E[\|\xi\|^2\mid\theta]
=\frac{N-b}{b(N-1)}\operatorname{tr}\Sigma(\theta).

$$
Here $\Sigma=N^{-1}\sum_i(g_i-f)(g_i-f)^\top$ and batches are uniform,
without replacement within each batch. Independently sampled gradients at one
fixed state give the norm-concentration approximation
$$
\mathbb E[\rho\mid\theta\ \mathrm{fixed}]
\approx
\frac{\|f(\theta)\|^2}
{\|f(\theta)\|^2+\frac{N-b}{b(N-1)}\operatorname{tr}\Sigma(\theta)}.

$$
This is not an exact expectation of a ratio or an evolving-trajectory guarantee.

The endpoint trapezoidal approximation for the incoming plain-SGD step is
$$
L_s-L_{s-1}\approx
-\frac{\eta}{2}\bigl(\|g_{s-1}\|^2+g_{s-1}^\top g_s\bigr)
+\frac{\eta}{2}g_{s-1}^\top(\xi_{s-1}+\xi_s).

$$
For a quadratic full-batch objective this becomes the exact relation
$$
L_s-L_{s-1}=\frac{\eta}{2}\|g_{s-1}\|^2(Q_s-2),\qquad
L_s>L_{s-1}\ \Longleftrightarrow\
Q_s>2\ \Longleftrightarrow\ \rho_s<-1/r_s .

$$
The strict inequality implies reversal and growth; the conjunction is weaker.
For nonlinear losses report the residual; this does not predict the outgoing step.
