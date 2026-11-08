- **Status:** consolidated notation for later integration; the manuscript is unchanged.
- **Editable source:** [equations.tex](equations.tex). See the section README for source records and claim boundaries.

# V-C. Functional loss accounting and local affine cap
For the MLP, $F_s,Y\in\mathbb R^{N\times D}$, $N=5000$, $D=10$,
and $L_s=\|F_s-Y\|_F^2/(2N)$. The actual output change obeys
$$
L_{s+1}-L_s=
\frac{\langle F_s-Y,\Delta F_s\rangle}{N}
+\frac{\|\Delta F_s\|_F^2}{2N},\qquad
\frac{\|\Delta F_s\|_F^2}{2N}=\frac D2\operatorname{RMS}(\Delta F_s)^2.

$$
This is exact in output space, not a parameter-linear Taylor approximation.
The first term need not equal $g_s^\top u_s$. Do not apply it unchanged to
cross-entropy.

For one affine module at state $s$, let
$H_s\in\mathbb R^{n\times d}$, $W_s\in\mathbb R^{m\times d}$,
$b_s\in\mathbb R^m$. Rows of $H_s$ are probe examples or token/spatial positions.
Native stored Adam increments $U_s,v_s$ have the shapes of $W_s,b_s$.
$$
D_s=H_sU_s^\top+\mathbf1_n v_s^\top\in\mathbb R^{n\times m},\qquad
c_{s,j}=\sqrt{\frac1n\sum_iD_{s,ij}^2}.

$$
The score is one RMS per output unit across all rows, not a mean response or
a per-example worst-case bound. For positive module budget $\tau_\ell$:
$$
a_{s,j}=
\begin{cases}\min(1,\tau_\ell/c_{s,j}),&c_{s,j}>0,\\1,&c_{s,j}=0,\end{cases}
\qquad
\widetilde U_{s,j:}=a_{s,j}U_{s,j:},\qquad
\widetilde v_{s,j}=a_{s,j}v_{s,j}.

$$
One threshold per module is calibrated from a quantile (q50 in the distributed
comparison) pooled over units and eight untreated preceding updates. A zero
calibrated threshold disables that module's cap. Native moments and clock advance;
final stored FP32 increments require the recorded rounding check.

Let $\widetilde H_s=[H_s,\mathbf1_n]$, $\bar u_{s,j}=[U_{s,j:}^\top,v_{s,j}]^\top$.
$$
c_{s,j}^2=\bar u_{s,j}^\top G_s\bar u_{s,j},\qquad
G_s=\frac1n\widetilde H_s^\top\widetilde H_s .

$$
$G_s$ is uncentered. The cap rescales along each native unit proposal;
it neither inverts $G_s$ nor computes the nearest Euclidean projection.
Changing upstream inputs leaves the extra terms
$$
\Delta Z_s=
\underbrace{H_s\widetilde U_s^\top+\mathbf1_n\widetilde v_s^\top}_{\text{capped direct term}}
+\underbrace{\Delta H_sW_s^\top+\Delta H_s\widetilde U_s^\top}_{\text{unconstrained incoming and interaction terms}}.

$$
No whole-network loss or output bound follows. The controller uses local products,
not the temporal SVD. Convolution scores use receptive-field patches implicitly;
Transformer QKV scores do not bound complete attention outputs.
