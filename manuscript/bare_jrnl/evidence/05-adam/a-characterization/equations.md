- **Status:** consolidated notation for later integration; the manuscript is unchanged.
- **Editable source:** [equations.tex](equations.tex). See the section README for source records and claim boundaries.

# V opening / V-A. Optimizer-dependent response and outcome timing
For a fixed smooth objective, use the actual incoming increment
$u_{s-1}=\theta_s-\theta_{s-1}$ and raw gradients $g_s=\nabla L(\theta_s)$.
$$
g_s-g_{s-1}
=\int_0^1H(\theta_{s-1}+a u_{s-1})u_{s-1}\,da,\qquad
Q_s=-\frac{g_{s-1}^\top(g_s-g_{s-1})}{\|g_{s-1}\|^2}.

$$
Writing $u_{s-1}=-\eta_{s-1}d_{s-1}$ gives
$$
\widehat\kappa_s=\frac{Q_s}{\eta_{s-1}}
=\int_0^1
\frac{g_{s-1}^\top H(\theta_{s-1}-a\eta_{s-1}d_{s-1})d_{s-1}}
{\|g_{s-1}\|^2}\,da .

$$
Adam determines the traversal; the raw gradient supplies the projection.
This is a mixed Hessian response, not generally a Rayleigh quotient.
Use the actual stored increment when clipping, decay or caps modify the proposal.
Changing sampled objectives add endpoint sampling noise. Smoothness is required.

For positive losses, the distinct future outcomes used in the retained audit are
$$
\begin{aligned}
z_{s,h}^{\rm step}&=\frac{L_{s+h}-L_{s+h-1}}{L_{s+h-1}},&
z_{s,h}^{\rm net}&=\frac{L_{s+h}-L_s}{L_s},&
z_{s,h}^{\rm max}&=\max_{1\leq j\leq h}\frac{L_{s+j}-L_s}{L_s}.

\end{aligned}
$$
The flag at $s$ uses no future value. Report the target, threshold, eligible
states, true positives, non-target alarms, precision and base rate.
All retained losses are positive; a future application must specify
zero-loss denominators explicitly. No universal Adam threshold follows.
