# Problem formulation

# Ideia central

Uma breve introdução que deve ser simplificada / resumida (porque ja está descrita na introdução) \-\> Our central claim is that stochastic learning involves an inherent limitation of incorporating new information and overwriting previously learned structure (interference). We argue that this interference is not specific to continual learning settings (catastrophic forgetting); it arises even under standard pre-training i.i.d. sampling and can be quantified by the successive interactions among gradients from different mini-batches. The intuition is that, when a network already stores many concepts in its weights, back-propagating the error to those weights using a new mini-batch necessarily modifies previously learned structure. If the new gradients align with past updates, the change reinforces existing representations; if they do not, it partially overwrites them. When gradients interfere destructively, successive updates partially undo one another, leading to oscillatory parameter trajectories, a reduced effective step size, and slower convergence. The interference problem is directly connected with curvature stability.

Essa descrição acima é centrada na escrita de informação da rede. Abaixo descrevemos uma perspectiva baseada em otimização (percorrendo o espaço da loss).

### **Two phased behaviour**

**Phase 1 — Positive alignment and constructive behaviour.** No regime inicial, gradientes relativamente alinhados conduzem o otimizador em direção à bacia, com queda da loss.  
**Fase 2 — Oscilação convergente (spiralling across the vale).** As similaridades cosseno dos gradientes brutos nos lags 1 e 2 tornam-se negativas. O gradiente frequentemente inverte direção; a trajetória pode curvar-se ou oscilar ao redor do vale.

**Additional observation:** Após valores baixos de ||gt||, os máximos sucessivos voltam a crescer de forma sustentada, enquanto ||mt|| permanece aproximadamente estável. A razão qt\=||gt||/||mt|| aumenta. Há também uma queda observada de similaridade do momentum. Quanto maior o learning rate, mais precoce e intenso o descolamento entre g e m.

## Parameter displacement under sthocastic trajectory training

### 1. Displacement under SGD
Let \(g_t\) denote the mini-batch gradient evaluated at \(\theta_{t-1}\). Consider mini-batch stochastic gradient descent (SGD) with a constant learning rate \(\eta>0\), without momentum or weight decay:
\[
\theta_t=\theta_{t-1}-\eta g_t.
\]

After \(T\) updates, the parameter displacement from initialization is
\[
\Delta_T:=\theta_T-\theta_0
=-\eta\sum_{t=1}^{T}g_t.
\]

The norm \(\|\Delta_T\|\) measures the net Euclidean displacement between the initial and final parameters. Thus, the final displacement can be expressed as the cumulative sum of gradient updates over the optimization trajectory. Expanding its squared norm yields

\[
\boxed{
\|\Delta_T\|^2
=\eta^2\sum_{t=1}^{T}\|g_t\|^2
+2\eta^2\sum_{1\leq u<t\leq T}g_u^\top g_t.
}
\]

The first term aggregates the squared magnitudes of the gradients, while the second captures their pairwise alignment across training steps. This decomposition separates the contributions of individual gradient magnitudes from their temporal interactions. 

**A observação central aqui da equação acima é a seguinte**: no espaço de displacement de parametros, \(\|g_t\|^2\) é sempre positivo (displacement liquido), mas \(g_u^\top g_t\) pode ser negativo e depende do caminho da trajetória de convergência. To make it explicit ... (segue abaixo)


We further decompose the pairwise interaction term into contributions from adjacent gradients (lag 1) and gradients separated by two or more optimization steps (lags \(\geq 2\)). Define the lag-1 cosine similarity between consecutive gradients as
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

The decomposition reveals that the magnitude of parameter displacement depends not only on gradient magnitudes but also on their directional alignment across optimization steps. In particular, persistently negative lag-1 cosine similarities (\(\rho_t<0\)) introduce negative contributions to the squared displacement through the interaction of consecutive gradients, reflecting partial cancellation between successive updates. Persistent gradient anti-alignment acts as a mechanism that suppresses net parameter displacement. However, the overall displacement also depends on longer-lag interactions, which may reinforce or offset this effect.

**Spiriling around with SGD**: 2-lag negative cosine and 3rd positive may imply spiral around the minimum (not just jumping from one side of the basin to the other). >>> this is just a note for me to remember, but I'll move it to the experiment section later (less formal discussion).


#### Mini-batch Gradient Noise, Curvature and Consecutive-Gradient Alignment

For mini-batch SGD, we explicitly incorporate stochastic gradient noise into the evolution of consecutive gradients and examine its contribution to their directional alignment. Let \(\xi_t\) denote the stochastic gradient estimation error at step \(t\), such that

\[
g_t=\nabla\mathcal L(\theta_{t-1})+\xi_t,
\]

where \(\mathcal L\) denotes the full training objective. Applying a first-order Taylor expansion of the full gradient around \(\theta_{t-2}\), and substituting the SGD update \(\theta_{t-1}=\theta_{t-2}-\eta g_{t-1}\), gives

\[
\boxed{
g_t\approx
(I-\eta H_{t-1})g_{t-1}
+(\xi_t-\xi_{t-1}),
}
\]

where \(H_{t-1}=\nabla^2\mathcal L(\theta_{t-2})\) denotes the Hessian of the full objective. Taking the inner product with \(g_{t-1}\) and using the definition of lag-1 cosine similarity yields


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

This expression makes explicit how the local alignment of consecutive stochastic gradients relates with their magnitudes, the curvature of the objective along the preceding gradient direction, and the change in mini-batch gradient noise.


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

The quantity \(\kappa_t\) is the Rayleigh quotient of the Hessian along the preceding stochastic gradient direction, measuring the local directional curvature of the loss landscape. Equivalently, it is a weighted average of the Hessian eigenvalues, with weights determined by the gradient's projections onto the corresponding eigendirections. Consequently, \(\eta\kappa_t\) characterizes the first-order effect of local curvature on consecutive-gradient alignment, with larger positive values favoring curvature-induced gradient anti-alignment.

This makes the sign conditions explicit.

(put in a table below)
Positive alignment:  \(\rho_t>0 \quad \Leftrightarrow \quad 
\eta\kappa_t<1+\varepsilon_t
\)

Orthogonality:  \(\rho_t=0 
\quad \Leftrightarrow \quad 
\eta\kappa_t\approx1+\varepsilon_t
\)

Negative alignment: \(\rho_t<0 
\quad \Leftrightarrow \quad 
\eta\kappa_t>1+\varepsilon_t\)

When \(\rho_t\) is positive, the step size multiplied by the curvature of the objective along the preceding gradient direction is smaller than one plus the normalized noise contribution. This is the critical point that explains negative alignment and oscilatory. It does not inform however is the oscilation is dampened (amortecida) or unstable (cresce indefinidamente). De toda forma é um bom proxy diagnóstico para constructive accumulation, random drift or destructive interference.

##### Stochastic consequente
One particularly interesting stochastic consequence. Under fresh, conditionally unbiased mini-batch sampling, we have

\[
\mathbb E[g_{t-1}^{\top}\xi_t]=0.
\]

But because \(g_{t-1}\) already contains the previous noise \(\xi_{t-1}\),

\[
\mathbb E[g_{t-1}^{\top}\xi_{t-1}]
=
\mathbb E[\|\xi_{t-1}\|^2].
\]

Consequently,

\[
\boxed{
\mathbb E\left[
g_{t-1}^{\top}(\xi_t-\xi_{t-1})
\right]
=
-\mathbb E[\|\xi_{t-1}\|^2].
}
\]

Even unbiased mini-batch noise produces a nonpositive expected contribution to the unnormalized consecutive-gradient inner product. It suggests a systematic mechanism through which stochastic gradient noise can reduce successive-gradient alignment, independently of the local directional curvature.


#### Gradient accumulation and adjacent alignment

To effectively measure the the interference effect through long-lags, compare the sum of gradient norms with the norm of their sum:

\[
A_T=\sum_{t=1}^{T}\|g_t\|,\qquad
B_T=\left\|\sum_{t=1}^{T}g_t\right\|.
\]

For the constant-LR SGD case above, \(\eta A_T\) is the length of the discrete parameter path and \(\eta B_T=\|\Delta_T\|\) is its endpoint displacement. Thus

\[
\boxed{
\frac{\|\Delta_T\|}{\eta\sum_{t=1}^{T}\|g_t\|}
=\frac{B_T}{A_T}\in[0,1].
}
\]

The numerator is the net displacement; the denominator is the total length of all SGD steps. This normalized displacement expresses how much of the accumulated step length remains as net motion of parameters after the gradient directions reinforce or cancel one another. 

Its relation to cumulative directional interference is exact:

\[
\left(\frac{\|\Delta_T\|}{\eta\sum_{t=1}^{T}\|g_t\|}\right)^2
=1-
\frac{2\sum_{1\leq u<t\leq T}
\left(\|g_u\|\|g_t\|-g_u^\top g_t\right)}
{\left(\sum_{t=1}^{T}\|g_t\|\right)^2}.
\]


#### Empirical GD EoS-like CIFAR-10 experiment
Describe it here, cite the paper and add the analyses\019-2026-10-13-cifar10-eos-lr\figures\cifar10_gd_diagnostics.pdf figure (but make it more paper-like -> I feel its not good enough). I think this is a very strong result because we see that our \eta kappa approximation tracks the "edge of stability" which is the hoovering of \eta lambdamax around 2. But note that we add additional information there
...
when we reach this regime, the cumulative B/A drops aggressively. So the parameter displacement goes almost to zero. This means that the network steps are cancelling itself. Which is a oscilatory behaviour. 
-> in the other hand, raw-gradient norms peak again. We don't have the explanation for this.. only speculation. Maybe we're seeing the solutions oscilating around a basin.

keep a TODO note here:
- run an experiment applying a lr schedule based on \rho. Something like divide lr per 2 every time it goes to negative. My intuition: we may make \rho_t positive again, but it doesn't mean better convergence.

 
## 2. Displacement under Adam

For Adam with a constant learning rate \(\eta>0\), without weight decay, the parameter updates are

\[
\theta_t=\theta_{t-1}-\eta d_t,
\qquad
d_t:=\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon},
\]

where

\[
\begin{aligned}
m_t&=\beta_1m_{t-1}+(1-\beta_1)g_t,
&\widehat m_t&=\frac{m_t}{1-\beta_1^t},\\
v_t&=\beta_2v_{t-1}+(1-\beta_2)g_t^{\odot 2},
&\widehat v_t&=\frac{v_t}{1-\beta_2^t},
\end{aligned}
\]

with \(m_0=v_0=0\), \(\beta_1,\beta_2\in[0,1)\), \(\epsilon>0\), and elementwise operations in \(d_t\). After \(T\) updates, expanding the squared parameter displacement gives

\[
\boxed{
\|\Delta_T\|^2
=\eta^2\sum_{t=1}^{T}\|d_t\|^2
+2\eta^2\sum_{1\leq u<t\leq T}d_u^\top d_t.
}
\]

Although Adam's parameter trajectory is determined by the preconditioned directions \(d_t\), we focus on the momentum vectors \(m_t\) to characterize temporal gradient alignment and cross-step interference. This separates the directional coherence of the first-moment estimates from the additional geometric effects of adaptive preconditioning, while retaining the influence of the actual Adam trajectory on gradient evolution.

### Consecutive-momentum cosine similarity

Analogously to SGD, we define the consecutive-momentum cosine similarity as

\[
c_t:=
\frac{m_{t-1}^{\top}m_t}
{\|m_{t-1}\|\|m_t\|},
\qquad t=2,\ldots,T,
\]

assuming nonzero momentum norms. Using the momentum recurrence, we obtain the exact identity

\[
c_t=
\frac{
\beta_1\|m_{t-1}\|^2
+(1-\beta_1)m_{t-1}^{\top}g_t
}{
\|m_{t-1}\|\|m_t\|
}.
\]

To characterize the contributions of curvature and stochastic gradient noise, we use the first-order approximation along the Adam trajectory,

\[
g_t\approx g_{t-1}
-\eta H_{t-1}d_{t-1}
+(\xi_t-\xi_{t-1}),
\]

where \(H_{t-1}=\nabla^2\mathcal L(\theta_{t-2})\) and \(g_t=\nabla\mathcal L(\theta_{t-1})+\xi_t\). Defining the normalized momentum-tracking, curvature, and noise contributions as

\[
\begin{aligned}
a_t&:=
\frac{m_{t-1}^{\top}(g_{t-1}-m_{t-1})}
{\|m_{t-1}\|^2},\\[4pt]
\kappa_t^{(m)}&:=
\frac{m_{t-1}^{\top}H_{t-1}d_{t-1}}
{\|m_{t-1}\|^2},\\[4pt]
\varepsilon_t^{(m)}&:=
\frac{m_{t-1}^{\top}(\xi_t-\xi_{t-1})}
{\|m_{t-1}\|^2},
\end{aligned}
\]

we obtain

\[
\boxed{
c_t\approx
\frac{\|m_{t-1}\|}{\|m_t\|}
\left[
1+(1-\beta_1)
\left(
a_t-\eta\kappa_t^{(m)}
+\varepsilon_t^{(m)}
\right)
\right].
}
\]

Thus, consecutive-momentum alignment depends on momentum tracking, local curvature along the preconditioned update direction, and changes in stochastic gradient noise. The factor \(1-\beta_1\) controls how strongly these contributions modify the preceding momentum direction. Unlike SGD, the curvature contribution is not generally a Hessian Rayleigh quotient, since the momentum and adaptive update directions need not be parallel.

### Normalized momentum accumulation

To measure cumulative cross-step interference, we define the **normalized momentum accumulation** as

\[
\boxed{
R_T^{(m)}
:=
\frac{\left\|\sum_{t=1}^{T}m_t\right\|}
{\sum_{t=1}^{T}\|m_t\|}
= \frac{A^{(m)}}{B^{(m)}}
\in[0,1],
}
\]

assuming a nonzero denominator. Values close to one indicate that momentum vectors reinforce one another, whereas smaller values reflect greater cumulative cancellation across optimization steps. 




### Gradient growth and momentum reorientation

To relate gradient magnitudes to consecutive-momentum alignment, consider Adam's first-moment recurrence with \(\beta=\beta_1\in(0,1)\):

\[
(1-\beta)g_t=m_t-\beta m_{t-1}.
\]

For nonzero momentum vectors, define the gradient-to-momentum norm ratio \(q_t:=\|g_t\|/\|m_t\|\), the consecutive-momentum cosine similarity \(c_t\), and the momentum-norm ratio \(r_t:=\|m_{t-1}\|/\|m_t\|\). Taking squared norms and rearranging yields the exact identity

\[
\boxed{
c_t
=
1-
\frac{
(1-\beta)^2q_t^2-(1-\beta r_t)^2
}{
2\beta r_t
}.
}
\]

This relation shows that changes in the gradient-to-momentum ratio must be accommodated by changes in momentum magnitude, directional alignment, or both.

To isolate directional effects, we assume that consecutive momentum norms vary within a relative tolerance \(\delta\):

\[
\left|\|m_t\|-\|m_{t-1}\|\right|
\leq\delta\|m_t\|,
\qquad 0\leq\delta<1.
\]

Since \(1-\delta\leq r_t\leq1+\delta\), the exact identity implies

\[
\boxed{
c_t\leq
1-
\frac{
\left[(1-\beta)^2q_t^2
-(1-\beta+\beta\delta)^2\right]_+
}{
2\beta(1+\delta)
},
}
\]

where \([x]_+=\max(x,0)\). Thus, when momentum-norm variation is bounded, increasing \(q_t\) progressively lowers the upper bound on consecutive-momentum alignment. Larger gradients relative to the accumulated momentum must eventually produce stronger directional changes rather than simply increasing momentum magnitude.

In the limiting case of exactly constant momentum norms (\(\delta=0\)), the bound becomes an exact identity:

\[
\boxed{
c_t=
1-\frac{(1-\beta)^2}{2\beta}
\left(
\frac{\|g_t\|^2}{\|m_t\|^2}-1
\right).
}
\]

Therefore, when momentum magnitudes remain constant, increasing gradient norms necessarily reduce \(c_t\), indicating increasingly abrupt changes in the direction of the accumulated first moment. In particular, sufficiently large gradient-to-momentum ratios force consecutive momentum into opposing directions:

\[
\boxed{
q_t>\frac{\sqrt{1+\beta^2}}{1-\beta}
\quad\Longrightarrow\quad c_t<0.
}
\]

This provides a mechanism through which large gradient variations can produce strong momentum reorientation without a corresponding increase in momentum magnitude. The resulting decline in \(c_t\) indicates reduced directional coherence and, when negative, destructive cross-step interference. However, it does not independently establish whether the resulting oscillations are damped or unstable.



### Momentum reorientation and loss excursions

To connect cross-step interference with observable training behavior, we examine changes in the full training loss \(\ell_t:=\mathcal L(\theta_t)\). For a window of \(k\) Adam updates, let \(D_{t-k,s}:=\sum_{j=t-k+1}^{s}d_j\). A second-order expansion around \(\theta_{t-k}\) gives

\[
\boxed{
\begin{aligned}
\ell_s-\ell_{t-k}\approx{}&
-\eta\nabla\mathcal L(\theta_{t-k})^\top D_{t-k,s}\\
&+\frac{\eta^2}{2}
D_{t-k,s}^\top H_{t-k}D_{t-k,s},
\end{aligned}
}
\]

where \(H_{t-k}=\nabla^2\mathcal L(\theta_{t-k})\). The first term captures accumulated descent alignment, while the second includes curvature-weighted interactions between updates. Thus, cross-step interference can influence loss variation, although its effect depends on the local geometry and Adam's adaptive preconditioning.

We characterize two observable properties: **transient loss peaks** and **loss oscillation**, defined respectively as

\[
\boxed{
P_{t,k}:=
\left[
\max_{t-k<s<t}\ell_s
-\max(\ell_{t-k},\ell_t)
\right]_+,
\qquad k\geq2,
}
\]

\[
\boxed{
O_{t,k}:=
1-
\frac{|\ell_t-\ell_{t-k}|}
{\sum_{s=t-k+1}^{t}|\ell_s-\ell_{s-1}|}
\in[0,1],
}
\]

with \(O_{t,k}=0\) when the denominator vanishes. The first measures the prominence of an intermediate loss peak relative to both endpoints, whereas the second measures the fraction of accumulated loss variation canceled by subsequent changes. Unlike peak magnitude, oscillation can occur without exceeding the initial loss.

To relate these behaviors to momentum alignment, we consider two consecutive Adam updates. Expanding around \(\theta_{t-2}\), with \(f_{t-2}:=\nabla\mathcal L(\theta_{t-2})\), gives

\[
\begin{aligned}
\ell_t-\ell_{t-1}\approx{}&
-\eta f_{t-2}^\top d_t
+\eta^2d_{t-1}^\top H_{t-2}d_t\\
&+\frac{\eta^2}{2}d_t^\top H_{t-2}d_t.
\end{aligned}
\]

The cross-step curvature term provides a connection between successive update directions and loss variation. To make it explicit, consider locally isotropic positive curvature \(H_{t-2}\approx\lambda I\), approximately constant scalar preconditioning \(d_j\approx\alpha m_j\), and equal consecutive momentum norms \(\|m_{t-1}\|=\|m_t\|=M>0\). Near a stationary point, where \(f_{t-2}\approx0\), we obtain

\[
\begin{aligned}
\ell_{t-1}-\ell_{t-2}&\approx\frac{K}{2},\\
\ell_t-\ell_{t-1}&\approx K\left(c_t+\frac12\right),
\end{aligned}
\qquad
K:=\eta^2\lambda\alpha^2M^2.
\]

The first update increases the loss, while sufficiently negative momentum alignment (\(c_t<-1/2\)) reverses this increase in the following step. Consequently,

\[
\boxed{
P_{t,2}\approx
K\left[-c_t-\frac12\right]_+.
}
\]

This illustrates how momentum reorientation can produce a transient peak through recovery after an uphill step. The measured peak prominence increases with the subsequent recovery, even when the initial loss increase remains unchanged.

This behavior also connects to the gradient-to-momentum ratio. Under constant momentum norms, the exact identity

\[
\boxed{
c_t=
1-\frac{(1-\beta)^2}{2\beta}(q_t^2-1),
\qquad
q_t:=\frac{\|g_t\|}{\|m_t\|},
}
\]

shows that increasing \(q_t\) necessarily reduces consecutive-momentum alignment. In particular, the pure alternating mode at the frozen-preconditioner quadratic stability boundary satisfies

\[
\boxed{
c_t=-1,
\qquad
q_t=\frac{1+\beta}{1-\beta}.
}
\]

This establishes a conditional connection between gradient-to-momentum ratios, momentum reversal, and the oscillatory dynamics associated with the adaptive edge of stability. More generally, \(q_t\) and \(c_t\) provide diagnostics of momentum reorientation, while \(P_{t,k}\) and \(O_{t,k}\) characterize its possible manifestations in the loss trajectory. Their relationship is not universal: local curvature, stochastic noise, and adaptive preconditioning determine whether directional reversals amplify, sustain, or correct loss excursions.
