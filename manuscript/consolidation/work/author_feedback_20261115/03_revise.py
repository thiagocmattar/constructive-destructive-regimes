"""Apply the author-annotated revision to consolidation sources only.

Historical editorial script: do not rerun after subsequent manual polishing.
"""
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
DRAFT=HERE.parents[1]/'draft'
S=DRAFT/'sections'
def read(name): return (S/name).read_text(encoding='utf-8')
def write(name,text): (S/name).write_text(text,encoding='utf-8',newline='\n')
def replace(name,old,new):
    text=read(name)
    assert old in text, (name,old[:80])
    write(name,text.replace(old,new))

intro=read('01-introduction.tex')
start=intro.index('In practice, stability')
end=intro.index('Destructive alignment alone')
intro=intro[:start]+r'''In practice, learning-rate schedules, gradient clipping, and adjustments to momentum or adaptive second moments help control update size~\cite{smith2017cyclical,loshchilov2016sgdr,pascanu2013difficulty,kingma2014adam}. These choices affect both the smoothness of training and how rapidly the network learns. Understanding what they control is therefore necessary for deciding when to intervene.

Classical stability analysis relates update size to curvature: a step can overshoot a minimum when the learning rate is too large relative to a sharp direction~\cite{polyak1964methods,bertsekas1997nonlinear}. Stochastic extensions also account for curvature variability and the influence of batch size~\cite{jastrzebski2020break}. Yet the Edge-of-Stability literature shows that loss can continue to decrease while sharpness remains near or beyond the nominal $2/\eta$ threshold~\cite{cohen2021edge,arora2022understanding,damian2023self}. Under Adam, adaptive preconditioning can itself generate loss spikes, even without a progressively sharper underlying objective~\cite{bai2026spikes}. These results explain important aspects of optimization, but a curvature threshold alone does not describe which learned representations are moving back and forth, how large their changes become, or which changes an intervention should limit.

This distinction matters because training is also a process of reorganizing the function represented by a network. The loss geometry depends on the data and architecture; for example, Li et al.~\cite{li2018visualizing} show how skip connections alter the geometry of loss surfaces and improve their trainability. A scalar loss curve hides these internal changes. Moreover, an upward excursion can be followed by further loss reduction, so suppressing every excursion need not preserve progress. We therefore seek an interpretation that connects the optimizer's steps to changes in the learned function, and uses that connection to study interventions.

To address this question, we develop a stability perspective based on constructive and destructive training regimes. Starting from the SGD equations, we express accumulated parameter displacement in terms of gradient magnitudes and their alignment. Consecutive gradients with $\rho_t>0$ reinforce their parameter changes, whereas $\rho_t<0$ indicates that the next step partially undoes the preceding one. We connect these interactions to directional curvature and introduce an inexpensive estimate using consecutive gradients and the actual parameter change. The step-based formulation applies independently of the optimizer, including Adam, without additional Hessian computations. Its SGD specialization depends only on gradient norms and their inner product, with an explicit contribution from mini-batch sampling.

'''+intro[end:]
start=intro.index('Our central contribution')
intro=intro[:start]+r'''Our contributions connect the interaction between training steps to accumulated parameter change, loss excursions, and functional oscillation. Each part addresses a distinct limitation of interpreting stability through the loss curve alone.
\begin{itemize}
\item \textbf{Characterizing constructive and destructive training regimes.}
Edge-of-Stability studies show that oscillation can coexist with progress~\cite{cohen2021edge,damian2023self}, while gradient-product diagnostics identify changes toward stationarity~\cite{chee2018diagnostics,pesme2020diagnostics}. We explain how successive updates reinforce or undo parameter changes, and measure how much traveled distance remains as net displacement. Our exact decomposition includes interactions across the complete trajectory. Experiments show that persistent local reversal can leave only a small fraction of this distance as net displacement while the loss continues to fall (Sections~\ref{sec:displacement}--\ref{sec:gd-experiments}).

\item \textbf{Estimating curvature cheaply along the applied step.}
The Hessian eigenvalues used in spectral stability analysis~\cite{cohen2021edge} require additional derivative computations, limiting the frequency and data coverage of measurement. We obtain a step-direction estimate from two consecutive gradients and the parameter increment already applied. For smooth full objectives, it equals directional curvature averaged along that step for any optimizer. Its SGD form connects curvature to gradient cosine and norm growth; independent Hessian--vector products check the approximation to starting-point curvature and the effect of sampling noise (Section~\ref{subsec:curvature-diagnostic} and Supplementary Section S2).

\item \textbf{Connecting reversal and growth to loss increases.}
Adaptive-preconditioner analysis explains a mechanism of Adam spikes~\cite{bai2026spikes}. We provide a complementary characterization through quantities directly available during training. For quadratic full-batch GD, the loss rises when $\rho_t<-\|g_{t-1}\|/\|g_t\|$: a reversed gradient must grow enough to overcome the preceding descent. For nonlinear objectives this is a local approximation. Under Adam, we test these signals before the next update against every observed next-step outcome, including a loss-history baseline, and report both accuracy and warning lift (Sections~\ref{subsec:loss} and~\ref{subsec:characterization}).

\item \textbf{Tracing functional oscillation through the network.}
Gradient and curvature diagnostics describe parameter-space dynamics but do not reveal how internal representations change. We follow the same reference inputs through successive states and separate the direction, size, and dimensionality of their changes. In the selected MLP window, reversal extends from the first linear map to the output; ResNet and Transformer show stronger agreement with output oscillation toward later stages, with less persistent reversal at the Transformer embedding. The output changes concentrate 95\% of their squared movement in 5/20/15 patterns out of at most 51/61/49 observable directions, respectively. Reversing coefficients reveal functional oscillation along these patterns (Section~\ref{subsec:tracing}).

\item \textbf{Testing selective control of representation displacement.}
Representation-change correction~\cite{kosson2024warmup} and unit-wise gradient clipping~\cite{brock2021normalization} motivate controls beyond a global learning rate. We cap the direct change induced by Adam's proposal in selected output units, using absolute movement budgets calibrated before intervention. Matched comparisons with learning-rate reduction, second-moment changes, and global clipping improve mean training loss and accumulated upward loss variation for all tested cap quantiles in the three cases, while functional reversals persist. Endpoint, held-out, and computational results delimit this benefit (Section~\ref{sec:vc-interventions}).
\end{itemize}

Section~\ref{sec:background} reviews stability perspectives. Section~\ref{sec:displacement} develops the regimes and their diagnostics; Section~\ref{sec:gd-experiments} examines GD and mini-batch sampling. Section~\ref{sec:adam} follows Adam loss spikes into functional oscillation, and Section~\ref{sec:vc-interventions} tests interventions. Section~\ref{sec:conclusion} concludes.
'''
write('01-introduction.tex',intro)

p=DRAFT/'manuscript.tex'
t=p.read_text(encoding='utf-8').replace('A Framework for Interpreting SGD Stability','From Gradient Interactions to Functional Oscillations')
p.write_text(t,encoding='utf-8',newline='\n')

replace('03-regimes.tex','from initialization,','from initialization at the final step $T$,')
replace('03-regimes.tex','Define the curvature along the preceding gradient and the normalized\nnoise contribution as','Define the curvature along the preceding gradient, $\\kappa_t$, and the normalized\nnoise contribution, $\\varepsilon_t$, as')
replace('03-regimes.tex',r'''Consequently, the noise contribution to consecutive alignment has
nonpositive expectation:''',r'''To evaluate the remaining term, substitute
$g_{t-1}=\nabla\mathcal L(\theta_{t-1})+\xi_{t-1}$ into
$g_{t-1}^{\top}\xi_{t-1}$. Its expectation is
$\mathbb E\|\xi_{t-1}\|^2$, because the full-gradient/noise cross term
vanishes. Subtracting it from the zero expectation of
$g_{t-1}^{\top}\xi_t$ gives''')
replace('03-regimes.tex',r'''Reducing $b$ weakens this shared directional signal, moving the expected
cosine toward zero. Along an evolving SGD trajectory, sampling also
changes the next parameter state and its curvature. The negative term
in Eq.~\eqref{eq:noise-expectation} alone therefore does not determine
how a negative cosine changes. Section~\ref{sec:sgd-noise-experiments}
tests the attenuation of constructive and destructive alignment on
actual training trajectories.''',r'''The numerator is the direction shared by both sampled gradients; the
additional variance in the denominator increases their lengths without
providing shared alignment. Smaller batches therefore dilute this common
direction and bring the expected cosine closer to zero. This fixed-state
calculation concerns two estimates of the same full gradient. Along an
evolving trajectory, sampling also changes the next state and the full
gradient there. It can weaken a coherent back-and-forth motion, but the
negative term in Eq.~\eqref{eq:noise-expectation} alone does not prove
attenuation of a negative cosine. Section~\ref{sec:sgd-noise-experiments}
tests that behavior during training.''')
start=read('03-regimes.tex').index('from initialization. Equation~')
end=read('03-regimes.tex').index('\\subsection{Conditions')
t=read('03-regimes.tex')
t=t[:start]+r'''from initialization. Thus $R_T$ measures how much of the accumulated step
length is retained in the network as net parameter displacement, while
$\rho_t$ describes the interaction between neighboring steps.

Aligned steps give $R_T=1$, equal opposing pairs give zero, and
equal-length orthogonal steps give $R_T=1/\sqrt T$. Even without exact
cancellation, the orthogonal example retains only 10\% after 100 updates
and 1\% after 10,000. In a destructive regime, successive updates actively
undo components of preceding changes. A small $R_T$ makes this limited
retention visible across many steps; it measures displacement retention,
not a fraction of learned information or useful computation.

'''+t[end:]
write('03-regimes.tex',t)
replace('03-regimes.tex','Directional interference','Interference')
replace('03-regimes.tex',r'''The first term favors descent. Constructive alignment adds a further
decrease, whereas destructive alignment offsets part of it.''',r'''The first term favors descent. The interference term depends on $\rho_t$:
positive alignment adds a further loss decrease, whereas negative alignment
opposes descent and can outweigh it.''')
t=read('03-regimes.tex'); start=t.index('Two conditions are necessary:'); end=t.index('The equivalent curvature form',start)
t=t[:start]+r'''For a fixed negative cosine, the required norm ratio is
$\|g_t\|/\|g_{t-1}\|>-1/\rho_t$. At $\rho_t=-1$, any increase in norm
is enough; at $\rho_t=-0.5$, the norm must more than double; at
$\rho_t=-0.2$, it must grow by more than a factor of five. Thus a weaker
reversal needs stronger amplification to produce an upward loss step.

'''+t[end:]
start=t.index('It summarizes the same joint effect:')
t=t[:start]+r'''Together, these relations distinguish three cases within the full-batch
quadratic model. When $\eta\kappa_t<1$, consecutive gradients are
constructive and the loss decreases. When $1<\eta\kappa_t<2$, they are
destructive but the return step still reduces the loss. When
$\eta\kappa_t>2$, reversal is sufficiently amplified to increase the loss.
The boundaries give zero alignment and zero loss change, respectively,
when the relevant gradients are nonzero. This closes the connection
between curvature, gradient norms, and interference: the angle/norm
condition expresses the same directional loss balance through inexpensive
measurements. It is not a replacement of directional curvature by the
largest Hessian eigenvalue. For nonlinear losses the quadratic relation
has a finite-step remainder, and sampling adds the explicit noise term.
We next examine how these regimes appear in training.
'''
write('03-regimes.tex',t)

t=read('04-gd-sgd.tex'); start=t.index('We test how'); end=t.index('\\begin{table*}',start)
t=t[:start]+r'''We test how constructive and destructive regimes emerge, how their
curvature and gradient signatures relate, and how repeated reversal
affects accumulated parameter change. Table~\ref{tab:setup} summarizes
the three model--dataset cases and learning rates. Within each model,
all GD rates share one initialization and the complete training objective
for 3,000 updates. Full-batch GD isolates these dynamics from sampling
noise; subsequent mini-batch continuations test the effect of sampling.

'''+t[end:]
start=t.index('The retained Hessian measurements'); end=t.index('\\begin{figure*}',start)
t=t[:start]+r'''The dashed curves measure sharpness, the largest Hessian eigenvalue
$\lambda_{\max}$. Computing it repeatedly requires additional
Hessian--vector products. To keep this measurement feasible, we use all
MLP data but fixed subsets for ResNet and Transformer, evaluated every
100 steps; the caption gives their coverage. This follows the practical
use of sampled sharpness measurements in EoS experiments~\cite{cohen2021edge}.
Our solid curves instead use consecutive gradients of the complete
training objective at every step, without diagnostic subsampling or
additional derivative evaluations. They measure a different quantity:
curvature along the realized GD direction, averaged over the step.
Agreement with starting-point directional Hessian measurements is checked
separately in Supplementary Section S2. Subset spectra and finite-step
directional curvature need not coincide.

Near the oscillatory regime, the smoothed $\eta\widehat\kappa_t$ stays
close to two. This identifies the directional loss-balance boundary but
hides how the oscillation changes in size. Figure~\ref{fig:gd-alignment-displacement}
provides the missing information by separating reversal from gradient
growth and accumulated displacement.

'''+t[end:]
anchor='The complete displacement sums confirm'
idx=t.index(anchor)
t=t[:idx]+r'''These trajectories initially build parameter changes through constructive
alignment, then enter a regime in which nearly opposite gradients undo
much of the preceding step. In a quadratic picture this is overshooting
and returning across a directional minimum. In a neural network, declining
loss together with growing gradient norms also permits descent along a
changing valley while oscillating across its steeper directions. Neither
a move between distinct basins nor overwriting a learned representation
can be inferred from these parameter-space measurements alone. The
functional analysis below asks directly whether representations also
reverse. This separates an interpretable geometric possibility from an
observed change in the learned function.

'''+t[idx:]
t=t.replace('The all-pair sums above quantify the resulting accumulation.',
'''This persistent oscillation across multiple steps, including renewed
alignment at longer lags, reveals temporal structure beyond immediate
neighbors. Because plain GD has no momentum state, this apparent memory
comes from the evolving objective gradients, rather than an explicit
optimizer memory. The complete displacement sums quantify its accumulated effect.''')
start=t.index('To test the effect of sampling,'); end=t.index('\\subsection{Checking',start)
t=t[:start]+r'''To examine the directional dilution in Eq.~\eqref{eq:noise-cosine-approx}
during training, we vary batch size at initialization and after 2,000 GD
updates. Learning rates are $0.02$, $0.1$, and $0.2$ for MLP, ResNet, and
Transformer. Each starting state receives 200 updates, with three
initializations, eight sampling repetitions per stochastic condition,
and one full-GD control per initialization. Every condition therefore
has the same number of optimizer steps, but a larger batch exposes the
model to more examples within those steps.

Figure~\ref{fig:sgd-moving-geometry} shows that the smallest batches move
the strongly negative trained-state alignment toward zero in all three
cases. Sampling weakens coherent reversal, consistent with attenuation
of the destructive regime. However, it also changes gradient norms and
the trajectory, so weaker opposition alone is not a stability guarantee.
At initialization, smaller MLP batches can induce negative mean alignment
despite a positive full-GD mean; in the Transformer, the same small batch
has similar negative cosine from both starting states but decreasing loss
from initialization and increasing loss from the trained state. Batch
size therefore influences both the coherence of updates and which region
of the objective they reach. The joint angle/norm condition explains why
these effects must be considered together.
'''
# The independent validation remains in the supplement, not a redundant subsection.
write('04-gd-sgd.tex',t)

write('05-adam.tex',r'''\section{Loss Spikes and Functional Oscillations under Adam}
\label{sec:adam}\label{sec:adam-diagnostics}

The preceding experiments show why consecutive steps must be considered
together. Reversal describes opposition; gradient growth determines its
size. Under SGD the raw gradients also determine the applied increments.
Adam combines the current gradient with past gradients and rescales
coordinates using second moments~\cite{kingma2014adam}. Its recurrences are
\begin{equation}
 m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
 v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^{\odot2}.
 \label{eq:adam-moments}
\end{equation}
The bias-corrected direction and parameter update are
\begin{equation}
 d_t=\frac{m_t/(1-\beta_1^{t+1})}
 {\sqrt{v_t/(1-\beta_2^{t+1})}+\epsilon},\qquad
 \theta_{t+1}=\theta_t-\eta_t d_t.
 \label{eq:adam-update}
\end{equation}
Moments start at zero before $t=0$, and the operations on second moments
are coordinatewise. The experiments use the native Adam update with
$(\beta_1,\beta_2)=(.9,.999)$ and $\epsilon=10^{-8}$, without weight decay.

Although Adam transforms the raw gradient, consecutive gradients are
evaluated at states separated by the actual Adam update. Their alignment
and relative magnitudes therefore characterize the gradient response
along the trajectory followed by the optimizer. For full gradients of a
smooth objective, the fundamental theorem of calculus gives
\begin{equation}
 g_t-g_{t-1}=-\eta_{t-1}\int_0^1
 H(\theta_{t-1}-a\eta_{t-1}d_{t-1})d_{t-1}\,da.
 \label{eq:adam-gradient-response}
\end{equation}
Projecting this difference onto $g_{t-1}$ recovers the same measured
combination as Eq.~\eqref{eq:kappa-estimate}. For the constant-rate Adam
trajectories considered here, we retain its notation,
\begin{equation}
 \begin{aligned}
 \eta\widehat\kappa_t
 &=1-\rho_t\frac{\|g_t\|}{\|g_{t-1}\|}\\
 &=\eta\int_0^1
 \frac{g_{t-1}^{\top}H(\theta_{t-1}-a\eta d_{t-1})d_{t-1}}
 {\|g_{t-1}\|^2}\,da.
 \end{aligned}
 \label{eq:adam-projected-response}
\end{equation}
This is a gradient-projected response to the applied direction. Under
GD, $d_{t-1}=g_{t-1}$ and it is the scaled step-direction curvature.
Under Adam those directions generally differ: curvature along the
actual increment is instead $\widehat\kappa_t^{\mathrm{step}}$ in
Eq.~\eqref{eq:kappa-step}. Thus the plotted quantity still combines
reversal and amplification, but the GD loss threshold is not an Adam
stability theorem. A value above two means that the component of the new
gradient opposing its predecessor exceeds the predecessor's magnitude.

We extend the same three full-objective learning problems to Adam, with
one matched initialization per model and 1,024 updates. The tested rates
are $.0001/.0003/.001$ for MLP and $.0003/.001/.003$ for ResNet and
Transformer. Figure~\ref{fig:adam-overview} displays the highest rate in
each grid, selected to expose pronounced excursions; the complete grid
is retained in Supplementary Section S3. We first ask how gradient
reversal and growth precede loss increases, then trace the corresponding
functional changes. The following section tests interventions from
matched training states.

\input{sections/05a-characterization}
\input{sections/05b-tracing}
\input{sections/05c-interventions}
''')

t=read('05a-characterization.tex').replace('$Q_t$',r'$\eta\widehat\kappa_t$').replace('$Q_t\\geq2$',r'$\eta\widehat\kappa_t\geq2$')
t=t.replace('the reversal-and-growth quantity', 'the gradient-projected response')
start=t.index('The panels show why'); end=t.index('A loss curve and a whole-network gradient',start)
t=t[:start]+r'''Unlike the sustained GD oscillations, the Adam trajectories exhibit
distinct excursions and recoveries. This is consistent with the adaptive
preconditioner mechanism analyzed by Bai et al.~\cite{bai2026spikes}:
changing second moments can alter the effective step size independently
of changes in the underlying Hessian. Our measurements describe how the
resulting motion appears in consecutive gradients. Each high-rate run
begins with constructive alignment and later develops reversal.
Prominent excursions occur with negative cosine and growing gradient
norms, producing large peaks in $\eta\widehat\kappa_t$. In ResNet and
Transformer, recovery often includes renewed constructive alignment and
falling norms. MLP reversals persist through both increases and recovery,
so their magnitude is essential to interpreting the loss.

The loss can reach lower values after an excursion. This shows why
removing an excursion and preserving subsequent progress are separate
objectives; it does not by itself establish a transition between basins,
a flatter destination, or premature convergence at a lower rate.
Similarly, peaks in gradient norm and projected response accompany
large excursions, but the panels do not establish a universal proportional
relation. The matched interventions below test the effect of limiting
movement rather than inferring it from the loss curve.

For a direct before-update check, let $L_t=\mathcal L(\theta_t)$ and flag
whether $(L_{t+1}-L_t)/L_t\geq0.001$. At decision step $t$, $g_t$ and
$L_t$ are available, but the next update has not been applied.
Table~\ref{tab:severity} compares the gradient conditions with the naive
rule that the preceding step already increased the loss, $L_t>L_{t-1}$.
Accuracy counts correct predictions of both increases and non-increases.
The warning hit rate answers a different question: how often an increase
follows a warning. Its lift divides this rate by the unconditional event
frequency. Reporting both prevents the many non-increasing steps from
making a weak warning rule appear effective.

\input{tables/diagnostic-0p1pct}

The combined angle/norm condition gives higher accuracy and warning
lift than the previous-loss-rise rule in all three cases. Its warning
lifts are $1.28\times$, $6.78\times$, and $3.87\times$ for MLP, ResNet,
and Transformer. Reversal or norm growth alone is less selective. The
angle/norm inequality and $\eta\widehat\kappa_t\geq2$ are equivalent
on these data, not independent predictors. These inexpensive signals
therefore contain information about the next loss increase before the
update is applied, although accuracy improves only modestly over always
predicting no increase. The threshold and trajectories were inspected
retrospectively; prospective forecasting needs independent validation.
Supplementary Section S3 retains the confusion counts and comparisons
across larger increases, future steps, and warning histories.

'''+t[end:]
write('05a-characterization.tex',t)

# Keep notation consistent in all explanatory supplement prose.
for p in S.glob('s*.tex'):
    t=p.read_text(encoding='utf-8').replace('Q_t',r'\eta\widehat\kappa_t').replace('$Q\\geq2$',r'$\eta\widehat\kappa\geq2$')
    p.write_text(t,encoding='utf-8',newline='\n')

# No colon immediately introducing a displayed equation in either document.
for p in S.glob('*.tex'):
    t=p.read_text(encoding='utf-8')
    t=re.sub(r':\s*\n(\\begin\{(?:equation|align)\*?\})',r'\n\1',t)
    p.write_text(t,encoding='utf-8',newline='\n')
