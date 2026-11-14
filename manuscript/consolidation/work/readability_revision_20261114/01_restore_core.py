"""Restore the author's baseline exposition, with explicit, limited scientific edits."""
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
ROOT=BASE.parents[1]
OLD=ROOT/'manuscript/draft/sections'
NEW=BASE/'draft/sections'

def write(name,text):
    (NEW/name).write_text(text.rstrip()+'\n',encoding='utf-8',newline='\n')

related=(OLD/'02-related-works.tex').read_text(encoding='utf-8')
related=related.replace('Let $\\theta_t\\in\\mathbb{R}^p$ denote the parameters after update $t$ and\n$g_t$ a mini-batch gradient evaluated at $\\theta_{t-1}$.',
    'Let $\\theta_t\\in\\mathbb{R}^p$ denote the parameters after $t$ completed updates and\n$g_t$ a mini-batch gradient evaluated at $\\theta_t$. Thus $g_t$ is available before\nthe next update, from $t$ to $t+1$.')
related=related.replace(r'\theta_t=\theta_{t-1}-\eta g_t.',r'\theta_{t+1}=\theta_t-\eta g_t.')
related=related.replace(r'\theta_{t-1}',r'\theta_t')
related=related.replace('An online diagnostic\nthat recovers curvature information from those existing gradients would\nmake this geometric analysis accessible during routine training.',
    'An online diagnostic\nthat recovers curvature information from those existing gradients makes\nthis geometric analysis accessible during routine training. Consecutive-gradient\nproducts have also been used in stationarity diagnostics and learning-rate\nadaptation~\\cite{chee2018diagnostics,pesme2020diagnostics,baydin2018hypergradient}.\nHere we use their direction and magnitude to explain the interaction between\nsteps and connect it to the changes observed inside the network.')
related=related.replace('Gradient norms are thus already useful monitoring signals.',
    'For Adam, Bai et al.~\\cite{bai2026spikes} connect loss spikes to the ability\nof the second-moment estimate to track changing curvature. Their analysis\nconcerns the geometry after adaptive scaling, which differs from the\nuntransformed gradient. This motivates examining both the gradient\nresponse and the parameter change that the optimizer actually applies.\n\nGradient norms are thus already useful monitoring signals.')
write('02-related-work.tex',related)

core=(OLD/'03-methods.tex').read_text(encoding='utf-8')
core=core.replace(r'\label{sec:displacement}',r'\label{sec:displacement}\label{sec:regimes}')
core=core.replace(r'\sum_{t=1}^T',r'\sum_{t=0}^{T-1}').replace(r'1\leq u<t\leq T',r'0\leq u<t<T')
core=core.replace(r'$g_t=\nabla\mathcal L(\theta_{t-1})+\xi_t$',r'$g_t=\nabla\mathcal L(\theta_t)+\xi_t$')
core=core.replace(r'\theta_{t-2}',r'\theta_{t-1}')
core=core.replace('where $r_t$ is the Taylor remainder.','where $\\mathcal R_t$ is the Taylor remainder.').replace('+r_t,',r'+\mathcal R_t,')
core=core.replace('We now connect these regimes',r'\label{subsec:curvature}'+'\n\nWe now connect these regimes')
core=core.replace('For nonzero consecutive gradients, write $\\rho_t:=\\rho_{t-1,t}$.',
    'For nonzero consecutive gradients, write $\\rho_t:=\\rho_{t-1,t}$.\nA zero gradient leaves its cosine undefined; it is not assigned zero alignment.')
core=core.replace('This relation supports two interpretations. From a representation-learning\nperspective, a destructive update opposes parameter changes made by its\npredecessor, providing a mechanism for repeatedly overwriting learned\nstructure. Even with iid data from a fixed distribution, successive\nupdates can repeatedly undo previously accumulated parameter changes.',
    'This relation supports two interpretations. From a representation-learning\nperspective, a destructive update opposes parameter changes made by its\npredecessor. Even with iid data from a fixed distribution, successive\nupdates can repeatedly undo components of accumulated parameter change.\nWhether the corresponding representations also reverse must be measured;\nSection~\\ref{subsec:tracing} follows these changes inside the network.')
start=core.index('The measured projected change has a precise geometric meaning.')
end=core.index(r'\subsection{Noise Contribution',start)
core=core[:start]+r'''The estimator has a direct geometric meaning. Under smooth full-batch GD,
it equals the directional Hessian curvature averaged over the preceding
step. If the curvature changes little during that step, it also estimates
the curvature at its starting point. This is the curvature encountered
along the direction actually followed; it need not equal the largest
Hessian eigenvalue, which describes the sharpest available direction.

The same construction extends to any optimizer by using its actual
parameter change. Let $u_{t-1}:=\theta_t-\theta_{t-1}\ne0$. We introduce
the step-direction estimate
\begin{equation}
 \boxed{\widehat\kappa_t^{\mathrm{step}}
   :=\frac{u_{t-1}^{\top}(g_t-g_{t-1})}{\|u_{t-1}\|^2}.}
 \label{eq:kappa-step}
\end{equation}
The numerator measures how much the gradient changes along the applied
step; division by squared step length gives curvature per unit displacement.
For full gradients and a smooth objective, this is exactly
\begin{equation}
 \widehat\kappa_t^{\mathrm{step}}=
 \int_0^1\frac{u_{t-1}^{\top}H(\theta_{t-1}+a u_{t-1})u_{t-1}}
 {\|u_{t-1}\|^2}\,da.
 \label{eq:step-path-average}
\end{equation}
No assumption about how the optimizer chose $u_{t-1}$ is needed. Thus the
estimate applies to momentum, Adam, and other parameter updates. It requires
the two consecutive gradients and the applied increment, but no additional
backward pass or Hessian--vector product. The required inner products and
storage scale linearly with the number of parameters.

For plain SGD, substituting $u_{t-1}=-\eta g_{t-1}$ recovers
Eq.~\eqref{eq:kappa-estimate}. For Adam, the actual increment must be used:
the raw gradient generally points in a different direction. With sampled
gradients, Eq.~\eqref{eq:kappa-step} also contains
$u_{t-1}^{\top}(\xi_t-\xi_{t-1})/\|u_{t-1}\|^2$.
Supplementary Section S1 derives the identity and its approximation bound.
A zero step leaves this estimate undefined. The estimator measures the
completed step; it does not by itself predict the next loss change.

''' + core[end:]
core=core.replace(r'\input{sections/04-controlled-figures}','')
core=core.replace(r'\label{subsec:loss-connection}',r'\label{subsec:loss-connection}\label{subsec:loss}')
core=core.replace(r'$\Delta\mathcal L_{t-1}:=\mathcal L(\theta_{t-1})-\n\mathcal L(\theta_{t-1})$',
    r'$\Delta\mathcal L_t:=\mathcal L(\theta_t)-\mathcal L(\theta_{t-1})$')
# The baseline's displayed definition contains a real newline.
core=core.replace('$\\Delta\\mathcal L_{t-1}:=\\mathcal L(\\theta_{t-1})-\n\\mathcal L(\\theta_{t-1})$',
    '$\\Delta\\mathcal L_t:=\\mathcal L(\\theta_t)-\\mathcal L(\\theta_{t-1})$')
core=core.replace(r'\Delta\mathcal L_{t-1}',r'\Delta\mathcal L_t')
core=core.replace(r'Appendix~\ref{appendix:loss-curvature}',r'Supplementary Section S1')
core=core.replace(r'\label{subsec:curvature-diagnostic}',r'\label{subsec:curvature-diagnostic}')
write('03-regimes.tex',core)

# Keep the baseline's empirical explanation and the improved setup/caption assets.
current=(NEW/'04-gd-sgd.tex').read_text(encoding='utf-8')
setup=re.search(r'\\begin\{table\*\}.*?\\end\{table\*\}',current,re.S)[0]
figures=re.findall(r'\\begin\{figure\*\}.*?\\end\{figure\*\}',current,re.S)
labels={'fig:gd-motion':'fig:gd-alignment-displacement','fig:gd-lags':'fig:gd-backward','fig:sgd':'fig:sgd-moving-geometry'}
for old,new in labels.items():
    figures=[f.replace(old,new) for f in figures]
body=(OLD/'04-gd-experiments.tex').read_text(encoding='utf-8')
body=body.replace(r'\label{sec:gd-experiments}',r'\label{sec:gd-experiments}\label{sec:gd}')
body=body.replace('Appendix~\\ref{appendix:gd-raw-alignment} specifies the objectives and\nmeasurement conventions.',
    'Table~\\ref{tab:setup} summarizes the objectives and measurement coverage;\nSupplementary Section S2 gives the numerical and sampling conventions.\nTraining uses FP32, diagnostic reductions use FP64, and data augmentation,\ndropout, and TF32 are disabled.\n\n'+setup+'\n\n'+figures[0])
body=body.replace('\\subsection{Destructive Regimes and Displacement Retention}',
    figures[1]+'\n\n\\subsection{Destructive Regimes and Displacement Retention}')
body=body.replace('Destructive regimes therefore sustain convergence\ndespite extensive local reversal.',
    'These destructive regimes therefore coexist with continued loss reduction\ndespite extensive local reversal.')
body=body.replace('\\subsection{Oscillation across Temporal Lags}',
    'The complete displacement sums confirm that local reversal need not\ndominate accumulated motion. Across all nine trajectories, the sum of\nall cross-step interactions is positive, including the seven whose\nadjacent-pair sum is negative. Interactions at longer lags more than offset\nthe adjacent cancellations (Supplementary Table S1).\n\n'+figures[2]+'\n\n\\subsection{Oscillation across Temporal Lags}')
body=body.replace('These recurrences\nexplain how local destructive interference coexists with reinforcement\nover longer lags and why net progress survives repeated reversals.',
    'These recurrences\nillustrate how local destructive interference coexists with reinforcement\nover longer lags. The all-pair sums above quantify the resulting accumulation.')
sampling=(OLD/'05-sgd-noise-experiments.tex').read_text(encoding='utf-8')
sampling=sampling.replace('Appendix~\\ref{appendix:gd-raw-alignment}',r'Supplementary Section S2')
validation=r'''
\subsection{Checking the Curvature Estimate}
\label{subsec:gd-validation}

The smooth tanh MLP allows a direct check of the estimated curvature.
We independently compute Hessian--vector products along the same sampled
gradient direction and compare them with Eq.~\eqref{eq:kappa-estimate}.
The comparison shows why the sampling term matters: correcting for the
measured noise brings the two-gradient estimate much closer to the
independent curvature calculation across all stochastic batch sizes.
Supplementary Table S2 and Figs. S1--S2 give the paired errors and the
separate frozen-parameter sampling check.

Curvature agreement and accurate loss prediction are different questions.
On a nonlinear objective, curvature can change during a large step, and
the local quadratic loss relation can then give the wrong sign. We retain
the complete sign comparison in Supplementary Table S1. Accordingly, the
figures are interpreted through direction, magnitude, and observed loss
together; the reference value two is not used as an unconditional
prediction of the next loss increase.
'''
write('04-gd-sgd.tex',body+'\n'+figures[3]+'\n'+sampling+'\n'+validation)
print('Restored Sections II–IV from the active draft with targeted additions.')
