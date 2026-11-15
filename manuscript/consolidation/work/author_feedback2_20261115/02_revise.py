"""Apply round-two editorial changes to the previously reviewed source."""
from pathlib import Path
import re
HERE=Path(__file__).resolve().parent
D=HERE.parents[1]/'draft'
def read(name): return (D/'sections'/name).read_text(encoding='utf8')
def write(name,s): (D/'sections'/name).write_text(s,encoding='utf8',newline='\n')
def cut(s,start,end,replacement=''):
    i=s.index(start);j=s.index(end,i);return s[:i]+replacement+s[j:]

s=read('05-adam.tex')
s=s.replace(r' m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad',r' \begin{aligned}'+'\n'+r' m_t&=\beta_1m_{t-1}+(1-\beta_1)g_t,\\')
s=s.replace(r' v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^{\odot2}.',r' v_t&=\beta_2v_{t-1}+(1-\beta_2)g_t^{\odot2}.'+'\n'+r' \end{aligned}')
s=cut(s,'This is a gradient-projected response','\\input{sections/05a',r'''We first characterize the gradient response along Adam's trajectory,
then trace the corresponding functional changes through the network.
The following section tests interventions from matched training states.
The complete experimental grid is specified in Supplementary Section S3.

''')
write('05-adam.tex',s)
s=read('05a-characterization.tex')
s=cut(s,'Figure~\\ref{fig:adam-overview} brings','\\begin{figure*}')
s=s.replace('Early constructive\nintervals give way to persistent reversal; the MLP also has an isolated\nnegative cosine at its first measured pair, before its constructive interval.','Early constructive\nintervals give way to persistent reversal.')
s=s.replace(' MLP reversals persist through both increases and recovery,\nso their magnitude is essential to interpreting the loss.','')
s=cut(s,'The loss can reach lower values','\\input{tables/diagnostic-0p1pct}',r'''The loss can reach lower values after an excursion. Such an excursion
could accompany movement into another basin, while momentum can delay
the update's response to a newly changing gradient direction. Whether
that movement helps depends on the current landscape, architecture,
and optimizer state. Related work on SGD catapults shows that spikes can
accompany improved feature learning~\cite{zhu2024catapults}; their role
under Adam must likewise be evaluated together with subsequent progress.

After computing the raw gradient, but before applying the next update,
we may want to classify whether that update will increase the loss.
Let $L_t=\mathcal L(\theta_t)$ and define an increase as
$(L_{t+1}-L_t)/L_t\geq0.001$. At decision step $t$, both $g_t$ and $L_t$
are available. However, the pair $(g_{t-1},g_t)$ describes the response
to the preceding update, from $\theta_{t-1}$ to $\theta_t$. Using it to
classify the outgoing update tests whether that response carries forward.
For quadratic full-batch plain GD, the strict condition
$\eta\widehat\kappa_t>2$ exactly characterizes a positive loss change
on the preceding step. This exact retrospective equivalence does not
extend to a finite nonlinear step or to Adam, whose applied direction
also depends on its moments.

Table~\ref{tab:severity} compares the gradient conditions with the
simple history rule $L_t>L_{t-1}$. Accuracy is the fraction of all
decisions classified correctly. Warning hit rate is the fraction of
warnings followed by an increase; lift divides this rate by the overall
frequency of increases. Together they distinguish correct classification
from how informative an issued warning is.

''')
s=cut(s,'The combined angle/norm condition','A loss curve',r'''The \emph{destructive loss-spike condition} combines reversal with
sufficient gradient growth, $\rho_t<0$ and
$\|g_t\|/\|g_{t-1}\|>-1/\rho_t$. It gives higher accuracy and warning
lift than the previous-loss-rise rule in all three cases. Reversal or
norm growth alone is less selective. The angle/norm condition is
algebraically equivalent to $\eta\widehat\kappa_t>2$; their identical
table entries verify the two implementations, rather than constitute
independent evidence. The non-strict threshold gives the same flags
here because no observed value equals two. The warning lifts are
$1.28\times$, $6.78\times$, and $3.87\times$ for MLP, ResNet, and
Transformer. These signals contain information before the next update,
although the gains in overall accuracy over always predicting no
increase are modest.

''')
write('05a-characterization.tex',s)
s=read('s03-adam-audit.tex')
s=s[:s.index('Decisions at')]
s=s.replace('Adam Timing, Severity, and Warning Memory','Adam Experimental Setup and Event Selection')
s+='The main comparison uses the highest rate in each grid, selected to expose pronounced excursions. Each next-step classification uses the 1,023 decisions with consecutive gradients available; the target is an individual loss increase of at least 0.1\\%, rather than a recovered excursion.\n'
write('s03-adam-audit.tex',s)

s=read('05b-tracing.tex')
algorithm=s[s.index('\\begin{algorithm}'):s.index('\\end{algorithm}')+len('\\end{algorithm}')]
fig1=s[s.index('\\begin{figure*}'):s.index('\\end{figure*}')+len('\\end{figure*}')]
start2=s.index('\\begin{figure*}',s.index('\\end{figure*}')+1)
end2=s.index('\\end{figure*}',start2)+len('\\end{figure*}')
fig2=s[start2:end2]
definitions=s[s.index('A measured'):s.index('\\begin{figure*}')]
definitions=definitions.replace('Rows identify reference examples and, where relevant, spatial or token\npositions;', 'The $N_\\ell$ rows identify reference examples and, where relevant, spatial or token\npositions;')
definitions=definitions.replace('This gives a\nrepresentation-level counterpart of destructive and constructive\ngradient regimes.',r'''As $\rho_t$ measures alignment between consecutive raw gradients,
$c_t^\ell$ measures alignment between consecutive functional changes at
site $\ell$, giving a representation counterpart of constructive and
destructive regimes.''')
svd=s[s.index('We next ask how many'):s.index('\\begin{figure*}',s.index('We next ask how many'))]
svd=svd.replace('We next ask how many independent patterns are needed to describe this\noscillating functional change.', 'To describe patterns shared across updates,')
svd=cut(svd,'We call each fixed pattern','The total squared movement',r'''A \emph{mode} is the right singular vector
$v_{\ell,j}\in\mathbb R^{P_\ell}$: the $j$th fixed pattern at site $\ell$
over the whole window. Its coordinates span all reference input positions
and features; its indices identify the site and pattern, not a sample
or training step. Equivalently, it is an eigenvector of
$D_\ell^\top D_\ell$ with eigenvalue $\sigma_{\ell,j}^2$.
The rectangular matrix $D_\ell$ itself need not have eigenvectors.
At most $\min(T_w,P_\ell)$ independent patterns can be observed.

''')
overlap=s[s.index('To compare timing across layers'):s.index('Finally, the lower row')]
overlap=overlap[:overlap.index('The middle row')]+r'''This measures shared timing, without backpropagating outputs or
equating feature directions at different sites.

'''
overlap=overlap.replace('With $F$ denoting the output','With $F$ denoting the final network output')
overlap=overlap.replace('This averages squared projections of one set of orthonormal temporal\npatterns onto the other.',"It measures how much of a site's leading time histories lies in the\nsubspace spanned by the final output's leading time histories.")
intro=r'''\subsection{Tracing Oscillations through the Learned Function}
\label{subsec:tracing}

A parameter update changes the representations computed inside the
network and, eventually, its output. We trace these changes backwards
from the output toward the input-side layers, asking where they reverse,
how large they become, and whether different parts of the network share
their timing. We compare the same inputs at successive training states;
otherwise, a representation difference could arise simply because the
examples changed. The reference sets are all 5,000 MLP training images,
one original 1,000-image ResNet normalization group, and 64 Transformer
sequences with 2,240 unchanged token positions (Table~\ref{tab:setup}).

A training state $\theta_t$ is the parameter state after $t$ completed
updates. For a window $a{:}b$, we analyze the $T_w=b-a+1$ representation
changes ending at steps $a,\ldots,b$. Computing their consecutive
cosines requires states $a-2,\ldots,b$: the first cosine compares the
changes ending at $a-1$ and $a$.

'''
retention=r'''The representation counterpart of normalized displacement is
\begin{equation}
 R_{a:b}^\ell=
 \frac{\|\sum_{t=a}^b\delta_t^\ell\|}{\sum_{t=a}^b\|\delta_t^\ell\|}
 =\frac{\|A_b^\ell-A_{a-1}^\ell\|_F}
 {\sum_{t=a}^b\|A_t^\ell-A_{t-1}^\ell\|_F}.
 \label{eq:representation-retention}
\end{equation}
This ratio lies in $[0,1]$ whenever movement occurs: it compares net
displacement with total distance traveled. In the windows below, output
retention is only 1.71\%/2.44\%/3.51\% for MLP/ResNet/Transformer;
the first measured sites retain 3.25\%/3.72\%/24.33\%, respectively.
Together with negative cosines, these values show substantial
back-and-forth movement with little remaining net displacement,
motivating control of its magnitude. Retention measures displacement,
rather than learning quality; orthogonal changes can also produce a
small ratio. All values use the full changes, without mode truncation.

'''
discussion1=r'''Figure~\ref{fig:trace-motion} shows where functional oscillation appears
and how its size changes. The top row locates the loss excursion.
Reading the cosine maps backwards from output to input reveals how
widely reversal is shared. In the MLP, every displayed change has
negative cosine at every measured site, from the first linear input
map to the final output. Oscillation is therefore homogeneous across
the measured depth, even while loss rises and falls. The RMS maps add
its changing amplitude. Comparing pre- and post-nonlinearity measurements
shows that tanh attenuates displacement; each site's separate
normalization in the figure highlights timing rather than absolute
differences in magnitude across sites.

ResNet and Transformer exhibit a less homogeneous pattern. Their
output-side representations reverse while earlier representations can
continue to reinforce their preceding change. The first displayed output
reversal precedes the 10\% loss marker by 15 steps in ResNet and five
in Transformer. In ResNet, the first convolution reaches its largest
displacement at the loss peak, followed one step later by the remaining
displayed sites. In Transformer, the embedding becomes more oscillatory
near the large excursion and returns to constructive movement sooner
than the later sites. Larger backpropagated errors could recruit earlier
representations into stronger changes, while constructive intervals
could accumulate smaller changes before that reorganization. This
architecture-dependent interpretation motivates examining both the
direction and magnitude throughout the network.

'''
discussion2=r'''Figure~\ref{fig:trace-structure} asks whether these changes repeatedly
use a small set of patterns. The top row shows concentration of squared
movement in the leading output modes, alongside the corresponding
gradient decomposition. Concentration can reflect both repeated
directions and a few large changes; the equal-length and orthogonal
controls in Supplementary Section S4 separate these effects.
The middle row extends the analysis through the network: $E_1^\ell$
and $E_{10}^\ell$ show concentration at each site, whereas $O_{10}$ shows
how much its leading temporal structure is shared with the final output.
The selected ResNet window has stronger temporal agreement toward the
output, with shared structure still extending to the input-side sites.
Finally, the bottom row establishes the back-and-forth component that
energy fractions alone cannot establish. Repeated changes in the sign
of $\alpha_{t,1}^F/\sqrt{P_F}$ show reversal along one fixed output
pattern, and its amplitude shows how strongly that pattern moves.

Together, the two figures and the small retention values characterize
coordinated, approximately low-dimensional functional oscillation in
these windows. The temporal Gram matrix $D_\ell D_\ell^\top$ permits
the analysis without storing all spatial modes; implementation and
numerical checks are in Supplementary Section S4. The resulting
intervention question is whether limiting the magnitude of local
representation changes can reduce loss excursions while preserving
subsequent training progress.
'''
algorithm=algorithm.replace(r'\State Stack changes $a{:}b$ as rows of $D_\ell$.',r'\State Stack changes $a{:}b$ as rows of $D_\ell$ and compute $R_{a:b}^\ell$.')
algorithm=algorithm.replace("\\State Compare each stage's timing and direction with the output.",r'\State Compute temporal overlap $O_k(\ell,F)$ with the final network output.')
s=intro+definitions+retention+svd+overlap+'Algorithm~\\ref{alg:trace} assembles these measurements.\n\n'+algorithm+'\n\n'+fig1+'\n\n'+discussion1+fig2+'\n\n'+discussion2
write('05b-tracing.tex',s)

# Terminology in prose; paths and asset identifiers are deliberately preserved.
for p in (D/'sections').glob('*.tex'):
    s=p.read_text(encoding='utf8')
    s='\n'.join(line if '\\includegraphics' in line else re.sub(r'\bstages?\b',lambda m:'sites' if m[0]=='stages' else 'site',line) for line in s.splitlines())+'\n'
    s=s.replace('Stages are','Sites are')
    p.write_text(s,encoding='utf8',newline='\n')
