"""Apply approved appendix edits, recording each decision before mutation."""
from pathlib import Path
import json
import re

HERE = Path(__file__).resolve().parent
DRAFT = HERE.parents[1] / 'draft'
AUDIT = HERE / 'PARAGRAPH_REVIEW.md'
DECISIONS = HERE / 'decisions.json'
assert not AUDIT.exists(), 'One-time edit script; use a new recorded decision for refinements.'
AUDIT.write_text('# Appendix restructuring: sequential decisions\n\n'
                 '15 November 2026. Original, diagnosis, proposed solution, and replacement '
                 'are recorded before each mutation. Structural transfers are identified explicitly.\n', encoding='utf-8')
decisions = []


def edit(file, name, before, after, diagnosis, solution):
    path = DRAFT / file
    current = path.read_text(encoding='utf-8') if path.exists() else ''
    assert (current.count(before) == 1) if before else not current, name
    d = dict(file=file, id=name, before=before, after=after,
             diagnosis=diagnosis, solution=solution)
    decisions.append(d)
    DECISIONS.write_text(json.dumps(decisions, indent=2) + '\n', encoding='utf-8')
    with AUDIT.open('a', encoding='utf-8', newline='\n') as f:
        f.write(f'\n## {name}\n\nSource: `{file}`\n\n**Read**\n\n```tex\n{before}\n```\n\n'
                f'**Diagnosis:** {diagnosis}\n\n**Proposed solution:** {solution}\n\n'
                f'**Rewrite:**\n\n```tex\n{after}\n```\n')
        f.flush()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(current.replace(before, after, 1) if before else after, encoding='utf-8', newline='\n')
    print(name)


F = 'sections/appendix.tex'
original = (DRAFT / F).read_text(encoding='utf-8')


def between(start, end):
    s = (DRAFT / F).read_text(encoding='utf-8')
    return s[s.index(start):s.index(end, s.index(start))]


edit(F, 'A title', r'\section{Mathematical Scope of the Diagnostics}',
     r'\section{Exact Identities and Approximation Limits}',
     'The title does not tell readers which mathematical distinctions the appendix establishes.',
     'Name exactness and approximation limits explicitly.')
edit(F, 'A1 exact curvature conditions', between('The distinction between', r'\begin{equation}'),
     r'''Full gradients of a fixed objective recover directional curvature
averaged along an update. Let $f_t=\nabla\mathcal L(\theta_t)$ and
$u_{t-1}=\theta_t-\theta_{t-1}\ne0$ be the actual applied increment.
For a twice continuously differentiable objective along this segment,
''', 'The opening emphasizes interpretation without first stating the full-gradient and fixed-objective conditions.',
     'State the conditions and the recovered quantity before the integral.')
edit(F, 'A1 starting-point comparison', between('Taking its inner product', r'\begin{equation}'),
     r'''Taking the inner product with $u_{t-1}/\|u_{t-1}\|^2$ gives
Eq.~\eqref{eq:kappa-step} with $g_t=f_t$, regardless of how the optimizer
selected the update. This step average need not equal curvature at the
starting state. If the Hessian is $L_H$-Lipschitz in operator norm along
the segment, their difference satisfies
''', 'The estimator connection silently assumes full gradients and the referent of its difference is unclear.',
     'Make the full-gradient substitution explicit and introduce the bound as a comparison of two quantities.')
edit(F, 'A1 bound and applicability', between('Indeed, the integral', 'The loss-rise condition'),
     r'''The bound follows by integrating $L_Ha\|u_{t-1}\|$ over $a\in[0,1]$.
For Adam, projection onto the applied increment must be distinguished
from the raw-gradient projection in Eq.~\eqref{eq:adam-projected-response}.
Across nonsmooth activation boundaries, the smooth derivation need not
apply even though the diagnostic remains computable. A zero increment
leaves step-direction curvature undefined.

''', 'One paragraph interleaves a bound, sampling correction, optimizer distinction, and applicability limits.',
     'Keep the short bound justification and applicability limits here; move the unchanged sampling correction beside the sampling assumptions.')
edit(F, 'A2 loss remainder purpose', between('The loss-rise condition', r'\begin{equation}'),
     r'''The quadratic loss-sign criterion can fail on a finite nonlinear step.
To quantify the discrepancy, set $x=\theta_{t-1}$ and $u=-\eta g_{t-1}$.
Integration by parts gives the remainder
''', 'The remainder is introduced as another formula rather than a limit on an existing interpretation.',
     'State the question it answers before defining the unchanged remainder.')
edit(F, 'A2 exact quadratic and nonlinear cases', between('Substituting $f_t', 'Sampling identities'),
     r'''Substituting $f_t=g_t-\xi_t$ gives the loss identity in the main text.
For a constant Hessian the remainder vanishes, and full-gradient plain
GD obeys $\Delta L_t=\eta\|g_{t-1}\|^2(\eta\widehat\kappa_t-2)/2$
exactly. Under the Lipschitz assumption, subtracting the midpoint Hessian
inside the integral gives $|\mathcal E_t|\leq L_H\|u\|^3/12$.
The remainder can therefore change the predicted loss sign on a nonlinear
step, particularly when the quadratic prediction is near zero.

For sampled gradients $g_t=f_t+\xi_t$, the curvature measurement contains
the additional term $u_{t-1}^{\top}(\xi_t-\xi_{t-1})/\|u_{t-1}\|^2$.
Under plain SGD, $u_{t-1}=-\eta g_{t-1}$, this term is
$-\varepsilon_t/\eta$.

''', 'The mathematical consequence can be stated more directly, and the sampling correction needs to precede its assumptions.',
     'Preserve the formulas, explain the sign limitation, and relocate the original noise term here.')
edit(F, 'A3 trajectory versus fixed-state sampling', between('Sampling identities', r'\section{Reproduction'),
     r'''Along the SGD trajectory, conditional unbiasedness does not require
the new state to be independent of the preceding noise. Fresh sampling
gives $\E[g_{t-1}^{\top}\xi_t]=0$, whereas
$\E[g_{t-1}^{\top}\xi_{t-1}]=\E\|\xi_{t-1}\|^2$.

Separately, at a fixed state $\theta$, uniform subsets of size $b$ from
$N$ examples, sampled without replacement, have
$\operatorname{Cov}(\xi\mid\theta)=(N-b)\Sigma/[b(N-1)]$
when $\Sigma$ uses denominator $N$. For two independent draws at that
state, the expected-cosine formula additionally approximates their random
product of norms by a moment-based value. It is not an exact identity
obtained by taking a ratio of expectations.

''', 'The paragraph moves from dependent trajectory states to independent fixed-state draws without marking the change.',
     'Separate the two experiments and retain the covariance and denominator-approximation qualifications.')
edit(F, 'B structure', '\\section{Reproduction Details and Independent Checks}\n\\label{app:validation}\n\n\\subsection{Objectives, sampling, and measurement timing}',
     '\\section{Experimental Definitions and Diagnostic Checks}\n\\label{app:validation}\n\n\\subsection{Objectives and experimental design}',
     'The existing headings combine methods and checks without identifying the objects being verified.',
     'Separate experimental definitions, gradient checks, and representation checks.')
edit(F, 'B-A1 objective-defining choices', between('The MLP is a', 'Parameters, gradients'),
     r'''Table~\ref{tab:setup} gives architectures, objective reductions, and
data coverage. MLP inputs use coordinate-wise standardization; ResNet
uses channel-wise standardization, both from training-split statistics.
ResNet retains fixed training-mode BatchNorm groups throughout training
and measurement. There is no augmentation or dropout.

The Transformer forms sequences of at most 35 tokens from twenty streams.
Its 33,278-word dictionary follows the pinned PyTorch example
revision~\cite{pytorchWordLanguageModelPinned}, including words from
training, validation, and test text. This is a fixed-objective dynamics
experiment rather than a train-only-vocabulary benchmark.

''', 'The paragraph repeats Table I and buries objective-defining preprocessing among architecture counts.',
     'Refer to Table I, retain normalization and vocabulary choices, and split image and language preprocessing. Full architecture counts remain in Supplement S2.')
edit(F, 'B-A2 numerical conventions', between('Parameters, gradients', 'MLP mini-batches'),
     r'''Parameters, gradients, and Adam moments are FP32; diagnostic products
and reductions use FP64, with TF32 disabled. Full-batch GD and Adam
updates use the complete training objective. Figure~\ref{fig:gd-loss}
specifies the separate spectral measurement coverage and interpolation.

''', 'A blanket full-objective statement is followed by mini-batch methods, and empirical findings interrupt numerical settings.',
     'Scope the full-objective claim and move the interaction/sign findings into B-B with their evidence.')
edit(F, 'B-A3 sampling design', between('MLP mini-batches', 'The Adam grid'),
     r'''MLP mini-batches contain 32, 128, 512, 2048, or 5000 images;
ResNet uses 1, 5, 20, or 50 complete normalization groups; Transformer
batches contain 8, 32, 128, 512, or 59680 sequences. At each draw within
a repetition, sizes use nested prefixes of a common random permutation;
a fresh permutation is drawn for each sampling event. For sequence-summed
losses $\ell_i$, the Transformer estimator is
$N(bM)^{-1}\sum_{i\in B}\ell_i$, where $N=59680$ and $M=2088600$.
The fixed denominator preserves the token-weighted objective.

Three initializations provide matched states at steps 0 and 2000,
with 200-update continuations and eight stochastic repetitions per cell.
Equal update counts do not imply equal data exposure. The two smallest
ResNet batches in initialization 2 use a different runtime; batch size
and runtime remain confounded in that block. Supplementary Section S2
retains the numerical bridge check and complete sampling protocol.

''', 'Sampling independence and across-size coupling are compressed together; the runtime bridge number distracts from its unresolved confound.',
     'Explain permutation coupling directly and retain the confound while pointing to the existing supplementary bridge result.')
edit(F, 'B-A4 Adam design and clock', between('The Adam grid', r'\subsection{Curvature'),
     r'''The Adam grid uses MLP rates $.0001/.0003/.001$ and ResNet/Transformer
rates $.0003/.001/.003$, for 1,024 updates with one initialization per
model. Initial parameters match across rates; moments start at zero
with the coefficients specified in Section~\ref{sec:adam}. These runs
use no weight decay, clipping, or schedule; the overview displays the
highest rates. An excursion starts at a strict loss rise and ends on
return to its pre-rise baseline. Recovery not observed before the run
ends is treated as right-censored. These events differ from the next-step
targets in Table~\ref{tab:severity}. At state $t$, gradients precede
update $t\to t+1$; representation changes ending at $t$ describe the
preceding update.

''', 'Coefficients repeat the main text and censoring is named without a definition.',
     'Retain design and clock, reference the existing coefficients, and explain censoring.')
edit(F, 'B-B1 accumulated-interaction evidence', r'\subsection{Curvature and representation checks}',
     r'''\subsection{Gradient and curvature checks}
The accumulated-interaction check tests whether opposition between
consecutive updates dominates the complete trajectory.
Table~\ref{tab:app-gd-audit} reports positive total interactions in all
nine GD trajectories, including seven with negative adjacent sums.
Its loss-sign comparison also exposes finite-step limits: the highest-rate
Transformer has 1,325 disagreements between the incoming quadratic
prediction and the observed loss change among 2,999 paired transitions.

\input{tables/appendix-gd-audit}
''', 'The central accumulated-reinforcement finding is repeated in prose without its compact numerical support in the main appendix.',
     'Expose the existing nine-row evidence, projected from the supplementary audit, without introducing new measurements.')
edit(F, 'B-B2 independent curvature comparison', between('Independent directional', r'\begin{table}'),
     r'''Independent Hessian--vector products measure directional curvature
at the starting state. Comparing them with the gradient-difference
measurement checks its finite-step approximation and sampling contribution.
The moving MLP study supplies 1,188 pairs from 297 continuations over
three starts and five batch sizes. Full gradients at both states supply
the realized noise correction. Table~\ref{tab:app-hvp} retains raw and
corrected discrepancies; finite-step differences can remain after
correction. This is an offline check, not an additional online estimator,
and measurements within a trajectory are dependent.

''', 'Cheap estimate versus curvature obscures step-average versus starting-point quantities.',
     'Identify both quantities and explain the scope and residual discrepancy of the offline check.')
edit(F, 'B-B3 fixed-state approximation check', between('Frozen-state sampling', 'Representation measurements'),
     r'''Independent resampling at nine fixed MLP states checks the moment
identities and expected-cosine approximation (Supplementary Section S2).
Maximum relative discrepancies in the two moment identities are .50\%
and .74\%; maximum absolute expected-cosine error is .00745. The
approximation lies outside 21 of 36 stochastic conditional 95\% bootstrap
intervals. It captures the scale of the expected cosine but is not exact
at the resolution of these checks.

\subsection{Representation measurements and spectral controls}
''', 'Protocol counts bury the distinction between small absolute error and statistical agreement.',
     'Keep every accuracy/coverage result, refer to the existing full sampling protocol, and separate representation checks structurally.')
edit(F, 'B-C1 measurement validity', between('Representation measurements', 'Temporal Grams'),
     r'''Representation measurements preserve input correspondence, masks,
normalization groups, model mode, random-number-generator state, buffers,
and gradients. Repeated-state and plain-forward checks verify that
measurement leaves the relevant training state unchanged. The selected
ResNet trace is checked against its own native continuation, not claimed
as bitwise recovery of another runtime's trajectory (Supplementary
Section S4). MLP outputs are raw values; ResNet/Transformer outputs are
log probabilities over all classes or words. These remove sensitivity
to a common logit shift but weight coordinates equally, rather than by
loss sensitivity.

''', 'Observer invariance and a detailed replay inventory interrupt the meaning of the measurement.',
     'Explain non-perturbation directly, retain native-continuation scope and output geometry, and leave exact replay counts in Supplement S4.')
edit(F, 'B-C2 temporal reduction', between('Temporal Grams', 'Table~\\ref{tab:app-spectral}'),
     r'''Temporal Gram matrices use all coordinates in FP64, without centering,
projection, or primary row normalization. The extra change needed for
the first cosine is excluded from the fitted 51/61/49-row matrices.
Matrix and reconstruction checks verify the reduction (Supplementary
Section S4). Retention uses
$\sqrt{\mathbf1^\top K\mathbf1}/\sum_i\sqrt{K_{ii}}$ without truncating
modes; direct and full-eigendecomposition calculations agree.

''', 'An implementation-check inventory distracts from the reduction and full-mode retention contract.',
     'Retain the measurement-defining operations and summarize numerical checks with a supplementary pointer.')
edit(F, 'B-C3 alternative explanations for concentration', between('Table~\\ref{tab:app-spectral}', r'\begin{table}'),
     r'''Could spectral concentration arise mainly from a few large steps?
The orthogonal control in Table~\ref{tab:app-spectral} zeros off-diagonal
entries of $K$, preserving step norms while removing directional
recurrence. Does concentration persist when all changes have equal
length? Unit-row normalization uses $B^{-1/2}KB^{-1/2}$, with
$B=\operatorname{diag}(\operatorname{diag}K)$, excluding zero rows.
Normalization retains strong concentration in the MLP but increases
the modes needed for 95\% in ResNet and Transformer. These descriptive
controls are not randomized significance tests. Signed coefficients
and consecutive cosines, rather than concentration alone, establish
reversal.

''', 'The controls are defined as matrix operations before explaining the alternative explanations they test.',
     'Introduce each control through its scientific question and interpret the existing architecture-dependent table values.')
edit(F, 'C structure', '\\section{Intervention Integrity, Endpoint Sensitivity, and Cost}\n\\label{app:interventions}',
     '\\section{Intervention Verification, Endpoint Sensitivity, and Cost}\n\\label{app:interventions}\n\n\\subsection{Matched conditions and the local displacement bound}',
     'The section mixes implementation validity and outcome robustness.',
     'Separate the matched-comparison contract from endpoint and cost checks.')
edit(F, 'C-A1 matched-state verification', between('All 12 conditions', 'Late $\\beta_2$'),
     r'''All 12 conditions within a model share the complete pre-intervention
parameter, optimizer, random, and data-order state. Eight preceding
untreated updates supply calibration; budgets are frozen before the
common start. Checks verify common starting states, unchanged-Adam
gradient and update calculations, reference evaluations, calibration
values, and bounds on stored increments. Supplementary Section S5
retains the audit counts and full coverage. These checks verify execution;
they are not statistical replications.

''', 'Counts and shorthand audit names obscure what was matched and verified.',
     'State the actual comparison guarantees and move the inventory to its existing Supplement S5 record.')
edit(F, 'C-A2 intervention semantics', between('Late $\\beta_2$', r'\begin{equation}'),
     r'''The interventions follow Section~\ref{sec:vc-interventions}: late
$\beta_2$ changes retain optimizer state, clipping precedes Adam, and
the cap scales proposed increments without recomputing moments. The cap
targets two hidden MLP linear maps, 31 ResNet convolutions, and two
query/key/value and four feed-forward Transformer maps; other increments
remain unchanged. When upstream inputs $H$ and local weights change
together, the preactivation change is
''', 'Implementation rules repeat Section VI and QKV remains shorthand.',
     'Retain site counts and state semantics concisely, and introduce the unchanged propagated-change equation.')
eqs = re.findall(r'\\begin\{equation\}.*?\\end\{equation\}', (DRAFT/F).read_text(), re.S)
edit(F, 'C-A2 propagated-change reference', eqs[-1], eqs[-1].replace('\\end{equation}', '\\label{eq:app-propagated-change}\n\\end{equation}'),
     'The local-bound equation lacks a reference label.', 'Add only a label; preserve its mathematics byte for byte.')
edit(F, 'C-A2 meaning of the bounded terms', between('Only the first line', r'\input{tables/appendix-last32}'),
     r'''The first line is the direct local change and is bounded by the cap.
The second contains the change arriving through upstream inputs and
its interaction with the local weight update. Neither is bounded by
the local rule, so Eq.~\eqref{eq:app-propagated-change} gives no guarantee
on total propagated output or loss.

\subsection{Endpoint sensitivity and computational cost}
''', 'The equation terms are not interpreted and edge conventions repeat Section VI.',
     'Explain direct, upstream, and interaction terms; retain the scope limit and remove duplicate zero conventions.')
edit(F, 'C-B1 endpoint sensitivity', between('The final loss in', r'\input{tables/appendix-costs}'),
     r'''The final loss in Fig.~\ref{fig:vc-tradeoff} depends on the phase of
remaining oscillation. Table~\ref{tab:app-late} averages states
$t_f-31,\ldots,t_f$ for every setting, without selecting a favorable
checkpoint. The endpoint advantage of q25 persists for all three models.
ResNet q50 remains worse than unchanged Adam, whereas q25 and q75
improve the late mean. Neither a fixed endpoint nor a short late average
establishes asymptotic convergence; comparisons use one initialization
per model and unequal effective intervention strengths.

''', 'The averaging interval is implicit and arms is unnecessary jargon.',
     'Specify the exact states while retaining the complete result and limitations.')
edit(F, 'C-B2 measured cost', between('The recorded q50', r'\section{Applied Updates'),
     r'''Table~\ref{tab:app-cost} reports recorded q50 component times. Gradient
and reference means cover evaluated parameter states, including the
final state; controller means cover applied updates. The reference pass
collects current cap inputs in addition to the local products and
reductions. Other diagnostics and I/O add wall time. Hosts and
instrumentation differ, so these measurements do not establish a
time-to-loss advantage. In the MLP, cap computation exceeds the
full-gradient calculation.

''', 'Retain the cost caveat is document commentary, and final boundary is undefined.',
     'State the timing units, what work they include, and the supported limitation directly.')
gallery = original[original.index('\\section{Applied Updates'):]
edit(F, 'D transfer out of main appendix', between(r'\section{Applied Updates', '\\clearpage'), '',
     'The complete trajectory gallery consumes main-paper space without a new central inference.',
     'Transfer all three existing assets to Supplement S7, preserving data and measurement captions.')
edit(F, 'Remove obsolete float-page reset', '\\clearpage\n% Reset column height after the full-width float pages before the bibliography.\n\\onecolumn\\twocolumn\n', '',
     'The column reset was specific to the removed full-width float pages.',
     'Remove it provisionally and inspect the rebuilt appendix-to-bibliography transition.')

# Project the existing numerical audit; do not recompute its scientific values.
source = (DRAFT/'tables/gd-audit.tex').read_text()
rows=[]
for line in source.splitlines():
    if line.startswith(('MLP &', 'ResNet-32 &', 'Transformer &')):
        cells=[c.strip() for c in line.removesuffix('\\\\').split('&')]
        rows.append(' & '.join(cells[i] for i in (0,1,4,5,6)) + r' \\')
assert len(rows)==9
table=r'''\begin{table}[!t]
\caption{Complete GD interaction and loss-sign audit. $E=\sum_t\|g_t\|^2$, $I_{\rm all}=2\sum_{i<j}g_i^\top g_j$, and $I_1=2\sum_tg_{t-1}^\top g_t$ use all applied gradients. Ratios remove the common $\eta^2$ factor. Sign disagreements compare the incoming quadratic prediction with 2,999 observed loss changes; the final unpaired update is excluded.}
\label{tab:app-gd-audit}
\centering\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrr}
\toprule
Model & $\eta$ & $I_{\rm all}/E$ & $I_1/E$ & Sign disagreements \\
\midrule
'''+'\n'.join(rows)+r'''
\bottomrule
\end{tabular}
\end{table}
'''
edit('tables/appendix-gd-audit.tex','Compact existing GD audit','',table,
     'The full supplementary audit includes loss and retention columns already covered in the main paper.',
     'Copy all nine rows exactly, selecting model, rate, interaction ratios, and sign disagreements.')
figures=re.findall(r'\\begin\{figure\*\}.*?\\end\{figure\*\}',gallery,re.S)
assert len(figures)==3
for i,s in enumerate(figures):
    s=s.replace('\\begin{figure*}[!t]','\\begin{figure}[H]').replace('\\end{figure*}','\\end{figure}')
    s=s.replace('width=\\textwidth','width=\\linewidth,height=.78\\textheight,keepaspectratio')
    s=s.replace('Sites match Fig.~\\ref{fig:trace-motion}.', 'Sites match the main paper\'s representation-motion figure.')
    figures[i]=s
s7=r'''\clearpage
\section{Extended Intervention Trajectories}
\label{supp:extended-interventions}
Figures~\ref{fig:app-updates}--\ref{fig:app-selected} retain complete
post-intervention update histories, intervention activity, and selected
representation trajectories. The applied increments differ from raw
gradients; nominal coefficients do not match effective intervention
strength. The selected conventional settings illustrate contrasting
behaviors, rather than rank each family's best setting. Section S5
retains the complete matched outcomes.

'''+ ('\n\n\\clearpage\n'.join(figures))+'\n\\clearpage\n'
edit('sections/s07-intervention-trajectories.tex','S7 transferred trajectory gallery','',s7,
     'Transferred figures need a standalone supplementary introduction and valid cross-document wording.',
     'Reuse the original assets and captions, adapting only float sizing and the main-paper figure reference.')
edit('supplementary.tex','Include S7',r'\input{sections/s06-reproducibility}',
     '\\input{sections/s06-reproducibility}\n\\input{sections/s07-intervention-trajectories}',
     'The supplementary wrapper has no destination for the transferred gallery.',
     'Append S7 after S6, retaining the existing section numbering.')
edit('supplementary.tex','Supplement roadmap',
     'S5 supplies intervention methods and outcomes. Historical MLP replication',
     'S5 supplies intervention methods and outcomes; S6 records reproducibility and scope; S7 retains extended intervention trajectories. Historical MLP replication',
     'The roadmap does not mention the new gallery or existing provenance section.',
     'Extend the roadmap without changing prior content.')
edit('sections/05c-interventions.tex','VI full-history pointer',
     r'Appendix~\ref{app:plots} retains all applied-update histories and selected functional trajectories.',
     'Supplementary Section S7 provides complete applied-update histories\nand selected representation trajectories.',
     'The referenced main appendix has moved.', 'Point to its explicit supplementary destination.')
edit('sections/05c-interventions.tex','VI motion pointer',
     'Appendix~\\ref{app:plots} compares this functional motion with selected\nconventional interventions.',
     'Supplementary Section S7 shows representation trajectories for selected\nconventional interventions.',
     'The referenced main appendix has moved.', 'Retain the supporting pointer without implying a new comparison or ranking.')
edit('sections/04-gd-sgd.tex','IV accumulated-interaction evidence pointer',
     r'the adjacent cancellations (Appendix~\ref{app:validation}).',
     r'the adjacent cancellations (Table~\ref{tab:app-gd-audit}).',
     'A broad appendix pointer obscures the specific evidence for the accumulated-interaction finding.',
     'Reference the newly exposed existing audit table directly.')
