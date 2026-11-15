"""Revise intervention exposition; preserve the frozen comparison and all arms."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
D=HERE.parents[1]/'draft'
p=D/'sections/05c-interventions.tex'
s=p.read_text(encoding='utf8')
def cut(start,end,text):
 global s
 i=s.index(start);j=s.index(end,i);s=s[:i]+text+s[j:]
s=s.replace('Interventions Guided by Representation Changes','Selective Interventions Guided by Representation Changes')
cut('The preceding figures suggest','% Declare',r'''The preceding analysis shows substantial functional movement repeatedly
reversing with little net displacement. Our working assumption is that
limiting unusually large local changes can reduce the disruptive
component of this oscillation while preserving subsequent progress.
We test a distributed displacement cap that selectively scales Adam's
proposed unit updates according to the representation change they would
produce. Learning-rate reductions, second-moment changes, and gradient
clipping provide complementary controls at other points in the update.

''')
cut('\\caption{Excursion suppression','\\label{fig:vc-tradeoff}',r'''\caption{Mean training loss $\bar L$ and accumulated upward loss variation $\mathcal P$ for unchanged Adam and all 33 treated continuations. Both metrics are lower-is-better; gray lines mark the control's values. Axes differ across models. The vertical axis is linear close to zero and logarithmic above the indicated transition, retaining the zero-variation Transformer continuation. Every tested setting is shown. One initialization is used per model; settings are not independent replications and their effective strengths are not matched.}
''')
s=s.replace(r'\input{\vcassetroot/tables/matched-main-outcomes.tex}',r'\input{tables/matched-main-outcomes.tex}')
s=s.replace("module's", "site's").replace('module','site')
s=s.replace('a proposal that would move','an update that would move').replace('A smaller proposal','A smaller update')
s=s.replace('before treatment begins. At starting step $f$', 'before the cap begins. At starting step $t_0$')
s=s.replace('t=f-8,\\ldots,f-1','t=t_0-8,\\ldots,t_0-1')
s=s.replace('We preselect the\nmedian, q50, as the principal setting; q25 and q75 examine sensitivity.', 'The matched experiment prespecified q50 and also tested q25 and q75;\nthe illustration below uses q25, with all three retained in the comparison.')
cut('The computation requires','\\begin{figure*}',r'''The score evaluates only the \emph{local} effect of the proposed
increment, holding the current site's inputs fixed. Once these inputs
are available, Eq.~\eqref{eq:vc-score} requires one linear product
(a convolution for convolutional sites), followed by a squared reduction
per output unit. It does not forward the proposed update through the
whole network. The implemented controller also evaluates the reference
examples to collect current inputs and verifies the rounded increments;
its total overhead therefore exceeds that local product alone.
The temporal SVD is an analysis tool and is not computed by the cap.
Supplementary Section S5 separates the bounded direct change from
upstream changes and their interactions.

The closest representation-based comparator is the relative-change
correction of Kosson et al.~\cite{kosson2024warmup}, developed to regulate
updates during warmup. Here the target is the \emph{absolute} per-unit
displacement during an established oscillatory trajectory, using a budget
calibrated from its preceding updates. This avoids dividing by a possibly
small current activation scale and leaves smaller unit updates intact.
AGC instead compares raw-gradient and weight norms~\cite{brock2021normalization};
FOOF constructs a direction for descent in neuron space~\cite{benzing2022neurons}.
Our contribution is to control a different quantity while retaining
Adam's proposed direction within each unit and its evolving moments.
The experiment below tests this choice against conventional interventions;
it does not rank these other representation-based optimizers.

''')
s=s.replace('25-main-q50-functional-motion.pdf','25-main-q25-functional-motion.pdf')
cut('\\caption{What the preselected','\\label{fig:vc-motion}',r'''\caption{Functional motion under unchanged Adam and the q25 cap. Top: full training loss on logarithmic axes. Remaining rows compare consecutive representation-change cosine and RMS displacement on the same reference examples. Each RMS pair is divided by the same site-specific maximum under unchanged Adam; the color scale includes all values. Sites are the measured outputs used in Fig.~\ref{fig:trace-motion}. Incoming changes are displayed at their completed-step endpoint $t$, without smoothing or averaging over layers. q25 illustrates a setting that improves all three reported training outcomes in all models; all quantiles remain in Table~\ref{tab:vc-outcomes}.}
''')
s=s.replace('We test immediate learning-rate multipliers $.75/.5/.1$;\nlate $\\beta_2$ values $.95/.9/.5$; global gradient-norm clipping at $.5/1$;',r'''Learning-rate schedules motivate changing the step scale~\cite{smith2017cyclical,loshchilov2016sgdr};
we test immediate multipliers $.75/.5/.1$. Following the second-moment
mechanism studied for Adam spikes~\cite{bai2026spikes}, we reduce
$\beta_2$ to $.95/.9/.5$. Global gradient clipping~\cite{pascanu2013difficulty}
uses norm thresholds $.5/1$;''')
s=s.replace('both hidden MLP affines','both hidden MLP linear maps').replace("Transformer's QKV and feed-forward affines","Transformer's QKV and feed-forward linear maps").replace('parameter proposal','proposed update')
cut('To distinguish smoother loss','For the squared-loss MLP',r'''To distinguish smoother loss from faster training progress, define
\begin{equation}
 \bar L=\frac{1}{t_f-t_0}\sum_{t=t_0+1}^{t_f}L_t,\qquad
 \mathcal P=\sum_{t=t_0+1}^{t_f}\max(L_t-L_{t-1},0).
 \label{eq:vc-outcomes}
\end{equation}
Here $t_0$ and $t_f$ are the start and end steps. Mean loss excludes the
shared starting state; $\mathcal P$ sums all upward loss movement,
rather than counting peaks. Final training loss $L_{t_f}$ records the
attained endpoint. All three are reported in absolute units, with lower
values favorable. The fixed windows make their comparison meaningful
within each model.

Figure~\ref{fig:vc-tradeoff} separates suppression of loss fluctuations
from training progress. A point below and to the left of the control
improves both metrics. Strong learning-rate reduction can almost remove
upward movement while retaining higher mean loss than other settings,
especially in ResNet. Late second-moment changes and clipping are more
model-dependent: an intervention that helps the Transformer can worsen
the MLP. Supplementary Section S5 retains their complete trajectories.

All three cap quantiles improve both mean loss and positive variation
over unchanged Adam in all three cases. The lowest observed mean loss
in each model belongs to a cap setting. Table~\ref{tab:vc-outcomes}
also exposes the endpoint tradeoff: ResNet q50 finishes above the
control, whereas q25 and q75 finish below it. No quantile is best across
all metrics and models. We use q25 in Fig.~\ref{fig:vc-motion} because
it improves all three training outcomes in every case; q50 remains the
experiment's originally prespecified setting.

The top row of Fig.~\ref{fig:vc-motion} shows the resulting loss
trajectories, and the paired maps show what changes inside the network.
Large output jumps become smaller, while the cosine maps remain strongly
oscillatory. The q25 cap can therefore improve training while retaining
back-and-forth movement. Its effect is on the size and distribution of
that movement, rather than eliminating every reversal. The complete
window also matters: attenuating the traced ResNet peak does not remove
all later oscillations. Supplementary Section S5 reports output-motion
measurements and maps for every setting.

''')
s=s.replace('baseline and q50 loss rise','baseline and q25 loss rise')
i=s.index('The comparison establishes')
s=s[:i]+r'''These comparisons support selective control of local movement in the
tested training trajectories. The implementation adds measurable cost;
the recorded MLP cap computation exceeds its full-gradient calculation
(Supplementary Section S5). One initialization per model and unequal
intervention strengths limit conclusions about a universal optimizer
ranking. The consistent reductions in mean loss and upward variation
nevertheless show that persistent oscillation can be made less disruptive
without requiring consecutive changes to become constructive.
'''
p.write_text(s,encoding='utf8',newline='\n')

p=D/'sections/s05-intervention-methods.tex';s=p.read_text(encoding='utf8')
s=s.replace('q50 was selected before the matched outcomes; its retention as the principal method is not a test-set or endpoint choice.', 'q50 was selected before the matched outcomes. The main motion illustration now uses q25 after inspection of training outcomes; it is not presented as a prespecified selection or a universal optimum.')
s=s.replace('The main q50 plate uses the identical values with a more compact layout.', 'The main q25 plate uses the same underlying q25 values with a compact layout, logarithmic losses, and no peak markers.')
s=s.replace('The absolute unchanged-Adam mean/PV/test references also appear in the main compact table;', 'The main compact table reports absolute training outcomes;')
s=s.replace('$f-8,\\ldots,f-1$','$t_0-8,\\ldots,t_0-1$').replace('$f$, $f+32$, $(f+e)/2$, $e-32$, and $e$','$t_0$, $t_0+32$, $(t_0+t_f)/2$, $t_f-32$, and $t_f$')
s=s.replace('$f+1,\\ldots,e$','$t_0+1,\\ldots,t_f$').replace('$e-31,\\ldots,e$','$t_f-31,\\ldots,t_f$').replace('$L_e$','$L_{t_f}$')
s=s.replace('PV reduction is $100(1-\\mathrm{PV}/\\mathrm{PV}_{\\rm Adam})$', r'Positive-variation reduction is $100(1-\mathcal P/\mathcal P_{\rm Adam})$').replace('Lowering PV','Lowering $\\mathcal P$')
p.write_text(s,encoding='utf8',newline='\n')
p=D/'sections/06-conclusion.tex';s=p.read_text(encoding='utf8').replace('case, higher q50 terminal test losses, and additional computation limit','case and additional computation limit');p.write_text(s,encoding='utf8',newline='\n')
