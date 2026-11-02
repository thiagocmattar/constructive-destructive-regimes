# Loss spikes as fixed-level excursions

A spike is an upward excursion of the training loss, from the first departure
above its pre-rise level until the first return to that level. It can contain
several local maxima. We retain every excursion as a candidate and measure its
amplitude continuously, without using optimizer diagnostics to select events.

This implements the author's selected **fixed pre-rise reference**, with zero
tolerance, no smoothing, and no minimum amplitude. The supplied draft is
preserved in [archive/loss_spike_definition_original.md](archive/loss_spike_definition_original.md).
No slowly varying baseline is estimated.

## 1. Definition and deterministic extraction

Let $\theta_t$ denote the parameters after $t$ accepted optimizer updates, and
write $\mathcal L_t:=\mathcal L(\theta_t)$ for the observed full training-objective
loss, $t=0,\ldots,T$. Here $T=1024$ for each of the nine Run215 trajectories.

Starting at the beginning of a trajectory, let $a_i$ be the earliest state not
already inside a preceding excursion for which

$$
\mathcal L_{a_i+1}>\mathcal L_{a_i}.
$$

The reference loss is $\mathcal L_{a_i}$, held constant throughout excursion $i$.
The first elevated state is $a_i+1$. Define the first recovery by

$$
c_i:=\inf\{t\in\{a_i+1,\ldots,T\}:\mathcal L_t\leq\mathcal L_{a_i}\},
$$

where $\inf\varnothing=+\infty$. For a recovered excursion, the defining
conditions are

$$
\mathcal L_t>\mathcal L_{a_i}\quad(a_i<t<c_i),
\qquad \mathcal L_{c_i}\leq\mathcal L_{a_i}.
$$

Thus $(a_i,c_i)$ is a maximal consecutive interval above this fixed reference.
The peak is the earliest maximizer in its interior:

$$
p_i:=\min\operatorname*{arg\,max}_{a_i<t<c_i}\mathcal L_t.
$$

The event is represented by $(a_i,p_i,c_i)$. In the dataset, `before_state` is
$a_i$, `onset_state` is $a_i+1$, `peak_state` is $p_i$, and `recovery_state` is
$c_i$. This distinguishes the last pre-rise observation from the first elevated
observation. After recovery, resume the search at $c_i$; a recovery state may
also be the pre-rise boundary of the next event. Interior states never belong
to two different excursions.

A local maximum followed by a partial decline remains inside the same event
until the loss returns to its reference. Plateaus at a maximum use their first
state as the peak. A flat interval before departure uses its final state as
$a_i$. All states, including initialization and the final observation, are
eligible; no startup exclusion or retrospective choice of baseline is used.

## 2. Loss magnitudes and durations

For each recovered event, retain the three observed losses
$\mathcal L_{a_i}$, $\mathcal L_{p_i}$ and $\mathcal L_{c_i}$. Their ratios to the
pre-rise loss, when $\mathcal L_{a_i}>0$, are

$$
\left(1,\frac{\mathcal L_{p_i}}{\mathcal L_{a_i}},
         \frac{\mathcal L_{c_i}}{\mathcal L_{a_i}}\right).
$$

The corresponding percentage change at any recorded state $t$ is
$100(\mathcal L_t/\mathcal L_{a_i}-1)$. For example, $(1,1.5,1)$ denotes a
50% peak increase and exact recovery. Recovery may undershoot the baseline;
its ratio is then below one. We retain that observed value rather than replace
it with one. The peak-to-recovery percentage change uses the peak loss as its
denominator: $100(\mathcal L_{c_i}/\mathcal L_{p_i}-1)$.

The durations, in accepted optimizer updates, are

$$
T_i^{\mathrm{rise}}=p_i-a_i,\qquad
T_i^{\mathrm{recovery}}=c_i-p_i,\qquad
T_i^{\mathrm{total}}=c_i-a_i.
$$

The rise includes the transition from the last pre-rise state to the first
elevated state. Recovery loss means loss **at the first return**, not the
minimum over a later window. The dataset also records the first elevated loss,
the observation immediately after recovery (state $c_i+1$, when available),
absolute peak increase, and the discrete sum
$\sum_{t=a_i+1}^{c_i-1}(\mathcal L_t-\mathcal L_{a_i})$ of excess loss
(loss units times updates).

A later amplitude selection can be made using
$\mathcal L_{p_i}/\mathcal L_{a_i}\geq 1+h$, with a declared $h>0$.
No such selection is applied in the current dataset. Candidate counts therefore
include very small fluctuations.

## 3. Incomplete excursions

If $c_i=+\infty$, recovery is right-censored at $T$. Store recovery state, loss,
and duration as missing, with `right_censored=True`; do not substitute $T$ as a
recovery. Use the earliest observed maximum over $a_i<t\leq T$ as an observed
peak, marked `peak_is_provisional=True`: a later, unobserved maximum could
replace it. The stored rise time for such an event is therefore to its observed
peak, not an established completed-event rise duration.

`end_state=T`, `observed_total_steps=T-a_i`, and
`observed_post_peak_steps=T-p_i` describe available follow-up. The loss sum is
restricted to the observed elevated states. Completed and censored events
remain distinguishable in every summary.

## 4. Joining optimizer diagnostics without shifting the loss clock

The manuscript defines $g_t$ at $\theta_{t-1}$, with Adam first moment $m_t$
and adaptive direction $d_t$ formed from that gradient. Therefore, at a loss
state $s$ the contemporaneous diagnostic row contains

$$
g_{s+1},\quad m_{s+1},\quad d_{s+1},\quad
\rho_{s+1}=\cos(g_s,g_{s+1}),\quad
\eta\widehat\kappa_{s+1}
=1-\rho_{s+1}\frac{\|g_{s+1}\|}{\|g_s\|}.
$$

Momentum and adaptive cosines similarly compare consecutive $m$ and $d$ vectors.
All are unsmoothed, with the full parameter scope and reductions of Run215.
The first cosine/projected-change observation is undefined. At the terminal
state, the unapplied raw-gradient norm and projected change were measured, but
the next Adam moment/direction and their cosines were not; missing observations
remain missing.

The actual-update retention at state $s$ is

$$
R_s=\frac{\|\theta_s-\theta_0\|}
           {\sum_{j=1}^{s}\|\theta_j-\theta_{j-1}\|},
$$

using recorded applied increments and the cumulative prefix from initialization.
It excludes the prospective update at $s+1$ and is undefined at initialization.
It is not reset at an excursion boundary.

The event table joins these diagnostics at $a_i,p_i,c_i$. It also records the
earliest rise-window maxima of gradient norm, momentum norm and projected
change, and minima of the three consecutive cosines over $[a_i,p_i]$, with
their state and lead to the loss peak. The companion state table preserves all
recorded scalar streams and cross-stream statistics throughout each event,
including both boundaries. Shared boundary observations appear once per event.
They are not independent additional samples.

Under Adam, $\eta\widehat\kappa_t$ denotes normalized projected raw-gradient
change along the adaptive trajectory; it does not identify a Hessian eigenvalue
or an Adam stability boundary. Events from one trajectory can also be temporally
dependent. This extraction supplies descriptive event samples; it does not by
itself establish prediction, causation, or uncertainty across initializations.
