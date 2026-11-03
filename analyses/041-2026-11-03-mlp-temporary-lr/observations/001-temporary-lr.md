# Observation001: fixed temporary reduction using $\eta\widehat\kappa_t$

Question: does temporary LR reduction using available $\eta\widehat\kappa_t$
preserve training progress relative to always-half while reducing excursion
severity? Source: Run219 attempts001-local and002-control-recovery; unchanged
Run218 parent and overlapping controls; fixed plan5107aff. Seed0, one selected
state657, three367-update full5,000-image objective continuations.

Reduction:01_report.py. Unchanged states below881 come from the interrupted
attempt;881..1024 from exact replay. All368 boundary loss/norm values match
historical Run215. All129 overlapping values match Run218 for both fixed-LR
controls. Raw failure/recovery evidence is preserved.

Caption: Unsmoothed training loss, available $\eta\widehat\kappa_t$ with the1.8
threshold, and actual LR/base LR for all three methods on the same calendar-state
axis. The zoom673..745 is fixed before results; the full view657..1024 exposes
later/replacement events and return to base LR. Loss uses a linear zoom axis and
log full-view axis. Shade exactly temporary half-LR intervals[s,s+1). At state s,
manuscript t=s+1; the available $\eta\widehat\kappa_t$ and outgoing LR share the
same decision clock. No temporary half-LR interval lies within the fixed zoom.
The last LR drawing endpoint at1024 closes the preceding step only.

Coverage:1,104 canonical boundary observations and1,101 native-Adam updates;
no smoothing, recentering or post-result threshold adjustment. Outcome reference
is unchanged loss689; peak/area cover689..1024 and mean loss658..1024. The existing
fixed-level excursion detector retains censored events. Endpoint test splits are
complete and descriptive only. Direct plot values, decision clocks, source
identity and scalar sums are checked; both one-page PDFs rendered/visually reviewed.

Finding: temporary feedback gives18 one-update reductions and36 LR switches,
but larger excursions and worse mean/final loss than always-half. The late
938/950 excursion is unrecovered at1024. One seed/state does not establish
robustness or generalization; no frequency-matched timing control is supplied.
See REPORT.md for8 findings, exact values and infrastructure cost accounting.
