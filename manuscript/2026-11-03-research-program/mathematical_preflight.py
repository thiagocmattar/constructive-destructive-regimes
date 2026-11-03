"""NumPy-only design validation. No model training or empirical research claims."""
from pathlib import Path
import json
import math
import numpy as np

HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261103)
ERRORS = {}
COUNTS = {}


def check(name, observed, expected, tol=2e-11):
    a, b = np.asarray(observed, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    err = float(np.max(np.abs(a-b)/(1.0+np.abs(a)+np.abs(b))))
    assert np.isfinite(err) and err <= tol, (name, err)
    ERRORS[name] = max(ERRORS.get(name, 0.0), err)
    COUNTS[name] = COUNTS.get(name, 0) + 1


def spectral_candidate(s, y, g, d, cap):
    """A guarded proposal; cap bounds alpha, not future loss or stability."""
    vectors = [np.asarray(v, dtype=np.float64) for v in [s, y, g, d]]
    if not all(np.isfinite(v).all() for v in vectors):
        return dict(status='unavailable', reason='nonfinite_input')
    s, y, g, d = vectors
    if not np.isfinite(cap) or cap <= 0:
        return dict(status='unavailable', reason='invalid_cap')
    ss, sy, dd, gd = float(s@s), float(s@y), float(d@d), float(g@d)
    if ss == 0:
        return dict(status='unavailable', reason='zero_previous_step')
    if dd == 0:
        return dict(status='unavailable', reason='zero_current_direction')
    if gd <= 0:
        return dict(status='unavailable', reason='non_descending_current_direction')
    if sy <= 0:
        return dict(status='unavailable', reason='nonpositive_secant_curvature')
    kappa = sy/ss
    alpha = gd/(kappa*dd)
    if not math.isfinite(alpha) or alpha <= 0:
        return dict(status='unavailable', reason='invalid_proposal')
    return dict(status='valid_proposal', kappa_hat=kappa, alpha_raw=alpha,
                alpha_capped=min(alpha, cap), clamp_active=alpha > cap,
                guarantee='No future loss-decrease or stability guarantee')


def spd(n):
    q, _ = np.linalg.qr(RNG.normal(size=(n, n)))
    return (q*np.geomspace(0.2, 100.0, n))@q.T


quadratic_cases = 0
bound_checks = 0
for n in range(2, 9):
    for _ in range(20):
        h = spd(n)
        theta = RNG.normal(size=n)
        b = RNG.normal(size=n)
        g = h@theta+b
        d_prev = g+0.3*RNG.normal(size=n)
        if g@d_prev <= 0:
            d_prev = g.copy()
        eta_prev = float(RNG.uniform(0.001, 0.05))
        s = -eta_prev*d_prev
        g_next = h@(theta+s)+b
        y = g_next-g
        k_sec = float(s@y/(s@s))
        check('quadratic_actual_step_secant', k_sec, d_prev@h@d_prev/(d_prev@d_prev))
        k_chat = d_prev@(g-g_next)/(eta_prev*(d_prev@d_prev))
        check('chat_direction_formula_equals_actual_secant', k_chat, k_sec)
        alpha_old = (g@d_prev)/(d_prev@h@d_prev)
        alpha_old_from_secant = eta_prev*(g@d_prev)/(d_prev@(g-g_next))
        check('completed_direction_line_optimum', alpha_old_from_secant, alpha_old)
        q_step = -float(s@y/(s@g))
        delta_exact = float(g@s+0.5*s@h@s)
        check('quadratic_loss_change_q_step', delta_exact, (s@g)*(1-q_step/2))
        # A scalar spectral proposal for a new direction is algebraically BB1 times slope correction.
        d_current = g_next.copy()
        if np.linalg.norm(d_current) == 0:
            continue
        bb1 = float(s@s/(s@y))
        alpha_spec = float((g_next@d_current)/(k_sec*(d_current@d_current)))
        check('bb1_current_direction_equivalence', alpha_spec,
              bb1*(g_next@d_current)/(d_current@d_current))
        # Positive and negative parallel directions have identical directional curvature.
        u = s/np.linalg.norm(s)
        m = float(np.max(np.abs(np.linalg.eigvalsh(h))))
        for v in [u, -u, RNG.normal(size=n)]:
            v = v/np.linalg.norm(v)
            c = float(np.clip(u@v, -1.0, 1.0))
            sinphi = math.sqrt(max(0.0, 1-c*c))
            mismatch = abs(float(v@h@v)-k_sec)
            bound = 2*m*sinphi
            assert mismatch <= bound+2e-9, (mismatch, bound)
            bound_checks += 1
        # Secant noise contamination is explicit, and increases as the step shrinks.
        delta_noise = RNG.normal(size=n)*0.01
        noisy_k = float(s@(y+delta_noise)/(s@s))
        check('endpoint_noise_projection', noisy_k-k_sec, s@delta_noise/(s@s))
        assert abs(noisy_k-k_sec) <= np.linalg.norm(delta_noise)/np.linalg.norm(s)+2e-9
        quadratic_cases += 1

# Smooth nonquadratic test of the endpoint-drift part of the transport bound.
nonquadratic_cases = 0
for n in [2, 3, 5]:
    for _ in range(30):
        a = spd(n)
        gamma = float(RNG.uniform(0.05, 2.0))
        x = RNG.normal(size=n)
        s = RNG.normal(size=n)*0.2
        grad = lambda z: a@z+gamma*z**3
        hessian = lambda z: a+np.diag(3*gamma*z*z)
        y = grad(x+s)-grad(x)
        u = s/np.linalg.norm(s)
        h_bar = a+np.diag(3*gamma*(x*x+x*s+s*s/3))
        k_sec = float(s@y/(s@s))
        check('quartic_path_average_identity', k_sec, u@h_bar@u)
        h_now = hessian(x+s)
        m_now = float(np.max(np.abs(np.linalg.eigvalsh(h_now))))
        radius = float(max(np.max(np.abs(x)), np.max(np.abs(x+s))))
        lh = 6*gamma*radius
        for v in [u, -u, RNG.normal(size=n)]:
            v = v/np.linalg.norm(v)
            c = float(np.clip(u@v, -1.0, 1.0))
            bound = 2*m_now*math.sqrt(max(0.0, 1-c*c))+0.5*lh*np.linalg.norm(s)
            assert abs(float(v@h_now@v)-k_sec) <= bound+2e-9
            bound_checks += 1
        nonquadratic_cases += 1

# Identical old states and gradients do not identify curvature in an unseen direction.
counterexamples = []
x_old, x_now, d_now = np.array([0.25, 0.0]), np.zeros(2), np.array([0.0, 1.0])
for unseen_curvature in [1.0, 100.0, 10000.0]:
    h = np.diag([1.0, unseen_curvature])
    b = np.array([0.0, 1.0])
    g_old, g_now = h@x_old+b, h@x_now+b
    check('nonidentifiability_same_old_gradient', g_old, [0.25, 1.0])
    check('nonidentifiability_same_new_gradient', g_now, [0.0, 1.0])
    s, y = x_now-x_old, g_now-g_old
    proposal = spectral_candidate(s, y, g_now, d_now, cap=100.0)
    alpha_true = float(g_now@d_now/(d_now@h@d_now))
    delta_proposed = float(-proposal['alpha_raw']*(g_now@d_now)
                           +0.5*proposal['alpha_raw']**2*(d_now@h@d_now))
    check('unseen_curvature_secant_is_one', proposal['kappa_hat'], 1.0)
    counterexamples.append(dict(unseen_curvature=unseen_curvature,
        identical_previous_secant=proposal['kappa_hat'],
        scalar_transported_alpha=proposal['alpha_raw'], exact_current_line_alpha=alpha_true,
        exact_loss_change_at_transported_alpha=delta_proposed))

# Exact bounded two-cycle for normalized-gradient descent, not default Adam or plain GD.
x_values = [0.25]
for _ in range(127):
    x_values.append(x_values[-1]-np.sign(x_values[-1]))
grads = np.asarray(x_values)
norms = np.abs(grads)
cosines = np.sign(grads[1:]*grads[:-1])
growth1 = norms[1:]/norms[:-1]
growth2 = norms[2:]/norms[:-2]
q_raw = 1-cosines*growth1
loss = 0.5*grads*grads
pair_envelope = loss.reshape(-1, 2).max(axis=1)
assert np.all(cosines == -1) and np.all(growth2 == 1)
assert np.all(pair_envelope == pair_envelope[0])
bounded_cycle = dict(
    method='Normalized-gradient descent, eta=1 on f(x)=x^2/2, x0=0.25',
    equivalent_toy='Adam beta1=beta2=0, epsilon=0, all gradients nonzero; not default Adam',
    states=len(grads), gradients=[0.25, -0.75], losses=[0.03125, 0.28125],
    adjacent_sign_and_growth_gate_hits=int(np.sum((cosines < 0)&(growth1 > 1))),
    q_raw_above_two_hits=int(np.sum(q_raw > 2)),
    lag2_growth_min=float(growth2.min()), lag2_growth_max=float(growth2.max()),
    pair_loss_envelope=float(pair_envelope[0]),
    scope='Negative control for progressive envelope growth or runaway instability. '
          'Each cycle DOES meet the current >10% recovered-excursion definition; '
          'therefore these are not false positives for that detector.')

guards = {}
examples = {
    'zero_previous_step': ([0, 0], [1, 0], [1, 0], [1, 0]),
    'zero_current_direction': ([1, 0], [1, 0], [1, 0], [0, 0]),
    'non_descending_current_direction': ([1, 0], [1, 0], [1, 0], [-1, 0]),
    'zero_secant_curvature': ([1, 0], [0, 0], [1, 0], [1, 0]),
    'negative_secant_curvature': ([1, 0], [-1, 0], [1, 0], [1, 0]),
    'nonfinite_input': ([1, 0], [float('nan'), 0], [1, 0], [1, 0]),
}
for name, args in examples.items():
    guards[name] = spectral_candidate(*args, cap=1.0)
    assert guards[name]['status'] == 'unavailable'
small_positive = spectral_candidate([1, 0], [1e-20, 0], [1, 0], [1, 0], cap=0.1)
assert small_positive['status'] == 'valid_proposal' and small_positive['clamp_active']
check('tiny_positive_curvature_cap', small_positive['alpha_capped'], 0.1)
guards['tiny_positive_curvature'] = small_positive

receipt = dict(
    status='passed', purpose='Mathematical design preflight only; no empirical finding or training',
    numpy_version=np.__version__, random_seed=20261103,
    quadratic_cases=quadratic_cases, smooth_nonquadratic_cases=nonquadratic_cases,
    transport_bound_checks=bound_checks, identity_check_counts=COUNTS,
    maximum_normalized_identity_errors=ERRORS,
    nonidentifiability_counterexamples=counterexamples, bounded_period2_control=bounded_cycle,
    explicit_guard_cases=guards,
    guarantees=['Exact quadratic scalar-line and secant identities within numerical tolerance',
                'Analytic transport bound numerically checked on synthetic smooth objectives'],
    limitations=['Finite random checks do not prove a theorem',
                 'No guarantee that previous secant curvature is predictive for current Adam direction',
                 'No guarantee of future decrease from scalar caps or empirical reliability gates',
                 'No default-Adam training, neural-network training, HVP workload or cloud action'])
(HERE/'mathematical_preflight_receipt.json').write_text(
    json.dumps(receipt, indent=2, allow_nan=False)+'\n', encoding='utf-8')
print(json.dumps({k:receipt[k] for k in ['status','quadratic_cases','smooth_nonquadratic_cases',
    'transport_bound_checks','maximum_normalized_identity_errors']}, indent=2))
