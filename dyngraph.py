"""Regime-switching dynamic Erdős-Rényi graphs: simulation and method-of-moments estimation from
star counts (Mandjes and Wang, 2023, 2025).

Model. Two dynamic Erdős-Rényi graphs on N vertices evolve in parallel. In graph i every vertex
pair alternates between on-times X_i and off-times Y_i. A hidden regime process alternates between
modes 1 and 2 with sojourn times Z_1 ~ G(p0) and Z_2 ~ G(q0); at time k one observes the graph of
the current mode, and only through aggregate counts: edges A_N(k) and l-stars S_{N,l}(k).

Distributions. Z ~ G(p) geometric, P(Z >= k) = (1 - p)^(k-1); Z ~ W(lam, alpha) Weibull,
P(Z >= k) = exp(-lam (k - 1)^alpha), k = 1, 2, ...

Estimator (two steps).
  step1  snapshot moments E A_N, E S_{N,la}, E S_{N,lb}            -> pi_1, rho_1, rho_2
  step2  lag-one cross moments E A_N(k) A_N(k+1), same for stars    -> E X_1, E X_2, E Z_1
  then   E Y_i = (1 - rho_i) E X_i / rho_i, E Z_2 = pi_2 E Z_1 / pi_1, and the parameters
         (p0, q0, alpha, q1, beta, q2) from the means, with the Weibull scale lam known.
"""

import numpy as np
from numba import njit
from scipy.optimize import brentq, fsolve
from scipy.special import comb


# --------------------------------------------------------------------------------------
# means of the duration distributions
# --------------------------------------------------------------------------------------

@njit
def mean_geom(p):
    return 1.0 / p


@njit
def mean_weib(lam, alpha):
    """E Z = sum_{k >= 1} P(Z >= k) = sum_{j >= 0} exp(-lam j^alpha), truncated at j < 10000."""
    k = np.arange(0, 10000)
    return np.sum(np.exp(-lam * k ** alpha))


def solve_alpha(lam, target):
    """Weibull shape alpha whose mean, with scale lam, equals `target` (nan if none in [0.01, 10])."""
    try:
        return brentq(lambda a: mean_weib(lam, a) - target, 0.01, 10.0)
    except ValueError:
        return np.nan


def _mean(p):
    return mean_geom(p['p']) if p['type'] == 'geometric' else mean_weib(p['lam'], p['alpha'])


# --------------------------------------------------------------------------------------
# simulation (inverse-transform sampling, compiled with numba)
# --------------------------------------------------------------------------------------

@njit
def seed_numba(seed):
    """Seed numba's random generator (np.random.seed outside njit code does not reach it)."""
    np.random.seed(seed)


@njit
def _geom(p):
    u = np.random.random()
    return int(np.ceil(np.log(1.0 - u) / np.log(1.0 - p)))


@njit
def _weib(lam, alpha):
    u = np.random.random()
    return max(int(np.ceil((-np.log(1.0 - u) / lam) ** (1.0 / alpha))), 1)


@njit
def _sim_regime(T, p0, q0, pi1):
    """Mode M(k) in {1, 2} for k = 0..T-1. The first sojourn is started at a uniform point of a
    fresh duration, so that the process is observed already running."""
    modes = np.zeros(T, dtype=np.int8)
    mode = 1 if np.random.random() < pi1 else 2
    full = _geom(p0) if mode == 1 else _geom(q0)
    rem = int(np.random.random() * full) + 1
    for t in range(T):
        modes[t] = mode
        rem -= 1
        if rem == 0:
            if mode == 1:
                mode, rem = 2, _geom(q0)
            else:
                mode, rem = 1, _geom(p0)
    return modes


@njit
def _sim_edges(n_edges, T, lam, alpha, p_off, rho, use_weib):
    """On/off state of every vertex pair over time (rows: pairs, columns: times). On-times are
    W(lam, alpha) if use_weib, else G(lam); off-times are G(p_off); the initial state is on with
    the stationary probability rho."""
    edges = np.zeros((n_edges, T), dtype=np.int8)
    for e in range(n_edges):
        state = 1 if np.random.random() < rho else 0
        if state == 1:
            full = _weib(lam, alpha) if use_weib else _geom(lam)
        else:
            full = _geom(p_off)
        rem = int(np.random.random() * full) + 1
        for t in range(T):
            edges[e, t] = state
            rem -= 1
            if rem == 0:
                if state == 1:
                    state, rem = 0, _geom(p_off)
                else:
                    state = 1
                    rem = _weib(lam, alpha) if use_weib else _geom(lam)
    return edges


@njit
def _count(N, T, modes, e1, e2, pi_arr, pj_arr, max_star):
    """Edge count and l-star counts, l = 3..max_star, of the observed graph at every time. An
    l-star has one centre and l - 1 leaves, so a vertex of degree d is the centre of
    C(d, l - 1) of them; the binomial coefficients are built recursively."""
    ns = max_star - 2
    ec = np.zeros(T, dtype=np.int64)
    sc = np.zeros((ns, T), dtype=np.int64)
    ne = len(pi_arr)
    for t in range(T):
        e = e1[:, t] if modes[t] == 1 else e2[:, t]
        deg = np.zeros(N, dtype=np.int64)
        na = 0
        for i in range(ne):
            if e[i] == 1:
                deg[pi_arr[i]] += 1
                deg[pj_arr[i]] += 1
                na += 1
        ec[t] = na
        for v in range(N):
            d = deg[v]
            if d < 2:
                continue
            prod = 1.0
            for k in range(1, max_star):
                if d < k:
                    break
                prod = prod * (d - k + 1) / k          # C(d, k)
                si = k - 2                             # C(d, k) counts (k + 1)-stars
                if 0 <= si < ns:
                    sc[si, t] += int(prod)
    return ec, sc


def simulate(N, T, X1, Y1, X2, Y2, p0, q0, stars):
    """Simulate T steps; returns {'edges': A_N(k), 'S3': S_{N,3}(k), ...} for l in `stars`."""
    ne = N * (N - 1) // 2
    ms = max(stars)
    EX1, EY1 = _mean(X1), _mean(Y1)
    EX2, EY2 = _mean(X2), _mean(Y2)
    r1, r2 = EX1 / (EX1 + EY1), EX2 / (EX2 + EY2)
    pi1 = (1.0 / p0) / (1.0 / p0 + 1.0 / q0)
    modes = _sim_regime(T, p0, q0, pi1)
    w1 = X1['type'] == 'weibull'
    w2 = X2['type'] == 'weibull'
    l1 = X1['lam'] if w1 else X1['p']
    a1 = X1['alpha'] if w1 else 0.0
    l2 = X2['lam'] if w2 else X2['p']
    a2 = X2['alpha'] if w2 else 0.0
    e1 = _sim_edges(ne, T, l1, a1, Y1['p'], r1, w1)
    e2 = _sim_edges(ne, T, l2, a2, Y2['p'], r2, w2)
    pi_arr = np.zeros(ne, dtype=np.int32)
    pj_arr = np.zeros(ne, dtype=np.int32)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            pi_arr[idx], pj_arr[idx] = i, j
            idx += 1
    ec, sc = _count(N, T, modes, e1, e2, pi_arr, pj_arr, ms)
    res = {'edges': ec, 'modes': modes}
    for s in stars:
        res[f'S{s}'] = sc[s - 3]
    return res


def moments(data, stars):
    """Empirical snapshot means (A0, S{l}_0) and lag-one cross moments (A1, S{l}_1)."""
    e = data['edges'].astype(np.float64)
    m = {'A0': np.mean(e), 'A1': np.mean(e[:-1] * e[1:])}
    for s in stars:
        v = data[f'S{s}'].astype(np.float64)
        m[f'S{s}_0'] = np.mean(v)
        m[f'S{s}_1'] = np.mean(v[:-1] * v[1:])
    return m


# --------------------------------------------------------------------------------------
# theoretical moments (report, equations (1), (2) and (4))
# --------------------------------------------------------------------------------------

def alm(N, l, m):
    """Ordered pairs of l-vertex sets that share exactly m vertices."""
    return comb(N, l, exact=True) * comb(N - l, l - m, exact=True) * comb(l, m, exact=True)


def th_edge(N, pi, r1, r2):
    return comb(N, 2, exact=True) * (pi * r1 + (1 - pi) * r2)


def th_star(N, l, pi, r1, r2):
    return l * comb(N, l, exact=True) * (pi * r1 ** (l - 1) + (1 - pi) * r2 ** (l - 1))


def th_edge_x(N, r1, r2, f1, f2, rn, s1, s2):
    """E A_N(k) A_N(k+1); f_i = f̄_i(1) = 1/E X_i, rn = P(mode changes), s_i = P(stays in i)."""
    n = comb(N, 2, exact=True)
    a0, a1, a2 = alm(N, 2, 0), alm(N, 2, 1), alm(N, 2, 2)
    return (rn * n ** 2 * r1 * r2
            + s1 * (a2 * r1 * (1 - f1) + (a0 + a1) * r1 ** 2)
            + s2 * (a2 * r2 * (1 - f2) + (a0 + a1) * r2 ** 2))


def th_star_x(N, l, r1, r2, f1, f2, rn, s1, s2):
    """E S_{N,l}(k) S_{N,l}(k+1), summing over the number m of shared vertices (Table 1)."""
    E = rn * (l * comb(N, l, exact=True)) ** 2 * r1 ** (l - 1) * r2 ** (l - 1)
    for m in range(l + 1):
        am = alm(N, l, m)
        for rho, fb, r in [(r1, f1, s1), (r2, f2, s2)]:
            E += r * am * ((l - m) * (l + m) * rho ** (2 * (l - 1))
                           + m * (m - 1) * rho ** (2 * l - 3) * (1 - fb)
                           + (m * rho ** (2 * l - m - 1) * (1 - fb) ** (m - 1) if m >= 1 else 0.0))
    return E


# --------------------------------------------------------------------------------------
# method of moments
# --------------------------------------------------------------------------------------

def step1(N, mom, la, lb):
    """(pi_1, rho_1, rho_2) from the snapshot moments of edges, la-stars and lb-stars: fsolve
    from a grid of starting points, keeping the converged root with the smallest residual. The
    labels are fixed so that graph 1 is the sparser one."""
    def eq(v):
        p, r1, r2 = v
        return [th_edge(N, p, r1, r2) - mom['A0'],
                th_star(N, la, p, r1, r2) - mom[f'S{la}_0'],
                th_star(N, lb, p, r1, r2) - mom[f'S{lb}_0']]
    best, br = None, np.inf
    for p in np.arange(0.15, 0.95, 0.12):
        for a in np.arange(0.08, 0.92, 0.12):
            for b in np.arange(0.08, 0.92, 0.12):
                try:
                    x, info, ier, _ = fsolve(eq, [p, a, b], full_output=True)
                    res = np.sum(info['fvec'] ** 2)
                    if ier == 1 and res < 1e-10 and all(0.01 < xi < 0.99 for xi in x):
                        pi, r1, r2 = x
                        if r1 > r2:
                            pi, r1, r2 = 1 - pi, r2, r1
                        if res < br and 0.01 < pi < 0.99:
                            best, br = (pi, r1, r2), res
                except Exception:
                    pass
    return best


def step2(N, mom, pi1, rho1, rho2, la, lb):
    """(E X_1, E X_2, E Z_1) from the lag-one cross moments, given step 1."""
    pi2 = 1 - pi1

    def eq(v):
        ex1, ex2, ez1 = v
        if ex1 <= 0 or ex2 <= 0 or ez1 <= 0:
            return [1e10] * 3
        ez2 = pi2 * ez1 / pi1
        f1, f2 = 1.0 / ex1, 1.0 / ex2
        h1, h2 = 1.0 / ez1, 1.0 / ez2
        rn = pi1 * h1 + pi2 * h2
        r1, r2 = pi1 * (1 - h1), pi2 * (1 - h2)
        return [th_edge_x(N, rho1, rho2, f1, f2, rn, r1, r2) - mom['A1'],
                th_star_x(N, la, rho1, rho2, f1, f2, rn, r1, r2) - mom[f'S{la}_1'],
                th_star_x(N, lb, rho1, rho2, f1, f2, rn, r1, r2) - mom[f'S{lb}_1']]
    best, br = None, np.inf
    for a in [1, 2, 4, 8]:
        for b in [1, 2, 4, 8]:
            for c in [1.5, 3, 5, 10]:
                try:
                    x, info, ier, _ = fsolve(eq, [a, b, c], full_output=True)
                    res = np.sum(info['fvec'] ** 2)
                    if ier == 1 and res < br and all(xi > 0.5 for xi in x):
                        best, br = x.copy(), res
                except Exception:
                    pass
    if best is None:
        return None
    ex1, ex2, ez1 = best
    return {'EX1': ex1, 'EX2': ex2, 'EZ1': ez1,
            'EZ2': pi2 * ez1 / pi1,
            'EY1': (1 - rho1) * ex1 / rho1, 'EY2': (1 - rho2) * ex2 / rho2}


def estimate(N, mom, la=3, lb=4, lam1=1.5, lam2=1.5):
    """Estimates of (p0, q0, alpha, q1, beta, q2) from the la- and lb-star moments, or None."""
    s = step1(N, mom, la, lb)
    if s is None:
        return None
    pi1, rho1, rho2 = s
    s2 = step2(N, mom, pi1, rho1, rho2, la, lb)
    if s2 is None:
        return None
    return {'p0': 1 / s2['EZ1'], 'q0': 1 / s2['EZ2'],
            'alpha': solve_alpha(lam1, s2['EX1']),
            'q1': 1 / s2['EY1'], 'beta': solve_alpha(lam2, s2['EX2']),
            'q2': 1 / s2['EY2']}
