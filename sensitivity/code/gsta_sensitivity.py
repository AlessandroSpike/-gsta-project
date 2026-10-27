"""
Gradient-based sensitivity of G-STA (new Section 4.5 and Supplement S8).

Implements
  * Proposition 11: gradient of the debiased state discrepancy (both self-comparison arguments)
  * Proposition 12: gradients of anomaly scores and of the doubly debiased G-STA discrepancy
  * Proposition 13: gradient of the level-adjusted lead-lag index (soft-DTW Hessian products)
  * Proposition 14: frozen-coupling remainder and first-order accuracy diagnostics
  * Corollary 2: budgeted first-order perturbations (fractional knapsack)
plus influence scores, symmetric/antisymmetric channels, regularity (branch) diagnostics,
a Hessian non-degeneracy certificate and delta-method uncertainty propagation.
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from gsta_core import State, Problem, solve_V, envelope_grads, gcg
from softdtw import soft_dtw, leadlag, leadlag_grad

BLOCKS = ("C", "F", "mu")


# ----------------------------------------------------------------------------------------
# small helpers for gradient dictionaries {C: (n,n), F: (n,K), mu: (n,)}
# ----------------------------------------------------------------------------------------
def gzero(G: State):
    return dict(C=np.zeros_like(G.C), F=np.zeros_like(G.F), mu=np.zeros_like(G.mu))


def gaxpy(acc, g, w=1.0):
    for k in BLOCKS:
        acc[k] = acc[k] + w * g[k]
    return acc


def gscale(g, w):
    return {k: w * g[k] for k in BLOCKS}


def gdot(g, d):
    return float(sum(np.sum(g[k] * d[k]) for k in BLOCKS if k in d))


# ----------------------------------------------------------------------------------------
# selection diagnostics for one state-level problem
# ----------------------------------------------------------------------------------------
class PairSolution:
    """Selected candidate of the portfolio and diagnostics of the regularity of the selection."""

    def __init__(self, best, allres, rtol=1e-7):
        self.P, self.J, self.start = best.P, best.J, best.start
        self.gap, self.residual, self.iters = best.gap, best.residual, best.iters
        self.phi, self.psi = best.phi, best.psi
        self.values = {k: r.J for k, r in allres.items()}
        thr = rtol * max(1.0, abs(best.J))
        tied = [r for r in allres.values() if r.J - best.J <= thr]
        self.n_tied = len(tied)
        scale = max(np.abs(best.P).max(), 1e-300)
        # do the tied candidates sit on the same stationary point (single branch)?
        self.tied_spread = max(np.abs(r.P - best.P).max() for r in tied) / scale
        others = sorted(r.J for r in allres.values() if r.J - best.J > thr)
        self.second_gap = (others[0] - best.J) if others else np.inf

    @property
    def regular(self):
        """Single branch: all candidates within tolerance converge to the same coupling."""
        return self.tied_spread < 1e-3


def solve_pair(G, H, prm, same_nodes=None, n_random=0):
    best, allr = solve_V(G, H, prm["alpha"], prm["gamma"], prm["eps"], same_nodes=same_nodes,
                         tol=prm.get("tol", 1e-9), n_random=n_random, return_all=True)
    return PairSolution(best, allr)


# ----------------------------------------------------------------------------------------
# Proposition 11: debiased state-level gradient
# ----------------------------------------------------------------------------------------
def grad_delta_first(G, H, P_GH, S_GG, alpha, gamma, variant="correct"):
    """Gradient of hat Delta(G,H) wrt the inputs of G.

    variant = "correct": d1 V(G,H) - 1/2 [d1 V(G,G) + d2 V(G,G)]       (Proposition 11)
              "first_only": d1 V(G,H) - 1/2 d1 V(G,G)   (omits the second self argument)
              "raw":        d1 V(G,H)                    (raw, non-debiased value)
    """
    g1, _ = envelope_grads(G, H, P_GH, alpha, gamma)
    if variant == "raw":
        return g1
    a1, a2 = envelope_grads(G, G, S_GG, alpha, gamma)
    if variant == "first_only":
        return gaxpy(dict(g1), a1, -0.5)
    out = gaxpy(dict(g1), a1, -0.5)
    return gaxpy(out, a2, -0.5)


# ----------------------------------------------------------------------------------------
# a trajectory compared with itself (anomaly scores, self-alignment)
# ----------------------------------------------------------------------------------------
class SelfTrajectory:
    def __init__(self, states, prm, n_random=0, verbose=False):
        self.states, self.prm = states, prm
        T = self.T = len(states)
        self.self_sol = [solve_pair(G, G, prm, same_nodes=True, n_random=n_random) for G in states]
        self.cross = {}
        D = np.zeros((T, T))
        for t in range(T):
            for s in range(t + 1, T):
                sol = solve_pair(states[t], states[s], prm, same_nodes=True, n_random=n_random)
                self.cross[(t, s)] = sol
                D[t, s] = D[s, t] = sol.J - 0.5 * self.self_sol[t].J - 0.5 * self.self_sol[s].J
            if verbose:
                print(f"  row {t} done")
        self.Delta = D

    def coupling(self, t, s):
        return self.cross[(t, s)].P if t < s else self.cross[(s, t)].P.T

    def grad_delta(self, t, s, variant="correct"):
        """d Delta(G_t, G_s) / d inputs of G_t."""
        a = self.prm
        return grad_delta_first(self.states[t], self.states[s], self.coupling(t, s),
                                self.self_sol[t].P, a["alpha"], a["gamma"], variant)

    def anomaly(self):
        return self.Delta.sum(1) / (self.T - 1)

    def grad_anomaly(self, t, wrt, variant="correct"):
        """d a(t) / d inputs of G_wrt."""
        T = self.T
        if wrt == t:
            acc = gzero(self.states[t])
            for s in range(T):
                if s != t:
                    gaxpy(acc, self.grad_delta(t, s, variant), 1.0 / (T - 1))
            return acc
        return gscale(self.grad_delta(wrt, t, variant), 1.0 / (T - 1))

    def regularity(self):
        sols = list(self.self_sol) + list(self.cross.values())
        return dict(all_regular=all(s.regular for s in sols),
                    max_tied_spread=max(s.tied_spread for s in sols),
                    min_second_gap=min(s.second_gap for s in sols),
                    max_gap=max(s.gap for s in sols), max_residual=max(s.residual for s in sols))


# ----------------------------------------------------------------------------------------
# two trajectories (regional G-STA and lead-lag)
# ----------------------------------------------------------------------------------------
class CrossTrajectory:
    def __init__(self, A: SelfTrajectory, B: SelfTrajectory, prm, n_random=0):
        self.A, self.B, self.prm = A, B, prm
        T, S = A.T, B.T
        self.cross = {}
        D = np.zeros((T, S))
        for t in range(T):
            for s in range(S):
                sol = solve_pair(A.states[t], B.states[s], prm, same_nodes=False, n_random=n_random)
                self.cross[(t, s)] = sol
                D[t, s] = sol.J - 0.5 * A.self_sol[t].J - 0.5 * B.self_sol[s].J
        self.Dab = D
        self.beta = 0.1 * np.median(D - D.min())       # temperature of Section 4.3

    # --- state-level pieces
    def grad_cross_A(self, t, s, variant="correct"):
        a = self.prm
        return grad_delta_first(self.A.states[t], self.B.states[s], self.cross[(t, s)].P,
                                self.A.self_sol[t].P, a["alpha"], a["gamma"], variant)

    def grad_cross_B(self, t, s, variant="correct"):
        a = self.prm
        return grad_delta_first(self.B.states[s], self.A.states[t], self.cross[(t, s)].P.T,
                                self.B.self_sol[s].P, a["alpha"], a["gamma"], variant)

    # --- Proposition 12
    def gsta(self, beta=None):
        b = self.beta if beta is None else beta
        return (soft_dtw(self.Dab, b)[0] - 0.5 * soft_dtw(self.A.Delta, b)[0]
                - 0.5 * soft_dtw(self.B.Delta, b)[0])

    def alignments(self, beta=None):
        b = self.beta if beta is None else beta
        return soft_dtw(self.Dab, b)[1], soft_dtw(self.A.Delta, b)[1], soft_dtw(self.B.Delta, b)[1]

    def grad_gsta_A(self, t, beta=None):
        Eab, Eaa, _ = self.alignments(beta)
        acc = gzero(self.A.states[t])
        for s in range(self.B.T):
            gaxpy(acc, self.grad_cross_A(t, s), Eab[t, s])
        for u in range(self.A.T):
            if u != t:
                gaxpy(acc, self.A.grad_delta(t, u), -Eaa[t, u])
        return acc

    def grad_gsta_B(self, s, beta=None):
        Eab, _, Ebb = self.alignments(beta)
        acc = gzero(self.B.states[s])
        for t in range(self.A.T):
            gaxpy(acc, self.grad_cross_B(t, s), Eab[t, s])
        for v in range(self.B.T):
            if v != s:
                gaxpy(acc, self.B.grad_delta(s, v), -Ebb[s, v])
        return acc

    # --- Proposition 13
    def tau(self):
        return leadlag(self.Dab)[0]

    def tau_grad_matrix(self):
        tau, G, b, E = leadlag_grad(self.Dab)
        return tau, G

    def grad_tau_A(self, t, Gtau=None):
        if Gtau is None:
            Gtau = self.tau_grad_matrix()[1]
        acc = gzero(self.A.states[t])
        for s in range(self.B.T):
            gaxpy(acc, self.grad_cross_A(t, s), Gtau[t, s])
        return acc

    def grad_tau_B(self, s, Gtau=None):
        if Gtau is None:
            Gtau = self.tau_grad_matrix()[1]
        acc = gzero(self.B.states[s])
        for t in range(self.A.T):
            gaxpy(acc, self.grad_cross_B(t, s), Gtau[t, s])
        return acc

    def regularity(self):
        sols = list(self.cross.values())
        return dict(n_cells=len(sols), n_regular=sum(s.regular for s in sols),
                    min_second_gap=min(s.second_gap for s in sols),
                    winners={k: sum(s.start == k for s in sols) for k in
                             ("product", "attribute", "quantile", "anneal")})


# ----------------------------------------------------------------------------------------
# Proposition 14(a): frozen-coupling remainder (explicit second-order term)
# ----------------------------------------------------------------------------------------
def frozen_remainder(G, H, P, alpha, gamma, dC=None, dF=None, dmu=None):
    """J_eps(P; G + d, H) - V - <grad V, d> for a perturbation d of the first state at fixed P."""
    r = P.sum(1)
    R = 0.0
    if dC is not None:
        R += alpha * float(np.sum(G.W * (P @ H.W @ P.T) * dC * dC))
    if dF is not None:
        R += (1 - alpha) * float(np.sum(r[:, None] * dF * dF))
    if dmu is not None:
        x = dmu / G.mu
        R += gamma * float(np.sum(r * (x - np.log1p(x))))
    return R


# ----------------------------------------------------------------------------------------
# non-degeneracy certificate: smallest eigenvalue of the Fisher-scaled Hessian
# ----------------------------------------------------------------------------------------
def hessian_certificate(G, H, P, alpha, gamma, eps, k=1):
    """lambda_min of diag(P)^{1/2} Hess_P J_eps diag(P)^{1/2}; positive => strict local minimizer."""
    prob = Problem(G, H, alpha, gamma, eps)
    n, m = P.shape
    r, c = P.sum(1), P.sum(0)
    sq = np.sqrt(P)

    def mv(y):
        Y = y.reshape(n, m)
        X = sq * Y
        HX = (alpha * (prob.ops.T(X) + prob.ops.Ts(X)) + gamma * (X.sum(1) / r)[:, None]
              + gamma * (X.sum(0) / c)[None, :] + eps * X / P)
        return (sq * HX).ravel()

    op = LinearOperator((n * m, n * m), matvec=mv, dtype=float)
    vals = eigsh(op, k=k, which="SA", tol=1e-8, maxiter=20000, return_eigenvectors=False)
    return float(np.min(vals))


# ----------------------------------------------------------------------------------------
# Corollary 2: budgeted first-order perturbations
# ----------------------------------------------------------------------------------------
def budget_allocation(score, budget, umax, direction=+1, order=None, signs=None, weights=None):
    """Solve max_u direction*<score,u> s.t. sum q|u| <= budget, |u| <= umax (fractional knapsack).

    order: optional ranking of coordinates (benchmarks); default ranks by |score|/q.
    signs: optional sign vector (default: direction*sign(score)).
    """
    s = np.asarray(score, float).ravel()
    q = np.ones_like(s) if weights is None else np.asarray(weights, float).ravel()
    if order is None:
        order = np.argsort(-np.abs(s) / q, kind="stable")
    sg = direction * np.sign(s) if signs is None else np.asarray(signs, float).ravel()
    u = np.zeros_like(s)
    rem = budget
    for k in order:
        if rem <= 0:
            break
        if sg[k] == 0:
            continue
        step = min(umax, rem / q[k])
        u[k] = sg[k] * step
        rem -= q[k] * step
    return u


# ----------------------------------------------------------------------------------------
# influence scores and symmetric/antisymmetric channels
# ----------------------------------------------------------------------------------------
def node_scores(EC, W, eta=1e-12):
    A = np.abs(EC) * (W > 0)
    np.fill_diagonal(A, 0.0)
    out, inn = A.sum(1), A.sum(0)
    return dict(S_out=out, S_in=inn, S=out + inn, B=(out - inn) / (out + inn + eta))


def sa_channels(gC, W):
    """Directional derivatives along symmetric / antisymmetric pair perturbations (Section 4.5).

    Returns D_S, D_A on pairs i<k observed in both directions and the antisymmetric share of
    the squared gradient norm on those pairs."""
    n = gC.shape[0]
    both = (W > 0) & (W.T > 0)
    iu = np.triu_indices(n, 1)
    sel = both[iu]
    gik, gki = gC[iu][sel], gC.T[iu][sel]
    DS, DA = gik + gki, gik - gki
    share = float(np.sum(DA ** 2) / max(np.sum(DS ** 2) + np.sum(DA ** 2), 1e-300))
    return DS, DA, share


def concentration(x, q=0.5):
    """Smallest share of coordinates whose |x| accounts for a fraction q of sum |x|."""
    a = np.sort(np.abs(np.asarray(x).ravel()))[::-1]
    a = a[a > 0]
    if a.size == 0:
        return np.nan
    cs = np.cumsum(a) / a.sum()
    return float((np.searchsorted(cs, q) + 1) / a.size)


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(np.ravel(a), np.ravel(b)).correlation)


def topk_overlap(a, b, k):
    ia = set(np.argsort(-np.abs(np.ravel(a)))[:k])
    ib = set(np.argsort(-np.abs(np.ravel(b)))[:k])
    return len(ia & ib) / k
