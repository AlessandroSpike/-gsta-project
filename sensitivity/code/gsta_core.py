"""
Corrected G-STA state-level solver (Section 3 of the manuscript).

* directed, masked structural energy with the adjoint-corrected gradient T + T* (Lemma 1)
* entropic unbalanced scaling oracle (Supplement S2)
* generalized conditional gradient with the closed-form safeguarded step (Algorithm 1)
* deterministic, exchange-equivariant multi-start portfolio (Section 3.3)
* closed-form envelope gradients (Proposition 6)
"""
import hashlib
import numpy as np

LOG_FLOOR = 1e-300


class State:
    """Network state G = (mu, F, C, W)."""
    __slots__ = ("mu", "F", "C", "W")

    def __init__(self, mu, F, C, W):
        self.mu = np.asarray(mu, float)
        F = np.asarray(F, float)
        self.F = F[:, None] if F.ndim == 1 else F
        self.C = np.asarray(C, float)
        self.W = np.asarray(W, float)

    @property
    def n(self):
        return self.mu.shape[0]

    def copy(self):
        return State(self.mu.copy(), self.F.copy(), self.C.copy(), self.W.copy())

    def key(self):
        h = hashlib.sha1()
        for a in (self.mu, self.F, self.C, self.W):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()

    def sub(self, idx):
        idx = np.asarray(idx)
        return State(self.mu[idx], self.F[idx], self.C[np.ix_(idx, idx)], self.W[np.ix_(idx, idx)])


# ----------------------------------------------------------------------------------------
# structural operators (Lemma 1)
# ----------------------------------------------------------------------------------------
class StructOps:
    """T(P) = (W o C^2) P W'^T - 2 (W o C) P (W' o D)^T + W P (W' o D^2)^T and its adjoint."""

    def __init__(self, C, D, W, Wp):
        self.A1, self.A2, self.A3 = W * C * C, W * C, W
        self.B1, self.B2, self.B3 = Wp, Wp * D, Wp * D * D

    def T(self, P):
        return self.A1 @ P @ self.B1.T - 2.0 * (self.A2 @ P @ self.B2.T) + self.A3 @ P @ self.B3.T

    def Ts(self, P):
        return self.A1.T @ P @ self.B1 - 2.0 * (self.A2.T @ P @ self.B2) + self.A3.T @ P @ self.B3

    def Q(self, P):
        return float(np.sum(self.T(P) * P))


def feature_cost(F, G):
    return (np.sum(F * F, 1)[:, None] + np.sum(G * G, 1)[None, :] - 2.0 * F @ G.T).clip(min=0.0)


def kl(a, b):
    return float(np.sum(a * np.log(np.maximum(a, LOG_FLOOR) / b) - a + b))


def entropy(P):
    return float(np.sum(P * (np.log(np.maximum(P, LOG_FLOOR)) - 1.0)))


# ----------------------------------------------------------------------------------------
# unbalanced entropic scaling oracle (Supplement S2)
# ----------------------------------------------------------------------------------------
def _lse(X, axis):
    m = X.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(X - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def uot_oracle(K, mu, nu, gamma, eps, phi=None, psi=None, tol=1e-12, max_iter=200000, ti=False):
    """argmin_Q <K,Q> + gamma KL(Q1|mu) + gamma KL(Q^T1|nu) + eps E(Q).

    Log-domain unbalanced scaling (Chizat et al. 2018) with lam = gamma/(gamma+eps). With ti=True
    each sweep is followed by the closed-form optimal dual translation (phi+k, psi-k) of
    Sejourne et al. (2022), which leaves the fixed point unchanged and removes the slow
    lam^2 contraction of the near-balanced regime. Stops when the sup-norm change of the
    potentials is below tol.
    """
    lam = gamma / (gamma + eps)
    lmu, lnu = np.log(mu), np.log(nu)
    Ke = K / eps
    phi = np.zeros(K.shape[0]) if phi is None else phi.copy()
    psi = np.zeros(K.shape[1]) if psi is None else psi.copy()
    it = 0
    for it in range(max_iter):
        phi_n = lam * eps * (lmu - _lse(psi[None, :] / eps - Ke, 1))
        psi_n = lam * eps * (lnu - _lse(phi_n[:, None] / eps - Ke, 0))
        if ti:
            la = _lse((lmu - phi_n / gamma)[None, :], 1)[0]
            lb = _lse((lnu - psi_n / gamma)[None, :], 1)[0]
            k = 0.5 * gamma * (la - lb)
            phi_n, psi_n = phi_n + k, psi_n - k
        err = max(np.abs(phi_n - phi).max(), np.abs(psi_n - psi).max())
        phi, psi = phi_n, psi_n
        if err < tol:
            break
    Qm = np.exp(phi[:, None] / eps + psi[None, :] / eps - Ke)
    return Qm, phi, psi, it + 1


# ----------------------------------------------------------------------------------------
# objective
# ----------------------------------------------------------------------------------------
class Problem:
    def __init__(self, G: State, H: State, alpha, gamma, eps):
        self.G, self.H = G, H
        self.alpha, self.gamma, self.eps = alpha, gamma, eps
        self.M = feature_cost(G.F, H.F)
        self.ops = StructOps(G.C, H.C, G.W, H.W)
        self.mu, self.nu = G.mu, H.mu

    def g(self, P):
        return (self.gamma * kl(P.sum(1), self.mu) + self.gamma * kl(P.sum(0), self.nu)
                + self.eps * entropy(P))

    def parts(self, P):
        """(feature, structural, mass-KL, entropic) terms of J_eps(P)."""
        a = self.alpha
        return np.array([(1 - a) * float(np.sum(self.M * P)), a * self.ops.Q(P),
                         self.gamma * (kl(P.sum(1), self.mu) + kl(P.sum(0), self.nu)),
                         self.eps * entropy(P)])

    def J(self, P):
        return float(self.parts(P).sum())

    def grad_f(self, P):
        return (1 - self.alpha) * self.M + self.alpha * (self.ops.T(P) + self.ops.Ts(P))

    def residual(self, P):
        r, c = P.sum(1), P.sum(0)
        R = (self.grad_f(P) + self.gamma * np.log(r / self.mu)[:, None]
             + self.gamma * np.log(c / self.nu)[None, :] + self.eps * np.log(np.maximum(P, LOG_FLOOR)))
        return float(np.abs(R).max())


# ----------------------------------------------------------------------------------------
# Algorithm 1: generalized conditional gradient with safeguarded step
# ----------------------------------------------------------------------------------------
def _phi_factory(prob, P, Dd, a, b):
    rP, cP, rD, cD = P.sum(1), P.sum(0), Dd.sum(1), Dd.sum(0)
    gam, eps, mu, nu = prob.gamma, prob.eps, prob.mu, prob.nu
    gP = prob.g(P)

    def g_tau(t):
        X = P + t * Dd
        r, c = rP + t * rD, cP + t * cD
        return gam * kl(r, mu) + gam * kl(c, nu) + eps * entropy(X)

    def phi(t):
        return b * t + a * t * t + g_tau(t) - gP

    def dphi(t):
        X = P + t * Dd
        r, c = rP + t * rD, cP + t * cD
        return (b + 2 * a * t + gam * np.sum(rD * np.log(np.maximum(r, LOG_FLOOR) / mu))
                + gam * np.sum(cD * np.log(np.maximum(c, LOG_FLOOR) / nu))
                + eps * np.sum(Dd * np.log(np.maximum(X, LOG_FLOOR))))
    return phi, dphi


def _refined_step(phi, dphi, a, grid=33, n_bis=60):
    if a >= 0:                                   # convex 1-d problem: bisection on phi'
        if dphi(1.0) <= 0:
            return 1.0
        lo, hi = 0.0, 1.0
        for _ in range(n_bis):
            mid = 0.5 * (lo + hi)
            if dphi(mid) > 0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)
    ts = np.linspace(0.0, 1.0, grid)             # negative curvature: grid + golden section
    vals = np.array([phi(t) for t in ts])
    k = int(np.argmin(vals))
    lo, hi = ts[max(k - 1, 0)], ts[min(k + 1, grid - 1)]
    gr = (np.sqrt(5) - 1) / 2
    x1, x2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
    f1, f2 = phi(x1), phi(x2)
    for _ in range(n_bis):
        if f1 <= f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - gr * (hi - lo)
            f1 = phi(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + gr * (hi - lo)
            f2 = phi(x2)
    cand = [(vals[k], ts[k]), (f1, x1), (f2, x2)]
    return min(cand)[1]


class GCGResult:
    __slots__ = ("P", "J", "gap", "iters", "residual", "history", "phi", "psi", "start")


def gcg(prob: Problem, P0, tol=1e-9, max_iter=3000, grid=33, phi=None, psi=None,
        record=False):
    P = np.maximum(P0, 1e-300).astype(float).copy()
    hist = []
    gap = np.inf
    J = prob.J(P)
    for r in range(max_iter):
        K = prob.grad_f(P)
        Qm, phi, psi, _ = uot_oracle(K, prob.mu, prob.nu, prob.gamma, prob.eps, phi, psi)
        gP, gQ = prob.g(P), prob.g(Qm)
        gap = float(np.sum(K * (P - Qm)) + gP - gQ)
        J = prob.J(P)
        if gap <= tol * max(1.0, abs(J)):
            break
        Dd = Qm - P
        a = prob.alpha * prob.ops.Q(Dd)
        b = float(np.sum(K * Dd))
        tau_ub = 1.0 if a <= 0 else min(1.0, gap / (2 * a))
        phi_f, dphi_f = _phi_factory(prob, P, Dd, a, b)
        tau_ref = _refined_step(phi_f, dphi_f, a, grid=grid)
        f_ref, f_ub = phi_f(tau_ref), phi_f(tau_ub)
        tau = tau_ref if f_ref <= f_ub else tau_ub
        if record:
            hist.append(dict(it=r, J=J, gap=gap, a=a, tau=tau, refined=bool(f_ref <= f_ub),
                             decrease=-phi_f(tau),
                             bound=(min(gap / 2, gap ** 2 / (4 * a)) if a > 0 else gap)))
        P = P + tau * Dd
    res = GCGResult()
    res.P, res.J, res.gap, res.iters = P, prob.J(P), gap, r + 1
    res.residual = prob.residual(P)
    res.history, res.phi, res.psi, res.start = hist, phi, psi, None
    return res


# ----------------------------------------------------------------------------------------
# deterministic portfolio (Section 3.3, Supplement S2)
# ----------------------------------------------------------------------------------------
def _quantile_coupling(G, H):
    a = G.mu / G.mu.sum()
    b = H.mu / H.mu.sum()
    oi = np.lexsort((np.arange(G.n), G.F[:, 0]))
    oj = np.lexsort((np.arange(H.n), H.F[:, 0]))
    P = np.zeros((G.n, H.n))
    i = j = 0
    ra, rb = a[oi].copy(), b[oj].copy()
    while i < G.n and j < H.n:
        m = min(ra[i], rb[j])
        P[oi[i], oj[j]] += m
        ra[i] -= m
        rb[j] -= m
        if ra[i] <= 1e-15:
            i += 1
        if rb[j] <= 1e-15:
            j += 1
    return P * np.sqrt(G.mu.sum() * H.mu.sum()) + 1e-12


def _random_start(G, H, k):
    kg, kh = G.key(), H.key()
    flip = kg > kh
    a, b = (H, G) if flip else (G, H)
    seed = int(hashlib.sha1((min(kg, kh) + max(kg, kh) + str(k)).encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    Z = rng.lognormal(0.0, 1.0, size=(a.n, b.n))
    P = Z * np.outer(a.mu, b.mu)
    P *= np.sqrt(a.mu.sum() * b.mu.sum()) / P.sum()
    return P.T if flip else P


def portfolio_starts(prob: Problem, same_nodes: bool, n_random=0):
    G, H = prob.G, prob.H
    mu, nu = G.mu, H.mu
    S = {}
    if same_nodes:
        S["identity"] = np.diag(np.sqrt(mu * nu)) + 1e-12
    S["product"] = np.outer(mu, nu) / np.sqrt(mu.sum() * nu.sum())
    S["attribute"] = uot_oracle(prob.M, mu, nu, prob.gamma, prob.eps)[0]
    if not same_nodes:
        S["quantile"] = _quantile_coupling(G, H)
        S["anneal"] = None
    for k in range(n_random):
        S[f"random{k}"] = _random_start(G, H, k)
    return S


def solve_V(G: State, H: State, alpha, gamma, eps, same_nodes=None, tol=1e-9,
            n_random=0, starts=None, return_all=False, warm=None):
    """Portfolio estimate hat V_eps(G,H). Returns the best GCGResult (with .start)."""
    if same_nodes is None:
        same_nodes = (G.n == H.n)
    prob = Problem(G, H, alpha, gamma, eps)
    if warm is not None:                            # branch-following re-solve (diagnostics / FD)
        res = gcg(prob, warm, tol=tol)
        res.start = "warm"
        return (res, {"warm": res}) if return_all else res
    S = portfolio_starts(prob, same_nodes, n_random)
    if starts is not None:
        S = {k: v for k, v in S.items() if k in starts}
    out = {}
    for name, P0 in S.items():
        if name == "anneal":
            P = np.outer(mu_ := G.mu, H.mu) / np.sqrt(G.mu.sum() * H.mu.sum())
            for e in (1.0, 0.5, 0.2, 0.1):
                if e > eps:
                    P = gcg(Problem(G, H, alpha, gamma, e), P, tol=tol).P
            res = gcg(prob, P, tol=tol)
        else:
            res = gcg(prob, P0, tol=tol)
        res.start = name
        out[name] = res
    best = min(out.values(), key=lambda r: r.J)
    return (best, out) if return_all else best


# ----------------------------------------------------------------------------------------
# Proposition 6: envelope gradients of a branch value V(G,H) at coupling P
# ----------------------------------------------------------------------------------------
def envelope_grads(G: State, H: State, P, alpha, gamma):
    """Gradients wrt the inputs of the first (G) and second (H) argument."""
    r, c = P.sum(1), P.sum(0)
    gC = 2 * alpha * G.W * (G.C * (P @ H.W @ P.T) - P @ (H.W * H.C) @ P.T)
    gD = 2 * alpha * H.W * (H.C * (P.T @ G.W @ P) - P.T @ (G.W * G.C) @ P)
    for X in (gC, gD):
        np.fill_diagonal(X, 0.0)                      # c_ii = 0 is not a free parameter
    gF = 2 * (1 - alpha) * (r[:, None] * G.F - P @ H.F)
    gG = 2 * (1 - alpha) * (c[:, None] * H.F - P.T @ G.F)
    gmu = gamma * (1 - r / G.mu)
    gnu = gamma * (1 - c / H.mu)
    return dict(C=gC, F=gF, mu=gmu), dict(C=gD, F=gG, mu=gnu)
