"""
Unbalanced fused Gromov-Wasserstein (UFGW) discrepancy between attributed,
weighted, possibly directed relational spaces.

Objective (entropic, counting-measure reference), for a coupling P in R_+^{n x m}:

    F_eps(P) = (1-alpha) <M, P>
             + alpha * sum_{ijkl} w_ik w'_jl (c_ik - d_jl)^2 p_ij p_kl
             + gamma * KL(P 1 | mu) + gamma * KL(P^T 1 | nu)
             + eps * sum_ij p_ij (log p_ij - 1),

with M_ij = ||f_i - g_j||^2 (or any user-supplied non-negative feature cost),
generalised KL(a|b) = sum a log(a/b) - a + b, and optional observation masks
W (n x n), W' (m x m) in {0,1} (or weights) for missing edges.

The value reported is the *regularised* optimal value V_eps = F_eps(P*), for
which the envelope (implicit-function) gradients below are exact at a
non-degenerate stationary point.

Solver: generalised conditional gradient (GCG; Bredies et al. 2009;
Rakotomamonjy et al. 2015).  Smooth part f = feature + GW term, convex part
g = KL penalties + entropy.  Each iteration solves the strictly convex
entropic unbalanced OT problem  Q = argmin <grad f(P), Q> + g(Q)  by
log-domain scaling (Chizat et al. 2018) and performs an exact line search on
f + g along Q - P.  The Frank-Wolfe gap
    gap(P) = <grad f(P), P - Q> + g(P) - g(Q) >= 0
is zero iff P is a stationary point and is used as stopping criterion.

All computations are in float64.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numba import njit

__all__ = [
    "UFGWProblem", "UFGWResult", "solve_ufgw", "ufgw_multistart", "stationarity_residuals", "residual_bound",
    "feature_cost", "gw_tensor_direct",
]

_TINY = 1e-300


# --------------------------------------------------------------------------
# Low-level kernels
# --------------------------------------------------------------------------
@njit(cache=True, fastmath=False)
def _uot_scaling(K, logmu, lognu, eps, lam, phi, psi, tol, maxit):
    """Log-domain scaling iterations for
        min_Q <K,Q> + eps*sum q(log q - 1) + gamma KL(Q1|mu) + gamma KL(Q^T1|nu)
    with lam = gamma/(gamma+eps).  Q_ij = exp((phi_i + psi_j - K_ij)/eps).
    phi, psi are updated in place (warm start).  Returns #iterations and the
    final sup-norm change of the potentials."""
    n, m = K.shape
    err = 0.0
    for it in range(maxit):
        err = 0.0
        for i in range(n):
            mx = -1e308
            for j in range(m):
                v = (psi[j] - K[i, j]) / eps
                if v > mx:
                    mx = v
            s = 0.0
            for j in range(m):
                s += math.exp((psi[j] - K[i, j]) / eps - mx)
            new = lam * eps * (logmu[i] - (mx + math.log(s)))
            d = abs(new - phi[i])
            if d > err:
                err = d
            phi[i] = new
        for j in range(m):
            mx = -1e308
            for i in range(n):
                v = (phi[i] - K[i, j]) / eps
                if v > mx:
                    mx = v
            s = 0.0
            for i in range(n):
                s += math.exp((phi[i] - K[i, j]) / eps - mx)
            new = lam * eps * (lognu[j] - (mx + math.log(s)))
            d = abs(new - psi[j])
            if d > err:
                err = d
            psi[j] = new
        if err < tol:
            return it + 1, err
    return maxit, err


def _xlogx_minus_x(P):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(P > 0, P * np.log(np.maximum(P, _TINY)) - P, 0.0)
    return out.sum()


def _gkl(a, b):
    """Generalised KL(a|b) for a >= 0, b > 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(a > 0, a * np.log(np.maximum(a, _TINY) / b), 0.0)
    return float(np.sum(t - a + b))


def feature_cost(F, G):
    """Squared Euclidean cost M_ij = ||f_i - g_j||^2 (F: n x K, G: m x K)."""
    F = np.atleast_2d(np.asarray(F, float))
    G = np.atleast_2d(np.asarray(G, float))
    if F.shape[0] == 1 and F.shape[1] != G.shape[1]:
        F = F.T
    if G.shape[0] == 1 and G.shape[1] != F.shape[1]:
        G = G.T
    sq = (F * F).sum(1)[:, None] + (G * G).sum(1)[None, :] - 2.0 * F @ G.T
    return np.maximum(sq, 0.0)


def gw_tensor_direct(P, C, D, Wc=None, Wd=None):
    """O(n^2 m^2) reference: T_ij = sum_kl w_ik w'_jl (c_ik - d_jl)^2 p_kl."""
    n, m = P.shape
    Wc = np.ones((n, n)) if Wc is None else Wc
    Wd = np.ones((m, m)) if Wd is None else Wd
    L = (Wc[:, None, :, None] * Wd[None, :, None, :]
         * (C[:, None, :, None] - D[None, :, None, :]) ** 2)
    return np.einsum("ijkl,kl->ij", L, P)


# --------------------------------------------------------------------------
# Problem definition
# --------------------------------------------------------------------------
class UFGWProblem:
    """Container for one UFGW comparison G=(mu,F,C) vs H=(nu,G,D)."""

    def __init__(self, mu, nu, M, C, D, alpha=0.5, gamma=1.0, eps=0.05,
                 Wc=None, Wd=None):
        self.mu = np.asarray(mu, float).ravel()
        self.nu = np.asarray(nu, float).ravel()
        if np.any(self.mu <= 0) or np.any(self.nu <= 0):
            raise ValueError("masses must be strictly positive")
        self.M = np.asarray(M, float)
        self.C = np.asarray(C, float)
        self.D = np.asarray(D, float)
        self.alpha, self.gamma, self.eps = float(alpha), float(gamma), float(eps)
        self.n, self.m = self.M.shape
        self.masked = (Wc is not None) or (Wd is not None)
        if self.masked:
            self.Wc = np.ones((self.n, self.n)) if Wc is None else np.asarray(Wc, float)
            self.Wd = np.ones((self.m, self.m)) if Wd is None else np.asarray(Wd, float)
            self._A1 = self.Wc * self.C * self.C        # W o C^2
            self._A2 = self.Wc * self.C                 # W o C
            self._B2 = self.Wd * self.D                 # W' o D
            self._B3 = self.Wd * self.D * self.D        # W' o D^2
        else:
            self.Wc = self.Wd = None
            self._C2 = self.C * self.C
            self._D2 = self.D * self.D
        self.symmetric = (np.allclose(self.C, self.C.T) and np.allclose(self.D, self.D.T)
                          and (not self.masked or (np.allclose(self.Wc, self.Wc.T)
                                                   and np.allclose(self.Wd, self.Wd.T))))
        self.logmu = np.log(self.mu)
        self.lognu = np.log(self.nu)
        self.lam = self.gamma / (self.gamma + self.eps)

    # ---- GW linear operator T(P) (O(N^3)) ----
    # T(P)_ij  = sum_kl w_ik w'_jl (c_ik - d_jl)^2 p_kl
    # Tt(P)_ij = sum_kl w_ki w'_lj (c_ki - d_lj)^2 p_kl   (adjoint, transposed inputs)
    # For directed (asymmetric) C, D the gradient of Q(P) = <T(P), P> is
    #   T(P) + Tt(P),  NOT 2 T(P)  (the tensor is not symmetric under (ij)<->(kl)).
    def Tt(self, P):
        if self.alpha == 0.0:
            return np.zeros_like(P)
        if not self.masked:
            r = P.sum(1)
            c = P.sum(0)
            return (self._C2.T @ r)[:, None] + (self._D2.T @ c)[None, :] \
                - 2.0 * (self.C.T @ P @ self.D)
        return (self._A1.T @ P @ self.Wd) - 2.0 * (self._A2.T @ P @ self._B2) \
            + (self.Wc.T @ P @ self._B3)

    def gradQ(self, P):
        """Gradient of Q(P) = <T(P),P> (valid for asymmetric geometries)."""
        if self.symmetric:
            return 2.0 * self.T(P)
        return self.T(P) + self.Tt(P)

    def T(self, P):
        if self.alpha == 0.0:
            return np.zeros_like(P)
        if not self.masked:
            r = P.sum(1)
            c = P.sum(0)
            return (self._C2 @ r)[:, None] + (self._D2 @ c)[None, :] \
                - 2.0 * (self.C @ P @ self.D.T)
        return (self._A1 @ P @ self.Wd.T) - 2.0 * (self._A2 @ P @ self._B2.T) \
            + (self.Wc @ P @ self._B3.T)

    # ---- objective pieces ----
    def smooth(self, P, TP=None):
        TP = self.T(P) if TP is None else TP
        return (1 - self.alpha) * float((self.M * P).sum()) \
            + self.alpha * float((TP * P).sum())

    def convex(self, P):
        return self.gamma * (_gkl(P.sum(1), self.mu) + _gkl(P.sum(0), self.nu)) \
            + self.eps * _xlogx_minus_x(P)

    def objective(self, P):
        return self.smooth(P) + self.convex(P)

    def unregularised(self, P):
        """F_0(P): objective without the entropy term (the 'sharp' value)."""
        return self.smooth(P) + self.gamma * (
            _gkl(P.sum(1), self.mu) + _gkl(P.sum(0), self.nu))

    def components(self, P):
        TP = self.T(P)
        return dict(feature=(1 - self.alpha) * float((self.M * P).sum()),
                    structure=self.alpha * float((TP * P).sum()),
                    kl=self.gamma * (_gkl(P.sum(1), self.mu) + _gkl(P.sum(0), self.nu)),
                    entropy=self.eps * _xlogx_minus_x(P))

    def grad_P(self, P, TP=None):
        """Gradient of the full objective in P (for stationarity checks)."""
        GQ = self.gradQ(P)
        r, c = P.sum(1), P.sum(0)
        return (1 - self.alpha) * self.M + self.alpha * GQ \
            + self.gamma * np.log(r / self.mu)[:, None] \
            + self.gamma * np.log(c / self.nu)[None, :] \
            + self.eps * np.log(np.maximum(P, _TINY))

    # ---- entropic UOT oracle ----
    def oracle(self, K, phi, psi, tol=1e-12, maxit=20000):
        it, err = _uot_scaling(np.ascontiguousarray(K), self.logmu, self.lognu,
                               self.eps, self.lam, phi, psi, tol, maxit)
        Q = np.exp((phi[:, None] + psi[None, :] - K) / self.eps)
        return Q, it, err


@dataclass
class UFGWResult:
    value: float                 # regularised optimal value V_eps = F_eps(P*)
    P: np.ndarray                # stationary coupling
    gap: float                   # final Frank-Wolfe gap
    n_iter: int
    converged: bool
    sharp: float                 # F_0(P*) (no entropy term)
    components: dict = field(default_factory=dict)
    init: str = ""
    history: list = field(default_factory=list)


# --------------------------------------------------------------------------
# GCG solver
# --------------------------------------------------------------------------
@njit(cache=True)
def _g_along(P, D, rP, rD, cP, cD, mu, nu, gam, eps, t):
    """g(P + t D) = gam*KL(r|mu) + gam*KL(c|nu) + eps*sum x(log x - 1)."""
    n, m = P.shape
    s = 0.0
    for i in range(n):
        r = rP[i] + t * rD[i]
        if r > 0.0:
            s += gam * (r * math.log(r / mu[i]) - r + mu[i])
        else:
            s += gam * mu[i]
    for j in range(m):
        c = cP[j] + t * cD[j]
        if c > 0.0:
            s += gam * (c * math.log(c / nu[j]) - c + nu[j])
        else:
            s += gam * nu[j]
    e = 0.0
    for i in range(n):
        for j in range(m):
            x = P[i, j] + t * D[i, j]
            if x > 0.0:
                e += x * math.log(x) - x
    return s + eps * e


@njit(cache=True)
def _dg_along(P, D, rP, rD, cP, cD, mu, nu, gam, eps, t):
    n, m = P.shape
    s = 0.0
    for i in range(n):
        r = max(rP[i] + t * rD[i], 1e-300)
        s += gam * math.log(r / mu[i]) * rD[i]
    for j in range(m):
        c = max(cP[j] + t * cD[j], 1e-300)
        s += gam * math.log(c / nu[j]) * cD[j]
    e = 0.0
    for i in range(n):
        for j in range(m):
            x = max(P[i, j] + t * D[i, j], 1e-300)
            e += D[i, j] * math.log(x)
    return s + eps * e


@njit(cache=True)
def _line_search_nb(P, D, mu, nu, gam, eps, b, a, ng=33):
    rP = P.sum(axis=1); cP = P.sum(axis=0)
    rD = D.sum(axis=1); cD = D.sum(axis=0)
    if a >= 0.0:                      # phi convex on [0,1]: bisection on phi'
        if b + 2.0 * a + _dg_along(P, D, rP, rD, cP, cD, mu, nu, gam, eps, 1.0) <= 0.0:
            return 1.0
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if b + 2.0 * a * mid + _dg_along(P, D, rP, rD, cP, cD, mu, nu, gam, eps, mid) > 0.0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)
    # possibly non-convex: dense grid + golden-section refinement
    best_k = 0
    best_v = 1e308
    vals = np.empty(ng)
    for k in range(ng):
        t = k / (ng - 1.0)
        v = b * t + a * t * t + _g_along(P, D, rP, rD, cP, cD, mu, nu, gam, eps, t)
        vals[k] = v
        if v < best_v:
            best_v = v
            best_k = k
    lo = max(best_k - 1, 0) / (ng - 1.0)
    hi = min(best_k + 1, ng - 1) / (ng - 1.0)
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = hi - gr * (hi - lo)
    x2 = lo + gr * (hi - lo)
    f1 = b * x1 + a * x1 * x1 + _g_along(P, D, rP, rD, cP, cD, mu, nu, gam, eps, x1)
    f2 = b * x2 + a * x2 * x2 + _g_along(P, D, rP, rD, cP, cD, mu, nu, gam, eps, x2)
    for _ in range(60):
        if f1 < f2:
            hi = x2; x2 = x1; f2 = f1
            x1 = hi - gr * (hi - lo)
            f1 = b * x1 + a * x1 * x1 + _g_along(P, D, rP, rD, cP, cD, mu, nu, gam, eps, x1)
        else:
            lo = x1; x1 = x2; f1 = f2
            x2 = lo + gr * (hi - lo)
            f2 = b * x2 + a * x2 * x2 + _g_along(P, D, rP, rD, cP, cD, mu, nu, gam, eps, x2)
    t = 0.5 * (lo + hi)
    vt = b * t + a * t * t + _g_along(P, D, rP, rD, cP, cD, mu, nu, gam, eps, t)
    if vt <= best_v:
        return t
    return best_k / (ng - 1.0)


def _line_search(prob, P, D, b, a):
    """Exact line search of phi(tau) = b*tau + a*tau^2 + g(P + tau D) on [0,1]."""
    return float(_line_search_nb(np.ascontiguousarray(P), np.ascontiguousarray(D),
                                 prob.mu, prob.nu, prob.gamma, prob.eps, float(b), float(a), 33))


def stationarity_residuals(prob: UFGWProblem, P):
    """Independently evaluated stationarity residuals of J_eps at P:
    max-norm and Frobenius norm of grad_P J_eps, and the mass-weighted residual
    sum_ij p_ij |d J/d p_ij| (insensitive to vanishing entries)."""
    G = prob.grad_P(P)
    return dict(res_inf=float(np.abs(G).max()), res_fro=float(np.linalg.norm(G)),
                res_weighted=float(np.abs(P * G).sum()))


def residual_bound(prob: UFGWProblem, P, gap):
    """Bound of Proposition 4(c): ||grad J(P)||_F <= L_g sqrt(2 p_max gap / eps), with
    L_g = eps/p_min + gamma m / r_min + gamma n / c_min over P and Q(P)."""
    K = (1 - prob.alpha) * prob.M + prob.alpha * prob.gradQ(P)
    Q, _, _ = prob.oracle(K, np.zeros(prob.n), np.zeros(prob.m))
    pmin, pmax = min(P.min(), Q.min()), max(P.max(), Q.max())
    rmin = min(P.sum(1).min(), Q.sum(1).min()); cmin = min(P.sum(0).min(), Q.sum(0).min())
    Lg = prob.eps / pmin + prob.gamma * prob.m / rmin + prob.gamma * prob.n / cmin
    return float(Lg * math.sqrt(2.0 * pmax * max(gap, 0.0) / prob.eps))


def solve_ufgw(prob: UFGWProblem, P0=None, tol=1e-10, max_iter=3000,
               inner_tol=1e-12, record=False, init_name="", grid=33, refine=True):
    """Generalised conditional gradient for UFGW with a SAFEGUARDED step.

    Along D = Q - P the smooth part is exactly quadratic, f(P + t D) = f(P) + b t + a t^2,
    and convexity of g gives the majorant U(t) = J(P) - t gap + a t^2.  The step
    t_ub = argmin_[0,1] U (closed form) guarantees J(P + t_ub D) <= U(t_ub), which is the
    sufficient decrease used in Proposition 4.  A refined step (bisection on phi' when
    a >= 0, where phi is convex; grid + golden section when a < 0) is accepted only if it
    does at least as well as t_ub.  Returns UFGWResult; with record=True the history
    contains one dict per iteration (objective, gap, residuals, step diagnostics)."""
    n, m = prob.n, prob.m
    if P0 is None:
        P = np.outer(prob.mu, prob.nu) / math.sqrt(prob.mu.sum() * prob.nu.sum())
    else:
        P = np.maximum(np.asarray(P0, float), 1e-300)
    phi = np.zeros(n)
    psi = np.zeros(m)
    hist = []
    converged = False
    gap = np.inf
    TP = prob.T(P)
    GP = prob.gradQ(P)
    it = 0
    for it in range(1, max_iter + 1):
        grad_f = (1 - prob.alpha) * prob.M + prob.alpha * GP
        Q, _, _ = prob.oracle(grad_f, phi, psi, tol=inner_tol)
        D = Q - P
        gP, gQ = prob.convex(P), prob.convex(Q)
        gap = float((grad_f * (P - Q)).sum()) + gP - gQ
        Fval = prob.smooth(P, TP) + gP
        if gap <= tol * max(1.0, abs(Fval)):
            converged = True
            if record:
                hist.append(dict(it=it, J=Fval, gap=gap, **stationarity_residuals(prob, P)))
            break
        TD = prob.T(D)
        GD = prob.gradQ(D)
        a = prob.alpha * float((TD * D).sum())
        b = float((grad_f * D).sum())
        g_pos = max(gap, 0.0)
        t_ub = 1.0 if a <= 0.0 else min(1.0, g_pos / (2.0 * a))
        Pc, Dc = np.ascontiguousarray(P), np.ascontiguousarray(D)
        rP, cP, rD, cD = Pc.sum(1), Pc.sum(0), Dc.sum(1), Dc.sum(0)
        phi_of = lambda t: b * t + a * t * t + _g_along(Pc, Dc, rP, rD, cP, cD, prob.mu, prob.nu,
                                                         prob.gamma, prob.eps, t) - gP
        f_ub = phi_of(t_ub)
        tau, accepted = t_ub, False
        if refine:
            t_ref = float(_line_search_nb(Pc, Dc, prob.mu, prob.nu, prob.gamma, prob.eps, b, a, grid))
            f_ref = phi_of(t_ref)
            if f_ref <= f_ub:
                tau, accepted = t_ref, True
        f_new = phi_of(tau)
        if record:
            hist.append(dict(it=it, J=Fval, gap=gap, a=a, negative_curvature=bool(a < 0), tau=tau,
                             tau_ub=t_ub, refined_accepted=accepted, decrease=-f_new,
                             majorant_decrease=t_ub * g_pos - a * t_ub * t_ub,
                             sufficient_decrease_ok=bool(f_new <= -(t_ub * g_pos - a * t_ub * t_ub) + 1e-13),
                             **stationarity_residuals(prob, P)))
        if tau <= 0.0:
            break
        P = P + tau * D
        TP = TP + tau * TD          # T and gradQ are linear in P
        GP = GP + tau * GD
    res = UFGWResult(value=prob.objective(P), P=P, gap=gap, n_iter=it,
                     converged=converged, sharp=prob.unregularised(P),
                     components=prob.components(P), init=init_name, history=hist)
    return res


# --------------------------------------------------------------------------
# Deterministic multi-start
# --------------------------------------------------------------------------
def _init_couplings(prob: UFGWProblem, same_nodes: bool, which=("identity", "product", "feature")):
    out = []
    mu, nu = prob.mu, prob.nu
    if "identity" in which and same_nodes and prob.n == prob.m:
        P0 = np.full((prob.n, prob.m), 1e-12)
        P0[np.diag_indices(prob.n)] = np.sqrt(mu * nu)
        out.append(("identity", P0))
    if "product" in which:
        out.append(("product", np.outer(mu, nu) / math.sqrt(mu.sum() * nu.sum())))
    if "quantile" in which:
        # monotone (north-west corner) coupling of nodes sorted by first feature coordinate
        fi = np.argsort(prob._f1 if hasattr(prob, "_f1") and prob._f1 is not None else np.arange(prob.n))
        gj = np.argsort(prob._g1 if hasattr(prob, "_g1") and prob._g1 is not None else np.arange(prob.m))
        a = mu[fi] / mu.sum(); b = nu[gj] / nu.sum()
        Q = np.zeros((prob.n, prob.m)); i = j = 0; ra, rb = a[0], b[0]
        while i < prob.n and j < prob.m:
            q = min(ra, rb); Q[fi[i], gj[j]] += q; ra -= q; rb -= q
            if ra <= 1e-15:
                i += 1; ra = a[i] if i < prob.n else 0.0
            if rb <= 1e-15:
                j += 1; rb = b[j] if j < prob.m else 0.0
        Q = Q * math.sqrt(mu.sum() * nu.sum()) + 1e-12
        out.append(("quantile", Q))
    if "feature" in which:
        # exact solution of the alpha = 0 (convex) problem with the same cost M
        K = prob.M.copy()
        phi, psi = np.zeros(prob.n), np.zeros(prob.m)
        Q, _, _ = prob.oracle(K, phi, psi)
        out.append(("feature", Q))
    return out


def _pair_seed(prob):
    """Seed derived from the (unordered) pair of input states, so that random starts are a
    deterministic function of the pair and transposition-equivariant."""
    import hashlib
    hG = hashlib.sha1(np.ascontiguousarray(np.r_[prob.mu, prob.C.ravel()]).tobytes()).hexdigest()
    hH = hashlib.sha1(np.ascontiguousarray(np.r_[prob.nu, prob.D.ravel()]).tobytes()).hexdigest()
    a, b = sorted([hG, hH])
    return int(hashlib.sha1((a + b).encode()).hexdigest()[:12], 16), (hG <= hH)


def _random_inits(prob, K):
    seed, canonical = _pair_seed(prob)
    rng = np.random.default_rng(seed)
    out = []
    for k in range(K):
        n_, m_ = (prob.n, prob.m) if canonical else (prob.m, prob.n)
        Z = np.exp(rng.normal(size=(n_, m_)))
        Z = Z if canonical else Z.T
        P0 = Z * np.outer(prob.mu, prob.nu)
        P0 *= math.sqrt(prob.mu.sum() * prob.nu.sum()) / P0.sum()
        out.append((f"random{k}", P0))
    return out


def ufgw_multistart(prob: UFGWProblem, same_nodes=True, inits=("identity", "product", "feature"),
                    tol=1e-10, max_iter=3000, extra_inits=None, n_random=0, return_all=False):
    """Run Algorithm 1 from a fixed, data-determined portfolio of initial couplings and return
    the stationary point with the smallest regularised objective (the portfolio estimate
    hat V_eps >= V_eps).  Deterministic in the pair of states and equivariant under exchanging
    them.  With return_all=True also returns the list of all candidates (name, value, gap)."""
    best = None
    cands = _init_couplings(prob, same_nodes, inits)
    if extra_inits:
        cands = cands + list(extra_inits)
    if n_random:
        cands = cands + _random_inits(prob, n_random)
    allres = []
    for name, P0 in cands:
        r = solve_ufgw(prob, P0=P0, tol=tol, max_iter=max_iter, init_name=name)
        allres.append((name, r.value, r.gap))
        if best is None or r.value < best.value:
            best = r
    if "anneal" in inits:
        P = None
        for e in [x for x in (1.0, 0.5, 0.2, 0.1) if x > prob.eps] + [prob.eps]:
            sub = UFGWProblem(prob.mu, prob.nu, prob.M, prob.C, prob.D, alpha=prob.alpha,
                              gamma=prob.gamma, eps=e, Wc=prob.Wc if prob.masked else None,
                              Wd=prob.Wd if prob.masked else None)
            r = solve_ufgw(sub, P0=P, tol=tol, max_iter=max_iter, init_name="anneal")
            P = r.P
        r = solve_ufgw(prob, P0=P, tol=tol, max_iter=max_iter, init_name="anneal")
        allres.append(("anneal", r.value, r.gap))
        if r.value < best.value:
            best = r
    return (best, allres) if return_all else best


# --------------------------------------------------------------------------
# Envelope gradients
# --------------------------------------------------------------------------
def envelope_gradients(prob: UFGWProblem, P, F=None, G=None):
    """Closed-form gradients of V_eps at a stationary coupling P.

    Returns dict with keys C, D, mu, nu and (if F,G given) F, G.
      dV/dC  = 2 alpha [ C o (P W' P^T) - P (W' o D) P^T ] o W   (masked)
             = 2 alpha [ C o r r^T - P D P^T ]                  (unmasked)
      dV/dD  = 2 alpha [ D o c c^T - P^T C P ]                  (unmasked)
      dV/dmu = gamma (1 - r/mu),   dV/dnu = gamma (1 - c/nu)
      dV/dF  = 2 (1-alpha) [ diag(r) F - P G ]   (squared Euclidean M)
    """
    a, gam = prob.alpha, prob.gamma
    r, c = P.sum(1), P.sum(0)
    out = {}
    if not prob.masked:
        out["C"] = 2 * a * (prob.C * np.outer(r, r) - P @ prob.D @ P.T)
        out["D"] = 2 * a * (prob.D * np.outer(c, c) - P.T @ prob.C @ P)
    else:
        out["C"] = 2 * a * prob.Wc * (prob.C * (P @ prob.Wd @ P.T) - P @ prob._B2 @ P.T)
        out["D"] = 2 * a * prob.Wd * (prob.D * (P.T @ prob.Wc @ P) - P.T @ prob._A2 @ P)
    out["mu"] = gam * (1 - r / prob.mu)
    out["nu"] = gam * (1 - c / prob.nu)
    if F is not None and G is not None:
        F = np.asarray(F, float).reshape(prob.n, -1)
        G = np.asarray(G, float).reshape(prob.m, -1)
        out["F"] = 2 * (1 - a) * (r[:, None] * F - P @ G)
        out["G"] = 2 * (1 - a) * (c[:, None] * G - P.T @ F)
    return out
