"""Automated checks of the mathematical identities used in the paper (run: python tests/test_identities.py)."""
import os, sys, itertools
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gsta.ufgw import UFGWProblem, solve_ufgw, ufgw_multistart, feature_cost, gw_tensor_direct, envelope_gradients
from gsta.softdtw import soft_dtw_grad, lead_lag, delannoy
from gsta.data import GraphState
from gsta.core import Trajectory, gsta
rng = np.random.default_rng(0)
def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name); assert ok, name
# Lemma 1: factorization (masked, n != m) and adjoint gradient vs explicit tensor
n, m = 6, 4
C, D, P = rng.uniform(0, 2, (n, n)), rng.uniform(0, 2, (m, m)), rng.uniform(size=(n, m))
W = (rng.uniform(size=(n, n)) > .3).astype(float); Wp = (rng.uniform(size=(m, m)) > .3).astype(float)
np.fill_diagonal(W, 1); np.fill_diagonal(Wp, 1)
pr = UFGWProblem(np.ones(n), np.ones(m), np.zeros((n, m)), C, D, alpha=.5, Wc=W, Wd=Wp)
L = W[:, None, :, None] * Wp[None, :, None, :] * (C[:, None, :, None] - D[None, :, None, :]) ** 2
check("Lemma 1(c) masked factorization", np.allclose(pr.T(P), gw_tensor_direct(P, C, D, W, Wp), atol=1e-12))
check("Lemma 1(a) gradient T+T* = tensor gradient", np.allclose(pr.gradQ(P), np.einsum("ijkl,kl->ij", L, P) + np.einsum("klij,kl->ij", L, P), atol=1e-12))
check("Lemma 1(b) 2T differs for directed geometries", np.abs(2 * pr.T(P) - pr.gradQ(P)).max() > 1e-3)
# Proposition 2: directed energy decomposition (symmetric masks)
Ws = np.ones((n, n)); Wps = np.ones((m, m))
q = lambda A, B: float((UFGWProblem(np.ones(n), np.ones(m), np.zeros((n, m)), A, B, alpha=.5).T(P) * P).sum())
check("Proposition 2 decomposition", abs(q(C, D) - q((C + C.T) / 2, (D + D.T) / 2) - q((C - C.T) / 2, (D - D.T) / 2)) < 1e-10)
# Propositions 4-5: stationarity and envelope gradient (one entry, re-solved central difference)
k = 6; mu, nu = rng.uniform(.05, .25, k), rng.uniform(.05, .25, k); F = rng.normal(size=(k, 1)); G = F + .3 * rng.normal(size=(k, 1))
C = rng.uniform(0, 1, (k, k)); D = np.clip(C + .3 * rng.normal(size=(k, k)), 0, None); np.fill_diagonal(C, 0); np.fill_diagonal(D, 0)
mk = lambda Cx: UFGWProblem(mu, nu, feature_cost(F, G), Cx, D, alpha=.5, gamma=1., eps=.02)
r = solve_ufgw(mk(C), tol=1e-15, max_iter=50000)
check("Proposition 4 gap at termination", r.gap < 1e-12 and r.converged)
g = envelope_gradients(mk(C), r.P)["C"]; h = 1e-5; Cp, Cm = C.copy(), C.copy(); Cp[0, 1] += h; Cm[0, 1] -= h
fd = (solve_ufgw(mk(Cp), P0=r.P, tol=1e-15).value - solve_ufgw(mk(Cm), P0=r.P, tol=1e-15).value) / (2 * h)
check("Proposition 5 envelope gradient", abs(fd - g[0, 1]) < 1e-6 * max(1, abs(fd)))
# symmetry of the portfolio estimate
p1 = UFGWProblem(mu, nu, feature_cost(F, G), C, D, alpha=.5, gamma=1., eps=.02); p2 = UFGWProblem(nu, mu, feature_cost(G, F), D, C, alpha=.5, gamma=1., eps=.02)
check("symmetry of the portfolio estimate", abs(ufgw_multistart(p1, n_random=2).value - ufgw_multistart(p2, n_random=2).value) < 1e-10)
# Proposition 6: soft-DTW gradient = Gibbs expectation (enumeration), Delannoy count
def paths(T, S):
    out = []
    def rec(i, j, cur):
        if (i, j) == (T - 1, S - 1): out.append(cur[:]); return
        for di, dj in ((1, 0), (0, 1), (1, 1)):
            if i + di < T and j + dj < S: cur.append((i + di, j + dj)); rec(i + di, j + dj, cur); cur.pop()
    rec(0, 0, [(0, 0)]); return out
Dm = rng.uniform(size=(4, 5)); ps = paths(4, 5); beta = .3
c = np.array([sum(Dm[x] for x in p) for p in ps]); wgt = np.exp(-(c - c.min()) / beta); wgt /= wgt.sum()
E = np.zeros_like(Dm)
for w_, p in zip(wgt, ps):
    for x in p: E[x] += w_
check("Proposition 6(a) Delannoy count", len(ps) == delannoy(3, 4))
check("Proposition 6(c) gradient = expected alignment", np.allclose(soft_dtw_grad(Dm, beta)[1], E, atol=1e-12))
# Proposition 9: antisymmetry, pathwise bound, idealized shift
E1 = soft_dtw_grad(Dm, beta)[1]; E2 = soft_dtw_grad(Dm.T.copy(), beta)[1]
check("Proposition 9(a) antisymmetry", abs(lead_lag(E1) + lead_lag(E2)) < 1e-12)
ok = True
for _ in range(200):
    T = int(rng.integers(3, 10)); X = rng.exponential(size=(T, T)); Eb = soft_dtw_grad(X, float(rng.choice([.05, .5, 2.])))[1]
    ok &= abs(lead_lag(Eb)) <= (T - 1) * (1 - T / Eb.sum()) + 1e-12
check("Proposition 9(c) pathwise bound", ok)
T = 10; t = np.arange(T)
for l, target in [(2, 2 * 9 / 12), (6, 0.0)]:
    Ds = np.where(t[None, :] - t[:, None] == l, 0., 1.)
    check("Proposition 9(d) idealized shift l=%d" % l, abs(lead_lag(soft_dtw_grad(Ds, .02)[1]) - target) < 1e-6)
# G-STA of a trajectory against an independent copy is exactly zero
states = [GraphState(rng.uniform(.1, 1, 5), rng.normal(size=(5, 1)), rng.uniform(size=(5, 5)) * (1 - np.eye(5)), None) for _ in range(4)]
copy = [GraphState(s.mu.copy(), s.F.copy(), s.C.copy(), None) for s in states]
check("G-STA(G, copy of G) = 0", abs(gsta(Trajectory(states), Trajectory(copy), beta=.05)["value"]) < 1e-12)
print("all checks passed")
