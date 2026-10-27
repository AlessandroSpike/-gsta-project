"""Additional verification requested in review (Supplement S3).

  A  tensor_gradient      explicit L_ijkl: grad Q by tensor formula and by finite differences of
                          Q(P), versus T + T* and versus 2T (masks, unequal sizes)
  B  leadlag_enumeration  exhaustive enumeration of all monotone paths: exact Gibbs tau versus the
                          soft-DTW gradient, the zero-temperature value l(T-1)/(T+l), and the
                          pathwise bound |tau| <= (T-1)(1 - T/sum E)
  C  convergence_audit    per-iteration logs of Algorithm 1: monotonicity, negative-curvature
                          frequency, accepted refined steps, sufficient decrease, grid sensitivity,
                          gap versus independently computed residuals and the KL bound
  D  multistart_stability extended portfolios with deterministic random starts: best versus
                          second-best candidates, improvements, winner switches under perturbation
Outputs: results/verif2_*.csv
"""
import os, sys, time, math, itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gsta.ufgw import (UFGWProblem, solve_ufgw, ufgw_multistart, feature_cost, gw_tensor_direct,
                       envelope_gradients, stationarity_residuals)
from gsta.softdtw import soft_dtw_grad, lead_lag, delannoy, boundary_corrected_lag
from gsta.sim import base_network, pulse, pulse_trajectory

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


# ---------------------------------------------------------------------------- A
def tensor_gradient(rng):
    rows = []
    for (n, m, masked) in [(5, 5, False), (5, 5, True), (6, 4, True)]:
        for rep in range(3):
            C, D = rng.uniform(0, 2, (n, n)), rng.uniform(0, 2, (m, m))
            np.fill_diagonal(C, 0); np.fill_diagonal(D, 0)
            W = (rng.uniform(size=(n, n)) > 0.3).astype(float) if masked else np.ones((n, n))
            Wp = (rng.uniform(size=(m, m)) > 0.3).astype(float) if masked else np.ones((m, m))
            np.fill_diagonal(W, 1); np.fill_diagonal(Wp, 1)
            P = rng.uniform(0, 1, (n, m))
            L = W[:, None, :, None] * Wp[None, :, None, :] * (C[:, None, :, None] - D[None, :, None, :]) ** 2
            Qf = lambda X: float(np.einsum("ijkl,ij,kl->", L, X, X))
            g_tensor = np.einsum("ijkl,kl->ij", L, P) + np.einsum("klij,kl->ij", L, P)
            g_fd = np.zeros_like(P); h = 1e-6
            for idx in np.ndindex(P.shape):
                Pp, Pm = P.copy(), P.copy(); Pp[idx] += h; Pm[idx] -= h
                g_fd[idx] = (Qf(Pp) - Qf(Pm)) / (2 * h)
            pr = UFGWProblem(np.ones(n), np.ones(m), np.zeros((n, m)), C, D, alpha=0.5,
                             Wc=W if masked else None, Wd=Wp if masked else None)
            g_fact = pr.gradQ(P)
            g_2T = 2 * pr.T(P)
            rows.append(dict(n=n, m=m, masked=masked, rep=rep,
                             err_factorised_vs_tensor=np.abs(g_fact - g_tensor).max() / np.abs(g_tensor).max(),
                             err_factorised_vs_fd=np.abs(g_fact - g_fd).max() / np.abs(g_fd).max(),
                             err_2T_vs_tensor=np.abs(g_2T - g_tensor).max() / np.abs(g_tensor).max()))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "verif2_tensor_gradient.csv"), index=False)
    print(df.groupby(["n", "m", "masked"]).max().drop(columns="rep").to_string())


# ---------------------------------------------------------------------------- B
def all_paths(T, S):
    out = []
    def rec(i, j, cur):
        if (i, j) == (T - 1, S - 1):
            out.append(cur[:]); return
        for di, dj in ((1, 0), (0, 1), (1, 1)):
            a, b = i + di, j + dj
            if a < T and b < S:
                cur.append((a, b)); rec(a, b, cur); cur.pop()
    rec(0, 0, [(0, 0)])
    return out


def idealised_shift(T, l, kappa=1.0):
    t = np.arange(T)
    return np.where((t[None, :] - t[:, None]) == l, 0.0, kappa)


def leadlag_enumeration(rng):
    rows = []
    for T in [3, 4, 5, 6]:
        ps = all_paths(T, T)
        offs = np.array([sum(s - t for (t, s) in p) for p in ps])
        lens = np.array([len(p) for p in ps])
        for l in range(T):
            Dm = idealised_shift(T, l)
            costs = np.array([sum(Dm[c] for c in p) for p in ps])
            for beta in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
                w = np.exp(-(costs - costs.min()) / beta); w /= w.sum()
                tau_enum = float((w * offs).sum() / (w * lens).sum())
                _, E = soft_dtw_grad(Dm, beta)
                rows.append(dict(T=T, lag=l, beta=beta, n_paths=len(ps), tau_enum=tau_enum,
                                 tau_softdtw=lead_lag(E), zero_temp=l * (T - 1) / (T + l),
                                 n_minimisers=int((costs == costs.min()).sum()),
                                 bound=(T - 1) * (1 - T / E.sum()),
                                 corrected=boundary_corrected_lag(lead_lag(E), T)))
    for T in [10, 20, 40]:
        for l in [1, 2, 3, 5]:
            for beta in [0.05, 0.1, 0.2, 0.5]:
                _, E = soft_dtw_grad(idealised_shift(T, l), beta)
                rows.append(dict(T=T, lag=l, beta=beta, n_paths=np.nan, tau_enum=np.nan,
                                 tau_softdtw=lead_lag(E), zero_temp=l * (T - 1) / (T + l), n_minimisers=np.nan,
                                 bound=(T - 1) * (1 - T / E.sum()),
                                 corrected=boundary_corrected_lag(lead_lag(E), T)))
    # pathwise bound on random cost matrices
    viol = 0
    for _ in range(2000):
        T = int(rng.integers(3, 12)); Dm = rng.exponential(size=(T, T)); beta = float(rng.choice([0.05, 0.3, 1, 3]))
        _, E = soft_dtw_grad(Dm, beta)
        viol += abs(lead_lag(E)) > (T - 1) * (1 - T / E.sum()) + 1e-12
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "verif2_leadlag_enumeration.csv"), index=False)
    e = df[df.tau_enum.notna()]
    print("max |enumeration - softdtw| = %.1e ; unique minimiser in all idealised cases: %s ; bound violations %d/2000"
          % (np.abs(e.tau_enum - e.tau_softdtw).max(), bool((e.n_minimisers == 1).all()), viol))
    piv = df[df["T"].isin([6, 10, 40])].pivot_table(index=["T", "lag"], columns="beta", values="tau_softdtw")
    piv["zero_temp"] = df[df["T"].isin([6, 10, 40])].groupby(["T", "lag"]).zero_temp.first()
    print(piv.round(4).to_string())


# ---------------------------------------------------------------------------- C
def instance(rng, kind):
    if kind == "same-node directed":
        n = m = 10
    elif kind == "same-node masked":
        n = m = 10
    else:
        n, m = 9, 6
    mu, nu = rng.uniform(0.05, 0.25, n), rng.uniform(0.05, 0.25, m)
    F, G = rng.normal(size=(n, 1)), rng.normal(size=(m, 1))
    if n == m:
        G = F + 0.3 * rng.normal(size=(n, 1))
    C, D = rng.uniform(0, 2, (n, n)), rng.uniform(0, 2, (m, m))
    np.fill_diagonal(C, 0); np.fill_diagonal(D, 0)
    W = Wp = None
    if kind == "same-node masked":
        W = (rng.uniform(size=(n, n)) > 0.2).astype(float); np.fill_diagonal(W, 1)
        Wp = (rng.uniform(size=(m, m)) > 0.2).astype(float); np.fill_diagonal(Wp, 1)
    return UFGWProblem(mu, nu, feature_cost(F, G), C, D, alpha=0.5, gamma=1.0, eps=0.05, Wc=W, Wd=Wp)


def convergence_audit(rng):
    it_rows, run_rows = [], []
    for kind in ["same-node directed", "same-node masked", "different node sets"]:
        for rep in range(15):
            pr = instance(rng, kind)
            for grid in [33, 513]:
                for init in ["product", "random"]:
                    P0 = None if init == "product" else np.exp(rng.normal(size=(pr.n, pr.m))) * np.outer(pr.mu, pr.nu)
                    t0 = time.perf_counter()
                    r = solve_ufgw(pr, P0=P0, tol=1e-12, record=True, grid=grid)
                    el = time.perf_counter() - t0
                    H = r.history
                    J = np.array([h["J"] for h in H])
                    steps = [h for h in H if "tau" in h]
                    run_rows.append(dict(kind=kind, rep=rep, grid=grid, init=init, iters=r.n_iter, converged=r.converged,
                                         value=r.value, final_gap=r.gap, time=el,
                                         monotone=bool(np.all(np.diff(J) <= 1e-13)),
                                         frac_negative_curvature=np.mean([h["negative_curvature"] for h in steps]) if steps else np.nan,
                                         frac_refined_accepted=np.mean([h["refined_accepted"] for h in steps]) if steps else np.nan,
                                         all_sufficient_decrease=all(h["sufficient_decrease_ok"] for h in steps),
                                         **stationarity_residuals(pr, r.P),
                                         kl_gap_check=r.gap >= 0))
                    if rep == 0 and grid == 33:
                        for h in H:
                            it_rows.append(dict(kind=kind, init=init, **{k: v for k, v in h.items()}))
    rr = pd.DataFrame(run_rows); rr.to_csv(os.path.join(OUT, "verif2_convergence_runs.csv"), index=False)
    pd.DataFrame(it_rows).to_csv(os.path.join(OUT, "verif2_convergence_iterations.csv"), index=False)
    g = rr.pivot_table(index=["kind", "rep", "init"], columns="grid", values="value")
    print("max |value(grid 513) - value(grid 33)| = %.1e" % np.abs(g[513] - g[33]).max())
    print(rr.groupby("kind")[["converged", "monotone", "all_sufficient_decrease", "frac_negative_curvature",
                              "frac_refined_accepted", "iters", "final_gap", "res_inf", "res_weighted", "time"]]
          .agg(["median", "max"]).T.round(6).to_string())


# ---------------------------------------------------------------------------- D
def multistart_stability(rng, n_pairs=12, n_random=20):
    rows = []
    for k in range(n_pairs):
        r_ = np.random.default_rng(500 + k)
        T = 6
        G = pulse_trajectory(base_network(8, r_), r_, pulse(T, 2, 1.0), sigma=0.25)
        H = pulse_trajectory(base_network(6, r_), r_, pulse(T, 3, 1.0), sigma=0.25)
        g, h = G[int(r_.integers(6))], H[int(r_.integers(6))]
        pr = UFGWProblem(g.mu, h.mu, feature_cost(g.F, h.F), g.C, h.C, alpha=0.5, gamma=1.0, eps=0.05)
        pr._f1, pr._g1 = g.F[:, 0], h.F[:, 0]
        base, cand = ufgw_multistart(pr, same_nodes=False, inits=("product", "feature", "quantile", "anneal"),
                                     return_all=True)
        ext, cand_e = ufgw_multistart(pr, same_nodes=False, inits=("product", "feature", "quantile", "anneal"),
                                      n_random=n_random, return_all=True)
        vals = np.sort([v for _, v, _ in cand_e])
        # perturbation: tiny relative noise on the geometry; does the winner change?
        switches, grad_jumps = 0, []
        gC0 = envelope_gradients(pr, base.P)["C"]
        for rep in range(5):
            Cp = g.C * (1 + 1e-6 * r_.normal(size=g.C.shape)); np.fill_diagonal(Cp, 0)
            prp = UFGWProblem(g.mu, h.mu, feature_cost(g.F, h.F), Cp, h.C, alpha=0.5, gamma=1.0, eps=0.05)
            prp._f1, prp._g1 = g.F[:, 0], h.F[:, 0]
            bp = ufgw_multistart(prp, same_nodes=False, inits=("product", "feature", "quantile", "anneal"))
            switches += bp.init != base.init
            grad_jumps.append(np.abs(envelope_gradients(prp, bp.P)["C"] - gC0).max() / np.abs(gC0).max())
        rows.append(dict(pair=k, baseline_best=base.value, baseline_winner=base.init, extended_best=ext.value,
                         extended_winner=ext.init, improvement=base.value - ext.value,
                         rel_improvement=(base.value - ext.value) / abs(base.value),
                         best_minus_second=vals[1] - vals[0], n_distinct_values=int(len(np.unique(np.round(vals, 8)))),
                         winner_switches_under_perturbation=switches, max_rel_gradient_change=max(grad_jumps)))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "verif2_multistart.csv"), index=False)
    print(df.round(6).to_string())


if __name__ == "__main__":
    rng = np.random.default_rng(31)
    todo = sys.argv[1:] or ["A", "B", "C", "D"]
    for key, f in [("A", tensor_gradient), ("B", leadlag_enumeration), ("C", convergence_audit), ("D", multistart_stability)]:
        if key in todo:
            t0 = time.time(); print("=" * 60, f.__name__); f(rng); print("(%.0fs)" % (time.time() - t0))
