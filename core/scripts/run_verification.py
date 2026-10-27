"""S1: numerical verification of the UFGW / soft-DTW components (Section 4.1, Supplement S3).

Outputs (results/):
  verif_gradients.csv     envelope gradients vs re-solved central differences
  verif_algorithms.csv    stationarity of GCG vs (i) original update, (ii) eq.(8) 2T linearisation
  verif_scaling.csv       O(N^4) vs O(N^3) contraction time, full solve time vs N
  verif_limits.csv        alpha=0 vs POT, gamma and epsilon sweeps, beta->0
  verif_softdtw.csv       soft-DTW vs brute-force path enumeration
  verif_symmetry.csv      V(G,H) vs V(H,G); Delta(G,G') for copies; tau antisymmetry
  verif_convergence.csv   objective and gap traces (GCG vs original update)
"""
import os, sys, time, math, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gsta.ufgw import (UFGWProblem, solve_ufgw, ufgw_multistart, feature_cost,
                       gw_tensor_direct, envelope_gradients)
from gsta.softdtw import soft_dtw, soft_dtw_grad, hard_dtw, delannoy, lead_lag
from gsta.data import GraphState
from gsta.core import Trajectory, gsta, value_matrix

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
PARS = dict(alpha=0.5, gamma=1.0, eps=0.02)


def instance(rng, n, m, masked=False, asym=True):
    mu = rng.uniform(0.05, 0.25, n)
    nu = rng.uniform(0.05, 0.25, m)
    F = rng.normal(size=(n, 1))
    G = rng.normal(size=(m, 1)) if m != n else F + 0.3 * rng.normal(size=(n, 1))
    C = rng.uniform(0, 1, (n, n))
    D = rng.uniform(0, 1, (m, m)) if m != n else np.clip(C + 0.3 * rng.normal(size=(n, n)), 0, None)
    if not asym:
        C = 0.5 * (C + C.T)
        D = 0.5 * (D + D.T)
    np.fill_diagonal(C, 0)
    np.fill_diagonal(D, 0)
    Wc = Wd = None
    if masked:
        Wc = (rng.uniform(size=(n, n)) > 0.2).astype(float); np.fill_diagonal(Wc, 1)
        Wd = (rng.uniform(size=(m, m)) > 0.2).astype(float); np.fill_diagonal(Wd, 1)
    return dict(mu=mu, nu=nu, F=F, G=G, C=C, D=D, Wc=Wc, Wd=Wd)


def problem(d, **over):
    a = {**d, **over}
    return UFGWProblem(a["mu"], a["nu"], feature_cost(a["F"], a["G"]), a["C"], a["D"],
                       Wc=a["Wc"], Wd=a["Wd"], **PARS)


# ------------------------------------------------------------------ gradients
def exp_gradients(rng):
    rows = []
    for label, (n, m, masked) in {"directed": (8, 8, False), "directed+mask": (8, 8, True),
                                  "different sizes": (8, 6, False)}.items():
        d = instance(rng, n, m, masked)
        pr = problem(d)
        r = solve_ufgw(pr, tol=1e-15, max_iter=50000)
        g = envelope_gradients(pr, r.P, d["F"], d["G"])
        h = 1e-5
        for name in ["C", "D", "F", "G", "mu", "nu"]:
            arr = d[name]
            num = np.zeros_like(arr)
            for idx in np.ndindex(arr.shape):
                ap, am = arr.copy(), arr.copy()
                ap[idx] += h; am[idx] -= h
                vp = solve_ufgw(problem(d, **{name: ap}), P0=r.P, tol=1e-15, max_iter=50000).value
                vm = solve_ufgw(problem(d, **{name: am}), P0=r.P, tol=1e-15, max_iter=50000).value
                num[idx] = (vp - vm) / (2 * h)
            ana = g[name].reshape(arr.shape)
            rows.append(dict(instance=label, block=name, max_abs_err=np.abs(ana - num).max(),
                             rel_err=np.abs(ana - num).max() / np.abs(num).max(),
                             gap=r.gap))
        if not masked and n == m:
            rr, cc = r.P.sum(1), r.P.sum(0)
            wrong = 2 * PARS["alpha"] * (d["C"] * np.outer(rr, cc) - r.P @ d["D"] @ r.P.T)
            rows.append(dict(instance=label, block="C (mixed marginals r c^T)",
                             max_abs_err=np.abs(wrong - g["C"]).max(),
                             rel_err=np.abs(wrong - g["C"]).max() / np.abs(g["C"]).max(), gap=r.gap))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "verif_gradients.csv"), index=False)
    print(pd.DataFrame(rows).to_string())


# ------------------------------------------------------------------ algorithms
def original_update(pr, tol=1e-12, max_iter=5000, trace=None):
    """The ORIGINAL update rule (float64): oracle cost (1-a)M + a T(P) (no factor 2, no adjoint),
    step size from the quadratic model of the smooth part only, stop on max|dP|."""
    P = np.outer(pr.mu, pr.nu) / math.sqrt(pr.mu.sum() * pr.nu.sum())
    phi, psi = np.zeros(pr.n), np.zeros(pr.m)
    for it in range(max_iter):
        TP = pr.T(P)
        K = (1 - pr.alpha) * pr.M + pr.alpha * TP
        Q, _, _ = pr.oracle(K, phi, psi)
        D = Q - P
        TD = pr.T(D)
        A = pr.alpha * float((TD * D).sum())
        B = (1 - pr.alpha) * float((pr.M * D).sum()) + pr.alpha * (float((TP * D).sum()) + float((TD * P).sum()))
        tau = 1.0 if A <= 1e-15 else float(np.clip(-B / (2 * A), 0, 1))
        Pn = P + tau * D
        if trace is not None:
            trace.append((it, pr.objective(Pn)))
        if np.abs(Pn - P).max() < tol:
            P = Pn
            break
        P = Pn
    return P


def symmetric_linearisation(pr, tol=1e-13, max_iter=5000):
    """GCG with the paper's eq.(8) gradient 2*alpha*T(P) (exact only for symmetric C, D)."""
    P = np.outer(pr.mu, pr.nu) / math.sqrt(pr.mu.sum() * pr.nu.sum())
    phi, psi = np.zeros(pr.n), np.zeros(pr.m)
    for it in range(max_iter):
        K = (1 - pr.alpha) * pr.M + 2 * pr.alpha * pr.T(P)
        Q, _, _ = pr.oracle(K, phi, psi)
        if np.abs(Q - P).max() < tol:
            return Q
        P = 0.5 * P + 0.5 * Q            # damped fixed-point iteration
    return P


def fw_gap(pr, P):
    grad_f = (1 - pr.alpha) * pr.M + pr.alpha * pr.gradQ(P)
    Q, _, _ = pr.oracle(grad_f, np.zeros(pr.n), np.zeros(pr.m))
    return float((grad_f * (P - Q)).sum()) + pr.convex(P) - pr.convex(Q)


def exp_algorithms(rng):
    rows, traces = [], []
    for rep in range(20):
        d = instance(rng, 10, 10, masked=False, asym=True)
        pr = problem(d)
        r = solve_ufgw(pr, tol=1e-13, max_iter=50000, record=(rep == 0))
        tr = []
        Po = original_update(pr, trace=tr if rep == 0 else None)
        Ps = symmetric_linearisation(pr)
        for name, P in [("GCG (this paper)", r.P), ("original update", Po), ("eq.(8) 2T linearisation", Ps)]:
            rows.append(dict(rep=rep, method=name, objective=pr.objective(P),
                             excess=pr.objective(P) - r.value, fw_gap=fw_gap(pr, P),
                             grad_norm=float(np.abs(pr.grad_P(P)).max())))
        if rep == 0:
            for h_ in r.history:
                traces.append(dict(method="GCG", iter=h_["it"], objective=h_["J"], gap=h_["gap"]))
            for it, F in tr:
                traces.append(dict(method="original", iter=it + 1, objective=F, gap=np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "verif_algorithms.csv"), index=False)
    pd.DataFrame(traces).to_csv(os.path.join(OUT, "verif_convergence.csv"), index=False)
    print(df.groupby("method")[["excess", "fw_gap", "grad_norm"]].agg(["median", "max"]).to_string())


# ------------------------------------------------------------------ scaling
def exp_scaling(rng):
    rows = []
    for n in [8, 16, 24, 32, 40, 48]:
        C, D, P = rng.uniform(size=(n, n)), rng.uniform(size=(n, n)), rng.uniform(size=(n, n))
        pr = UFGWProblem(np.ones(n), np.ones(n), np.zeros((n, n)), C, D, alpha=0.5)
        t0 = time.perf_counter(); Td = gw_tensor_direct(P, C, D); td = time.perf_counter() - t0
        reps = 50
        t0 = time.perf_counter()
        for _ in range(reps):
            Tf = pr.T(P)
        tf = (time.perf_counter() - t0) / reps
        rows.append(dict(kind="contraction", N=n, t_direct=td, t_factorised=tf,
                         max_err=np.abs(Td - Tf).max()))
    for n in [64, 128, 256, 512, 1024]:
        C, D, P = rng.uniform(size=(n, n)), rng.uniform(size=(n, n)), rng.uniform(size=(n, n))
        pr = UFGWProblem(np.ones(n), np.ones(n), np.zeros((n, n)), C, D, alpha=0.5)
        reps = max(2, int(2000 / n))
        t0 = time.perf_counter()
        for _ in range(reps):
            pr.T(P)
        rows.append(dict(kind="contraction", N=n, t_direct=np.nan,
                         t_factorised=(time.perf_counter() - t0) / reps, max_err=np.nan))
    for n in [25, 50, 100, 200, 400]:
        d = instance(rng, n, n, masked=True)
        d["mu"] = d["mu"] / d["mu"].sum(); d["nu"] = d["nu"] / d["nu"].sum()
        pr = UFGWProblem(d["mu"], d["nu"], feature_cost(d["F"], d["G"]), d["C"], d["D"],
                         Wc=d["Wc"], Wd=d["Wd"], alpha=0.5, gamma=1.0, eps=0.05)
        t0 = time.perf_counter(); r = solve_ufgw(pr, tol=1e-9); ts = time.perf_counter() - t0
        rows.append(dict(kind="full solve (masked)", N=n, t_direct=np.nan, t_factorised=ts,
                         max_err=r.gap, iters=r.n_iter))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "verif_scaling.csv"), index=False)
    for kind in df.kind.unique():
        s = df[df.kind == kind]
        b = np.polyfit(np.log(s.N), np.log(s.t_factorised), 1)[0]
        print(kind, "log-log slope (factorised/solve) %.2f" % b)
    s = df[(df.kind == "contraction") & df.t_direct.notna()]
    print("direct slope %.2f" % np.polyfit(np.log(s.N), np.log(s.t_direct), 1)[0])
    print(df.to_string())


# ------------------------------------------------------------------ limits
def exp_limits(rng):
    import ot
    rows = []
    n, m = 8, 10
    mu, nu = rng.uniform(0.5, 2, n), rng.uniform(0.5, 2, m)
    F, G = rng.normal(size=(n, 2)), rng.normal(size=(m, 2))
    M = feature_cost(F, G)
    pr = UFGWProblem(mu, nu, M, np.zeros((n, n)), np.zeros((m, m)), alpha=0.0, gamma=1.0, eps=0.05)
    r = solve_ufgw(pr, tol=1e-14)
    Ppot = ot.unbalanced.sinkhorn_unbalanced(mu, nu, M, 0.05, 1.0, reg_type="entropy",
                                             numItermax=200000, stopThr=1e-15)
    rows.append(dict(check="alpha=0 vs POT sinkhorn_unbalanced", param=0.0,
                     value=float(np.abs(r.P - Ppot).max())))
    d = instance(rng, 8, 8)
    for gam in [0.01, 0.1, 1, 10, 100, 1000]:
        prg = UFGWProblem(np.full(8, 1 / 8), np.full(8, 1 / 8), feature_cost(d["F"], d["G"]),
                          d["C"], d["D"], alpha=0.5, gamma=gam, eps=0.02)
        rg = ufgw_multistart(prg, tol=1e-12)
        rows.append(dict(check="gamma sweep: max marginal violation", param=gam,
                         value=float(max(np.abs(rg.P.sum(1) - 1 / 8).max(), np.abs(rg.P.sum(0) - 1 / 8).max()))))
    for eps in [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]:
        pre = UFGWProblem(d["mu"], d["nu"], feature_cost(d["F"], d["G"]), d["C"], d["D"],
                          alpha=0.5, gamma=1.0, eps=eps)
        t0 = time.perf_counter(); re = ufgw_multistart(pre, tol=1e-12); te = time.perf_counter() - t0
        Pn = re.P / re.P.sum()
        rows.append(dict(check="eps sweep: sharp value F0(P_eps)", param=eps, value=re.sharp))
        rows.append(dict(check="eps sweep: entropy of normalised coupling", param=eps,
                         value=float(-np.sum(np.where(Pn > 0, Pn * np.log(np.maximum(Pn, 1e-300)), 0.0)))))
        rows.append(dict(check="eps sweep: wall time (s)", param=eps, value=te))
    Dm = rng.uniform(size=(10, 12))
    hd = hard_dtw(Dm)
    for beta in [1.0, 0.1, 0.01, 0.001, 1e-4]:
        rows.append(dict(check="beta sweep: hard DTW - soft DTW", param=beta, value=hd - soft_dtw(Dm, beta)))
        rows.append(dict(check="beta sweep: bound beta*log(Delannoy)", param=beta,
                         value=beta * math.log(delannoy(9, 11))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "verif_limits.csv"), index=False)
    print(df.to_string())


# ------------------------------------------------------------------ soft-DTW
def exp_softdtw(rng):
    def paths(T, S):
        out = []
        def rec(i, j, cur):
            if (i, j) == (T - 1, S - 1):
                out.append(cur[:]); return
            for di, dj in [(1, 0), (0, 1), (1, 1)]:
                a, b = i + di, j + dj
                if a < T and b < S:
                    cur.append((a, b)); rec(a, b, cur); cur.pop()
        rec(0, 0, [(0, 0)])
        return out
    rows = []
    for (T, S) in [(3, 3), (4, 4), (4, 6), (5, 5), (6, 6)]:
        D = rng.uniform(0, 1, (T, S)); beta = 0.2
        ps = paths(T, S)
        costs = np.array([sum(D[a] for a in p) for p in ps])
        val = -beta * np.log(np.exp(-costs / beta).sum())
        w = np.exp(-(costs - costs.min()) / beta); w /= w.sum()
        E = np.zeros((T, S))
        for wk, p in zip(w, ps):
            for a in p:
                E[a] += wk
        v, Eh = soft_dtw_grad(D, beta)
        rows.append(dict(T=T, S=S, n_paths=len(ps), delannoy=delannoy(T - 1, S - 1),
                         value_err=abs(v - val), grad_err=np.abs(E - Eh).max()))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "verif_softdtw.csv"), index=False)
    print(df.to_string())


# ------------------------------------------------------------------ symmetry
def exp_symmetry(rng):
    rows = []
    for rep in range(10):
        d = instance(rng, 9, 9)
        p1 = UFGWProblem(d["mu"], d["nu"], feature_cost(d["F"], d["G"]), d["C"], d["D"], **PARS)
        p2 = UFGWProblem(d["nu"], d["mu"], feature_cost(d["G"], d["F"]), d["D"], d["C"], **PARS)
        v1, v2 = ufgw_multistart(p1).value, ufgw_multistart(p2).value
        rows.append(dict(check="|V(G,H)-V(H,G)|", rep=rep, value=abs(v1 - v2)))
    # trajectory level: Delta(G, G') with G' an independent copy is exactly zero; tau antisymmetry
    states = []
    for t in range(6):
        d = instance(rng, 7, 7)
        states.append(GraphState(d["mu"], d["F"], d["C"], None))
    copy = [GraphState(s.mu.copy(), s.F.copy(), s.C.copy(), None) for s in states]
    TG, TC = Trajectory(states, **PARS), Trajectory(copy, **PARS)
    res = gsta(TG, TC, beta=0.05)
    rows.append(dict(check="max|Delta(G,G copy)| diagonal", rep=0, value=float(np.abs(np.diag(res["Delta"])).max())))
    rows.append(dict(check="doubly debiased G-STA(G, G copy)", rep=0, value=abs(res["value"])))
    other = []
    for t in range(6):
        d = instance(rng, 7, 7)
        other.append(GraphState(d["mu"], d["F"], d["C"], None))
    TH = Trajectory(other, **PARS)
    r1 = gsta(TG, TH, beta=0.05)
    r2 = gsta(TH, TG, beta=0.05, Vcross=r1["V"].T)
    rows.append(dict(check="tau(G,H)+tau(H,G)", rep=0, value=r1["tau"] + r2["tau"]))
    rows.append(dict(check="tau(G,G)", rep=0, value=lead_lag(soft_dtw_grad(TG.Delta_self, 0.05)[1])))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "verif_symmetry.csv"), index=False)
    print(df.groupby("check").value.max().to_string())


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    t0 = time.time()
    todo = sys.argv[1:]
    for f in [exp_softdtw, exp_symmetry, exp_limits, exp_algorithms, exp_gradients, exp_scaling]:
        if todo and f.__name__ not in todo:
            continue
        print("=" * 70, f.__name__)
        f(rng)
    print("total %.0fs" % (time.time() - t0))
