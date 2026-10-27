"""Computational benchmark (Section 5.1, Supplement S3): (i) cost breakdown of Algorithm 1 by
component and scaling in N; (ii) comparison with POT (balanced fused GW with asymmetric
matrices, symmetric=False): run time, balanced FGW energy of the returned coupling, marginal error.
Hardware: 1 core Intel Xeon 2.1 GHz; OpenBLAS single thread.  Output: results/bench_*.csv"""
import os, sys, time, math
import numpy as np, pandas as pd
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import gsta.ufgw as U
import ot
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
TIMES = {}
def timed(name, f):
    def g(*a, **k):
        t0 = time.perf_counter(); r = f(*a, **k); TIMES[name] = TIMES.get(name, 0.0) + time.perf_counter() - t0; return r
    return g
class TimedProblem(U.UFGWProblem):
    def T(self, P): return timed("structural operator", super().T)(P)
    def gradQ(self, P): return timed("structural operator", super().gradQ)(P)
    def oracle(self, *a, **k): return timed("Sinkhorn oracle", super().oracle)(*a, **k)
U._line_search_nb = timed("line search", U._line_search_nb)
U._g_along = timed("line search", U._g_along)

def inst(rng, N, masked):
    mu = rng.uniform(0.5, 1.5, N); mu /= mu.sum(); nu = rng.uniform(0.5, 1.5, N); nu /= nu.sum()
    F = rng.normal(size=(N, 1)); G = F + 0.3 * rng.normal(size=(N, 1))
    C = rng.uniform(0, 1, (N, N)); D = np.clip(C + 0.2 * rng.normal(size=(N, N)), 0, None)
    np.fill_diagonal(C, 0); np.fill_diagonal(D, 0)
    W = Wp = None
    if masked:
        W = (rng.uniform(size=(N, N)) > 0.1).astype(float); Wp = (rng.uniform(size=(N, N)) > 0.1).astype(float)
        np.fill_diagonal(W, 1); np.fill_diagonal(Wp, 1)
    return mu, nu, F, G, C, D, W, Wp

rows, prow = [], []
rng = np.random.default_rng(11)
for N in ([] if os.environ.get("SKIP_BREAKDOWN") else [25, 50, 100, 200, 400]):
    for rep in range(3 if N < 400 else 1):
        mu, nu, F, G, C, D, W, Wp = inst(rng, N, True)
        pr = TimedProblem(mu, nu, U.feature_cost(F, G), C, D, alpha=0.5, gamma=1.0, eps=0.05, Wc=W, Wd=Wp)
        TIMES.clear(); t0 = time.perf_counter(); r = U.solve_ufgw(pr, tol=1e-9); tot = time.perf_counter() - t0
        rows.append(dict(N=N, rep=rep, iters=r.n_iter, total=tot, per_iter=tot / r.n_iter, gap=r.gap, **TIMES,
                         other=tot - sum(TIMES.values())))
        # multistart cost (identity, product, feature) for the same problem
        t0 = time.perf_counter(); U.ufgw_multistart(U.UFGWProblem(mu, nu, U.feature_cost(F, G), C, D, alpha=0.5, gamma=1.0,
                                                                  eps=0.05, Wc=W, Wd=Wp), tol=1e-9)
        rows[-1]["multistart_3"] = time.perf_counter() - t0
    print(N, "done", flush=True)
if rows:
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "bench_breakdown.csv"), index=False)
    print(pd.DataFrame(rows).groupby("N").median().round(4).to_string())

# ---- comparison with POT on balanced, unmasked, directed problems
PN = [int(x) for x in os.environ.get("POT_N", "25,50,100").split(",")]
if os.environ.get("SKIP_BREAKDOWN"):
    pass
for N in PN:
    for rep in range(3):
        mu, nu, F, G, C, D, _, _ = inst(rng, N, False)
        p = np.full(N, 1 / N); q = np.full(N, 1 / N); M = U.feature_cost(F, G)
        energy = lambda P: 0.5 * float((M * P).sum()) + 0.5 * float(np.einsum("ij,ij->", U.UFGWProblem(p, q, M, C, D, alpha=0.5).T(P), P))
        t0 = time.perf_counter(); Tp, lg = ot.gromov.fused_gromov_wasserstein(M, C, D, p, q, loss_fun="square_loss", alpha=0.5,
                                                                             symmetric=False, log=True); tp = time.perf_counter() - t0
        t0 = time.perf_counter(); Te = ot.gromov.entropic_fused_gromov_wasserstein(M, C, D, p, q, epsilon=0.01, alpha=0.5,
                                                                                   symmetric=False, max_iter=1000, tol=1e-9); te = time.perf_counter() - t0
        pr = U.UFGWProblem(p, q, M, C, D, alpha=0.5, gamma=1e3, eps=0.01)
        t0 = time.perf_counter(); ro = U.ufgw_multistart(pr, inits=("product",), tol=1e-9); to = time.perf_counter() - t0
        Po = ro.P / ro.P.sum()
        for name, P, tt in [("POT fused_gromov_wasserstein (CG, exact OT oracle)", Tp, tp),
                            ("POT entropic_fused_gromov_wasserstein (eps=0.01)", Te, te),
                            ("Algorithm 1 (gamma=1e3, eps=0.01)", Po, to)]:
            prow.append(dict(N=N, rep=rep, method=name, time=tt, fgw_energy=energy(P),
                             marginal_error=max(np.abs(P.sum(1) - p).max(), np.abs(P.sum(0) - q).max())))
    print("POT", N, flush=True)
    fn = os.path.join(OUT, "bench_pot.csv")
    old = pd.read_csv(fn) if os.path.exists(fn) else pd.DataFrame()
    cur = pd.DataFrame(prow)
    allp = pd.concat([old[~old.N.isin(cur.N.unique())] if len(old) else old, cur])
    allp.drop(columns=[c for c in allp.columns if c == "rel_energy_vs_POT_CG"]).to_csv(fn, index=False)
pdf = pd.read_csv(os.path.join(OUT, "bench_pot.csv"))
ref = pdf[pdf.method.str.startswith("POT fused")].set_index(["N", "rep"]).fgw_energy
pdf["rel_energy_vs_POT_CG"] = pdf.apply(lambda r: r.fgw_energy / ref[(r.N, r.rep)] - 1, axis=1)
pdf.to_csv(os.path.join(OUT, "bench_pot.csv"), index=False)
print(pdf.groupby(["N", "method"])[["time", "rel_energy_vs_POT_CG", "marginal_error"]].median().to_string())
