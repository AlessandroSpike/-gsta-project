"""S2: power to detect a one-period structural shock (Section 5.2).

Base design: N=12 nodes, T=8 periods (indexed 1..8), shock in period 5 on the four heaviest nodes.
Scenarios: reversal, corridor, feature, mass (Supplement S4).  Detection: the shock period has the
largest anomaly score.  Variants (env VARIANT): base (all scenarios x 3 effect sizes, R=60), N24,
sigma0.4, kappa1.0, masssd0.5 (reversal 0.5, corridor 0.05, null; R=40) and relabel (node labels
re-drawn every period; null, reversal 1.0, corridor 0.1; R=30).  Resumable: results/parts/det_*.csv.
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gsta.sim import base_network, shock_trajectory, relabel, summary_vector, spectral_signature
from gsta.core import value_matrix, debias, anomaly_scores
from gsta.data import GraphState

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
T, T0 = 8, 4                      # 0-based index 4 = period 5 of 8
DELTAS = {"reversal": [0.25, 0.5, 1.0], "corridor": [0.025, 0.05, 0.1],
          "feature": [0.05, 0.1, 0.2], "mass": [0.25, 0.5, 1.0]}
SCEN = ["reversal", "corridor", "feature", "mass"]
VARIANTS = {
    "base": dict(N=12, sigma=0.25, kappa=0.5, mass_sd=1.0, R=60, blocks="all"),
    "N24": dict(N=24, sigma=0.25, kappa=0.5, mass_sd=1.0, R=40, blocks="mid"),
    "sigma0.4": dict(N=12, sigma=0.4, kappa=0.5, mass_sd=1.0, R=40, blocks="mid"),
    "kappa1.0": dict(N=12, sigma=0.25, kappa=1.0, mass_sd=1.0, R=40, blocks="mid"),
    "masssd0.5": dict(N=12, sigma=0.25, kappa=0.5, mass_sd=0.5, R=40, blocks="mid"),
    "relabel": dict(N=12, sigma=0.25, kappa=0.5, mass_sd=1.0, R=30, blocks="relabel"),
}


def delta_ufgw(states, **kw):
    V = value_matrix(states, None, **kw)
    d = np.diag(V).copy()
    D = debias(V, d, d)
    np.fill_diagonal(D, 0.0)
    return D


def symmetrised(states):
    return [GraphState(s.mu, s.F, 0.5 * (s.C + s.C.T), s.W) for s in states]


def frobenius(states, weighted):
    T_ = len(states)
    D = np.zeros((T_, T_))
    if not weighted:
        Cs = np.array([s.C for s in states]); Fs = np.array([s.F for s in states]); Ms = np.array([s.mu for s in states])
        sc, sf, sm = Cs.std(), Fs.std(), Ms.std()
        for t in range(T_):
            for u in range(T_):
                D[t, u] = (((Cs[t] - Cs[u]) / sc) ** 2).sum() + (((Fs[t] - Fs[u]) / sf) ** 2).sum() \
                    + (((Ms[t] - Ms[u]) / sm) ** 2).sum()
        return D
    for t in range(T_):
        for u in range(T_):
            a, b = states[t], states[u]
            mbar = 0.5 * (a.mu + b.mu)
            D[t, u] = 0.5 * (np.outer(mbar, mbar) * (a.C - b.C) ** 2).sum() \
                + 0.5 * (mbar[:, None] * (a.F - b.F) ** 2).sum() \
                + 2.0 * ((np.sqrt(a.mu) - np.sqrt(b.mu)) ** 2).sum()
    return D


def vector_distance(vecs):
    X = np.array(vecs)
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    return ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)


def one_rep(variant, scenario, delta, rep):
    cfg = VARIANTS[variant]
    rng = np.random.default_rng(10_000 * (SCEN.index(scenario) + 1) + int(4000 * delta) + rep
                                + 1_000_003 * list(VARIANTS).index(variant))
    net = base_network(cfg["N"], rng, kappa=cfg["kappa"], mass_sd=cfg["mass_sd"])
    hubs = np.argsort(-net["mu"])[:4]
    states = shock_trajectory(net, rng, T, T0, scenario, delta, hubs, sigma=cfg["sigma"])
    inits = ("identity",)
    if variant == "relabel":
        states = relabel(states, rng)
        inits = ("feature", "product")
    kw = dict(alpha=0.5, gamma=1.0, eps=0.05, inits=inits, tol=1e-9)
    mats = {"G-STA (directed)": delta_ufgw(states, **kw)}
    if variant != "relabel":
        mats["G-STA (symmetrised)"] = delta_ufgw(symmetrised(states), **kw)
    mats["feature/mass only (alpha=0)"] = delta_ufgw(states, **{**kw, "alpha": 0.0})
    mats["stacked Frobenius"] = frobenius(states, False)
    mats["mass-weighted Frobenius"] = frobenius(states, True)
    mats["network summaries"] = vector_distance([summary_vector(s) for s in states])
    mats["geometry spectrum"] = vector_distance([spectral_signature(s) for s in states])
    rows = []
    for name, D in mats.items():
        a = anomaly_scores(D)
        others = np.delete(a, T0)
        rows.append(dict(variant=variant, scenario=scenario, delta=delta, rep=rep, method=name,
                         detected=int(np.argmax(a) == T0),
                         zscore=(a[T0] - others.mean()) / (others.std() + 1e-300)))
    return rows


def blocks_for(variant):
    b = VARIANTS[variant]["blocks"]
    if b == "all":
        return [("null", 0.0)] + [(sc, de) for sc in SCEN for de in DELTAS[sc]]
    if b == "mid":
        return [("null", 0.0), ("reversal", 0.5), ("corridor", 0.05)]
    return [("null", 0.0), ("reversal", 1.0), ("corridor", 0.1)]


if __name__ == "__main__":
    t0 = time.time()
    PART = os.path.join(OUT, "parts"); os.makedirs(PART, exist_ok=True)
    variants = os.environ.get("VARIANT", "base").split(",")
    for v in variants:
        for sc, de in blocks_for(v):
            fn = os.path.join(PART, f"det_{v}_{sc}_{de}.csv")
            if os.path.exists(fn):
                continue
            rows = []
            for rep in range(VARIANTS[v]["R"]):
                rows += one_rep(v, "reversal" if sc == "null" else sc, de, rep)
            for r_ in rows:
                r_["scenario"] = "none" if sc == "null" else sc
            pd.DataFrame(rows).to_csv(fn, index=False)
            print(v, sc, de, "done %.0fs" % (time.time() - t0), flush=True)
    import glob
    df = pd.concat([pd.read_csv(f, keep_default_na=False) for f in sorted(glob.glob(os.path.join(PART, "det_*_*_*.csv")))])
    df.to_csv(os.path.join(OUT, "sim_detection.csv"), index=False)
    print(df.groupby(["variant", "scenario", "delta", "method"]).detected.mean().unstack().round(2).to_string())
