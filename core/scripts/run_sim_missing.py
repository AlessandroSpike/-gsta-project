"""S4: missing edges -- masked UFGW versus imputation (Section 4.4).

N=20 nodes, T=6 periods with a gradual drift.  The oracle Delta uses the full
geometry; competing estimates use (i) observation masks, (ii) row-maximum
imputation, (iii) two-way fixed-effects imputation.  Missingness: MCAR (15%) or
structured (two nodes never observed as destination + MNAR on high costs and on
peripheral origins), mimicking the transport-cost panel.
Output: results/sim_missing.csv
"""
import os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gsta.sim import base_network
from gsta.data import GraphState, twoway_fe_impute
from gsta.core import value_matrix, debias, anomaly_scores

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
R = int(os.environ.get("R_MIS", 30))
N, T = 20, 6
KW = dict(alpha=0.5, gamma=1.0, eps=0.05, inits=("identity",), tol=1e-9)


def Delta(states):
    V = value_matrix(states, None, **KW)
    d = np.diag(V).copy()
    D = debias(V, d, d); np.fill_diagonal(D, 0)
    return D


MECHS = {"MCAR 5%": ("mcar", 0.05), "MCAR 15%": ("mcar", 0.15), "MCAR 30%": ("mcar", 0.30),
         "MCAR 45%": ("mcar", 0.45), "asymmetric": ("asym", None), "cost-dependent": ("mnar", None),
         "node-dependent": ("node", None), "structured": ("structured", None)}


def one(mech, rep):
    kind, rate = MECHS[mech]
    rng = np.random.default_rng(777 + 31 * rep + 5000 * list(MECHS).index(mech))
    net = base_network(N, rng)
    drift = rng.normal(size=(N, N)) * 0.15
    Cs, Fs, Ms = [], [], []
    for t in range(T):
        C = np.maximum(net["C0"] + t * drift + 0.25 * rng.normal(size=(N, N)), 0); np.fill_diagonal(C, 0)
        Cs.append(C); Fs.append(net["f"] + 0.05 * rng.normal(size=(N, 1)) + 0.05 * t)
        Ms.append(net["mu"] * np.exp(0.02 * rng.normal(size=N)))
    Cs = np.array(Cs)
    obs = np.ones((T, N, N), bool)
    periph = np.argsort(net["mu"])[:5]
    upper = np.triu(np.ones((N, N), bool), 1)
    for t in range(T):
        if kind == "mcar":
            obs[t] = rng.uniform(size=(N, N)) > rate
        elif kind == "asym":           # one direction of each pair much more often missing
            obs[t] = rng.uniform(size=(N, N)) > np.where(upper, 0.35, 0.05)
        elif kind == "mnar":           # missingness increases with the (unobserved) cost itself
            pm = 0.5 / (1 + np.exp(-(Cs[t] - np.quantile(Cs[t], 0.75)) * 3.0))
            obs[t] = rng.uniform(size=(N, N)) > pm
        elif kind == "node":           # peripheral (light) origins report much less
            pm = np.full((N, N), 0.05); pm[periph, :] = 0.5
            obs[t] = rng.uniform(size=(N, N)) > pm
        else:
            pm = 1 / (1 + np.exp(-(Cs[t] - np.median(Cs[t])) * 2.0)) * 0.2
            pm[periph, :] += 0.35
            obs[t] = rng.uniform(size=(N, N)) > pm
            obs[t][:, [0, 1]] = False            # two destinations never observed
        np.fill_diagonal(obs[t], False)
    full = [GraphState(Ms[t], Fs[t], Cs[t], None) for t in range(T)]
    W = obs.astype(float)
    for t in range(T):
        np.fill_diagonal(W[t], 1.0)
    masked = [GraphState(Ms[t], Fs[t], np.where(obs[t], Cs[t], 0.0), W[t]) for t in range(T)]
    rowmax = []
    for t in range(T):
        C = np.where(obs[t], Cs[t], np.nan)
        for i in range(N):
            v = C[i][np.isfinite(C[i])]
            C[i][~np.isfinite(C[i])] = v.max() if len(v) else 0.0
        np.fill_diagonal(C, 0)
        rowmax.append(GraphState(Ms[t], Fs[t], C, None))
    fe = twoway_fe_impute(np.where(obs, Cs, 0.0), obs)
    fes = [GraphState(Ms[t], Fs[t], fe[t], None) for t in range(T)]
    D0 = Delta(full)
    off = ~np.eye(T, dtype=bool)
    rows = []
    for name, st in [("mask", masked), ("row-max imputation", rowmax), ("two-way FE imputation", fes)]:
        D = Delta(st)
        rows.append(dict(mechanism=mech, rep=rep, method=name,
                         rel_err=np.linalg.norm(D - D0) / np.linalg.norm(D0),
                         spearman_offdiag=spearmanr(D[off], D0[off])[0],
                         spearman_anomaly=spearmanr(anomaly_scores(D), anomaly_scores(D0))[0],
                         missing_share=1 - obs[:, ~np.eye(N, dtype=bool)].mean()))
    return rows


if __name__ == "__main__":
    import glob
    t0 = time.time()
    PART = os.path.join(OUT, "parts"); os.makedirs(PART, exist_ok=True)
    for mech in MECHS:
        fn = os.path.join(PART, "mis_" + mech.replace(" ", "_").replace("%", "pct") + ".csv")
        if os.path.exists(fn):
            continue
        rows = []
        for rep in range(R):
            rows += one(mech, rep)
        pd.DataFrame(rows).to_csv(fn, index=False)
        print(mech, "done %.0fs" % (time.time() - t0), flush=True)
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(PART, "mis_*.csv")))])
    df.to_csv(os.path.join(OUT, "sim_missing.csv"), index=False)
    print(df.groupby(["mechanism", "method"])[["rel_err", "spearman_offdiag", "spearman_anomaly", "missing_share"]].mean().round(3).to_string())
