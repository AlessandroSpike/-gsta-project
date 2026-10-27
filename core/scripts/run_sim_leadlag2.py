"""S3b: lead-lag between two noisy realisations of the same network (Section 5.3).

Network with N=10 nodes; the second trajectory experiences the common pulse l periods later
(the first leads by l).  T in {6, 10, 16}, l in {0,...,4} (l >= T/2 is outside the identifiable
range of Proposition 9), level and spread shocks, 24 replications.  For each replication the
cross matrix is computed once and the estimator is evaluated for beta = c x median(level-adjusted
Delta), c in {0.03, 0.1, 0.3, 1}.  Output: results/sim_leadlag2.csv (resumable parts)."""
import os, sys, time, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gsta.sim import base_network, pulse, pulse_trajectory, spread_trajectory
from gsta.core import value_matrix, self_values, debias
from gsta.softdtw import soft_dtw_grad, lead_lag, boundary_corrected_lag

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
R = int(os.environ.get("R_LL2", 24))
KW = dict(alpha=0.5, gamma=1.0, eps=0.05, inits=("identity",), tol=1e-9)


def tau_max_identifiable(T):
    l = int(np.ceil(T / 2) - 1)
    return l * (T - 1) / (T + l)


def one(shock, T, lag, rep):
    rng = np.random.default_rng(3_000_000 + 100_000 * T + 1000 * lag + rep + (0 if shock == "level" else 50_000))
    net = base_network(10, rng)
    w = max(1.0, T / 6)
    c = (T - 1) / 2 - lag / 2
    traj = pulse_trajectory if shock == "level" else spread_trajectory
    G = traj(net, rng, pulse(T, c, w), sigma=0.25)
    H = traj(net, rng, pulse(T, c + lag, w), sigma=0.25)
    V = value_matrix(G, H, **KW)
    D = debias(V, self_values(G, **KW), self_values(H, **KW))
    Da = D - D.min()
    rows = []
    for cb in [0.03, 0.1, 0.3, 1.0]:
        beta = cb * np.median(Da)
        _, E = soft_dtw_grad(Da, beta)
        tau = lead_lag(E)
        rows.append(dict(shock=shock, T=T, lag=lag, rep=rep, beta_factor=cb, tau=tau,
                         tau_raw=lead_lag(soft_dtw_grad(D, beta)[1]),
                         corrected=boundary_corrected_lag(tau, T),
                         in_range=abs(tau) <= tau_max_identifiable(T) + 1e-12,
                         excess_length=float(E.sum() - T), identifiable=2 * lag < T))
    return rows


if __name__ == "__main__":
    t0 = time.time()
    PART = os.path.join(OUT, "parts"); os.makedirs(PART, exist_ok=True)
    for shock in ["level", "spread"]:
        for T in [6, 10, 16]:
            for lag in [0, 1, 2, 3, 4]:
                fn = os.path.join(PART, f"ll2_{shock}_{T}_{lag}.csv")
                if os.path.exists(fn):
                    continue
                rows = []
                for rep in range(R):
                    rows += one(shock, T, lag, rep)
                pd.DataFrame(rows).to_csv(fn, index=False)
                print(shock, T, lag, "done %.0fs" % (time.time() - t0), flush=True)
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(PART, "ll2_*.csv")))])
    df.to_csv(os.path.join(OUT, "sim_leadlag2.csv"), index=False)
    s = df[df.beta_factor == 0.1].groupby(["shock", "T", "lag"]).agg(tau=("tau", "mean"), corrected=("corrected", "mean"))
    print(s.round(2).to_string())
