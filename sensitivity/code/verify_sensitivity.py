"""
Verification of the sensitivity formulas (Section 5.5 and Supplement S8.3).

Every finite difference re-solves all affected state-level problems with the full deterministic
portfolio (Algorithm 1, tolerance 1e-13); no derivative is propagated through solver iterations.

Outputs (results/):
  verif_state.csv      Proposition 11 by instance and block, plus incorrect variants
  verif_traj.csv       Proposition 12 (anomaly, G-STA) and 13 (lead-lag) on small trajectories
  verif_branch.csv     value of the portfolio estimate along a path through a branch tie
"""
import os
import time
import numpy as np
import pandas as pd

from gsta_core import State, solve_V
from gsta_sensitivity import (grad_delta_first, SelfTrajectory, CrossTrajectory, solve_pair,
                              hessian_certificate, frozen_remainder)
from softdtw import soft_dtw, leadlag

OUT = "results"
os.makedirs(OUT, exist_ok=True)
PRM = dict(alpha=0.5, gamma=1.0, eps=0.05, tol=1e-13)
H_FD = 1e-5


# ----------------------------------------------------------------------------------------
# simulation design of Supplement S4
# ----------------------------------------------------------------------------------------
def base_network(n, rng, kappa=0.5):
    x = rng.standard_normal((n, 2))
    S = np.linalg.norm(x[:, None] - x[None], axis=2) + 2.0
    u = rng.standard_normal(n)
    C = S + kappa * (u[:, None] - u[None, :])
    np.fill_diagonal(C, 0.0)
    mu = rng.lognormal(0.0, 1.0, n)
    mu /= mu.sum()
    F = rng.standard_normal((n, 1))
    return C, mu, F


def observe(C, mu, F, rng, sigma=0.25, miss=0.0):
    n = C.shape[0]
    Cn = np.clip(C + sigma * rng.standard_normal(C.shape), 0.0, None)
    np.fill_diagonal(Cn, 0.0)
    W = np.ones((n, n))
    if miss > 0:
        W = (rng.random((n, n)) > miss).astype(float)
        np.fill_diagonal(W, 1.0)
    Fn = F + 0.05 * rng.standard_normal(F.shape)
    mun = mu * rng.lognormal(0.0, 0.02, n)
    return State(mun, Fn, Cn * W, W)


def Dhat(G, H, same):
    a, g, e, tol = PRM["alpha"], PRM["gamma"], PRM["eps"], PRM["tol"]
    return (solve_V(G, H, a, g, e, same_nodes=same, tol=tol).J
            - 0.5 * solve_V(G, G, a, g, e, same_nodes=True, tol=tol).J
            - 0.5 * solve_V(H, H, a, g, e, same_nodes=True, tol=tol).J)


def perturbed(G, block, idx, h):
    X = G.copy()
    if block == "C":
        X.C[idx] += h
    elif block == "F":
        X.F[idx] += h
    else:
        X.mu[idx] += h
    return X


def coords(G, block, rng, k):
    if block == "C":
        obs = np.argwhere((G.W > 0) & ~np.eye(G.n, dtype=bool))
        pick = obs[rng.choice(len(obs), size=min(k, len(obs)), replace=False)]
        return [tuple(p) for p in pick]
    if block == "F":
        return [(i, 0) for i in rng.choice(G.n, size=min(k, G.n), replace=False)]
    return [int(i) for i in rng.choice(G.n, size=min(k, G.n), replace=False)]


def fd_state(G, H, same, block, idx, h=H_FD):
    return (Dhat(perturbed(G, block, idx, h), H, same)
            - Dhat(perturbed(G, block, idx, -h), H, same)) / (2 * h)


# ----------------------------------------------------------------------------------------
# 1. Proposition 11 on four instance types (both arguments)
# ----------------------------------------------------------------------------------------
def verify_state(seed=11, k=8):
    rng = np.random.default_rng(seed)
    rows = []
    insts = {}
    C, mu, F = base_network(8, rng)
    insts["directed"] = (observe(C, mu, F, rng), observe(C * 1.15, mu, F + 0.3, rng), True)
    C, mu, F = base_network(8, rng)
    insts["directed+mask"] = (observe(C, mu, F, rng, miss=0.25),
                              observe(C * 1.15, mu, F + 0.3, rng, miss=0.25), True)
    C1, mu1, F1 = base_network(8, rng)
    C2, mu2, F2 = base_network(6, rng)
    insts["different sizes"] = (observe(C1, mu1, F1, rng, miss=0.2),
                                observe(C2, mu2, F2, rng, miss=0.2), False)
    # near-copy: H is a small perturbation of G (gradient of Delta vanishes at H = G)
    C, mu, F = base_network(8, rng)
    G0 = observe(C, mu, F, rng, miss=0.2)
    H0 = G0.copy(); H0.C = H0.C + 0.02 * rng.standard_normal(H0.C.shape) * H0.W
    np.fill_diagonal(H0.C, 0.0)
    insts["near copy"] = (G0, H0, True)

    a, g, e = PRM["alpha"], PRM["gamma"], PRM["eps"]
    for name, (G, H, same) in insts.items():
        t0 = time.time()
        for arg, (X, Y) in (("first", (G, H)), ("second", (H, G))):
            sxy = solve_pair(X, Y, PRM, same_nodes=same)
            sxx = solve_pair(X, X, PRM, same_nodes=True)
            cert = hessian_certificate(X, Y, sxy.P, a, g, e)
            grads = {v: grad_delta_first(X, Y, sxy.P, sxx.P, a, g, v)
                     for v in ("correct", "first_only", "raw")}
            for block in ("C", "F", "mu"):
                idxs = coords(X, block, rng, k)
                fd = np.array([fd_state(X, Y, same, block, i) for i in idxs])
                for v, gr in grads.items():
                    an = np.array([gr[block][i] for i in idxs])
                    rows.append(dict(instance=name, argument=arg, block=block, variant=v,
                                     n_coords=len(idxs), max_abs_err=np.abs(an - fd).max(),
                                     rel_err=np.abs(an - fd).max() / max(np.abs(fd).max(), 1e-300),
                                     max_abs_fd=np.abs(fd).max(), gap=sxy.gap,
                                     residual=sxy.residual, start=sxy.start,
                                     n_tied=sxy.n_tied, hess_min=cert))
            # masked entries: derivative must be exactly zero (Corollary 1)
            if (X.W == 0).any():
                i = tuple(np.argwhere(X.W == 0)[0])
                rows.append(dict(instance=name, argument=arg, block="C (unobserved)",
                                 variant="correct", n_coords=1,
                                 max_abs_err=abs(fd_state(X, Y, same, "C", i)), rel_err=np.nan,
                                 max_abs_fd=0.0, gap=sxy.gap, residual=sxy.residual,
                                 start=sxy.start, n_tied=sxy.n_tied, hess_min=cert))
        print(f"  {name}: {time.time() - t0:.1f}s")
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/verif_state.csv", index=False)
    return df


# ----------------------------------------------------------------------------------------
# 2. Propositions 12 and 13 on small trajectories
# ----------------------------------------------------------------------------------------
def make_trajectory(n, T, rng, miss=0.15, shift=0):
    C0, mu, F = base_network(n, rng)
    B = 0.15 * rng.standard_normal((n, n))
    pulse = np.exp(-0.5 * ((np.arange(T) - (T - 1) / 2 - shift) / 1.0) ** 2)
    out = []
    for t in range(T):
        Ct = C0 * (1 + 0.6 * pulse[t]) + 0.3 * t * B
        out.append(observe(np.clip(Ct, 0, None), mu * (1 - 0.2 * pulse[t]), F - 0.5 * pulse[t],
                           rng, sigma=0.1, miss=miss))
    return out


def verify_traj(seed=5, k=6):
    rng = np.random.default_rng(seed)
    T = 5
    GA = make_trajectory(8, T, rng, shift=-0.5)
    GB = make_trajectory(6, T, rng, shift=+0.5)
    prm = PRM
    A, B = SelfTrajectory(GA, prm), SelfTrajectory(GB, prm)
    X = CrossTrajectory(A, B, prm)
    beta_fixed = X.beta
    t0_ = 2

    def quantities(GA_, GB_):
        A_, B_ = SelfTrajectory(GA_, prm), SelfTrajectory(GB_, prm)
        X_ = CrossTrajectory(A_, B_, prm)
        return dict(anomaly_t=A_.anomaly()[t0_], anomaly_other=A_.anomaly()[0],
                    gsta=X_.gsta(beta_fixed), tau=X_.tau())

    tau0, Gtau = X.tau_grad_matrix()
    an = dict(anomaly_t=A.grad_anomaly(t0_, t0_), anomaly_other=A.grad_anomaly(0, t0_),
              gsta=X.grad_gsta_A(t0_, beta_fixed), tau=X.grad_tau_A(t0_, Gtau))
    # alternative "frozen alignment" approximation: identity alignment / fixed E
    rows = []
    for block in ("C", "F", "mu"):
        idxs = coords(GA[t0_], block, rng, k)
        for i in idxs:
            fds = {}
            qp = quantities([perturbed(G, block, i, H_FD) if s == t0_ else G for s, G in enumerate(GA)], GB)
            qm = quantities([perturbed(G, block, i, -H_FD) if s == t0_ else G for s, G in enumerate(GA)], GB)
            for q in qp:
                fd = (qp[q] - qm[q]) / (2 * H_FD)
                rows.append(dict(quantity=q, block=block, coord=str(i), analytic=an[q][block][i], fd=fd))
    df = pd.DataFrame(rows)
    summ = (df.assign(err=lambda d: np.abs(d.analytic - d.fd))
              .groupby(["quantity", "block"])
              .apply(lambda d: pd.Series(dict(n=len(d), max_abs_err=d.err.max(),
                                              rel_err=d.err.max() / max(np.abs(d.fd).max(), 1e-300),
                                              max_abs_fd=np.abs(d.fd).max())))
              .reset_index())
    summ["tau0"] = tau0
    summ["regular_cells"] = X.regularity()["n_regular"]
    summ.to_csv(f"{OUT}/verif_traj.csv", index=False)
    df.to_csv(f"{OUT}/verif_traj_coords.csv", index=False)
    return summ


# ----------------------------------------------------------------------------------------
# 3. a branch tie: portfolio estimate along a path, one-sided derivatives
# ----------------------------------------------------------------------------------------
def verify_branch(seed=3, n=6, npts=21):
    """H(lam) interpolates between a network and a copy in which two interchangeable nodes
    exchange their relations; the identity-type and swap-type stationary branches tie at
    lam = 1/2, where the portfolio estimate has a kink."""
    rng = np.random.default_rng(seed)
    C, mu, F = base_network(n, rng)
    mu[1] = mu[0]; F[1] = F[0]
    G = State(mu, F, C, np.ones((n, n)))
    perm = np.arange(n); perm[[0, 1]] = [1, 0]
    Cs = C[np.ix_(perm, perm)]
    a, g, e = PRM["alpha"], PRM["gamma"], PRM["eps"]
    P_id = np.diag(mu) + 1e-12
    P_sw = np.eye(n)[perm] * mu[:, None] + 1e-12
    rows = []
    track = {"identity branch": P_id, "swap branch": P_sw}
    VGG = solve_V(G, G, a, g, e, tol=1e-13).J
    lams = np.linspace(0.3, 0.7, npts)
    for lam in lams:
        H = State(mu, F, (1 - lam) * C + lam * Cs, np.ones((n, n)))
        VHH = solve_V(H, H, a, g, e, tol=1e-13).J
        best, allr = solve_V(G, H, a, g, e, same_nodes=True, tol=1e-13,
                             n_random=6, return_all=True)
        row = dict(lam=lam, portfolio=best.J - 0.5 * VGG - 0.5 * VHH, winner=best.start)
        dH = State(mu, F, Cs - C, np.ones((n, n)))   # direction of the path in the C-block
        for name in track:
            res = solve_V(G, H, a, g, e, warm=track[name], tol=1e-13)
            track[name] = res.P
            row[name] = res.J - 0.5 * VGG - 0.5 * VHH
            # envelope directional derivative along the path (second argument moves)
            sHH = solve_V(H, H, a, g, e, tol=1e-13)
            gH = grad_delta_first(H, G, res.P.T, sHH.P, a, g)
            row[name + " slope"] = float(np.sum(gH["C"] * dH.C))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/verif_branch.csv", index=False)
    return df


if __name__ == "__main__":
    t0 = time.time()
    print("state-level verification")
    s = verify_state()
    print(s.groupby(["instance", "variant"]).rel_err.max().unstack().to_string())
    print("trajectory-level verification")
    print(verify_traj().to_string())
    print("branch tie")
    b = verify_branch()
    print(b.iloc[::5].to_string())
    print(f"total {time.time() - t0:.0f}s")
