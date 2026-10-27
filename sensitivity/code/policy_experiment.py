"""
Structural influence and gradient-based counterfactual perturbations
for the international transport-cost network, 2016-2021 (new Section 6.5, Supplement S8).

Run stage by stage (each stage caches its results in results/):
    python policy_experiment.py base        # year-level trajectory, couplings, certificates
    python policy_experiment.py maps        # anomaly-score sensitivity maps (WP2-WP3)
    python policy_experiment.py regional    # regional G-STA / lead-lag sensitivity (WP4-WP5)
    python policy_experiment.py fd          # re-solved finite differences on the data (WP1)
    python policy_experiment.py mitigation  # counterfactual: reduce Delta(2020, 2021)  (WP6)
    python policy_experiment.py stress      # counterfactual: make 2020 distinctive     (WP6)
    python policy_experiment.py timing      # counterfactual: remove the EAP-SSEA lead  (WP6)
    python policy_experiment.py uncertainty # delta method vs Monte Carlo, masks         (WP7)
The interpretation throughout is counterfactual sensitivity and stress testing, not causal
policy evaluation.
"""
import os
import sys
import time
import pickle
import numpy as np
import pandas as pd

from data_panel import build_panel, REGIONS
from gsta_core import State, solve_V, gcg, Problem, envelope_grads
from gsta_sensitivity import (SelfTrajectory, CrossTrajectory, solve_pair, gzero, gaxpy,
                              node_scores, sa_channels, concentration, spearman, topk_overlap,
                              budget_allocation, hessian_certificate, frozen_remainder,
                              grad_delta_first)
from softdtw import leadlag, leadlag_grad

OUT = "results"
os.makedirs(OUT, exist_ok=True)
PRM = dict(alpha=0.5, gamma=1.0, eps=0.05, tol=1e-12)     # tighter than 1e-9 for gradients
PRM_CF = dict(alpha=0.5, gamma=1.0, eps=0.05, tol=1e-10)  # counterfactual re-solves
YEARS = list(range(2016, 2022))
T = len(YEARS)
UMAX = 0.5            # admissible |log-change| of a corridor cost (x1.65 / x0.61)
RNG_SEED = 20240611


def save(obj, name):
    with open(f"{OUT}/{name}.pkl", "wb") as f:
        pickle.dump(obj, f)


def load(name):
    with open(f"{OUT}/{name}.pkl", "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------------------------------
# the measurement scale: log costs, log LSCI and log TEU
# ----------------------------------------------------------------------------------------
class Scales:
    """Maps the transformed model inputs back to log costs, log LSCI and log TEU."""

    def __init__(self, panel):
        self.p = panel
        info = panel["info"]
        self.qlo, self.qhi = np.log1p(info["qlo_usd"]), np.log1p(info["qhi_usd"])
        self.s_c, self.s_f = info["s_c"], info["logLSCI_sd"]
        raw = panel["raw_cost"]
        lc = np.log1p(np.where(panel["obs"], raw, 0.0))
        inside = panel["obs"] & (lc > self.qlo) & (lc < self.qhi)
        self.dc = np.where(inside, np.where(panel["obs"], raw, 0) / (1 + np.where(panel["obs"], raw, 0)) / self.s_c, 0.0)
        self.eligible = inside

    def C_from_cost(self, cost, W):
        C = np.where(W > 0, np.clip(np.log1p(cost), self.qlo, self.qhi) / self.s_c, 0.0)
        np.fill_diagonal(C, 0.0)
        return C

    def elastic(self, g, t, idx=None):
        """Semi-elasticities: dS/dlog cost_ik, dS/dlog LSCI_i, dS/dlog TEU_i."""
        dc = self.dc[t] if idx is None else self.dc[t][np.ix_(idx, idx)]
        mu = self.p["mu"][t] if idx is None else self.p["mu"][t][idx]
        return dict(C=g["C"] * dc, F=g["F"][:, 0] / self.s_f, mu=g["mu"] * mu / 2.0)

    def perturbed_state(self, t, U=None, v=None, w=None, idx=None, W=None):
        p = self.p
        sel = slice(None) if idx is None else idx
        cost = np.where(p["obs"][t], p["raw_cost"][t], 0.0)
        Wt = p["W"][t] if W is None else W
        if idx is not None:
            cost, Wt = cost[np.ix_(idx, idx)], Wt[np.ix_(idx, idx)]
        if U is not None:
            cost = cost * np.exp(U)
        C = self.C_from_cost(cost, Wt)
        F = p["F"][t][sel].copy()
        mu = p["mu"][t][sel].copy()
        if v is not None:
            F[:, 0] += v / self.s_f
        if w is not None:
            mu = mu * np.exp(w / 2.0)
        return State(mu, F, C, Wt)


def year_states(panel, masks=None):
    return [State(panel["mu"][t], panel["F"][t], panel["C"][t],
                  panel["W"][t] if masks is None else masks[t]) for t in range(T)]


def region_index(panel):
    ci = {c: i for i, c in enumerate(panel["countries"])}
    return {r: np.array([ci[c] for c in cs]) for r, cs in REGIONS.items()}


# ----------------------------------------------------------------------------------------
# stage: base
# ----------------------------------------------------------------------------------------
def stage_base():
    t0 = time.time()
    panel = build_panel()
    base = SelfTrajectory(year_states(panel), PRM, verbose=True)
    print("Delta:\n", np.round(base.Delta, 4))
    print("anomaly:", np.round(base.anomaly(), 4), "regularity:", base.regularity())
    cert = []
    for (t, s), sol in list(base.cross.items()) + [((t, t), base.self_sol[t]) for t in range(T)]:
        G, H = base.states[t], base.states[s]
        lam = hessian_certificate(G, H, sol.P, PRM["alpha"], PRM["gamma"], PRM["eps"])
        cert.append(dict(t=YEARS[t], s=YEARS[s], J=sol.J, gap=sol.gap, residual=sol.residual,
                         start=sol.start, n_tied=sol.n_tied, tied_spread=sol.tied_spread,
                         hess_min=lam))
    pd.DataFrame(cert).to_csv(f"{OUT}/base_certificates.csv", index=False)
    save(dict(panel=panel, base=base), "base")
    print(f"base done in {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------------------
# stage: maps (anomaly-score sensitivity, WP2-WP3)
# ----------------------------------------------------------------------------------------
def anomaly_gradients(base, variant="correct"):
    """grads[t][u] = d a(t) / d inputs of G_u."""
    return [[base.grad_anomaly(t, u, variant) for u in range(T)] for t in range(T)]


def stage_maps():
    d = load("base")
    panel, base = d["panel"], d["base"]
    sc = Scales(panel)
    countries = np.array(panel["countries"])
    n = len(countries)
    a = base.anomaly()
    grads = anomaly_gradients(base)
    raw = [base.grad_anomaly(t, t, "raw") for t in range(T)]
    first = [base.grad_anomaly(t, t, "first_only") for t in range(T)]
    reg_of = {}
    for r, cs in REGIONS.items():
        for c in cs:
            reg_of[c] = r
    rows, corr_rows, node_rows = [], [], []
    EC_own = []
    for t in range(T):
        e = sc.elastic(grads[t][t], t)
        EC_own.append(e["C"])
        W = panel["W"][t]
        ns = node_scores(e["C"], W)
        _, _, shareA = sa_channels(grads[t][t]["C"], W)
        L1 = {k: np.abs(e[k]).sum() for k in ("C", "F", "mu")}
        L2 = {k: np.sqrt(np.sum(e[k] ** 2)) for k in ("C", "F", "mu")}
        # leverage of the other years' inputs on a(t)
        L1_other = sum(np.abs(sc.elastic(grads[t][u], u)["C"]).sum() for u in range(T) if u != t)
        er = sc.elastic(raw[t], t)
        ef = sc.elastic(first[t], t)
        pos = np.sum(e["C"] > 0) / max(np.sum(e["C"] != 0), 1)
        mass_pair = np.outer(panel["mu"][t], panel["mu"][t])
        nz = e["C"] != 0
        rows.append(dict(
            year=YEARS[t], anomaly=a[t], rank=int((-a).argsort().argsort()[t] + 1),
            lev_cost=0.01 * L1["C"], lev_lsci=0.01 * L1["F"], lev_teu=0.01 * L1["mu"],
            lev_cost_other_years=0.01 * L1_other,
            l2_cost=L2["C"], l2_lsci=L2["F"], l2_teu=L2["mu"],
            conc50_corr=concentration(e["C"][nz], 0.5), conc50_nodes=concentration(ns["S"], 0.5),
            share_pos=pos, anti_share=shareA,
            rho_mass_pair=spearman(np.abs(e["C"][nz]), mass_pair[nz]),
            rho_node_mass=spearman(ns["S"], panel["mu"][t]),
            rho_raw=spearman(np.abs(er["C"][nz]), np.abs(e["C"][nz])),
            relerr_raw=np.linalg.norm(er["C"] - e["C"]) / np.linalg.norm(e["C"]),
            relerr_first_only=np.linalg.norm(ef["C"] - e["C"]) / np.linalg.norm(e["C"]),
            top_country=countries[np.argmax(ns["S"])],
            top_country_B=ns["B"][np.argmax(ns["S"])]))
        for i in range(n):
            node_rows.append(dict(year=YEARS[t], country=countries[i], region=reg_of[countries[i]],
                                  S_out=ns["S_out"][i], S_in=ns["S_in"][i], S=ns["S"][i],
                                  B=ns["B"][i], e_lsci=e["F"][i], e_teu=e["mu"][i],
                                  mass=panel["mu"][t][i]))
        idx = np.argwhere(nz)
        vals = e["C"][nz]
        order = np.argsort(-np.abs(vals))[:60]
        for o in order:
            i, k = idx[o]
            corr_rows.append(dict(year=YEARS[t], origin=countries[i], destination=countries[k],
                                  elasticity=vals[o], grad_c=grads[t][t]["C"][i, k],
                                  cost=panel["raw_cost"][t][i, k],
                                  rev_elasticity=e["C"][k, i]))
    summ = pd.DataFrame(rows)
    # stability across years of the corridor sensitivity profile
    stab = np.zeros((T, T)); ov = np.zeros((T, T))
    common = np.all([EC_own[t] != 0 for t in range(T)], axis=0)
    for t in range(T):
        for s in range(T):
            stab[t, s] = spearman(np.abs(EC_own[t][common]), np.abs(EC_own[s][common]))
            ov[t, s] = topk_overlap(np.where(common, EC_own[t], 0), np.where(common, EC_own[s], 0), 100)
    summ.to_csv(f"{OUT}/maps_summary.csv", index=False)
    pd.DataFrame(node_rows).to_csv(f"{OUT}/maps_nodes.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(f"{OUT}/maps_corridors.csv", index=False)
    pd.DataFrame(stab, index=YEARS, columns=YEARS).to_csv(f"{OUT}/maps_stability_spearman.csv")
    pd.DataFrame(ov, index=YEARS, columns=YEARS).to_csv(f"{OUT}/maps_stability_top100.csv")
    np.save(f"{OUT}/maps_EC_own.npy", np.array(EC_own))
    # regional aggregation of the corridor sensitivity of each year's anomaly
    ridx = region_index(panel)
    blk = []
    for t in range(T):
        A = np.abs(EC_own[t])
        tot = A.sum()
        for r1, i1 in ridx.items():
            for r2, i2 in ridx.items():
                blk.append(dict(year=YEARS[t], origin=r1, destination=r2,
                                share=A[np.ix_(i1, i2)].sum() / tot,
                                pairs=len(i1) * len(i2) - (len(i1) if r1 == r2 else 0)))
    pd.DataFrame(blk).to_csv(f"{OUT}/maps_region_blocks.csv", index=False)
    print(summ.round(4).to_string())
    print("stability (Spearman):\n", np.round(stab, 2))
    print("top-100 overlap:\n", np.round(ov, 2))


# ----------------------------------------------------------------------------------------
# stage: regional (WP4-WP5)
# ----------------------------------------------------------------------------------------
def regional_states(panel, r, ridx, symm=False):
    idx = ridx[r]
    out = []
    for t in range(T):
        C = panel["C"][t][np.ix_(idx, idx)].copy()
        W = panel["W"][t][np.ix_(idx, idx)].copy()
        if symm:
            W = W * W.T
            C = 0.5 * (C + C.T) * W
        out.append(State(panel["mu"][t][idx], panel["F"][t][idx], C, W))
    return out


def stage_regional():
    t0 = time.time()
    d = load("base")
    panel = d["panel"]
    sc = Scales(panel)
    ridx = region_index(panel)
    regs = list(REGIONS)
    selfs = {r: SelfTrajectory(regional_states(panel, r, ridx), PRM) for r in regs}
    print(f"regional self-trajectories {time.time() - t0:.0f}s", flush=True)
    pairs = {}
    rows = []
    countries = np.array(panel["countries"])
    for i, ra in enumerate(regs):
        for rb in regs[i + 1:]:
            t1 = time.time()
            X = CrossTrajectory(selfs[ra], selfs[rb], PRM)
            pairs[(ra, rb)] = X
            tau, Gtau = X.tau_grad_matrix()
            g_val = X.gsta()
            Eab, _, _ = X.alignments()
            res = dict(a=ra, b=rb, tau=tau, gsta=g_val, beta=X.beta, **{
                f"reg_{k}": v for k, v in X.regularity().items() if k != "winners"})
            for side, reg, tr in (("a", ra, selfs[ra]), ("b", rb, selfs[rb])):
                idx = ridx[reg]
                eg = {"C": [], "F": [], "mu": []}
                et = {"C": [], "F": [], "mu": []}
                ediag = []
                antiG, antiT = [], []
                for t in range(T):
                    gG = X.grad_gsta_A(t) if side == "a" else X.grad_gsta_B(t)
                    gT = X.grad_tau_A(t, Gtau) if side == "a" else X.grad_tau_B(t, Gtau)
                    gD = X.grad_cross_A(t, t) if side == "a" else X.grad_cross_B(t, t)
                    e1, e2, e3 = sc.elastic(gG, t, idx), sc.elastic(gT, t, idx), sc.elastic(gD, t, idx)
                    for k in ("C", "F", "mu"):
                        eg[k].append(e1[k]); et[k].append(e2[k])
                    ediag.append(e3["C"])
                    antiG.append(sa_channels(gG["C"], tr.states[t].W)[2])
                    antiT.append(sa_channels(gT["C"], tr.states[t].W)[2])
                EG = {k: np.array(v) for k, v in eg.items()}
                ET = {k: np.array(v) for k, v in et.items()}
                ED = np.array(ediag)
                nz = EG["C"] != 0
                res[f"{side}_lev_tau_cost"] = 0.01 * np.abs(ET["C"]).sum()
                res[f"{side}_lev_tau_lsci"] = 0.01 * np.abs(ET["F"]).sum()
                res[f"{side}_lev_tau_teu"] = 0.01 * np.abs(ET["mu"]).sum()
                res[f"{side}_lev_gsta_cost"] = 0.01 * np.abs(EG["C"]).sum()
                res[f"{side}_lev_gsta_lsci"] = 0.01 * np.abs(EG["F"]).sum()
                res[f"{side}_lev_gsta_teu"] = 0.01 * np.abs(EG["mu"]).sum()
                res[f"{side}_anti_gsta"] = float(np.mean(antiG))
                res[f"{side}_anti_tau"] = float(np.mean(antiT))
                res[f"{side}_rho_gsta_tau"] = spearman(np.abs(EG["C"][nz]), np.abs(ET["C"][nz]))
                res[f"{side}_rho_gsta_diag"] = spearman(np.abs(EG["C"][nz]), np.abs(ED[nz]))
                yr_tau = np.abs(ET["C"]).sum(axis=(1, 2)) + np.abs(ET["F"]).sum(1) + np.abs(ET["mu"]).sum(1)
                res[f"{side}_tau_year_profile"] = ";".join(f"{x:.4g}" for x in yr_tau / yr_tau.sum())
                ns = node_scores(np.abs(ET["C"]).sum(0), np.ones((len(idx), len(idx))))
                nsg = node_scores(np.abs(EG["C"]).sum(0), np.ones((len(idx), len(idx))))
                res[f"{side}_top_tau"] = countries[idx][np.argmax(ns["S"] + np.abs(ET["F"]).sum(0) / 1e9)]
                res[f"{side}_top_gsta"] = countries[idx][np.argmax(nsg["S"])]
                np.savez(f"{OUT}/regional_{ra}_{rb}_{side}.npz", EG_C=EG["C"], EG_F=EG["F"],
                         EG_mu=EG["mu"], ET_C=ET["C"], ET_F=ET["F"], ET_mu=ET["mu"], ED=ED,
                         idx=idx)
            # beta sensitivity of the lead-lag gradient
            for bf in (0.05, 0.2):
                tb, Gb, _, _ = leadlag_grad(X.Dab, beta_frac=bf)
                res[f"tau_beta{bf}"] = tb
                res[f"rho_Gtau_beta{bf}"] = spearman(np.abs(Gb), np.abs(Gtau))
            res["Gtau_sum"] = float(Gtau.sum())
            res["Gtau_dotD"] = float(np.sum(Gtau * X.Dab))
            rows.append(res)
            print(f"  {ra}-{rb}: tau={tau:+.3f} gsta={g_val:.3f} [{time.time() - t1:.0f}s]", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/regional_summary.csv", index=False)
    save(dict(selfs=selfs, pairs=pairs), "regional")
    print(df[["a", "b", "tau", "gsta", "a_anti_tau", "a_anti_gsta", "a_rho_gsta_tau"]].round(3).to_string())
    print(f"regional done in {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------------------
# stage: fd (re-solved finite differences on the data)
# ----------------------------------------------------------------------------------------
def stage_fd(h=1e-4, k_top=8, k_rand=8):
    t0 = time.time()
    d = load("base")
    panel, base = d["panel"], d["base"]
    rng = np.random.default_rng(RNG_SEED)
    prm = dict(PRM)
    rows = []
    t20, t21 = YEARS.index(2020), YEARS.index(2021)

    def Delta_pair(G, H):
        return (solve_V(G, H, prm["alpha"], prm["gamma"], prm["eps"], tol=prm["tol"]).J
                - 0.5 * solve_V(G, G, prm["alpha"], prm["gamma"], prm["eps"], tol=prm["tol"]).J
                - 0.5 * base.self_sol[t20].J)

    def pert(G, block, i, hh):
        X = G.copy()
        if block == "C":
            X.C[i] += hh
        elif block == "F":
            X.F[i] += hh
        else:
            X.mu[i] += hh
        return X

    # (i) Delta(2021, 2020) wrt the inputs of 2021
    g = base.grad_delta(t21, t20)
    G21, G20 = base.states[t21], base.states[t20]
    obs = np.argwhere((G21.W > 0) & ~np.eye(G21.n, dtype=bool))
    absg = np.abs(g["C"][obs[:, 0], obs[:, 1]])
    pick = list(obs[np.argsort(-absg)[:k_top]]) + list(obs[rng.choice(len(obs), k_rand, replace=False)])
    for i in pick:
        i = tuple(i)
        fd = (Delta_pair(pert(G21, "C", i, h), G20) - Delta_pair(pert(G21, "C", i, -h), G20)) / (2 * h)
        rows.append(dict(quantity="Delta(2021,2020)", block="C", coord=str(i), analytic=g["C"][i], fd=fd))
    for i in rng.choice(G21.n, 4, replace=False):
        fd = (Delta_pair(pert(G21, "F", (i, 0), h), G20) - Delta_pair(pert(G21, "F", (i, 0), -h), G20)) / (2 * h)
        rows.append(dict(quantity="Delta(2021,2020)", block="F", coord=str(i), analytic=g["F"][i, 0], fd=fd))
        fd = (Delta_pair(pert(G21, "mu", i, h), G20) - Delta_pair(pert(G21, "mu", i, -h), G20)) / (2 * h)
        rows.append(dict(quantity="Delta(2021,2020)", block="mu", coord=str(i), analytic=g["mu"][i], fd=fd))
    print(f"  Delta FD done {time.time() - t0:.0f}s", flush=True)
    # (ii) anomaly a(2020) wrt C_2020
    ga = base.grad_anomaly(t20, t20)

    def a2020(G):
        Vs = solve_V(G, G, prm["alpha"], prm["gamma"], prm["eps"], tol=prm["tol"]).J
        tot = 0.0
        for s in range(T):
            if s != t20:
                tot += (solve_V(G, base.states[s], prm["alpha"], prm["gamma"], prm["eps"], tol=prm["tol"]).J
                        - 0.5 * Vs - 0.5 * base.self_sol[s].J)
        return tot / (T - 1)
    obs = np.argwhere((G20.W > 0) & ~np.eye(G20.n, dtype=bool))
    absg = np.abs(ga["C"][obs[:, 0], obs[:, 1]])
    pick = list(obs[np.argsort(-absg)[:4]]) + list(obs[rng.choice(len(obs), 2, replace=False)])
    for i in pick:
        i = tuple(i)
        fd = (a2020(pert(G20, "C", i, h)) - a2020(pert(G20, "C", i, -h))) / (2 * h)
        rows.append(dict(quantity="a(2020)", block="C", coord=str(i), analytic=ga["C"][i], fd=fd))
    print(f"  anomaly FD done {time.time() - t0:.0f}s", flush=True)
    # (iii) regional G-STA and lead-lag for EAP-SSEA wrt SSEA inputs of 2018
    R = load("regional")
    X = R["pairs"][("EAP", "SSEA")]
    A, B = X.A, X.B
    beta0 = X.beta
    tau0, Gtau = X.tau_grad_matrix()
    s0 = 2
    gG, gT = X.grad_gsta_B(s0), X.grad_tau_B(s0, Gtau)

    def quantities(Bs):
        Vb = solve_V(Bs, Bs, prm["alpha"], prm["gamma"], prm["eps"], tol=prm["tol"]).J
        Dab = X.Dab.copy()
        for t in range(T):
            Dab[t, s0] = (solve_V(A.states[t], Bs, prm["alpha"], prm["gamma"], prm["eps"], same_nodes=False,
                                  tol=prm["tol"]).J - 0.5 * A.self_sol[t].J - 0.5 * Vb)
        Dbb = B.Delta.copy()
        for v in range(T):
            if v != s0:
                Dbb[s0, v] = Dbb[v, s0] = (solve_V(Bs, B.states[v], prm["alpha"], prm["gamma"], prm["eps"],
                                                   tol=prm["tol"]).J - 0.5 * Vb - 0.5 * B.self_sol[v].J)
        from softdtw import soft_dtw
        gsta = (soft_dtw(Dab, beta0)[0] - 0.5 * soft_dtw(A.Delta, beta0)[0] - 0.5 * soft_dtw(Dbb, beta0)[0])
        return gsta, leadlag(Dab)[0]
    Bst = B.states[s0]
    obs = np.argwhere((Bst.W > 0) & ~np.eye(Bst.n, dtype=bool))
    absg = np.abs(gT["C"][obs[:, 0], obs[:, 1]])
    pick = list(obs[np.argsort(-absg)[:6]]) + list(obs[rng.choice(len(obs), 4, replace=False)])
    for i in pick:
        i = tuple(i)
        qp, qm = quantities(pert(Bst, "C", i, h)), quantities(pert(Bst, "C", i, -h))
        rows.append(dict(quantity="GSTA(EAP,SSEA)", block="C", coord=str(i), analytic=gG["C"][i],
                         fd=(qp[0] - qm[0]) / (2 * h)))
        rows.append(dict(quantity="tau(EAP,SSEA)", block="C", coord=str(i), analytic=gT["C"][i],
                         fd=(qp[1] - qm[1]) / (2 * h)))
    for i in range(3):
        qp, qm = quantities(pert(Bst, "F", (i, 0), h)), quantities(pert(Bst, "F", (i, 0), -h))
        rows.append(dict(quantity="GSTA(EAP,SSEA)", block="F", coord=str(i), analytic=gG["F"][i, 0],
                         fd=(qp[0] - qm[0]) / (2 * h)))
        rows.append(dict(quantity="tau(EAP,SSEA)", block="F", coord=str(i), analytic=gT["F"][i, 0],
                         fd=(qp[1] - qm[1]) / (2 * h)))
        qp, qm = quantities(pert(Bst, "mu", i, h)), quantities(pert(Bst, "mu", i, -h))
        rows.append(dict(quantity="GSTA(EAP,SSEA)", block="mu", coord=str(i), analytic=gG["mu"][i],
                         fd=(qp[0] - qm[0]) / (2 * h)))
        rows.append(dict(quantity="tau(EAP,SSEA)", block="mu", coord=str(i), analytic=gT["mu"][i],
                         fd=(qp[1] - qm[1]) / (2 * h)))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/fd_empirical_coords.csv", index=False)
    summ = (df.assign(err=lambda x: np.abs(x.analytic - x.fd))
              .groupby(["quantity", "block"])
              .apply(lambda x: pd.Series(dict(n=len(x), max_abs_err=x.err.max(),
                                              rel_err=x.err.max() / max(np.abs(x.fd).max(), 1e-300),
                                              max_abs_fd=np.abs(x.fd).max())))
              .reset_index())
    summ.to_csv(f"{OUT}/fd_empirical.csv", index=False)
    print(summ.to_string())
    print(f"fd done in {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------------------
# counterfactual machinery
# ----------------------------------------------------------------------------------------
def resolve(G, H, sol_base, same_nodes=True, prm=PRM_CF):
    """Portfolio estimate at the perturbed inputs and the branch-following value."""
    best, allr = solve_V(G, H, prm["alpha"], prm["gamma"], prm["eps"], same_nodes=same_nodes,
                         tol=prm["tol"], return_all=True)
    bf = solve_V(G, H, prm["alpha"], prm["gamma"], prm["eps"], warm=sol_base.P, tol=prm["tol"])
    return best, bf


def strategy_perturbation(name, E, budget, direction, elig, rng=None, aux=None, umax=UMAX):
    """Log-cost perturbation U (n x n) for a strategy; E are first-order semi-elasticities."""
    idx = np.argwhere(elig)
    s = E[elig]
    signs = direction * np.sign(s)
    if name == "gradient":
        u = budget_allocation(s, budget, umax, direction)
    elif name == "mass-weighted":
        order = np.argsort(-aux["mass_pair"][elig], kind="stable")
        u = budget_allocation(s, budget, umax, direction, order=order)
    elif name == "random":
        u = budget_allocation(s, budget, umax, direction, order=rng.permutation(len(s)))
    elif name == "uniform":
        u = signs * min(umax, budget / len(s))
    elif name == "cost-change":
        dev = aux["dev"][elig]
        order = np.argsort(-np.abs(dev), kind="stable")
        sg = aux["dev_sign"] * np.sign(dev)
        caps = np.minimum(umax, np.abs(dev)) if aux.get("cap_dev", False) else np.full(len(s), umax)
        u = np.zeros(len(s)); rem = budget
        for k in order:
            if rem <= 0:
                break
            step = min(caps[k], rem)
            u[k] = sg[k] * step; rem -= step
    else:
        raise ValueError(name)
    U = np.zeros_like(E)
    U[tuple(idx.T)] = u
    return U


def stage_mitigation(budgets=(1, 2, 4, 8, 16, 32, 64, 128), n_random=5):
    """Reduce Delta(G_2020, G_2021) by perturbing 2021 corridor costs."""
    t0 = time.time()
    d = load("base")
    panel, base = d["panel"], d["base"]
    sc = Scales(panel)
    rng = np.random.default_rng(RNG_SEED + 1)
    t20, t21 = YEARS.index(2020), YEARS.index(2021)
    g = base.grad_delta(t21, t20)
    E = sc.elastic(g, t21)["C"]
    elig = sc.eligible[t21] & (E != 0)
    lc = np.log(np.where(panel["obs"], panel["raw_cost"], 1.0))
    both = panel["obs"][t20] & panel["obs"][t21]
    dev = np.where(both, lc[t21] - lc[t20], 0.0)
    aux = dict(mass_pair=np.outer(panel["mu"][t21], panel["mu"][t21]), dev=dev, dev_sign=-1.0,
               cap_dev=True)
    D0 = base.Delta[t21, t20]
    sol_x, sol_s = base.cross[(t20, t21)], base.self_sol[t21]
    rows = []
    strategies = ["gradient", "cost-change", "mass-weighted", "uniform"] + [f"random{k}" for k in range(n_random)]
    for B in budgets:
        for st in strategies:
            name = "random" if st.startswith("random") else st
            U = strategy_perturbation(name, E, B, -1.0, elig, rng=rng, aux=aux)
            G21p = sc.perturbed_state(t21, U=U)
            pred = float(np.sum(E * U))
            bx, bfx = resolve(G21p, base.states[t20], sol_x)  # note: orientation (2021, 2020)
            bs, bfs = resolve(G21p, G21p, sol_s)
            real = bx.J - 0.5 * bs.J - 0.5 * base.self_sol[t20].J - D0
            real_bf = bfx.J - 0.5 * bfs.J - 0.5 * base.self_sol[t20].J - D0
            dC = G21p.C - base.states[t21].C
            P21 = base.coupling(t21, t20)
            R = frozen_remainder(base.states[t21], base.states[t20], P21,
                                 PRM["alpha"], PRM["gamma"], dC=dC)
            gx = envelope_grads(base.states[t21], base.states[t20], P21, PRM["alpha"], PRM["gamma"])[0]
            pred_c = float(np.sum(g["C"] * dC))
            Vx0 = sol_x.J
            rows.append(dict(budget=B, strategy=st, n_corr=int(np.sum(U != 0)),
                             mean_abs_u=float(np.abs(U[U != 0]).mean()) if np.any(U != 0) else 0.0,
                             pred=pred, pred_cscale=pred_c, realized=real, realized_bf=real_bf,
                             rel_change=real / D0, ratio=real / pred if pred != 0 else np.nan,
                             cross_change=bx.J - Vx0, cross_change_bf=bfx.J - Vx0,
                             cross_pred=float(np.sum(gx["C"] * dC)), frozen_R=R,
                             switch_cross=bool(abs(bx.J - bfx.J) > 1e-7 * max(1, abs(bx.J))),
                             switch_self=bool(abs(bs.J - bfs.J) > 1e-7 * max(1, abs(bs.J))),
                             winner_cross=bx.start, winner_self=bs.start))
        print(f"  budget {B} done [{time.time() - t0:.0f}s]", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/cf_mitigation.csv", index=False)
    print(df.groupby(["budget", df.strategy.str.replace(r"\d+", "", regex=True)]).rel_change.mean().unstack().round(4))
    print(f"mitigation done in {time.time() - t0:.0f}s")


def anomaly_after(base, t, Gp, prm=PRM_CF):
    """Anomaly scores after replacing G_t by Gp (portfolio and branch-following)."""
    D, Dbf = base.Delta.copy(), base.Delta.copy()
    bs, bfs = resolve(Gp, Gp, base.self_sol[t])
    switches = int(abs(bs.J - bfs.J) > 1e-7 * max(1, abs(bs.J)))
    for s in range(T):
        if s == t:
            continue
        sol = base.cross[(min(t, s), max(t, s))]
        warm = sol.P if t < s else sol.P.T
        best, allr = solve_V(Gp, base.states[s], prm["alpha"], prm["gamma"], prm["eps"], tol=prm["tol"],
                             return_all=True)
        bf = solve_V(Gp, base.states[s], prm["alpha"], prm["gamma"], prm["eps"], warm=warm, tol=prm["tol"])
        switches += int(abs(best.J - bf.J) > 1e-7 * max(1, abs(best.J)))
        D[t, s] = D[s, t] = best.J - 0.5 * bs.J - 0.5 * base.self_sol[s].J
        Dbf[t, s] = Dbf[s, t] = bf.J - 0.5 * bfs.J - 0.5 * base.self_sol[s].J
    return D.sum(1) / (T - 1), Dbf.sum(1) / (T - 1), switches


def stage_stress(budgets=(8, 32, 128, 512, 1024, 2048), n_random=2, channels=("C",)):
    """Stress test: smallest targeted perturbation of the 2020 network that makes 2020 the most
    distinctive year (largest anomaly score)."""
    t0 = time.time()
    d = load("base")
    panel, base = d["panel"], d["base"]
    sc = Scales(panel)
    rng = np.random.default_rng(RNG_SEED + 2)
    t20 = YEARS.index(2020)
    a0 = base.anomaly()
    others = [u for u in range(T) if u != t20]
    # objective: margin m = a(2020) - max_{u != 2020} a(u); first-order gradient uses the leader
    lead = others[int(np.argmax(a0[others]))]
    g_own = base.grad_anomaly(t20, t20)
    g_lead = base.grad_anomaly(lead, t20)
    g_m = gaxpy(dict(g_own), g_lead, -1.0)
    E_m = sc.elastic(g_m, t20)["C"]
    E_a = sc.elastic(g_own, t20)["C"]
    elig = sc.eligible[t20] & (E_m != 0)
    lc = np.log(np.where(panel["obs"], panel["raw_cost"], np.nan))
    other_mean = np.nanmean(np.where(panel["obs"][others], lc[others], np.nan), axis=0)
    dev = np.nan_to_num(lc[t20] - other_mean)
    aux = dict(mass_pair=np.outer(panel["mu"][t20], panel["mu"][t20]), dev=dev, dev_sign=+1.0)
    rows = []
    strategies = ["gradient", "cost-change", "mass-weighted", "uniform"] + [f"random{k}" for k in range(n_random)]
    for B in budgets:
        for st in strategies:
            name = "random" if st.startswith("random") else st
            U = strategy_perturbation(name, E_m, B, +1.0, elig, rng=rng, aux=aux)
            Gp = sc.perturbed_state(t20, U=U)
            a1, a1bf, sw = anomaly_after(base, t20, Gp)
            m0 = a0[t20] - a0[others].max()
            m1 = a1[t20] - a1[others].max()
            rows.append(dict(budget=B, strategy=st, n_corr=int(np.sum(U != 0)),
                             pred_margin=float(np.sum(E_m * U)), real_margin=m1 - m0,
                             pred_a2020=float(np.sum(E_a * U)), real_a2020=a1[t20] - a0[t20],
                             real_a2020_bf=a1bf[t20] - a0[t20], margin_after=m1,
                             rank2020=int((-a1).argsort().argsort()[t20] + 1), switches=sw,
                             **{f"a{YEARS[u]}": a1[u] for u in range(T)}))
        print(f"  budget {B} done [{time.time() - t0:.0f}s]", flush=True)
    # an across-the-board increase of all 2020 costs, for reference
    for x in (0.05, 0.10, 0.25, 0.5):
        U = np.where(sc.eligible[t20], np.log1p(x), 0.0)
        Gp = sc.perturbed_state(t20, U=U)
        a1, a1bf, sw = anomaly_after(base, t20, Gp)
        rows.append(dict(budget=float(np.abs(U).sum()), strategy=f"across-the-board +{int(100 * x)}%",
                         n_corr=int(np.sum(U != 0)), pred_margin=float(np.sum(E_m * U)),
                         real_margin=(a1[t20] - a1[others].max()) - (a0[t20] - a0[others].max()),
                         pred_a2020=float(np.sum(E_a * U)), real_a2020=a1[t20] - a0[t20],
                         real_a2020_bf=a1bf[t20] - a0[t20],
                         margin_after=a1[t20] - a1[others].max(),
                         rank2020=int((-a1).argsort().argsort()[t20] + 1), switches=sw,
                         **{f"a{YEARS[u]}": a1[u] for u in range(T)}))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/cf_stress.csv", index=False)
    print(df[["budget", "strategy", "n_corr", "pred_margin", "real_margin", "margin_after", "rank2020", "switches"]].round(4).to_string())
    print(f"stress done in {time.time() - t0:.0f}s")


def stage_timing(budgets=(0.5, 1, 2, 4, 8, 16), n_random=3, side="b"):
    """Counterfactual for the regional lead-lag index: perturb SSEA corridor costs (all years)
    to remove the alignment lead of EAP over SSEA."""
    t0 = time.time()
    d = load("base")
    panel = d["panel"]
    sc = Scales(panel)
    R = load("regional")
    X = R["pairs"][("EAP", "SSEA")]
    ridx = region_index(panel)
    idx = ridx["SSEA"]
    rng = np.random.default_rng(RNG_SEED + 3)
    tau0, Gtau = X.tau_grad_matrix()
    E = np.array([sc.elastic(X.grad_tau_B(s, Gtau), s, idx)["C"] for s in range(T)])  # (T,n,n)
    elig = np.array([sc.eligible[s][np.ix_(idx, idx)] for s in range(T)]) & (E != 0)
    mass = np.array([np.outer(panel["mu"][s][idx], panel["mu"][s][idx]) for s in range(T)])
    aux = dict(mass_pair=mass)
    direction = -np.sign(tau0)       # move tau toward zero
    rows = []
    strategies = ["gradient", "mass-weighted", "uniform"] + [f"random{k}" for k in range(n_random)]
    A = X.A
    for Bdg in budgets:
        for st in strategies:
            name = "random" if st.startswith("random") else st
            U = strategy_perturbation(name, E, Bdg, direction, elig, rng=rng, aux=aux)
            Bs = [sc.perturbed_state(s, U=U[s], idx=idx) for s in range(T)]
            Vb = [solve_V(b, b, PRM_CF["alpha"], PRM_CF["gamma"], PRM_CF["eps"], tol=PRM_CF["tol"]).J for b in Bs]
            Dab = np.zeros((T, T))
            Dab_bf = np.zeros((T, T))
            sw = 0
            for t in range(T):
                for s in range(T):
                    best = solve_V(A.states[t], Bs[s], PRM_CF["alpha"], PRM_CF["gamma"], PRM_CF["eps"],
                                   same_nodes=False, tol=PRM_CF["tol"])
                    bf = solve_V(A.states[t], Bs[s], PRM_CF["alpha"], PRM_CF["gamma"], PRM_CF["eps"],
                                 warm=X.cross[(t, s)].P, tol=PRM_CF["tol"])
                    sw += int(abs(best.J - bf.J) > 1e-7 * max(1, abs(best.J)))
                    Dab[t, s] = best.J - 0.5 * A.self_sol[t].J - 0.5 * Vb[s]
                    Dab_bf[t, s] = bf.J - 0.5 * A.self_sol[t].J - 0.5 * Vb[s]
            tau1, tau1bf = leadlag(Dab)[0], leadlag(Dab_bf)[0]
            rows.append(dict(budget=Bdg, strategy=st, n_corr=int(np.sum(U != 0)),
                             pred=float(np.sum(E * U)), realized=tau1 - tau0, realized_bf=tau1bf - tau0,
                             tau_after=tau1, switches=sw))
        print(f"  budget {Bdg} done [{time.time() - t0:.0f}s]", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/cf_timing.csv", index=False)
    print(df.round(4).to_string())
    print(f"timing done in {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------------------
# stage: uncertainty (WP7)
# ----------------------------------------------------------------------------------------
def stage_uncertainty(sigmas=(0.1, 0.25), n_mc=12, n_mask=4, drop=0.05):
    t0 = time.time()
    d = load("base")
    panel, base = d["panel"], d["base"]
    sc = Scales(panel)
    rng = np.random.default_rng(RNG_SEED + 4)
    grads = anomaly_gradients(base)
    a0 = base.anomaly()
    t16, t20, t21 = YEARS.index(2016), YEARS.index(2020), YEARS.index(2021)
    # delta-method standard deviations under independent multiplicative cost errors
    ES = [[sc.elastic(grads[t][u], u)["C"] for u in range(T)] for t in range(T)]
    rows = []
    for sig in sigmas:
        sd = [sig * np.sqrt(sum(np.sum(ES[t][u] ** 2) for u in range(T))) for t in range(T)]
        sd_m1 = sig * np.sqrt(sum(np.sum((ES[t16][u] - ES[t20][u]) ** 2) for u in range(T)))
        sd_m2 = sig * np.sqrt(sum(np.sum((ES[t21][u] - ES[t20][u]) ** 2) for u in range(T)))
        mc = []
        for r in range(n_mc):
            states = []
            for t in range(T):
                Z = rng.standard_normal(panel["C"][t].shape)
                U = np.where(panel["obs"][t], sig * Z, 0.0)
                states.append(sc.perturbed_state(t, U=U))
            trj = SelfTrajectory(states, PRM_CF)
            mc.append(trj.anomaly())
            print(f"   sigma={sig} draw {r} [{time.time() - t0:.0f}s]", flush=True)
        mc = np.array(mc)
        ranks = np.array([(-x).argsort().argsort() + 1 for x in mc])
        for t in range(T):
            rows.append(dict(sigma=sig, year=YEARS[t], a0=a0[t], sd_delta=sd[t], sd_mc=mc[:, t].std(ddof=1),
                             mean_shift_mc=mc[:, t].mean() - a0[t], p_rank1=np.mean(ranks[:, t] == 1),
                             mean_rank=ranks[:, t].mean()))
        rows.append(dict(sigma=sig, year="2016-2020", a0=a0[t16] - a0[t20], sd_delta=sd_m1,
                         sd_mc=(mc[:, t16] - mc[:, t20]).std(ddof=1),
                         mean_shift_mc=(mc[:, t16] - mc[:, t20]).mean() - (a0[t16] - a0[t20]),
                         p_rank1=np.mean(mc[:, t16] > mc[:, t20]), mean_rank=np.nan))
        rows.append(dict(sigma=sig, year="2021-2020", a0=a0[t21] - a0[t20], sd_delta=sd_m2,
                         sd_mc=(mc[:, t21] - mc[:, t20]).std(ddof=1),
                         mean_shift_mc=(mc[:, t21] - mc[:, t20]).mean() - (a0[t21] - a0[t20]),
                         p_rank1=np.mean(mc[:, t21] > mc[:, t20]), mean_rank=np.nan))
        np.save(f"{OUT}/unc_mc_sigma{sig}.npy", mc)
    pd.DataFrame(rows).to_csv(f"{OUT}/unc_delta_vs_mc.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string())
    # observation uncertainty: randomly drop observed pairs and recompute the influence rankings
    E0 = [sc.elastic(grads[t][t], t)["C"] for t in range(T)]
    S0 = [node_scores(E0[t], panel["W"][t])["S"] for t in range(T)]
    mrows = []
    for r in range(n_mask):
        masks = []
        for t in range(T):
            W = panel["W"][t].copy()
            off = (W > 0) & ~np.eye(W.shape[0], dtype=bool)
            kill = off & (rng.random(W.shape) < drop)
            W[kill] = 0.0
            masks.append(W)
        states = [State(panel["mu"][t], panel["F"][t], panel["C"][t] * masks[t], masks[t]) for t in range(T)]
        trj = SelfTrajectory(states, PRM_CF)
        a1 = trj.anomaly()
        for t in range(T):
            e1 = sc.elastic(trj.grad_anomaly(t, t), t)["C"]
            keep = (masks[t] > 0) & (E0[t] != 0)
            s1 = node_scores(e1, masks[t])["S"]
            mrows.append(dict(draw=r, year=YEARS[t], a=a1[t], rank=int((-a1).argsort().argsort()[t] + 1),
                              rho_corr=spearman(np.abs(e1[keep]), np.abs(E0[t][keep])),
                              top100=topk_overlap(np.where(keep, e1, 0), np.where(keep, E0[t], 0), 100),
                              rho_nodes=spearman(s1, S0[t]),
                              top10_nodes=topk_overlap(s1, S0[t], 10)))
        print(f"   mask draw {r} [{time.time() - t0:.0f}s]", flush=True)
    pd.DataFrame(mrows).to_csv(f"{OUT}/unc_masks.csv", index=False)
    print(pd.DataFrame(mrows).groupby("year").mean(numeric_only=True).round(3).to_string())
    print(f"uncertainty done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "base"
    dict(base=stage_base, maps=stage_maps, regional=stage_regional, fd=stage_fd,
         mitigation=stage_mitigation, stress=stage_stress, timing=stage_timing,
         uncertainty=stage_uncertainty)[stage]()
