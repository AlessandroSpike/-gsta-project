"""Empirical application: the international freight-cost network 2016-2021 (Section 5).

Stages (each writes results/emp_*.{csv,json,npz}; run all with no argument or a
subset, e.g. `python run_empirical.py e1 e2`):
  e0  data diagnostics (coverage, missingness, imputation shares, asymmetry)
  e1  main specification: Delta along the alpha path, components, directed
      decomposition, MDS coordinates, trend-adjusted anomaly, multistart and
      kernel positive-definiteness diagnostics
  e2  Shapley attribution of Delta_ts to geometry vs node attributes
  e3  robustness grid (missing-data handling, masses, features, eps, gamma,
      original preprocessing)
  e4  node-subsampling uncertainty (m = 80% of countries)
  e5  regional lead-lag (6 regions, directed and symmetrised geometry)
  e6  gradient-based corridor and node attribution (envelope gradients)
"""
import os, sys, time, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gsta.data import load_panel, build_states, restrict, GraphState
from gsta.ufgw import UFGWProblem, ufgw_multistart, feature_cost, envelope_gradients, solve_ufgw
from gsta.core import value_matrix, self_values, debias, anomaly_scores, ufgw_pair
from gsta.softdtw import soft_dtw, soft_dtw_grad, lead_lag, boundary_corrected_lag

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)
YEARS = list(range(2016, 2022))
MAIN = dict(alpha=0.5, gamma=1.0, eps=0.05)
INITS = ("identity", "product", "feature")

REGIONS = {
    "East Asia & Pacific": ["China", "China, Hong Kong SAR", "Japan", "Republic of Korea", "Australia",
                            "New Zealand", "Samoa"],
    "South & South-East Asia": ["Cambodia", "India", "Indonesia", "Malaysia", "Pakistan", "Philippines",
                                "Singapore", "Thailand", "Viet Nam"],
    "Europe": ["Albania", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Denmark", "Estonia", "Finland",
               "Germany", "Greece", "Iceland", "Ireland", "Italy", "Latvia", "Lithuania", "Malta",
               "Netherlands (Kingdom of the)", "Poland", "Portugal", "Romania", "Russian Federation",
               "Slovenia", "Spain", "Sweden", "Ukraine"],
    "Middle East & North Africa": ["Egypt", "Israel", "Jordan", "Lebanon", "Morocco", "Oman", "Qatar",
                                   "Saudi Arabia", "Sudan", "Tunisia", "Turkiye", "United Arab Emirates",
                                   "Yemen"],
    "Sub-Saharan Africa": ["Angola", "Cameroon", "Cote d'Ivoire", "Ghana", "Kenya", "Mauritius", "Namibia",
                           "Nigeria", "Senegal", "Togo", "United Republic of Tanzania"],
    "Latin America & Caribbean": ["Argentina", "Aruba", "Brazil", "Colombia", "Dominican Republic", "Ecuador",
                                  "Grenada", "Guatemala", "Guyana", "Honduras", "Jamaica", "Mexico",
                                  "Nicaragua", "Panama", "Peru", "Saint Lucia",
                                  "Saint Vincent and the Grenadines", "Suriname", "Trinidad and Tobago",
                                  "Uruguay", "Venezuela (Bolivarian Rep. of)"],
}


def log(*a):
    print(*a, flush=True)


def delta_self(states, kw, inits=INITS, keep=False):
    """6x6 debiased matrix for one trajectory; optionally keep the optimal results."""
    T = len(states)
    V = np.zeros((T, T)); res = {}
    for t in range(T):
        for s in range(t, T):
            v, r, pr = ufgw_pair(states[t], states[s], inits=inits, return_result=True, **kw)
            V[t, s] = V[s, t] = v
            if keep:
                res[(t, s)] = (r, pr)
    d = np.diag(V).copy()
    D = debias(V, d, d); np.fill_diagonal(D, 0.0)
    return (D, V, res) if keep else D


def trend_adjusted(D):
    T = D.shape[0]
    lag = np.abs(np.subtract.outer(np.arange(T), np.arange(T)))
    prof = np.array([D[lag == h].mean() for h in range(1, T)])
    Rz = D - np.where(lag > 0, prof[np.maximum(lag - 1, 0)], 0.0)
    off = lag > 0
    return np.array([Rz[t][off[t]].mean() for t in range(T)]), prof


def mds(D, k=2):
    T = D.shape[0]
    J = np.eye(T) - np.ones((T, T)) / T
    B = -0.5 * J @ np.maximum(D, 0) @ J
    w, U = np.linalg.eigh(B)
    o = np.argsort(-w)
    w, U = w[o], U[:, o]
    return U[:, :k] * np.sqrt(np.maximum(w[:k], 0)), w


def sym_antisym_energy(pr, P):
    """Exact split of the structural energy Q(P) into symmetric and antisymmetric parts."""
    def Q(C, D, Wc, Wd):
        p = UFGWProblem(pr.mu, pr.nu, pr.M, C, D, alpha=0.5, Wc=Wc, Wd=Wd)
        return float((p.T(P) * P).sum())
    Wc = pr.Wc if pr.masked else None
    Wd = pr.Wd if pr.masked else None
    if pr.masked:
        # masks enter symmetrically: use pairs observed in both directions for the split
        Wc2 = Wc * Wc.T; Wd2 = Wd * Wd.T
        CS, CA = 0.5 * (pr.C + pr.C.T), 0.5 * (pr.C - pr.C.T)
        DS, DA = 0.5 * (pr.D + pr.D.T), 0.5 * (pr.D - pr.D.T)
        return Q(CS, DS, Wc2, Wd2), Q(CA, DA, Wc2, Wd2), Q(pr.C, pr.D, Wc2, Wd2)
    CS, CA = 0.5 * (pr.C + pr.C.T), 0.5 * (pr.C - pr.C.T)
    DS, DA = 0.5 * (pr.D + pr.D.T), 0.5 * (pr.D - pr.D.T)
    return Q(CS, DS, None, None), Q(CA, DA, None, None), Q(pr.C, pr.D, None, None)


# ---------------------------------------------------------------------- e0
def e0(panel, states, meta):
    T, N = panel.teu.shape
    obs = meta["obs"]
    L = np.log(panel.rate)
    both = np.isfinite(L).all(0)
    Lp = L[:, both]
    rows = []
    for t, y in enumerate(YEARS):
        A = L[t]; ok = np.isfinite(A) & np.isfinite(A.T); np.fill_diagonal(ok, False)
        rows.append(dict(year=y, observed_share=float(obs[t][~np.eye(N, dtype=bool)].mean()),
                         median_rate_persistent=float(np.exp(np.median(Lp[t]))),
                         mean_log_rate_persistent=float(Lp[t].mean()),
                         total_TEU_million=float(panel.teu[t].sum() / 1e6),
                         mean_log_LSCI=float(np.log(panel.lsci[t]).mean()),
                         n_pairs_both_directions=int(ok.sum() // 2),
                         mean_abs_log_ratio_ij_ji=float(np.mean(np.abs(A[ok] - A.T[ok]))),
                         corr_ij_ji=float(np.corrcoef(A[ok], A.T[ok])[0, 1]),
                         AI_observed=float(np.linalg.norm(np.where(ok, states[t].C - states[t].C.T, 0))
                                           / np.linalg.norm(np.where(ok, states[t].C, 0)))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "emp_e0_descriptives.csv"), index=False)
    # imputation diagnostics of the ORIGINAL preprocessing
    p91 = load_panel(DATA, drop_incomplete_teu=False)
    st0, _ = build_states(p91, original_preprocessing=True)
    o91 = np.isfinite(p91.rate)
    share = [float((st0[t].C[~o91[t] & ~np.eye(len(p91.countries), dtype=bool)] ** 2).sum() / (st0[t].C ** 2).sum())
             for t in range(T)]
    teu_missing = {c: [YEARS[t] for t in range(T) if not np.isfinite(p91.teu[t, i])]
                   for i, c in enumerate(p91.countries) if not np.isfinite(p91.teu[:, i]).all()}
    colmiss = {p91.countries[j]: int((~o91[:, :, j]).sum() - T) for j in range(len(p91.countries))}
    worst_cols = dict(sorted(colmiss.items(), key=lambda kv: -kv[1])[:5])
    info = dict(N=N, N_original=len(p91.countries), countries=panel.countries, dropped=panel.dropped,
                winsor_log1p=list(meta["winsor"]), winsor_usd=list(np.expm1(meta["winsor"])),
                geometry_scale=meta["scale"], pairs_all_years=int(both.sum()), pairs_total=N * (N - 1),
                imputed_share_sqfrob_original=share, teu_missing_original=teu_missing,
                most_missing_destination_columns_original=worst_cols,
                yoy_sd_log_rate=list(np.diff(Lp, axis=0).std(1)),
                yoy_mean_log_rate=list(np.diff(Lp, axis=0).mean(1)),
                corr_log_rate_years=np.corrcoef(Lp).tolist(),
                eff_nodes_sqrt=float(1 / np.sum((states[3].mu / states[3].mu.sum()) ** 2)))
    json.dump(info, open(os.path.join(OUT, "emp_e0_info.json"), "w"), indent=1, default=float)
    log(df.round(3).to_string())


# ---------------------------------------------------------------------- e1
def e1(states):
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    summ, comp_rows, dir_rows, ms_rows = [], [], [], []
    for a in alphas:
        t0 = time.time()
        kw = dict(MAIN, alpha=a)
        D, V, res = delta_self(states, kw, keep=True)
        np.save(os.path.join(OUT, f"emp_e1_Delta_alpha{a:.2f}.npy"), D)
        an = anomaly_scores(D)
        ta, prof = trend_adjusted(D)
        X, ev = mds(D)
        lam_pos = ev[ev > 0].sum(); lam_abs = np.abs(ev).sum()
        dist = np.sqrt(np.maximum(D, 0)); emb = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
        iu = np.triu_indices(6, 1)
        stress = float(np.sqrt(((dist[iu] - emb[iu]) ** 2).sum() / (dist[iu] ** 2).sum()))
        json.dump(dict(alpha=a, eigenvalues=[float(x) for x in ev], share_first_two_of_positive=float(ev[:2].sum() / lam_pos),
                       negative_share_of_total=float(np.abs(ev[ev < 0]).sum() / lam_abs), kruskal_stress1=stress),
                  open(os.path.join(OUT, f"emp_e1_mds_alpha{a:.2f}.json"), "w"), indent=1)
        for t, y in enumerate(YEARS):
            summ.append(dict(alpha=a, year=y, anomaly=an[t], trend_adjusted=ta[t],
                             mds1=X[t, 0], mds2=X[t, 1]))
        # components (exact additive decomposition of Delta at the optimal couplings)
        comps = {k: np.zeros((6, 6)) for k in ["feature", "structure", "kl", "entropy"]}
        Qs = {k: np.zeros((6, 6)) for k in ["sym", "antisym", "both"]}
        for (t, s), (r, pr) in res.items():
            for k in comps:
                comps[k][t, s] = comps[k][s, t] = r.components[k]
            if a > 0:
                qs, qa, qb = sym_antisym_energy(pr, r.P)
                for k, v in zip(["sym", "antisym", "both"], [qs, qa, qb]):
                    Qs[k][t, s] = Qs[k][s, t] = v
            # multistart diagnostics
            if t != s and a in (0.5,):
                vals = {}
                for ini in INITS:
                    vals[ini] = ufgw_pair(states[t], states[s], inits=(ini,), **kw)
                best = min(vals.values())
                ms_rows.append(dict(alpha=a, t=YEARS[t], s=YEARS[s], **{f"V_{k}": v for k, v in vals.items()},
                                    max_excess=max(v - best for v in vals.values()), winner=r.init))
        for k, M in comps.items():
            d = np.diag(M).copy(); Dk = debias(M, d, d); np.fill_diagonal(Dk, 0)
            for t in range(6):
                for s in range(t + 1, 6):
                    comp_rows.append(dict(alpha=a, t=YEARS[t], s=YEARS[s], component=k, value=Dk[t, s]))
        if a > 0:
            for k, M in Qs.items():
                d = np.diag(M).copy(); Dk = a * debias(M, d, d); np.fill_diagonal(Dk, 0)
                for t in range(6):
                    for s in range(t + 1, 6):
                        dir_rows.append(dict(alpha=a, t=YEARS[t], s=YEARS[s], part=k, value=Dk[t, s]))
        # positive-definiteness of the local Gibbs kernel exp(-Delta/beta) on the observed states
        med = np.median(D[~np.eye(6, dtype=bool)])
        for bf in [0.05, 0.1, 0.5, 1.0]:
            K = np.exp(-D / (bf * med))
            summ.append(dict(alpha=a, year=-1, anomaly=np.nan, trend_adjusted=np.nan, mds1=bf,
                             mds2=float(np.linalg.eigvalsh(K).min()),
                             kernel_kk=float(np.linalg.eigvalsh(K / (1 + K)).min())))
        log("alpha %.2f: anomaly %s  (%.0fs)" % (a, np.round(an, 4), time.time() - t0))
    pd.DataFrame(summ).to_csv(os.path.join(OUT, "emp_e1_summary.csv"), index=False)
    pd.DataFrame(comp_rows).to_csv(os.path.join(OUT, "emp_e1_components.csv"), index=False)
    pd.DataFrame(dir_rows).to_csv(os.path.join(OUT, "emp_e1_directed.csv"), index=False)
    pd.DataFrame(ms_rows).to_csv(os.path.join(OUT, "emp_e1_multistart.csv"), index=False)


# ---------------------------------------------------------------------- e2
def hybrid(a, b, take):
    if take == "geometry":
        return GraphState(a.mu, a.F, b.C, b.W, a.year)
    return GraphState(b.mu, b.F, a.C, a.W, a.year)


def e2(states):
    kw = MAIN
    rows = []
    vself = [ufgw_pair(s, s, inits=INITS, **kw) for s in states]
    for t in range(6):
        for s in range(t + 1, 6):
            a, b = states[t], states[s]
            def Dpair(x, y, vx, vy=None):
                vy = ufgw_pair(y, y, inits=INITS, **kw) if vy is None else vy
                return ufgw_pair(x, y, inits=INITS, **kw) - 0.5 * vx - 0.5 * vy
            v_both = Dpair(a, b, vself[t], vself[s])
            v_geo = Dpair(a, hybrid(a, b, "geometry"), vself[t])
            v_node = Dpair(a, hybrid(a, b, "node"), vself[t])
            phi_g = 0.5 * v_geo + 0.5 * (v_both - v_node)
            phi_n = 0.5 * v_node + 0.5 * (v_both - v_geo)
            rows.append(dict(t=YEARS[t], s=YEARS[s], Delta=v_both, v_geometry_only=v_geo, v_node_only=v_node,
                             shapley_geometry=phi_g, shapley_node=phi_n, interaction=v_both - v_geo - v_node))
            log(rows[-1])
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "emp_e2_shapley.csv"), index=False)


# ---------------------------------------------------------------------- e3
def e3(panel, D_main):
    specs = {
        "main (mask, sqrt mass, log LSCI)": dict(),
        "two-way FE imputation": dict(missing="fe"),
        "row-maximum imputation": dict(missing="rowmax"),
        "raw TEU mass": dict(mass="raw"),
        "log TEU mass": dict(mass="log"),
        "raw LSCI feature": dict(feature="lsci"),
        "no winsorisation": dict(winsor=(0.0, 1.0)),
        "symmetrised geometry": dict(symmetrise=True),
        "common support (pairs observed in all years)": dict(common_support=True),
    }
    hyp = {"eps = 0.02": dict(eps=0.02), "eps = 0.1": dict(eps=0.1),
           "gamma = 0.5": dict(gamma=0.5), "gamma = 2": dict(gamma=2.0)}
    off = ~np.eye(6, dtype=bool)
    rows = []
    PART = os.path.join(OUT, "parts"); os.makedirs(PART, exist_ok=True)
    def cached(name, fun):
        fn = os.path.join(PART, "e3_" + "".join(ch if ch.isalnum() else "_" for ch in name) + ".npy")
        if os.path.exists(fn):
            return np.load(fn)
        D = fun(); np.save(fn, D); return D
    def record(name, D):
        an = anomaly_scores(D); ta, _ = trend_adjusted(D)
        rho = float(pd.Series(D[off]).corr(pd.Series(D_main[off]), method="spearman"))
        for t, y in enumerate(YEARS):
            rows.append(dict(spec=name, year=y, anomaly=an[t], rank=int((an > an[t]).sum()) + 1,
                             trend_adjusted=ta[t], spearman_vs_main=rho))
        log(name, np.round(an, 4), "rho=%.3f" % rho)
    for name, kwb in specs.items():
        st, _ = build_states(panel, **{**dict(mass="sqrt"), **kwb})
        record(name, D_main if not kwb else cached(name, lambda: delta_self(st, MAIN)))
    st, _ = build_states(panel, mass="sqrt")
    for name, kwh in hyp.items():
        record(name, cached(name, lambda: delta_self(st, {**MAIN, **kwh})))
    p91 = load_panel(DATA, drop_incomplete_teu=False)
    st0, _ = build_states(p91, original_preprocessing=True)
    D0 = cached("original", lambda: delta_self(st0, dict(alpha=0.7, gamma=1.0, eps=0.05)))
    an = anomaly_scores(D0)
    for t, y in enumerate(YEARS):
        rows.append(dict(spec="original preprocessing (91 countries), corrected solver, alpha=0.7",
                         year=y, anomaly=an[t], rank=int((an > an[t]).sum()) + 1,
                         trend_adjusted=trend_adjusted(D0)[0][t], spearman_vs_main=np.nan))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "emp_e3_robustness.csv"), index=False)


# ---------------------------------------------------------------------- e4
def e4(states, B=None, frac=0.8):
    B = int(os.environ.get("E4_B", 30)) if B is None else B
    rng = np.random.default_rng(42)
    N = states[0].mu.shape[0]
    m = int(round(frac * N))
    rows, mats = [], []
    t0 = time.time()
    PART = os.path.join(OUT, "parts"); os.makedirs(PART, exist_ok=True)
    for b in range(B):
        idx = np.sort(rng.choice(N, m, replace=False))
        fn = os.path.join(PART, f"e4_{b}.npy")
        if os.path.exists(fn):
            D = np.load(fn)
        else:
            D = delta_self(restrict(states, idx), MAIN, inits=("identity",))
            np.save(fn, D)
        mats.append(D)
        an = anomaly_scores(D); ta, _ = trend_adjusted(D)
        for t, y in enumerate(YEARS):
            rows.append(dict(b=b, year=y, anomaly=an[t], rank=int((an > an[t]).sum()) + 1, trend_adjusted=ta[t]))
        if b % 10 == 9:
            log("subsample %d (%.0fs)" % (b + 1, time.time() - t0))
            pd.DataFrame(rows).to_csv(os.path.join(OUT, "emp_e4_subsampling.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "emp_e4_subsampling.csv"), index=False)
    np.save(os.path.join(OUT, "emp_e4_Delta_subsamples.npy"), np.array(mats))


# ---------------------------------------------------------------------- e5
def _prob(g, h):
    p = UFGWProblem(g.mu, h.mu, feature_cost(g.F, h.F), g.C, h.C, Wc=g.W, Wd=h.W, **MAIN)
    p._f1, p._g1 = np.asarray(g.F)[:, 0], np.asarray(h.F)[:, 0]
    return p


def e5(panel, states, states_sym):
    idx = {r: [panel.countries.index(c) for c in cs if c in panel.countries] for r, cs in REGIONS.items()}
    assert sum(len(v) for v in idx.values()) == len(panel.countries), "regions must partition the panel"
    labels = list(REGIONS)
    rows = []
    PART = os.path.join(OUT, "parts"); os.makedirs(PART, exist_ok=True)
    geoms = os.environ.get("E5_GEOMS", "directed").split(",")
    for geom, st in [("directed", states), ("symmetrised", states_sym)]:
        if geom not in geoms:
            continue
        sub = {r: restrict(st, idx[r]) for r in labels}
        cf = os.path.join(PART, f"e5_self_{geom}.npz")
        if os.path.exists(cf):
            z = np.load(cf, allow_pickle=True)
            selfD, selfv = z["selfD"].item(), z["selfv"].item()
        else:
            selfD = {r: delta_self(sub[r], MAIN, inits=("identity",)) for r in labels}
            selfv = {r: np.array([ufgw_pair(s, s, inits=("identity",), **MAIN) for s in sub[r]]) for r in labels}
            np.savez(cf, selfD=np.array(selfD, dtype=object), selfv=np.array(selfv, dtype=object))
        inits = ("product", "feature", "quantile", "anneal") if geom == "directed" else ("product", "feature", "anneal")
        for i, a in enumerate(labels):
            for j, b in enumerate(labels):
                if j <= i:
                    continue
                fn = os.path.join(PART, f"e5_{geom}_{i}{j}.csv")
                if os.path.exists(fn):
                    continue
                t0 = time.time()
                V = value_matrix(sub[a], sub[b], inits=inits, **MAIN)
                D = debias(V, selfv[a], selfv[b])
                Dadj = D - D.min()
                beta = 0.1 * np.median(Dadj)
                _, E = soft_dtw_grad(Dadj, beta)
                tau = lead_lag(E)
                tau_raw = lead_lag(soft_dtw_grad(D, beta)[1])
                val = soft_dtw(D, beta) - 0.5 * soft_dtw(selfD[a], beta) - 0.5 * soft_dtw(selfD[b], beta)
                # positive-definiteness of the local Gibbs kernel on the 12 states of the two regions
                Big = np.block([[selfD[a], D], [D.T, selfD[b]]])
                Kg = np.exp(-Big / beta)
                pd_min, pdk_min = float(np.linalg.eigvalsh(Kg).min()), float(np.linalg.eigvalsh(Kg / (1 + Kg)).min())
                # sensitivity to the initialization portfolio: 10 additional deterministic random starts
                if geom == "directed":
                    Vx = np.array([[ufgw_multistart(_prob(sub[a][t], sub[b][s_]), same_nodes=False, inits=inits,
                                                    n_random=10).value for s_ in range(6)] for t in range(6)])
                    Dx = debias(Vx, selfv[a], selfv[b]); Dxa = Dx - Dx.min()
                    tau_x = lead_lag(soft_dtw_grad(Dxa, 0.1 * np.median(Dxa))[1])
                    impr = float((V - Vx).max()); impr_rel = float(((V - Vx) / np.abs(V)).max())
                else:
                    tau_x, impr, impr_rel = np.nan, np.nan, np.nan
                np.save(os.path.join(OUT, f"emp_e5_D_{geom}_{i}{j}.npy"), D)
                for (x, y, sg) in [(a, b, 1.0), (b, a, -1.0)]:
                    rows.append(dict(geometry=geom, region_a=x, region_b=y, tau=sg * tau,
                                     tau_raw=sg * tau_raw, lag_corrected=sg * boundary_corrected_lag(tau, 6),
                                     identifiable=abs(tau) <= 1.25, gsta=val, beta=beta, offset=float(D.min()),
                                     n_a=len(idx[x]), n_b=len(idx[y]), min_eig_K=pd_min, min_eig_K_over_1pK=pdk_min,
                                     tau_extended_portfolio=sg * tau_x, max_value_improvement=impr,
                                     max_rel_value_improvement=impr_rel))
                log(geom, a, "vs", b, "tau=%.3f raw=%.3f G-STA=%.4f (%.0fs)" % (tau, tau_raw, val, time.time() - t0))
                pd.DataFrame(rows[-2:]).to_csv(fn, index=False)
    import glob
    fs = sorted(glob.glob(os.path.join(PART, "e5_*.csv")))
    if fs:
        pd.concat([pd.read_csv(f) for f in fs]).to_csv(os.path.join(OUT, "emp_e5_leadlag.csv"), index=False)


# ---------------------------------------------------------------------- e6
def e6(panel, states):
    out = {}
    rows_c, rows_n = [], []
    for (t, s) in [(4, 5), (0, 1), (3, 4)]:
        a, b = states[t], states[s]
        va, ra, pa = ufgw_pair(a, a, inits=INITS, return_result=True, **MAIN)
        vb, rb, pb = ufgw_pair(b, b, inits=INITS, return_result=True, **MAIN)
        vab, rab, pab = ufgw_pair(a, b, inits=INITS, return_result=True, **MAIN)
        g_ab = envelope_gradients(pab, rab.P, a.F, b.F)
        g_bb = envelope_gradients(pb, rb.P, b.F, b.F)
        # d Delta_ab / d C_b = dV(a,b)/dD - 0.5 [dV(b,b)/dC + dV(b,b)/dD]
        gC = g_ab["D"] - 0.5 * (g_bb["C"] + g_bb["D"])
        gmu = g_ab["nu"] - 0.5 * (g_bb["mu"] + g_bb["nu"])
        gF = g_ab["G"] - 0.5 * (g_bb["F"] + g_bb["G"])
        dC = np.where((a.W > 0) & (b.W > 0), b.C - a.C, 0.0)
        np.fill_diagonal(dC, 0)
        contrib = gC * dC
        # finite-difference spot check of the geometry gradient on 3 corridors
        fd = []
        cand = np.dstack(np.unravel_index(np.argsort(-np.abs(gC * (b.W > 0)), axis=None)[:3], gC.shape))[0]
        for (i, j) in cand:
            h = 1e-5
            def Dval(Cb):
                bb = GraphState(b.mu, b.F, Cb, b.W, b.year)
                v1 = solve_ufgw(UFGWProblem(a.mu, bb.mu, feature_cost(a.F, bb.F), a.C, bb.C, Wc=a.W, Wd=bb.W, **MAIN),
                                P0=rab.P, tol=1e-12).value
                v2 = solve_ufgw(UFGWProblem(bb.mu, bb.mu, feature_cost(bb.F, bb.F), bb.C, bb.C, Wc=bb.W, Wd=bb.W, **MAIN),
                                P0=rb.P, tol=1e-12).value
                return v1 - 0.5 * v2
            Cp, Cm = b.C.copy(), b.C.copy(); Cp[i, j] += h; Cm[i, j] -= h
            fd.append(dict(i=panel.countries[i], j=panel.countries[j], analytic=gC[i, j],
                           finite_difference=(Dval(Cp) - Dval(Cm)) / (2 * h)))
        key = f"{YEARS[t]}-{YEARS[s]}"
        out[key] = dict(Delta=vab - 0.5 * va - 0.5 * vb, first_order_sum=float(contrib.sum()), fd_check=fd)
        order = np.argsort(-np.abs(contrib), axis=None)[:25]
        for k in order:
            i, j = np.unravel_index(k, contrib.shape)
            rows_c.append(dict(transition=key, origin=panel.countries[i], destination=panel.countries[j],
                               gradient=gC[i, j], change=dC[i, j], contribution=contrib[i, j],
                               rate_from=float(np.expm1(a.C[i, j] * 1)), rate_to=float(np.expm1(b.C[i, j] * 1))))
        for i, c in enumerate(panel.countries):
            rows_n.append(dict(transition=key, country=c, grad_mass=gmu[i], grad_feature=float(gF[i, 0]),
                               row_contribution=float(contrib[i].sum()), col_contribution=float(contrib[:, i].sum())))
        log(key, out[key])
    json.dump(out, open(os.path.join(OUT, "emp_e6_attribution.json"), "w"), indent=1, default=float)
    pd.DataFrame(rows_c).to_csv(os.path.join(OUT, "emp_e6_corridors.csv"), index=False)
    pd.DataFrame(rows_n).to_csv(os.path.join(OUT, "emp_e6_nodes.csv"), index=False)


if __name__ == "__main__":
    stages = sys.argv[1:] or ["e0", "e1", "e2", "e3", "e4", "e5", "e6"]
    panel = load_panel(DATA)
    states, meta = build_states(panel, mass="sqrt")
    states_sym, _ = build_states(panel, mass="sqrt", symmetrise=True)
    t0 = time.time()
    if "e0" in stages:
        e0(panel, states, meta)
    if "e1" in stages:
        e1(states)
    D_main = np.load(os.path.join(OUT, "emp_e1_Delta_alpha0.50.npy")) if os.path.exists(
        os.path.join(OUT, "emp_e1_Delta_alpha0.50.npy")) else None
    if "e2" in stages:
        e2(states)
    if "e3" in stages:
        e3(panel, D_main)
    if "e6" in stages:
        e6(panel, states)
    if "e5" in stages:
        e5(panel, states, states_sym)
    if "e4" in stages:
        e4(states)
    log("total %.0fs" % (time.time() - t0))
