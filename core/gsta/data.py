"""Construction of the maritime graph-valued trajectory from the three UNCTADstat
panels (bilateral transport costs, LSCI, container port throughput).

Main specification (see Section 5 of the paper):
  * node set: countries present in all three panels with complete TEU 2016-2021;
  * node mass  mu_{t,i} = TEU_{t,i} / TEUbar, TEUbar = (1/T) sum_t sum_i TEU_{t,i}
                (average total mass 1; year-to-year variation preserved);
  * node feature f_{t,i} = standardised log LSCI (pooled mean/sd);
  * geometry  c_{t,ij} = log(1 + rate_{t,ij}) winsorised at pooled 0.5%/99.5%
                quantiles and divided by the pooled sd; c_{t,ii} = 0;
  * missing directed pairs are NOT imputed: an observation mask W_t (W_t,ii = 1)
    enters the masked UFGW objective.
Alternative specifications (robustness): imputation by the original row maximum,
two-way fixed-effects imputation, sqrt / log mass transforms, symmetrised geometry.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

YEARS = list(range(2016, 2022))

# The UNCTADstat transport-cost file uses "Surname, Qualifier" economy labels for a few
# economies, whereas the LSCI and throughput files use the official short names.
# Without harmonisation these economies silently drop out of the intersection.
NAME_MAP = {
    "Korea, Republic of": "Republic of Korea",
    "Tanzania, United Republic of": "United Republic of Tanzania",
    "Congo, Dem. Rep. of the": "Dem. Rep. of the Congo",
}


@dataclass
class GraphState:
    mu: np.ndarray            # (n,)
    F: np.ndarray             # (n, K)
    C: np.ndarray             # (n, n)
    W: np.ndarray | None      # (n, n) observation mask or None
    year: int | None = None


@dataclass
class Panel:
    countries: list
    years: list
    teu: np.ndarray           # (T, N) raw TEU
    lsci: np.ndarray          # (T, N) raw LSCI
    rate: np.ndarray          # (T, N, N) raw USD/kg, NaN = missing, diag NaN
    dropped: dict


def load_panel(data_dir, years=YEARS, drop_incomplete_teu=True):
    fr = pd.read_csv(os.path.join(data_dir, "transport_costs_bilateral.csv"))
    ls = pd.read_csv(os.path.join(data_dir, "lsci_annual.csv"))
    te = pd.read_csv(os.path.join(data_dir, "container_throughput_masses.csv"))
    fr = fr.replace({"Origin": NAME_MAP, "Destination": NAME_MAP})
    fr = fr[fr.Year.isin(years)].dropna(subset=["FreightRate_USD_per_kg"])
    ls = ls[ls.Year.isin(years)].dropna(subset=["LSCI"])
    te = te[te.Year.isin(years)].dropna(subset=["TEU"])
    common = sorted((set(fr.Origin) | set(fr.Destination)) & set(ls.Country) & set(te.Country))
    dropped = {}
    if drop_incomplete_teu:
        cnt = te[te.Country.isin(common)].groupby("Country").Year.nunique()
        incomplete = sorted(set(common) - set(cnt[cnt == len(years)].index))
        dropped["incomplete_TEU"] = incomplete
        common = [c for c in common if c not in incomplete]
    cnt = ls[ls.Country.isin(common)].groupby("Country").Year.nunique()
    incomplete = sorted(set(common) - set(cnt[cnt == len(years)].index))
    dropped["incomplete_LSCI"] = incomplete
    common = [c for c in common if c not in incomplete]
    N, T = len(common), len(years)
    c2i = {c: i for i, c in enumerate(common)}
    teu = np.full((T, N), np.nan)
    lsci = np.full((T, N), np.nan)
    for r in te[te.Country.isin(common)].itertuples():
        teu[years.index(r.Year), c2i[r.Country]] = r.TEU
    for r in ls[ls.Country.isin(common)].itertuples():
        lsci[years.index(r.Year), c2i[r.Country]] = r.LSCI
    rate = np.full((T, N, N), np.nan)
    sub = fr[fr.Origin.isin(common) & fr.Destination.isin(common) & (fr.Origin != fr.Destination)]
    sub = sub[sub.FreightRate_USD_per_kg > 0]
    for r in sub.itertuples():
        rate[years.index(r.Year), c2i[r.Origin], c2i[r.Destination]] = r.FreightRate_USD_per_kg
    return Panel(common, list(years), teu, lsci, rate, dropped)


def _offdiag_mask(N):
    return ~np.eye(N, dtype=bool)


def twoway_fe_impute(L, obs, n_iter=200):
    """Impute log-rates by an additive model L_tij = a_ti + b_tj (per year),
    fitted by alternating least squares on observed off-diagonal cells."""
    T, N, _ = L.shape
    out = L.copy()
    off = _offdiag_mask(N)
    for t in range(T):
        o = obs[t] & off
        a = np.zeros(N)
        b = np.zeros(N)
        mu0 = L[t][o].mean()
        for _ in range(n_iter):
            R = np.where(o, L[t] - mu0 - b[None, :], 0.0)
            a = R.sum(1) / np.maximum(o.sum(1), 1)
            R = np.where(o, L[t] - mu0 - a[:, None], 0.0)
            b = R.sum(0) / np.maximum(o.sum(0), 1)
        fit = mu0 + a[:, None] + b[None, :]
        out[t] = np.where(o, L[t], fit)
        np.fill_diagonal(out[t], 0.0)
    return out


def build_states(panel: Panel, mass="raw", missing="mask", winsor=(0.005, 0.995),
                 symmetrise=False, feature="loglsci", geometry_scale="sd",
                 original_preprocessing=False, common_support=False):
    """Return list of GraphState, one per year, plus a dict of metadata."""
    T, N = panel.teu.shape
    off = _offdiag_mask(N)
    obs = np.isfinite(panel.rate)
    for t in range(T):
        np.fill_diagonal(obs[t], False)
    if common_support:
        # keep only pairs observed in every year: masks are then constant over time, so all
        # discrepancies come from changes in observed costs, masses and attributes
        obs = np.repeat(obs.all(0)[None], T, axis=0)

    if original_preprocessing:
        # Faithful re-implementation of the ORIGINAL code (for the audit only):
        # missing TEU -> 0, row-max imputation, log1p + global min-max, floor 1e-4.
        teu = np.nan_to_num(panel.teu, nan=0.0)
        lt = np.log1p(teu)
        mu = (lt - lt.min()) / (lt.max() - lt.min())
        mu = np.maximum(mu, 1e-4)
        ls = panel.lsci
        f = (ls - ls.min()) / (ls.max() - ls.min())
        R = panel.rate.copy()
        for t in range(T):
            for i in range(N):
                row = R[t, i][np.isfinite(R[t, i])]
                fb = row.max() if len(row) else 1.0
                miss = ~np.isfinite(R[t, i])
                miss[i] = False
                R[t, i, miss] = fb
            np.fill_diagonal(R[t], np.nan)
        Lg = np.log1p(R)
        Cm = (Lg - np.nanmin(Lg)) / (np.nanmax(Lg) - np.nanmin(Lg))
        Cm = np.nan_to_num(Cm, nan=0.0)
        states = [GraphState(mu[t].copy(), f[t][:, None].copy(), Cm[t].copy(), None, panel.years[t])
                  for t in range(T)]
        return states, dict(obs=obs, scale=None)

    # ---- masses ----
    teu = panel.teu
    if mass == "raw":
        base = teu
    elif mass == "sqrt":
        base = np.sqrt(teu)
    elif mass == "log":
        base = np.log1p(teu / 1e3)
    else:
        raise ValueError(mass)
    mu = base / (base.sum(1).mean())

    # ---- features ----
    if feature == "loglsci":
        x = np.log(panel.lsci)
    elif feature == "lsci":
        x = panel.lsci.astype(float)
    else:
        raise ValueError(feature)
    f = (x - x.mean()) / x.std()

    # ---- geometry ----
    Lg = np.log1p(panel.rate)                 # NaN where missing
    vals = Lg[obs]
    lo, hi = np.quantile(vals, winsor)
    Lg = np.clip(Lg, lo, hi)
    scale = float(np.nanstd(np.where(obs, Lg, np.nan))) if geometry_scale == "sd" else float(hi)
    Lg = Lg / scale
    if missing == "mask":
        C = np.where(obs, Lg, 0.0)
        W = obs.astype(float)
        for t in range(T):
            np.fill_diagonal(W[t], 1.0)
    elif missing == "fe":
        C = twoway_fe_impute(np.where(obs, Lg, 0.0), obs)
        W = None
    elif missing == "rowmax":
        C = np.where(obs, Lg, 0.0)
        for t in range(T):
            for i in range(N):
                o = obs[t, i]
                fb = C[t, i][o].max() if o.any() else 0.0
                C[t, i, (~o) & off[i]] = fb
        W = None
    else:
        raise ValueError(missing)
    for t in range(T):
        np.fill_diagonal(C[t], 0.0)
    if symmetrise:
        if W is None:
            C = 0.5 * (C + np.transpose(C, (0, 2, 1)))
        else:
            # symmetrise over pairs observed in both directions; one-sided pairs keep their value
            Wt = np.transpose(W, (0, 2, 1))
            Ct = np.transpose(C, (0, 2, 1))
            both = (W > 0) & (Wt > 0)
            C = np.where(both, 0.5 * (C + Ct), np.where(W > 0, C, Ct))
            W = np.maximum(W, Wt)
    states = []
    for t in range(T):
        states.append(GraphState(mu[t].copy(), f[t][:, None].copy(), C[t].copy(),
                                 None if W is None else W[t].copy(), panel.years[t]))
    meta = dict(obs=obs, winsor=(lo, hi), scale=scale, f_mean=x.mean(), f_sd=x.std(),
                mass_norm=base.sum(1).mean())
    return states, meta


def restrict(states, idx):
    idx = np.asarray(idx)
    out = []
    for s in states:
        W = None if s.W is None else s.W[np.ix_(idx, idx)].copy()
        out.append(GraphState(s.mu[idx].copy(), s.F[idx].copy(), s.C[np.ix_(idx, idx)].copy(), W, s.year))
    return out
