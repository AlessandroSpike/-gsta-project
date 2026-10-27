"""Construction of the 86-economy panel exactly as in Section 6.1 / Supplement S5."""
import numpy as np
import pandas as pd

YEARS = list(range(2016, 2022))
HARMONIZE = {
    "Korea, Republic of": "Republic of Korea",
    "Tanzania, United Republic of": "United Republic of Tanzania",
    "Congo, Dem. Rep. of the": "Dem. Rep. of the Congo",
}
REGIONS = {
    "EAP": ["China", "China, Hong Kong SAR", "Japan", "Republic of Korea", "Australia",
            "New Zealand", "Samoa"],
    "SSEA": ["Cambodia", "India", "Indonesia", "Malaysia", "Pakistan", "Philippines",
             "Singapore", "Thailand", "Viet Nam"],
    "EUR": ["Albania", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Denmark", "Estonia",
            "Finland", "Germany", "Greece", "Iceland", "Ireland", "Italy", "Latvia",
            "Lithuania", "Malta", "Netherlands (Kingdom of the)", "Poland", "Portugal", "Romania",
            "Russian Federation", "Slovenia", "Spain", "Sweden", "Ukraine"],
    "MENA": ["Egypt", "Israel", "Jordan", "Lebanon", "Morocco", "Oman", "Qatar",
             "Saudi Arabia", "Sudan", "Tunisia", "Turkiye", "United Arab Emirates", "Yemen"],
    "SSA": ["Angola", "Cameroon", "Cote d'Ivoire", "Ghana", "Kenya", "Mauritius", "Namibia",
            "Nigeria", "Senegal", "Togo", "United Republic of Tanzania"],
    "LAC": ["Argentina", "Aruba", "Brazil", "Colombia", "Dominican Republic", "Ecuador",
            "Grenada", "Guatemala", "Guyana", "Honduras", "Jamaica", "Mexico", "Nicaragua",
            "Panama", "Peru", "Saint Lucia", "Saint Vincent and the Grenadines", "Suriname",
            "Trinidad and Tobago", "Uruguay", "Venezuela (Bolivarian Rep. of)"],
}


def build_panel(path_cost="transport_costs_bilateral.csv", path_lsci="lsci_annual.csv",
                path_teu="container_throughput_masses.csv", years=YEARS, winsor=(0.005, 0.995),
                verbose=True):
    """Return dict with countries, mu (T,n), F (T,n,1), C (T,n,n), W (T,n,n), raw arrays."""
    tc = pd.read_csv(path_cost)
    ls = pd.read_csv(path_lsci)
    te = pd.read_csv(path_teu)
    for df in (tc, ls, te):
        df.columns = df.columns.str.strip()
    tc["Origin"] = tc["Origin"].replace(HARMONIZE)
    tc["Destination"] = tc["Destination"].replace(HARMONIZE)
    ls["Country"] = ls["Country"].replace(HARMONIZE)
    te["Country"] = te["Country"].replace(HARMONIZE)
    tc = tc[tc.Year.isin(years)]
    ls = ls[ls.Year.isin(years)]
    te = te[te.Year.isin(years)]

    inter = (set(tc.Origin) | set(tc.Destination)) & set(ls.Country) & set(te.Country)
    T = len(years)
    teu_p = te.pivot_table(index="Country", columns="Year", values="TEU", aggfunc="mean")
    full_teu = [c for c in inter if c in teu_p.index
                and np.all(np.isfinite(teu_p.loc[c, years].values.astype(float)))
                and np.all(teu_p.loc[c, years].values.astype(float) > 0)]
    excluded = sorted(inter - set(full_teu))
    countries = sorted(full_teu)
    n = len(countries)
    idx = {c: i for i, c in enumerate(countries)}
    if verbose:
        print(f"intersection: {len(inter)}; kept: {n}; excluded: {excluded}")

    teu = np.array([[teu_p.loc[c, y] for c in countries] for y in years], float)
    ls_p = ls.pivot_table(index="Country", columns="Year", values="LSCI", aggfunc="mean")
    lsci = np.array([[ls_p.loc[c, y] if (c in ls_p.index and y in ls_p.columns) else np.nan
                      for c in countries] for y in years], float)
    if np.any(~np.isfinite(lsci)) or np.any(lsci <= 0):
        raise ValueError("missing LSCI in retained panel")

    raw = np.full((T, n, n), np.nan)
    tc = tc[tc.Origin.isin(idx) & tc.Destination.isin(idx) & (tc.Origin != tc.Destination)]
    for y_i, y in enumerate(years):
        sub = tc[tc.Year == y]
        o = sub.Origin.map(idx).values
        d = sub.Destination.map(idx).values
        v = sub.FreightRate_USD_per_kg.values.astype(float)
        ok = np.isfinite(v) & (v > 0)          # one non-positive cost is treated as missing
        raw[y_i, o[ok], d[ok]] = v[ok]
    obs = np.isfinite(raw)
    for t in range(T):
        np.fill_diagonal(obs[t], False)

    # masses: sqrt(TEU), scaled so that the average total mass over the years is one
    sq = np.sqrt(teu)
    kappa = sq.sum() / T
    mu = sq / kappa
    # attributes: standardized log LSCI (pooled)
    ll = np.log(lsci)
    F = ((ll - ll.mean()) / ll.std())[:, :, None]
    # geometry: log(1+cost), winsorized at pooled quantiles, divided by pooled sd
    lc = np.log1p(raw)
    pooled = lc[obs]
    qlo, qhi = np.quantile(pooled, winsor[0]), np.quantile(pooled, winsor[1])
    lcw = np.clip(lc, qlo, qhi)
    s_c = lcw[obs].std()
    C = np.where(obs, lcw / s_c, 0.0)
    W = obs.astype(float)
    for t in range(T):
        np.fill_diagonal(C[t], 0.0)
        np.fill_diagonal(W[t], 1.0)
    info = dict(qlo_usd=np.expm1(qlo), qhi_usd=np.expm1(qhi), s_c=s_c, kappa=kappa,
                logLSCI_mean=ll.mean(), logLSCI_sd=ll.std(), excluded=excluded)
    return dict(countries=countries, years=list(years), mu=mu, F=F, C=C, W=W,
                raw_cost=raw, obs=obs, teu=teu, lsci=lsci, info=info)


def describe(panel):
    raw, obs, years = panel["raw_cost"], panel["obs"], panel["years"]
    n = raw.shape[1]
    allyrs = obs.all(axis=0)
    print("pairs observed in all years:", allyrs.sum(), " of", n * (n - 1))
    rows = []
    for t, y in enumerate(years):
        o = obs[t]
        both = o & o.T
        iu = np.triu_indices(n, 1)
        bmask = both[iu]
        r1 = raw[t][iu][bmask]
        r2 = raw[t].T[iu][bmask]
        rows.append(dict(Year=y, observed=o.sum() / (n * (n - 1)),
                         median_all=np.median(raw[t][allyrs]),
                         TEU_m=panel["teu"][t].sum() / 1e6,
                         mean_logLSCI=np.log(panel["lsci"][t]).mean(),
                         mean_abs_logratio=np.mean(np.abs(np.log(r1 / r2))),
                         corr=np.corrcoef(np.log(r1), np.log(r2))[0, 1]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = build_panel()
    print(p["info"])
    print(describe(p).round(3).to_string(index=False))
    regs = sum(REGIONS.values(), [])
    print("region coverage:", len(regs), len(set(regs) & set(p["countries"])),
          set(p["countries"]) - set(regs), set(regs) - set(p["countries"]))
