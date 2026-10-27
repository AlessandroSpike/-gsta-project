"""Write every new table (tables/*.tex) and figure (figures/*.pdf) from results/."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, FIG, TAB = "results", "figures", "tables"
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "serif"})
COL = {"gradient": "#1b4f9c", "cost-change": "#d1495b", "mass-weighted": "#2e8b57",
       "uniform": "#888888", "random": "#e8a33d"}
SHORT = {"China, Hong Kong SAR": "Hong Kong SAR", "Republic of Korea": "Korea, Rep.",
         "United Arab Emirates": "UAE", "Russian Federation": "Russia",
         "Netherlands (Kingdom of the)": "Netherlands", "Venezuela (Bolivarian Rep. of)": "Venezuela",
         "Saint Vincent and the Grenadines": "St Vincent \\& Gren.", "Dominican Republic": "Dominican Rep.",
         "Trinidad and Tobago": "Trinidad \\& Tobago", "United Republic of Tanzania": "Tanzania",
         "Cote d'Ivoire": "C\\^ote d'Ivoire", "Turkiye": "T\\\"urkiye"}


def short(c):
    return SHORT.get(c, c)


def sci(x, d=1):
    if x == 0 or not np.isfinite(x):
        return "0" if x == 0 else "--"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    return f"${m:.{d}f}\\times10^{{{e}}}$"


def write(name, s):
    with open(f"{TAB}/{name}.tex", "w") as f:
        f.write(s)


def base_strat(s):
    return "random" if s.startswith("random") else s


# =====================================================================================
# Table 6: anomaly-score sensitivity by year
# =====================================================================================
m = pd.read_csv(f"{R}/maps_summary.csv")
nodes = pd.read_csv(f"{R}/maps_nodes.csv")
lines = [r"\begin{tabular}{lrrrrrrrrrl}", r"\toprule",
         r" & & & \multicolumn{4}{c}{leverage of a 1\% change ($\times10^{-4}$)} & \multicolumn{2}{c}{share for half} & & \\",
         r"\cmidrule(lr){4-7}\cmidrule(lr){8-9}",
         r"Year & $a(t)$ & rank & costs & LSCI & TEU & other years & corridors & economies & antisym. & leading economy ($B_i$)\\",
         r"\midrule"]
for _, r in m.iterrows():
    lines.append(f"{int(r.year)} & {r.anomaly:.4f} & {int(r['rank'])} & {1e4*r.lev_cost:.1f} & {1e4*r.lev_lsci:.1f} & "
                 f"{1e4*r.lev_teu:.1f} & {1e4*r.lev_cost_other_years:.1f} & {r.conc50_corr:.3f} & {r.conc50_nodes:.2f} & "
                 f"{r.anti_share:.2f} & {short(r.top_country)} (${r.top_country_B:+.2f}$)\\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("tab6_maps", "\n".join(lines))

# supplement: top economies by year, top corridors
top = []
for y, d in nodes.groupby("year"):
    d = d.sort_values("S", ascending=False).head(8)
    top.append(d)
top = pd.concat(top)
lines = [r"\begin{tabular}{rlllrrrrr}", r"\toprule",
         r"Year & rank & economy & region & $S_i\times10^{4}$ & $B_i$ & $\partial a/\partial\log\mathrm{LSCI}_i$ & $\partial a/\partial\log\mathrm{TEU}_i$ & $\mu_i$\\", r"\midrule"]
for y, d in top.groupby("year"):
    for k, (_, r) in enumerate(d.iterrows()):
        lines.append(f"{y if k == 0 else ''} & {k+1} & {short(r.country)} & {r.region} & {1e4*r.S:.2f} & ${r.B:+.2f}$ & "
                     f"{sci(r.e_lsci)} & {sci(r.e_teu)} & {r.mass:.3f}\\\\")
    lines.append(r"\addlinespace")
lines[-1] = r"\bottomrule"
lines.append(r"\end{tabular}")
write("tabS_nodes", "\n".join(lines))

cor = pd.read_csv(f"{R}/maps_corridors.csv")
lines = [r"\begin{tabular}{rlllrrr}", r"\toprule",
         r"Year & origin & destination & cost (USD/kg) & $\partial a/\partial\log\mathrm{cost}_{ik}$ & reverse direction & $\partial a/\partial c_{ik}$\\", r"\midrule"]
for y in (2016, 2020, 2021):
    d = cor[cor.year == y].head(8)
    for k, (_, r) in enumerate(d.iterrows()):
        lines.append(f"{y if k == 0 else ''} & {short(r.origin)} & {short(r.destination)} & {r.cost:.3f} & {sci(r.elasticity, 2)} & "
                     f"{sci(r.rev_elasticity, 1)} & {sci(r.grad_c, 2)}\\\\")
    lines.append(r"\addlinespace")
lines[-1] = r"\bottomrule"
lines.append(r"\end{tabular}")
write("tabS_corridors", "\n".join(lines))

stab = pd.read_csv(f"{R}/maps_stability_spearman.csv", index_col=0)
ov = pd.read_csv(f"{R}/maps_stability_top100.csv", index_col=0)
lines = [r"\begin{tabular}{l" + "r" * 6 + "}", r"\toprule", " & " + " & ".join(str(c) for c in stab.columns) + r"\\", r"\midrule"]
for i, y in enumerate(stab.index):
    cells = []
    for j in range(6):
        cells.append("--" if i == j else (f"{stab.iloc[i, j]:.2f}" if j > i else f"{ov.iloc[i, j]:.2f}"))
    lines.append(f"{y} & " + " & ".join(cells) + r"\\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("tabS_stability", "\n".join(lines))

# =====================================================================================
# Table 7: regional trajectory-level and timing sensitivity
# =====================================================================================
g = pd.read_csv(f"{R}/regional_summary.csv")
lines = [r"\begin{tabular}{llrrrrrrrrl}", r"\toprule",
         r" & & & & \multicolumn{2}{c}{leverage ($\times10^{-2}$)} & \multicolumn{2}{c}{antisym. share} & \multicolumn{2}{c}{rank corr.} & \\",
         r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
         r"$a$ & $b$ & $\hat\tau$ & $\widehat{\mathrm{GSTA}}$ & $\hat\tau$ & GSTA & $\hat\tau$ & GSTA & $\hat\tau$/GSTA & GSTA/annual & leading economy for $\hat\tau$\\",
         r"\midrule"]
for _, r in g.iterrows():
    levt = 100 * (r.a_lev_tau_cost + r.b_lev_tau_cost + r.a_lev_tau_lsci + r.b_lev_tau_lsci + r.a_lev_tau_teu + r.b_lev_tau_teu)
    levg = 100 * (r.a_lev_gsta_cost + r.b_lev_gsta_cost + r.a_lev_gsta_lsci + r.b_lev_gsta_lsci + r.a_lev_gsta_teu + r.b_lev_gsta_teu)
    at = 0.5 * (r.a_anti_tau + r.b_anti_tau)
    ag = 0.5 * (r.a_anti_gsta + r.b_anti_gsta)
    rt = 0.5 * (r.a_rho_gsta_tau + r.b_rho_gsta_tau)
    rd = 0.5 * (r.a_rho_gsta_diag + r.b_rho_gsta_diag)
    lt = r.a_lev_tau_cost + r.a_lev_tau_lsci + r.a_lev_tau_teu
    lb = r.b_lev_tau_cost + r.b_lev_tau_lsci + r.b_lev_tau_teu
    lead = r.a_top_tau if lt >= lb else r.b_top_tau
    lines.append(f"{r.a} & {r.b} & ${r.tau:+.2f}$ & {r.gsta:.3f} & {levt:.2f} & {levg:.2f} & {at:.2f} & {ag:.2f} & {rt:.2f} & {rd:.2f} & {short(lead)}\\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("tab7_regional", "\n".join(lines))

# supplement: channel split, year profile and beta sensitivity
lines = [r"\begin{tabular}{llrrrrrrrrrr}", r"\toprule",
         r" & & \multicolumn{3}{c}{channel share of $\hat\tau$ leverage} & \multicolumn{6}{c}{share of $\hat\tau$ leverage by year} & $\rho$ ($\beta$ halved/doubled)\\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-11}",
         r"$a$ & $b$ & costs & LSCI & TEU & 2016 & 2017 & 2018 & 2019 & 2020 & 2021 & \\", r"\midrule"]
for _, r in g.iterrows():
    c = r.a_lev_tau_cost + r.b_lev_tau_cost
    f_ = r.a_lev_tau_lsci + r.b_lev_tau_lsci
    t_ = r.a_lev_tau_teu + r.b_lev_tau_teu
    tot = c + f_ + t_
    pa = np.array([float(x) for x in r.a_tau_year_profile.split(";")])
    pb = np.array([float(x) for x in r.b_tau_year_profile.split(";")])
    la = r.a_lev_tau_cost + r.a_lev_tau_lsci + r.a_lev_tau_teu
    lb = r.b_lev_tau_cost + r.b_lev_tau_lsci + r.b_lev_tau_teu
    prof = (la * pa + lb * pb) / (la + lb)
    lines.append(f"{r.a} & {r.b} & {c/tot:.2f} & {f_/tot:.2f} & {t_/tot:.2f} & " + " & ".join(f"{x:.2f}" for x in prof)
                 + f" & {r['rho_Gtau_beta0.05']:.2f}/{r['rho_Gtau_beta0.2']:.2f}\\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("tabS_regional", "\n".join(lines))

# =====================================================================================
# counterfactuals
# =====================================================================================
mit = pd.read_csv(f"{R}/cf_mitigation.csv")
mit["base"] = mit.strategy.map(base_strat)
st = pd.read_csv(f"{R}/cf_stress.csv")
st["base"] = st.strategy.map(base_strat)
tm = pd.read_csv(f"{R}/cf_timing.csv")
tm["base"] = tm.strategy.map(base_strat)
D0 = None

ORDER = ["gradient", "cost-change", "mass-weighted", "random", "uniform"]
mb = [8, 32, 128]
sb = sorted(st[~st.strategy.str.startswith("across")].budget.unique())
tb = sorted(tm.budget.unique())
lines = [r"\begin{tabular}{l" + "r" * (len(mb) + 2 + 2) + "}", r"\toprule",
         r" & \multicolumn{" + str(len(mb)) + r"}{c}{(i) change of $\Delta(G_{2020},G_{2021})$, \%} & \multicolumn{2}{c}{(ii) 2020 margin after} & \multicolumn{2}{c}{(iii) $\hat\tau$(EAP,SSEA) after}\\",
         r"\cmidrule(lr){2-" + str(1 + len(mb)) + r"}\cmidrule(lr){" + str(2 + len(mb)) + "-" + str(3 + len(mb)) + r"}\cmidrule(lr){" + str(4 + len(mb)) + "-" + str(5 + len(mb)) + "}",
         "Strategy & " + " & ".join(f"$B={b}$" for b in mb) + f" & $B={int(sb[-2])}$ & $B={int(sb[-1])}$ & $B={tb[-2]:g}$ & $B={tb[-1]:g}$" + r"\\", r"\midrule"]
for s_ in ORDER:
    row = [s_]
    for b in mb:
        d = mit[(mit.base == s_) & (mit.budget == b)]
        row.append(f"${100*d.rel_change.mean():+.1f}$" if len(d) else "--")
    for b in sb[-2:]:
        d = st[(st.base == s_) & (st.budget == b)]
        row.append(f"${d.margin_after.mean():+.4f}$" if len(d) else "--")
    for b in tb[-2:]:
        d = tm[(tm.base == s_) & (tm.budget == b)]
        row.append(f"${d.tau_after.mean():+.2f}$" if len(d) else "--")
    lines.append(" & ".join(row) + r"\\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("tab8_counterfactual", "\n".join(lines))

# supplement: first-order accuracy of the gradient strategy
lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r"Experiment & budget $B$ & corridors & predicted & realized & branch-following & realized/predicted & switches\\", r"\midrule"]
for _, r in mit[mit.strategy == "gradient"].iterrows():
    lines.append(f"(i) $\\Delta(G_{{2020}},G_{{2021}})$ & {r.budget:g} & {r.n_corr} & {sci(r.pred,2)} & {sci(r.realized,2)} & {sci(r.realized_bf,2)} & {r.realized/r.pred:.3f} & {int(r.switch_cross)+int(r.switch_self)}\\\\")
lines.append(r"\addlinespace")
for _, r in st[st.strategy == "gradient"].iterrows():
    lines.append(f"(ii) $a(2020)$ & {r.budget:g} & {r.n_corr} & {sci(r.pred_a2020,2)} & {sci(r.real_a2020,2)} & {sci(r.real_a2020_bf,2)} & {r.real_a2020/r.pred_a2020:.3f} & {int(r.switches)}\\\\")
lines.append(r"\addlinespace")
for _, r in tm[tm.strategy == "gradient"].iterrows():
    lines.append(f"(iii) $\\hat\\tau$(EAP,SSEA) & {r.budget:g} & {r.n_corr} & {r.pred:+.3f} & {r.realized:+.3f} & {r.realized_bf:+.3f} & {r.realized/r.pred:.3f} & {int(r.switches)}\\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("tabS_accuracy", "\n".join(lines))

# frozen-coupling bound check (Proposition 14a)
viol = (mit.cross_change_bf - (mit.cross_pred + mit.frozen_R)).max()
viol_p = (mit.cross_change - (mit.cross_pred + mit.frozen_R)).max()
with open(f"{TAB}/numbers.txt", "w") as f:
    f.write(f"max violation of frozen bound (branch-following) {viol:.3e}\n")
    f.write(f"max violation of frozen bound (portfolio) {viol_p:.3e}\n")
    f.write(f"mitigation switches {int(mit.switch_cross.sum() + mit.switch_self.sum())} of {2*len(mit)}\n")
    f.write(f"stress switches {int(st.switches.sum())} ; timing switches {int(tm.switches.sum())}\n")
    ab = st[st.strategy.str.startswith("across")]
    f.write(ab[["strategy", "margin_after", "rank2020", "real_a2020"]].to_string() + "\n")
    for s_ in ORDER:
        d = st[(st.base == s_)].groupby("budget").margin_after.mean()
        hit = d[d > 0]
        f.write(f"stress {s_}: first budget with 2020 rank 1: {hit.index.min() if len(hit) else 'none'}\n")
    g_ = st[st.strategy == "gradient"]
    f.write(g_[["budget", "n_corr", "pred_margin", "real_margin", "margin_after", "rank2020"]].to_string() + "\n")

# =====================================================================================
# Figure 6: influence maps
# =====================================================================================
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
d = nodes[nodes.year == 2020]
a_ = ax[0]
a_.scatter(1e4 * d.S_out, 1e4 * d.S_in, s=600 * d.mass, alpha=0.55, color="#1b4f9c", edgecolor="k", lw=0.3)
lim = 1e4 * max(d.S_out.max(), d.S_in.max()) * 1.08
a_.plot([0, lim], [0, lim], "k:", lw=0.7)
for _, r in d.sort_values("S", ascending=False).head(6).iterrows():
    a_.annotate(short(r.country).replace("\\", ""), (1e4 * r.S_out, 1e4 * r.S_in), fontsize=6, xytext=(3, 2), textcoords="offset points")
a_.set_xlabel(r"outgoing influence $S_i^{\rm out}$ ($\times10^{-4}$)")
a_.set_ylabel(r"incoming influence $S_i^{\rm in}$ ($\times10^{-4}$)")
a_.set_title("(a) economies, $a(2020)$", fontsize=8)
a_ = ax[1]
x = np.arange(len(m))
w = 0.2
for k, (c, lab, col) in enumerate([("lev_cost", "own costs", "#1b4f9c"), ("lev_cost_other_years", "other years' costs", "#9db9e3"),
                                    ("lev_lsci", "LSCI", "#d1495b"), ("lev_teu", "TEU", "#2e8b57")]):
    a_.bar(x + (k - 1.5) * w, 1e4 * m[c], w, label=lab, color=col)
a_.set_xticks(x); a_.set_xticklabels([f"{y}" for y in m.year], rotation=45)
a_.set_ylabel(r"change of $a(t)$ per 1\% ($\times10^{-4}$)".replace("\\%", "%"))
a_.legend(fontsize=6, frameon=False)
a_.set_title("(b) leverage by channel", fontsize=8)
a_ = ax[2]
blk = pd.read_csv(f"{R}/maps_region_blocks.csv")
regs = ["EAP", "SSEA", "EUR", "MENA", "SSA", "LAC"]
M = np.array([[blk[(blk.origin == o) & (blk.destination == de)].share.mean() for de in regs] for o in regs])
im = a_.imshow(M, cmap="Blues")
a_.set_xticks(range(6)); a_.set_xticklabels(regs, rotation=45, fontsize=6)
a_.set_yticks(range(6)); a_.set_yticklabels(regs, fontsize=6)
for i in range(6):
    for j in range(6):
        a_.text(j, i, f"{100*M[i, j]:.0f}", ha="center", va="center", fontsize=5.5, color="w" if M[i, j] > M.max() / 2 else "k")
a_.set_xlabel("destination"); a_.set_ylabel("origin")
a_.set_title("(c) corridor sensitivity (%)", fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIG}/fig6_sensitivity.pdf")
plt.close(fig)

# =====================================================================================
# Figure 7: counterfactual curves
# =====================================================================================
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6))
for a_, df, ycol, pcol, title, ylab in (
        (ax[0], mit, "rel_change", None, "(i) reduce $\\Delta(G_{2020},G_{2021})$", "relative change"),
        (ax[1], st[~st.strategy.str.startswith("across")], "margin_after", None, "(ii) make 2020 most distinctive",
         "margin of 2020 over leader"),
        (ax[2], tm, "tau_after", None, r"(iii) remove EAP lead over SSEA", r"$\hat\tau$(EAP,SSEA)")):
    for s_ in ORDER:
        dd = df[df.base == s_].groupby("budget")[ycol].agg(["mean", "min", "max"])
        if len(dd) == 0:
            continue
        a_.plot(dd.index, dd["mean"], marker="o", ms=2.5, lw=1, color=COL[s_], label=s_)
        if s_ == "random":
            a_.fill_between(dd.index, dd["min"], dd["max"], color=COL[s_], alpha=0.2, lw=0)
    a_.set_xscale("log"); a_.set_xlabel(r"budget $B=\sum|u_{ik}|$")
    a_.set_title(title, fontsize=8); a_.set_ylabel(ylab)
# first-order predictions of the gradient strategy
gm = mit[mit.strategy == "gradient"]
D0m = (gm.realized / gm.rel_change).iloc[0]
ax[0].plot(gm.budget, gm.pred / D0m, "--", color=COL["gradient"], lw=0.8, label="gradient, predicted")
gs = st[st.strategy == "gradient"]
m0 = gs.margin_after.iloc[0] - gs.real_margin.iloc[0]
ax[1].plot(gs.budget, m0 + gs.pred_margin, "--", color=COL["gradient"], lw=0.8)
ax[1].axhline(0, color="k", lw=0.6, ls=":")
gt = tm[tm.strategy == "gradient"]
t0 = gt.tau_after.iloc[0] - gt.realized.iloc[0]
ax[2].plot(gt.budget, t0 + gt.pred, "--", color=COL["gradient"], lw=0.8)
ax[0].legend(fontsize=5.5, frameon=False)
fig.tight_layout()
fig.savefig(f"{FIG}/fig7_counterfactual.pdf")
plt.close(fig)
print("main outputs written")

# =====================================================================================
# Supplement figures: branch tie (S6) and uncertainty (S7)
# =====================================================================================
b = pd.read_csv(f"{R}/verif_branch.csv")
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.4))
ax[0].plot(b.lam, b["identity branch"], "--", color=COL["mass-weighted"], lw=1, label="identity-type branch")
ax[0].plot(b.lam, b["swap branch"], "--", color=COL["cost-change"], lw=1, label="swap-type branch")
ax[0].plot(b.lam, b.portfolio, "o", ms=2.5, color="k", label="portfolio estimate")
ax[0].set_xlabel(r"path parameter $\lambda$"); ax[0].set_ylabel(r"$\widehat\Delta_\varepsilon(G,H_\lambda)$")
ax[0].legend(fontsize=6, frameon=False); ax[0].set_title("(a) values", fontsize=8)
sel = np.where(b["identity branch"] <= b["swap branch"], b["identity branch slope"], b["swap branch slope"])
ax[1].plot(b.lam, b["identity branch slope"], "--", color=COL["mass-weighted"], lw=1)
ax[1].plot(b.lam, b["swap branch slope"], "--", color=COL["cost-change"], lw=1)
ax[1].plot(b.lam, sel, "o", ms=2.5, color="k", label="slope of selected branch")
fd = np.gradient(b.portfolio.values, b.lam.values)
ax[1].plot(b.lam, fd, "x", ms=3, color=COL["gradient"], label="central difference of estimate")
ax[1].axhline(0, color="k", lw=0.5, ls=":")
ax[1].set_xlabel(r"path parameter $\lambda$"); ax[1].set_ylabel("directional derivative")
ax[1].legend(fontsize=6, frameon=False); ax[1].set_title("(b) Proposition 11 slopes", fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/figS6_branch.pdf"); plt.close(fig)

if os.path.exists(f"{R}/unc_delta_vs_mc.csv"):
    u = pd.read_csv(f"{R}/unc_delta_vs_mc.csv")
    yrs = u[~u.year.astype(str).str.contains("-")].copy(); yrs["year"] = yrs.year.astype(int)
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.4))
    for sig, mk in zip(sorted(yrs.sigma.unique()), ("o", "s")):
        d = yrs[yrs.sigma == sig]
        ax[0].plot(d.sd_delta, d.sd_mc, mk, ms=3.5, label=fr"$\sigma_m={sig}$")
        ax[1].plot(d.year, d.mean_shift_mc / d.sd_mc, mk + "-", ms=3, lw=0.8, label=fr"$\sigma_m={sig}$")
    lim = max(yrs.sd_delta.max(), yrs.sd_mc.max()) * 1.1
    ax[0].plot([0, lim], [0, lim], "k:", lw=0.7)
    ax[0].set_xlabel("delta-method s.d. of $a(t)$"); ax[0].set_ylabel("Monte Carlo s.d.")
    ax[0].legend(fontsize=6, frameon=False); ax[0].set_title("(a) spread", fontsize=8)
    ax[1].set_xlabel("year"); ax[1].set_ylabel("Monte Carlo mean shift / s.d.")
    ax[1].legend(fontsize=6, frameon=False); ax[1].set_title("(b) bias", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/figS7_uncertainty.pdf"); plt.close(fig)
print("supplement figures written")
