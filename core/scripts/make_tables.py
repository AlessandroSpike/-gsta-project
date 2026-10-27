"""Writes every LaTeX table of the paper and supplement from results/ (no hand transcription)."""
import os, json, glob
import numpy as np
import pandas as pd
HERE = os.path.dirname(__file__); RES = os.path.join(HERE, "..", "results"); TAB = os.path.join(HERE, "..", "tables")
os.makedirs(TAB, exist_ok=True)
YEARS = list(range(2016, 2022))
def R(n, **k): return pd.read_csv(os.path.join(RES, n), **k)
def sci(x):
    if not np.isfinite(x): return "--"
    if x == 0: return "0"
    e = int(np.floor(np.log10(abs(x)))); return r"$%.1f\times10^{%d}$" % (x / 10 ** e, e)
def w(name, lines): open(os.path.join(TAB, name), "w").write("\n".join(lines) + "\n"); print("wrote", name)
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
METH = ["G-STA (directed)", "G-STA (symmetrised)", "feature/mass only (alpha=0)", "mass-weighted Frobenius",
        "stacked Frobenius", "network summaries", "geometry spectrum"]
HEAD = ["directed", "symm.", "$\\alpha{=}0$", "w.\\,Frob.", "Frob.", "summ.", "spectr."]

# ---------------- Table 1: verification summary (main text)
g = R("verif_gradients.csv"); a = R("verif_algorithms.csv"); sy = R("verif_symmetry.csv"); li = R("verif_limits.csv")
sd = R("verif_softdtw.csv"); tg = R("verif2_tensor_gradient.csv"); en = R("verif2_leadlag_enumeration.csv")
cr = R("verif2_convergence_runs.csv"); ms = R("verif2_multistart.csv")
ok = g[~g.block.str.contains("mixed")]
e = en[en.tau_enum.notna()]
small = e[(e.beta <= 0.1) & (2 * e.lag < e["T"])]
big = e[(e.beta <= 0.1) & (2 * e.lag > e["T"])]
rows = [("$\\nabla Q=T+T^\\ast$ vs.\\ explicit tensor $L_{ijkl}$ (max.\\ rel.\\ error; masks, $n\\ne m$)", sci(tg.err_factorised_vs_tensor.max())),
        ("\\quad the symmetric-case expression $2T$ vs.\\ explicit tensor (max.\\ rel.\\ error)", "%.2f" % tg.err_2T_vs_tensor.max()),
        ("Envelope gradients vs.\\ re-solved central differences (max.\\ rel.\\ error, 18 blocks)", sci(ok.rel_err.max())),
        ("\\quad weighting $C$ by $rc^\\top$ instead of $rr^\\top$ (rel.\\ error)", "%.2f" % g[g.block.str.contains("mixed")].rel_err.max())]
for m_, lab in [("GCG (this paper)", "Algorithm~1"), ("eq.(8) 2T linearisation", "symmetric-case linearization"),
                ("original update", "half-gradient fixed point")]:
    s = a[a.method == m_]
    rows.append(("Frank--Wolfe gap at termination, %s (median; max)" % lab, "%s; %s" % (sci(s.fw_gap.median()), sci(s.fw_gap.max()))))
rows += [("Audit of Algorithm~1, %d runs: monotone / sufficient decrease at every step" % len(cr),
          "%d\\%% / %d\\%%" % (100 * cr.monotone.mean(), 100 * cr.all_sufficient_decrease.mean())),
         ("\\quad steps with negative curvature $a<0$; refined step accepted (median share)",
          "%.2f; %.2f" % (cr.frac_negative_curvature.median(), cr.frac_refined_accepted.median())),
         ("\\quad residual $\\|\\nabla J_\\varepsilon\\|_\\infty$ at termination (median; max)", "%s; %s" % (sci(cr.res_inf.median()), sci(cr.res_inf.max()))),
         ("Lead--lag: enumeration of all paths vs.\\ soft-DTW gradient (max.\\ abs.\\ difference)", sci(np.abs(e.tau_enum - e.tau_softdtw).max())),
         ("\\quad idealized shift, $2\\ell<T$, $\\beta/\\kappa\\le0.1$: $|\\tau-\\ell(T-1)/(T+\\ell)|$ (max.)", sci(np.abs(small.tau_enum - small.zero_temp).max())),
         ("\\quad idealized shift, $2\\ell>T$, $\\beta/\\kappa\\le0.1$: $|\\tau|$ (max.)", sci(big.tau_enum.abs().max())),
         ("Cross-network pairs improved by 20 extra random starts (of %d); max.\\ rel.\\ improvement" % len(ms),
          "%d; %.3f" % ((ms.improvement > 1e-9).sum(), ms.rel_improvement.max())),
         ("$|V(G,H)-V(H,G)|$ (max.); G-STA between a trajectory and a copy", "%s; %s" % (sci(sy[sy.check.str.startswith('|V')].value.max()), sci(sy[sy.check.str.startswith('doubly')].value.max()))),
         ("$\\alpha=0$ vs.\\ POT \\texttt{sinkhorn\\_unbalanced}; soft-DTW vs.\\ enumeration (max.\\ errors)",
          "%s; %s" % (sci(li[li.check.str.startswith('alpha=0')].value.max()), sci(max(sd.value_err.max(), sd.grad_err.max()))))]
w("tab_verification.tex", ["\\begin{tabular}{lr}", "\\toprule", "Check & Result\\\\", "\\midrule"] + ["%s & %s\\\\" % r for r in rows] + ["\\bottomrule", "\\end{tabular}"])

# ---------------- detection tables
d = R("sim_detection.csv", keep_default_na=False)
d["delta"] = d.delta.astype(float)
b = d[d.variant == "base"]
lines = ["\\begin{tabular}{ll" + "c" * len(METH) + "}", "\\toprule", "Scenario & $\\delta$ & " + " & ".join(HEAD) + "\\\\", "\\midrule"]
names = {"none": "no shock", "reversal": "reversal", "corridor": "corridor surge", "feature": "connectivity", "mass": "mass contraction"}
for sc in ["none", "reversal", "corridor", "feature", "mass"]:
    for k, de in enumerate(sorted(b[b.scenario == sc].delta.unique())):
        cells = []
        for m_ in METH:
            x = b[(b.scenario == sc) & (b.delta == de) & (b.method == m_)].detected
            lo, hi = wilson(x.sum(), len(x)); cells.append("%.2f {\\tiny[%.2f,%.2f]}" % (x.mean(), lo, hi))
        lines.append("%s & %g & " % (names[sc] if k == 0 else "", de) + " & ".join(cells) + "\\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
w("tab_detection.tex", lines)
VN = {"base": "base ($N{=}12$, $\\sigma{=}0.25$, $\\kappa{=}0.5$)", "N24": "$N=24$", "sigma0.4": "$\\sigma=0.4$", "kappa1.0": "$\\kappa=1$",
      "masssd0.5": "less mass heterogeneity", "relabel": "non-persistent identities"}
lines = ["\\begin{tabular}{lll" + "c" * len(METH) + "}", "\\toprule", "Variant & Scenario & $\\delta$ & " + " & ".join(HEAD) + "\\\\", "\\midrule"]
for v in ["base", "N24", "sigma0.4", "kappa1.0", "masssd0.5", "relabel"]:
    sub = d[d.variant == v]
    sel = [("none", 0.0), ("reversal", 0.5), ("corridor", 0.05)] if v != "relabel" else [("none", 0.0), ("reversal", 1.0), ("corridor", 0.1)]
    for k, (sc, de) in enumerate(sel):
        cells = []
        for m_ in METH:
            x = sub[(sub.scenario == sc) & (np.isclose(sub.delta, de)) & (sub.method == m_)].detected
            cells.append("--" if len(x) == 0 else "%.2f" % x.mean())
        lines.append("%s & %s & %g & " % (VN[v] if k == 0 else "", names[sc], de) + " & ".join(cells) + "\\\\")
    if v != "relabel": lines.append("\\addlinespace")
lines += ["\\bottomrule", "\\end{tabular}"]
w("tab_detection_variants.tex", lines)

# ---------------- lead-lag (same-node study)
ll = R("sim_leadlag2.csv"); l1 = ll[np.isclose(ll.beta_factor, 0.1)]
lines = ["\\begin{tabular}{lrrrrrr}", "\\toprule", "Shock & $T$ & $\\ell$ & mean $\\hat\\tau$ & mean $\\hat\\ell$ & RMSE $\\hat\\ell$ & sign correct\\\\", "\\midrule"]
for sh in ["level", "spread"]:
    for T in [6, 10, 16]:
        for lag in range(5):
            s = l1[(l1.shock == sh) & (l1["T"] == T) & (l1.lag == lag)]
            mark = "" if 2 * lag < T else "$^\\ast$"
            sign = "--" if lag == 0 else "%.2f" % (s.tau > 0).mean()
            lines.append("%s & %s & %d%s & %.2f & %.2f & %.2f & %s\\\\" % (sh if (T == 6 and lag == 0) else "", str(T) if lag == 0 else "", lag, mark,
                         s.tau.mean(), s.corrected.mean(), np.sqrt(((s.corrected - lag) ** 2).mean()), sign))
        if not (sh == "spread" and T == 16): lines.append("\\addlinespace")
lines += ["\\bottomrule", "\\end{tabular}"]
w("tab_leadlag.tex", lines)

# ---------------- missing edges
mi = R("sim_missing.csv")
order = ["MCAR 5%", "MCAR 15%", "MCAR 30%", "MCAR 45%", "asymmetric", "cost-dependent", "node-dependent", "structured"]
lines = ["\\begin{tabular}{lrcccccc}", "\\toprule", " & & \\multicolumn{3}{c}{rel.\\ error of $\\Delta$} & \\multicolumn{3}{c}{Spearman (entries)}\\\\",
         "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}", "Missingness & missing & mask & FE imp. & row-max & mask & FE imp. & row-max\\\\", "\\midrule"]
for mech in order:
    s = mi[mi.mechanism == mech]
    f = lambda m_, c: s[s.method == m_][c]
    lines.append(("%s & %.2f & " % (mech.replace("%", "\\%"), s.missing_share.mean())) + " & ".join(
        "%.2f (%.2f)" % (f(m_, "rel_err").mean(), f(m_, "rel_err").std()) for m_ in ["mask", "two-way FE imputation", "row-max imputation"]) + " & " +
        " & ".join("%.2f" % f(m_, "spearman_offdiag").mean() for m_ in ["mask", "two-way FE imputation", "row-max imputation"]) + "\\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
w("tab_missing.tex", lines)

# ---------------- robustness (ranks)
r = R("emp_e3_robustness.csv")
lines = ["\\begin{tabular}{l" + "c" * 6 + "r}", "\\toprule", "Specification & " + " & ".join(str(y) for y in YEARS) + " & $\\rho$\\\\", "\\midrule"]
for spec in r.spec.unique():
    s = r[r.spec == spec].sort_values("year"); rho = s.spearman_vs_main.iloc[0]
    lab = (spec.replace("main (mask, sqrt mass, log LSCI)", "main specification").replace("symmetrised", "symmetrized")
           .replace("winsorisation", "winsorization").replace("original preprocessing (91 countries), corrected solver, alpha=0.7", "earlier preprocessing$^\\dagger$"))
    lines.append("%s & " % lab + " & ".join(str(int(k)) for k in s["rank"]) + " & %s\\\\" % ("--" if not np.isfinite(rho) else "%.3f" % rho))
lines += ["\\bottomrule", "\\end{tabular}"]
w("tab_robustness.tex", lines)

# ---------------- regional
e5 = R("emp_e5_leadlag.csv")
SH = {"East Asia & Pacific": "EAP", "South & South-East Asia": "SSEA", "Europe": "EUR", "Middle East & North Africa": "MENA",
      "Sub-Saharan Africa": "SSA", "Latin America & Caribbean": "LAC"}
regs = list(SH)
lines = ["\\begin{tabular}{llrrrrr}", "\\toprule", "$a$ & $b$ & $\\hat\\tau(a,b)$ & $\\hat\\tau$ symm. & $\\hat\\ell(a,b)$ & $\\widehat{\\mathrm{GSTA}}$ & min.\\ eig.\\\\", "\\midrule"]
for i, a_ in enumerate(regs):
    for j, b_ in enumerate(regs):
        if j <= i: continue
        x = e5[(e5.geometry == "directed") & (e5.region_a == a_) & (e5.region_b == b_)].iloc[0]
        y = e5[(e5.geometry == "symmetrised") & (e5.region_a == a_) & (e5.region_b == b_)].iloc[0]
        lag = ("%+.2f" % x.lag_corrected) + ("$^\\dagger$" if abs(x.lag_corrected) >= 2.5 else "")
        lines.append("%s & %s & %+.2f & %+.2f & %s & %.3f & %.3f\\\\" % (SH[a_], SH[b_], x.tau, y.tau, lag, x.gsta, x.min_eig_K_over_1pK))
lines += ["\\bottomrule", "\\end{tabular}"]
w("tab_regional.tex", lines)

# ---------------- benchmark
bb = R("bench_breakdown.csv").groupby("N").median(numeric_only=True)
lines = ["\\begin{tabular}{rrrrrrr}", "\\toprule", "$N$ & iterations & s/iteration & oracle & structural & line search & 3-start portfolio (s)\\\\", "\\midrule"]
for N, x in bb.iterrows():
    tot = x["total"]
    lines.append("%d & %d & %.4f & %.0f\\%% & %.0f\\%% & %.0f\\%% & %.2f\\\\" % (N, x.iters, x.per_iter, 100 * x["Sinkhorn oracle"] / tot,
                 100 * x["structural operator"] / tot, 100 * x["line search"] / tot, x.multistart_3))
lines += ["\\bottomrule", "\\end{tabular}"]
w("tab_bench_breakdown.tex", lines)
po = R("bench_pot.csv")
ref = po[po.method.str.startswith("POT fused")].set_index(["N", "rep"]).fgw_energy
po["rel"] = [r_.fgw_energy / ref[(r_.N, r_.rep)] - 1 for r_ in po.itertuples()]
lab = {"POT fused_gromov_wasserstein (CG, exact OT oracle)": "POT CG (exact OT oracle)",
       "POT entropic_fused_gromov_wasserstein (eps=0.01)": "POT entropic ($\\varepsilon=0.01$)",
       "Algorithm 1 (gamma=1e3, eps=0.01)": "Algorithm~1 ($\\gamma=10^3$, $\\varepsilon=0.01$)"}
lines = ["\\begin{tabular}{rlrrr}", "\\toprule", "$N$ & method & time (s) & energy excess & marginal error\\\\", "\\midrule"]
for N in sorted(po.N.unique()):
    for m_ in lab:
        s = po[(po.N == N) & (po.method == m_)]
        lines.append("%s & %s & %.3f & %s & %s\\\\" % (str(N) if m_.startswith("POT fused") else "", lab[m_], s.time.median(), sci(s.rel.median()) if s.rel.median() != 0 else "0", sci(s.marginal_error.median())))
lines += ["\\bottomrule", "\\end{tabular}"]
w("tab_bench_pot.tex", lines)

# ---------------- MDS diagnostics
lines = ["\\begin{tabular}{rlccc}", "\\toprule", "$\\alpha$ & eigenvalues of $-\\frac12J\\Delta J$ & first two / positive & negative / total & stress\\\\", "\\midrule"]
for a_ in ["0.00", "0.25", "0.50", "0.75", "1.00"]:
    j = json.load(open(os.path.join(RES, "emp_e1_mds_alpha%s.json" % a_)))
    lines.append("%s & %s & %.2f & %.2f & %.2f\\\\" % (a_, ", ".join("%.3f" % x for x in j["eigenvalues"][:5]), j["share_first_two_of_positive"],
                 j["negative_share_of_total"], j["kruskal_stress1"]))
lines += ["\\bottomrule", "\\end{tabular}"]
w("tab_mds.tex", lines)

# ---------------- gradients, corridors, descriptives (supplement)
lines = ["\\begin{tabular}{llrr}", "\\toprule", "Instance & Block & max.\\ abs.\\ error & max.\\ rel.\\ error\\\\", "\\midrule"]
for rr in g.itertuples():
    bl = rr.block.replace("C (mixed marginals r c^T)", "$C$, with $rc^\\top$ (incorrect)")
    bl = bl if bl.startswith("$") else "$%s$" % ({"mu": "\\mu", "nu": "\\nu"}.get(bl, bl))
    lines.append("%s & %s & %s & %s\\\\" % (rr.instance, bl, sci(rr.max_abs_err), sci(rr.rel_err)))
w("tab_gradients.tex", lines + ["\\bottomrule", "\\end{tabular}"])
cor = R("emp_e6_corridors.csv")
lines = ["\\begin{tabular}{llrrr}", "\\toprule", "Origin & Destination & $\\partial\\Delta/\\partial c_{ij}$ & $\\Delta c_{ij}$ & contribution\\\\", "\\midrule"]
for rr in cor[cor.transition == "2020-2021"].head(12).itertuples():
    lines.append("%s & %s & %s & %+.2f & %s\\\\" % (rr.origin.replace("&", "\\&"), rr.destination.replace("&", "\\&"), sci(rr.gradient), rr.change, sci(rr.contribution)))
w("tab_corridors.tex", lines + ["\\bottomrule", "\\end{tabular}"])
de = R("emp_e0_descriptives.csv")
lines = ["\\begin{tabular}{lrrrrrr}", "\\toprule", "Year & observed & median USD/kg$^\\ast$ & TEU (m) & mean log LSCI & mean $|\\log(r_{ij}/r_{ji})|$ & corr$(\\log r_{ij},\\log r_{ji})$\\\\", "\\midrule"]
for rr in de.itertuples():
    lines.append("%d & %.3f & %.3f & %.1f & %.3f & %.2f & %.2f\\\\" % (rr.year, rr.observed_share, rr.median_rate_persistent, rr.total_TEU_million, rr.mean_log_LSCI, rr.mean_abs_log_ratio_ij_ji, rr.corr_ij_ji))
w("tab_descriptives.tex", lines + ["\\bottomrule", "\\end{tabular}"])

# ---------------- supplement: enumeration, audit, multistart
en = R("verif2_leadlag_enumeration.csv")
lines = ["\\begin{tabular}{rr" + "r" * 5 + "r}", "\\toprule", "$T$ & $\\ell$ & $\\beta/\\kappa=0.05$ & $0.1$ & $0.2$ & $0.5$ & $1$ & $\\ell(T-1)/(T+\\ell)$\\\\", "\\midrule"]
for T in [6, 10, 40]:
    for lag in sorted(en[en["T"] == T].lag.unique()):
        row = en[(en["T"] == T) & (en.lag == lag)].set_index("beta").tau_softdtw
        cells = ["%.4f" % row[b] if b in row.index else "--" for b in [0.05, 0.1, 0.2, 0.5, 1.0]]
        lines.append("%s & %d & " % (str(T) if lag == sorted(en[en["T"] == T].lag.unique())[0] else "", lag) + " & ".join(cells) + " & %.4f\\\\" % (lag * (T - 1) / (T + lag)))
    if T != 40: lines.append("\\addlinespace")
w("tab_enum.tex", lines + ["\\bottomrule", "\\end{tabular}"])
cr = R("verif2_convergence_runs.csv")
lines = ["\\begin{tabular}{lrrrrrr}", "\\toprule", "Instance type & runs & iterations & neg.\\ curvature & refined accepted & residual & time (s)\\\\", "\\midrule"]
for k, s_ in cr.groupby("kind"):
    lines.append("%s & %d & %d [%d] & %.2f & %.2f & %s & %.3f\\\\" % (k, len(s_), s_.iters.median(), s_.iters.max(), s_.frac_negative_curvature.median(),
                 s_.frac_refined_accepted.median(), sci(s_.res_inf.median()), s_.time.median()))
w("tab_audit.tex", lines + ["\\bottomrule", "\\end{tabular}"])
ms = R("verif2_multistart.csv")
lines = ["\\begin{tabular}{rlrlrrr}", "\\toprule", "pair & winner (portfolio) & value & winner (+20 random) & rel.\\ improvement & distinct values & switches\\\\", "\\midrule"]
for r_ in ms.itertuples():
    lines.append("%d & %s & %.4f & %s & %.4f & %d & %d\\\\" % (r_.pair, r_.baseline_winner, r_.baseline_best, r_.extended_winner, r_.rel_improvement,
                 r_.n_distinct_values, r_.winner_switches_under_perturbation))
w("tab_multistart.tex", lines + ["\\bottomrule", "\\end{tabular}"])
