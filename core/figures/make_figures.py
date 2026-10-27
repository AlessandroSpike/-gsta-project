"""Figures for the paper and the supplement.

This script only READS the files written by scripts/run_*.py (results/*.csv,
*.npy, *.json) and writes figures/*.pdf and figures/*.png.  Each figure is
produced by one self-contained function (fig1_..., figS1_...), so that it can
be ported to MATLAB one function at a time (readtable/readmatrix + plotting).

Design: every figure is readable in grayscale (distinct markers and line
styles in addition to colour).
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = HERE
YEARS = list(range(2016, 2022))

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "font.family": "DejaVu Sans",
})
# colour-blind safe palette (Okabe-Ito) + markers / dashes for grayscale printing
PAL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
MK = ["o", "s", "^", "D", "v", "P", "X"]
LS = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), "-"]


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300)
    plt.close(fig)
    print("wrote", name)


def read(name):
    return pd.read_csv(os.path.join(RES, name))


# ----------------------------------------------------------------------------
# Figure 1: computational verification
# ----------------------------------------------------------------------------
def fig1_verification():
    alg = read("verif_algorithms.csv")
    sc = read("verif_scaling.csv")
    conv = read("verif_convergence.csv")
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3))
    # (a) convergence trace of GCG
    g = conv[conv.method == "GCG"]
    ax[0].semilogy(g.iter, np.maximum(g.gap, 1e-16), color=PAL[0], marker="o", ms=3, label="Frank-Wolfe gap")
    exc = g.objective - g.objective.min()
    ax[0].semilogy(g.iter, np.maximum(exc, 1e-16), color=PAL[1], ls="--", marker="s", ms=3,
                   label="objective excess")
    ax[0].set_xlabel("GCG iteration"); ax[0].set_ylabel("value (log scale)")
    ax[0].set_title("(a) GCG convergence"); ax[0].legend(frameon=False)
    # (b) stationarity at termination for three update rules
    methods = ["GCG (this paper)", "eq.(8) 2T linearisation", "original update"]
    labels = ["GCG", "symm.\nlinear.", "half-\ngradient"]
    data = [np.maximum(alg[alg.method == m].fw_gap.values, 1e-16) for m in methods]
    bp = ax[1].boxplot(data, widths=0.5, patch_artist=True, medianprops=dict(color="k"))
    for p, c in zip(bp["boxes"], [PAL[0], PAL[3], PAL[1]]):
        p.set_facecolor(c); p.set_alpha(0.45)
    ax[1].set_yscale("log"); ax[1].set_xticks([1, 2, 3]); ax[1].set_xticklabels(labels, fontsize=6.5)
    ax[1].set_ylabel("Frank-Wolfe gap at termination"); ax[1].set_title("(b) stationarity (20 instances)")
    # (c) timing
    c = sc[sc.kind == "contraction"]
    d = c[c.t_direct.notna()]
    ax[2].loglog(d.N, d.t_direct, color=PAL[1], marker="s", ms=3, ls="--", label=r"direct $O(N^4)$ tensor")
    ax[2].loglog(c.N, c.t_factorised, color=PAL[0], marker="o", ms=3, label="factorized (Lemma 1)")
    s = sc[sc.kind == "full solve (masked)"]
    ax[2].loglog(s.N, s.t_factorised, color=PAL[2], marker="^", ms=3, ls="-.", label="full masked solve")
    ax[2].set_xlabel("number of nodes $N$"); ax[2].set_ylabel("seconds"); ax[2].set_title("(c) run time")
    ax[2].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig1_verification")


# ----------------------------------------------------------------------------
# Figure 2: detection power (S2), base design with Wilson 95% intervals
# ----------------------------------------------------------------------------
METHODS_DET = ["G-STA (directed)", "G-STA (symmetrised)", "feature/mass only (alpha=0)",
               "mass-weighted Frobenius", "stacked Frobenius", "network summaries", "geometry spectrum"]
LAB_DET = ["G-STA (directed)", "G-STA (symmetrized)", r"attribute/mass only ($\alpha=0$)",
           "mass-weighted Frobenius", "stacked Frobenius", "network summaries", "geometry spectrum"]


def _wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fig2_detection():
    df = pd.read_csv(os.path.join(RES, "sim_detection.csv"), keep_default_na=False)
    df["delta"] = df.delta.astype(float)
    df = df[df.variant == "base"]
    titles = {"reversal": "(a) direction reversal", "corridor": "(b) corridor surge",
              "feature": "(c) connectivity shock", "mass": "(d) mass contraction"}
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.45), sharey=True)
    for k, sc in enumerate(titles):
        for m, (meth, lab) in enumerate(zip(METHODS_DET, LAB_DET)):
            xs, ys, lo, hi = [], [], [], []
            for sce, de in [("none", 0.0)] + [(sc, x) for x in sorted(df[df.scenario == sc].delta.unique())]:
                x = df[(df.scenario == sce) & np.isclose(df.delta, de) & (df.method == meth)].detected
                l_, h_ = _wilson(x.sum(), len(x))
                xs.append(de); ys.append(x.mean()); lo.append(max(0.0, x.mean() - l_)); hi.append(max(0.0, h_ - x.mean()))
            xs = np.array(xs) + (m - 3) * 0.004 * max(xs)
            ax[k].errorbar(xs, ys, yerr=[lo, hi], color=PAL[m], marker=MK[m], ls=LS[m], ms=3, lw=1.0,
                           elinewidth=0.5, capsize=1, label=lab)
        ax[k].axhline(1 / 8, color="gray", lw=0.6, ls=":")
        ax[k].set_title(titles[sc], fontsize=8); ax[k].set_xlabel(r"effect size $\delta$")
        ax[k].set_ylim(-0.03, 1.03)
    ax[0].set_ylabel("detection rate")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "fig2_detection")


# ----------------------------------------------------------------------------
# Figure 3: lead-lag estimation (same-node study S3)
# ----------------------------------------------------------------------------
def fig3_leadlag():
    df = pd.read_csv(os.path.join(RES, "sim_leadlag2.csv"))
    df = df[np.isclose(df.beta_factor, 0.1)]
    fig, ax = plt.subplots(2, 3, figsize=(7.0, 4.0), sharex=True)
    for r_, sh in enumerate(["level", "spread"]):
        for c_, T in enumerate([6, 10, 16]):
            a = ax[r_, c_]
            d = df[(df.shock == sh) & (df["T"] == T)]
            lags = np.arange(5)
            for m, (col, lab) in enumerate([("tau", r"$\hat\tau$"), ("corrected", r"$\hat\ell$ (boundary-corrected)")]):
                g = d.groupby("lag")[col]
                a.errorbar(lags + (m - 0.5) * 0.08, g.mean().values, yerr=1.96 * (g.std() / np.sqrt(g.count())).values,
                           color=PAL[m], marker=MK[m], ms=3.5, lw=1.0, ls=LS[m], capsize=1.5, label=lab)
            a.plot(lags, lags, color="gray", lw=0.7, ls=":", label="truth")
            zt = np.where(2 * lags < T, lags * (T - 1) / (T + lags), 0.0)
            a.plot(lags, zt, color="k", lw=0.7, ls=(0, (1, 2)), label="zero-temperature value (Prop. 9d)")
            start = np.ceil(T / 2) - 0.5          # shade leads with 2l >= T
            if start < 4.4:
                a.axvspan(start, 4.4, color="0.9", zorder=0)
            a.set_xlim(-0.3, 4.4)
            a.set_title("%s shock, $T=%d$" % (sh, T), fontsize=8)
            if r_ == 1: a.set_xlabel(r"true lead $\ell$")
            a.set_xticks(lags)
    ax[0, 0].set_ylabel("estimate"); ax[1, 0].set_ylabel("estimate")
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig3_leadlag")


# ----------------------------------------------------------------------------
# Figure 4: empirical structural trajectory
# ----------------------------------------------------------------------------
def fig4_trajectory():
    D = np.load(os.path.join(RES, "emp_e1_Delta_alpha0.50.npy"))
    summ = read("emp_e1_summary.csv")
    summ = summ[summ.year > 0]
    sub = read("emp_e4_subsampling.csv") if os.path.exists(os.path.join(RES, "emp_e4_subsampling.csv")) else None
    fig = plt.figure(figsize=(7.2, 2.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.05])
    ax0 = fig.add_subplot(gs[0])
    im = ax0.imshow(D, cmap="Greys", vmin=0)
    for t in range(6):
        for s in range(6):
            ax0.text(s, t, "%.3f" % D[t, s], ha="center", va="center", fontsize=5.5,
                     color="w" if D[t, s] > 0.6 * D.max() else "k")
    ax0.set_xticks(range(6)); ax0.set_yticks(range(6))
    ax0.set_xticklabels(YEARS, rotation=90); ax0.set_yticklabels(YEARS)
    ax0.set_title(r"(a) $\Delta_{ts}$, $\alpha=0.5$")
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.03)
    ax1 = fig.add_subplot(gs[1])
    for m, a in enumerate([0.0, 0.5, 1.0]):
        s = summ[np.isclose(summ.alpha, a)].sort_values("year")
        X = s[["mds1", "mds2"]].values
        # orient for comparability: 2016 on the left, 2017 below the axis
        if X[0, 0] > X[-1, 0]:
            X[:, 0] *= -1
        if X[1, 1] > 0:
            X[:, 1] *= -1
        X = X / np.abs(X).max()
        ax1.plot(X[:, 0], X[:, 1], color=PAL[m], marker=MK[m], ms=3.5, ls=LS[m], lw=1.0,
                 label=r"$\alpha=%.1f$" % a)
        for t in [0, 4, 5]:
            ax1.annotate(str(YEARS[t])[2:], X[t], fontsize=6, color=PAL[m], xytext=(2, 2),
                         textcoords="offset points")
    ax1.set_title("(b) MDS trajectory maps (rescaled)"); ax1.legend(frameon=False, fontsize=6.5)
    ax1.set_xlabel("coordinate 1"); ax1.set_ylabel("coordinate 2")
    ax2 = fig.add_subplot(gs[2])
    s = summ[np.isclose(summ.alpha, 0.5)].sort_values("year")
    x = np.arange(6)
    rel_a = s.anomaly.values / s.anomaly.mean()
    rel_t = s.trend_adjusted.values / s.anomaly.mean()
    ax2.bar(x - 0.18, rel_a, width=0.36, color=PAL[0], alpha=0.7, label=r"$a(t)/\bar a$")
    ax2.bar(x + 0.18, rel_t, width=0.36, color=PAL[1], alpha=0.7, hatch="///", label=r"$\tilde a(t)/\bar a$")
    if sub is not None and len(sub):
        sub = sub.copy()
        mean_b = sub.groupby("b").anomaly.transform("mean")
        sub["ra"] = sub.anomaly / mean_b
        sub["rt"] = sub.trend_adjusted / mean_b
        for col, off in [("ra", -0.18), ("rt", 0.18)]:
            q = sub.groupby("year")[col].quantile([0.05, 0.95]).unstack()
            ax2.vlines(x + off, q[0.05].values, q[0.95].values, color="k", lw=1.0)
            ax2.hlines(q[0.05].values, x + off - 0.06, x + off + 0.06, color="k", lw=0.8)
            ax2.hlines(q[0.95].values, x + off - 0.06, x + off + 0.06, color="k", lw=0.8)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels(YEARS, rotation=90)
    ax2.set_title("(c) relative scores, 90% subsampling ranges"); ax2.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig4_trajectory")


# ----------------------------------------------------------------------------
# Figure 5: attribution (Shapley, components, directed energy)
# ----------------------------------------------------------------------------
def fig5_attribution():
    sh = read("emp_e2_shapley.csv")
    comp = read("emp_e1_components.csv")
    dr = read("emp_e1_directed.csv")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    lab = [f"{t % 100:02d}-{s % 100:02d}" for t, s in zip(sh.t, sh.s)]
    x = np.arange(len(sh))
    ax[0].bar(x, sh.shapley_geometry, color=PAL[0], alpha=0.75, label="geometry")
    ax[0].bar(x, sh.shapley_node, bottom=sh.shapley_geometry, color=PAL[1], alpha=0.75, hatch="///",
              label="node attributes")
    ax[0].plot(x, sh.Delta, "k_", ms=7, label=r"$\Delta_{ts}$")
    ax[0].set_xticks(x); ax[0].set_xticklabels(lab, rotation=90, fontsize=6)
    ax[0].set_title("(a) Shapley attribution of $\\Delta_{ts}$"); ax[0].legend(frameon=False, fontsize=6.5)
    c = comp[np.isclose(comp.alpha, 0.5)]
    piv = c.pivot_table(index=["t", "s"], columns="component", values="value").reset_index()
    bottom_pos = np.zeros(len(piv)); bottom_neg = np.zeros(len(piv))
    for m, k in enumerate(["structure", "feature", "kl", "entropy"]):
        v = piv[k].values
        b = np.where(v >= 0, bottom_pos, bottom_neg)
        ax[1].bar(np.arange(len(piv)), v, bottom=b, color=PAL[m], alpha=0.75, hatch=["", "///", "...", "xx"][m],
                  label={"structure": "structure", "feature": "feature", "kl": "mass (KL)", "entropy": "entropy"}[k])
        bottom_pos += np.where(v >= 0, v, 0); bottom_neg += np.where(v < 0, v, 0)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xticks(np.arange(len(piv)))
    ax[1].set_xticklabels([f"{t % 100:02d}-{s % 100:02d}" for t, s in zip(piv.t, piv.s)], rotation=90, fontsize=6)
    ax[1].set_title(r"(b) exact component split, $\alpha=0.5$"); ax[1].legend(frameon=False, fontsize=6)
    d = dr[np.isclose(dr.alpha, 0.5)].pivot_table(index=["t", "s"], columns="part", values="value").reset_index()
    share = d["antisym"] / (d["sym"] + d["antisym"])
    ax[2].bar(np.arange(len(d)), share, color=PAL[2], alpha=0.8)
    ax[2].set_ylim(0, 1)
    ax[2].set_xticks(np.arange(len(d)))
    ax[2].set_xticklabels([f"{t % 100:02d}-{s % 100:02d}" for t, s in zip(d.t, d.s)], rotation=90, fontsize=6)
    ax[2].set_title("(c) antisymmetric share of\nstructural divergence (Prop. 2)")
    fig.tight_layout()
    save(fig, "fig5_attribution")


# ----------------------------------------------------------------------------
# Figure 6: regional lead-lag
# ----------------------------------------------------------------------------
SHORT = {"East Asia & Pacific": "EAP", "South & South-East Asia": "SSEA", "Europe": "EUR",
         "Middle East & North Africa": "MENA", "Sub-Saharan Africa": "SSA", "Latin America & Caribbean": "LAC"}


def fig6_regional():
    df = read("emp_e5_leadlag.csv")
    regs = list(SHORT)
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.8))
    for k, geom in enumerate(["directed", "symmetrised"]):
        d = df[df.geometry == geom]
        M = np.full((6, 6), np.nan)
        for r in d.itertuples():
            M[regs.index(r.region_a), regs.index(r.region_b)] = r.tau
        lim = np.nanmax(np.abs(df.tau.values)) if len(df) else 1
        im = ax[k].imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim))
        for i in range(6):
            for j in range(6):
                if np.isfinite(M[i, j]):
                    ax[k].text(j, i, "%+.2f" % M[i, j], ha="center", va="center", fontsize=5.5)
        ax[k].set_xticks(range(6)); ax[k].set_yticks(range(6))
        ax[k].set_xticklabels([SHORT[r] for r in regs], rotation=90); ax[k].set_yticklabels([SHORT[r] for r in regs])
        ax[k].set_title("(%s) %s geometry" % ("ab"[k], geom))
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=r"$\hat\tau(a,b)$ ($>0$: row region leads)")
    save(fig, "fig6_regional")


# ----------------------------------------------------------------------------
# Supplementary figures
# ----------------------------------------------------------------------------
def figS1_missing():
    df = read("sim_missing.csv")
    order = ["MCAR 5%", "MCAR 15%", "MCAR 30%", "MCAR 45%", "asymmetric", "cost-dependent", "node-dependent", "structured"]
    meths = ["mask", "two-way FE imputation", "row-max imputation"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
    x = np.arange(len(order))
    for m, meth in enumerate(meths):
        for k, col in enumerate(["rel_err", "spearman_offdiag"]):
            g = df[df.method == meth].groupby("mechanism")[col]
            mu = g.mean().reindex(order); sd = g.std().reindex(order)
            ax[k].errorbar(x + (m - 1) * 0.18, mu.values, yerr=sd.values, fmt=MK[m], color=PAL[m], ms=3.5, capsize=1.5,
                           lw=0.8, label=meth)
    ax[0].set_yscale("log"); ax[0].set_ylabel(r"$\|\hat\Delta-\Delta\|_F/\|\Delta\|_F$ (mean $\pm$ sd)")
    ax[1].set_ylabel("Spearman correlation with full data")
    for a in ax:
        a.set_xticks(x); a.set_xticklabels([o.replace("dependent", "dep.") for o in order], rotation=60, fontsize=6.5)
    ax[1].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    save(fig, "figS1_missing")


def figS4_beta():
    df = pd.read_csv(os.path.join(RES, "sim_leadlag2.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(6.0, 2.3), sharey=True)
    for k, sh in enumerate(["level", "spread"]):
        d = df[(df.shock == sh) & (df["T"] == 10)]
        for m, lag in enumerate(range(5)):
            g = d[d.lag == lag].groupby("beta_factor").tau.mean()
            ax[k].plot(g.index, g.values, color=PAL[m], marker=MK[m], ls=LS[m], ms=3, label=r"$\ell=%d$" % lag)
        ax[k].set_xscale("log"); ax[k].set_xlabel(r"$\beta$ / median of level-adjusted $\Delta$")
        ax[k].set_title("%s shock, $T=10$" % sh, fontsize=8)
    ax[0].set_ylabel(r"mean $\hat\tau$"); ax[1].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    save(fig, "figS4_beta")


def figS2_alpha_path():
    summ = read("emp_e1_summary.csv")
    summ = summ[summ.year > 0]
    fig, ax = plt.subplots(1, 2, figsize=(6.0, 2.3))
    for m, a in enumerate(sorted(summ.alpha.unique())):
        s = summ[np.isclose(summ.alpha, a)].sort_values("year")
        ax[0].plot(YEARS, s.anomaly / s.anomaly.mean(), color=PAL[m], marker=MK[m], ls=LS[m], ms=3, label=r"$\alpha=%.2f$" % a)
        ax[1].plot(YEARS, s.trend_adjusted / s.anomaly.mean(), color=PAL[m], marker=MK[m], ls=LS[m], ms=3)
    ax[0].set_title("(a) anomaly score / mean"); ax[1].set_title("(b) trend-adjusted score / mean anomaly")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    save(fig, "figS2_alpha_path")


def figS3_corridors():
    df = read("emp_e6_corridors.csv")
    d = df[df.transition == "2020-2021"].head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    lab = [f"{o[:18]} $\\to$ {t[:18]}" for o, t in zip(d.origin, d.destination)]
    ax.barh(np.arange(len(d)), d.contribution, color=[PAL[1] if v > 0 else PAL[0] for v in d.contribution])
    ax.set_yticks(np.arange(len(d))); ax.set_yticklabels(lab, fontsize=6)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel(r"first-order contribution $\partial\Delta/\partial c_{ij}\times\Delta c_{ij}$")
    ax.set_title("Largest corridor contributions, 2020$\\to$2021")
    fig.tight_layout()
    save(fig, "figS3_corridors")


if __name__ == "__main__":
    for f in [fig1_verification, fig2_detection, fig3_leadlag, fig4_trajectory, fig5_attribution,
              fig6_regional, figS1_missing, figS2_alpha_path, figS3_corridors, figS4_beta]:
        try:
            f()
        except Exception as e:          # keep going if some results are not yet available
            print("SKIPPED", f.__name__, "->", repr(e))
