"""Trajectory-level G-STA: pairwise debiased UFGW matrices, soft-DTW alignment,
double debiasing, lead-lag estimation and anomaly scores."""
from __future__ import annotations

import numpy as np

from .ufgw import UFGWProblem, ufgw_multistart, feature_cost, envelope_gradients
from .softdtw import soft_dtw, soft_dtw_grad, lead_lag

DEFAULTS = dict(alpha=0.5, gamma=1.0, eps=0.05)


def ufgw_pair(g, h, alpha=0.5, gamma=1.0, eps=0.05, same_nodes=None,
              inits=("identity", "product", "feature"), tol=1e-9, max_iter=3000,
              return_result=False):
    """Regularised UFGW value V_eps(g, h) from a deterministic multi-start."""
    if same_nodes is None:
        same_nodes = (g.mu.shape[0] == h.mu.shape[0])
    prob = UFGWProblem(g.mu, h.mu, feature_cost(g.F, h.F), g.C, h.C,
                       alpha=alpha, gamma=gamma, eps=eps, Wc=g.W, Wd=h.W)
    prob._f1, prob._g1 = np.asarray(g.F)[:, 0], np.asarray(h.F)[:, 0]
    res = ufgw_multistart(prob, same_nodes=same_nodes, inits=inits, tol=tol,
                          max_iter=max_iter)
    return (res.value, res, prob) if return_result else res.value


def value_matrix(G, H=None, **kw):
    """V_ts = V_eps(G_t, H_s).  If H is None (self comparison), only t <= s is
    solved and the matrix is mirrored (V is symmetric in its arguments)."""
    if H is None:
        T = len(G)
        V = np.zeros((T, T))
        for t in range(T):
            for s in range(t, T):
                V[t, s] = ufgw_pair(G[t], G[s], **kw)
                V[s, t] = V[t, s]
        return V
    V = np.zeros((len(G), len(H)))
    for t in range(len(G)):
        for s in range(len(H)):
            V[t, s] = ufgw_pair(G[t], H[s], **kw)
    return V


def self_values(G, **kw):
    return np.array([ufgw_pair(g, g, **kw) for g in G])


def debias(V, vG, vH):
    """Graph Sinkhorn divergence Delta_ts = V_ts - (vG_t + vH_s)/2."""
    return V - 0.5 * vG[:, None] - 0.5 * vH[None, :]


class Trajectory:
    """Cache of self-quantities for one graph-valued trajectory."""

    def __init__(self, states, name="", **kw):
        self.states = states
        self.name = name
        self.kw = {**DEFAULTS, **kw}
        self._Vself = None

    @property
    def Vself(self):
        """Full T x T matrix of V(G_t, G_s) (needed for the temporal debiasing)."""
        if self._Vself is None:
            self._Vself = value_matrix(self.states, None, **self.kw)
        return self._Vself

    @property
    def diag(self):
        return np.diag(self.Vself).copy()

    @property
    def Delta_self(self):
        d = self.diag
        D = debias(self.Vself, d, d)
        np.fill_diagonal(D, 0.0)
        return D


def gsta(TG: Trajectory, TH: Trajectory, beta, project=False, Vcross=None):
    """Doubly debiased G-STA discrepancy, lead-lag and diagnostics.

    Returns dict with: value (doubly debiased), raw (A_beta(Delta^GH)),
    tau (positive => G leads H), E (soft alignment matrix), Delta (cross),
    Delta_GG, Delta_HH."""
    kw = TG.kw
    if Vcross is None:
        Vcross = value_matrix(TG.states, TH.states, **kw)
    D = debias(Vcross, TG.diag, TH.diag)
    DG, DH = TG.Delta_self, TH.Delta_self
    if project:
        D, DG, DH = np.maximum(D, 0), np.maximum(DG, 0), np.maximum(DH, 0)
    raw, E = soft_dtw_grad(D, beta)
    vG = soft_dtw(DG, beta)
    vH = soft_dtw(DH, beta)
    return dict(value=raw - 0.5 * vG - 0.5 * vH, raw=raw, raw_GG=vG, raw_HH=vH,
                tau=lead_lag(E), E=E, Delta=D, Delta_GG=DG, Delta_HH=DH, V=Vcross)


def anomaly_scores(Delta):
    """a(t) = mean_{s != t} Delta_ts for a self-comparison matrix."""
    T = Delta.shape[0]
    off = ~np.eye(T, dtype=bool)
    return np.array([Delta[t][off[t]].mean() for t in range(T)])
