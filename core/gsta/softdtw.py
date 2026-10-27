"""Soft dynamic time warping (Cuturi & Blondel, 2017) in float64.

A_beta(Delta) = -beta log sum_{A in paths} exp(-<A,Delta>/beta)
Gradient E = d A_beta / d Delta is the expected alignment matrix under the
Gibbs distribution over monotone paths, p(A) ~ exp(-<A,Delta>/beta).
"""
import math
import numpy as np
from numba import njit

__all__ = ["soft_dtw", "soft_dtw_grad", "hard_dtw", "lead_lag", "delannoy",
           "softdtw_divergence", "boundary_corrected_lag"]


@njit(cache=True)
def _softmin3(a, b, c, beta):
    m = min(a, min(b, c))
    if m == np.inf:
        return np.inf
    s = math.exp(-(a - m) / beta) + math.exp(-(b - m) / beta) + math.exp(-(c - m) / beta)
    return m - beta * math.log(s)


@njit(cache=True)
def _forward(D, beta):
    T, S = D.shape
    R = np.full((T + 2, S + 2), np.inf)
    R[0, 0] = 0.0
    for i in range(1, T + 1):
        for j in range(1, S + 1):
            R[i, j] = D[i - 1, j - 1] + _softmin3(R[i - 1, j - 1], R[i - 1, j], R[i, j - 1], beta)
    return R


@njit(cache=True)
def _backward(D, R, beta):
    T, S = D.shape
    Dp = np.zeros((T + 2, S + 2))
    Dp[1:T + 1, 1:S + 1] = D
    E = np.zeros((T + 2, S + 2))
    Rb = R.copy()
    for i in range(1, T + 1):
        Rb[i, S + 1] = -np.inf
    for j in range(1, S + 1):
        Rb[T + 1, j] = -np.inf
    Rb[T + 1, S + 1] = R[T, S]
    E[T + 1, S + 1] = 1.0
    for j in range(S, 0, -1):
        for i in range(T, 0, -1):
            a = math.exp((Rb[i + 1, j] - Rb[i, j] - Dp[i + 1, j]) / beta)
            b = math.exp((Rb[i, j + 1] - Rb[i, j] - Dp[i, j + 1]) / beta)
            c = math.exp((Rb[i + 1, j + 1] - Rb[i, j] - Dp[i + 1, j + 1]) / beta)
            E[i, j] = E[i + 1, j] * a + E[i, j + 1] * b + E[i + 1, j + 1] * c
    return E[1:T + 1, 1:S + 1]


def soft_dtw(Delta, beta):
    Delta = np.ascontiguousarray(Delta, dtype=float)
    return float(_forward(Delta, float(beta))[Delta.shape[0], Delta.shape[1]])


def soft_dtw_grad(Delta, beta):
    """Return (A_beta(Delta), E) with E the expected alignment matrix."""
    Delta = np.ascontiguousarray(Delta, dtype=float)
    R = _forward(Delta, float(beta))
    E = _backward(Delta, R, float(beta))
    return float(R[Delta.shape[0], Delta.shape[1]]), E


@njit(cache=True)
def _hard(D):
    T, S = D.shape
    R = np.full((T + 1, S + 1), np.inf)
    R[0, 0] = 0.0
    for i in range(1, T + 1):
        for j in range(1, S + 1):
            R[i, j] = D[i - 1, j - 1] + min(R[i - 1, j - 1], min(R[i - 1, j], R[i, j - 1]))
    return R[T, S]


def hard_dtw(Delta):
    return float(_hard(np.ascontiguousarray(Delta, dtype=float)))


def delannoy(m, n):
    """Number of monotone (1,0),(0,1),(1,1) lattice paths from (0,0) to (m,n);
    |paths(T,S)| = delannoy(T-1, S-1)."""
    D = np.zeros((m + 1, n + 1), dtype=object)
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                D[i, j] = 1
            else:
                D[i, j] = D[i - 1, j] + D[i, j - 1] + D[i - 1, j - 1]
    return int(D[m, n])


def lead_lag(E):
    """tau(G,H) = sum_ts (s - t) E_ts / sum_ts E_ts.
    Convention: tau > 0  <=>  the FIRST trajectory G leads H
    (states of G are matched to later states of H)."""
    T, S = E.shape
    t = np.arange(T)[:, None]
    s = np.arange(S)[None, :]
    return float(((s - t) * E).sum() / E.sum())


def boundary_corrected_lag(tau, T):
    """Invert the boundary attenuation tau = l (T-1)/(T+|l|) of the
    ramp-plateau-ramp path (T = S):  l = T tau / (T - 1 - |tau|)."""
    return T * tau / (T - 1 - abs(tau))


def softdtw_divergence(D_GH, D_GG, D_HH, beta):
    return soft_dtw(D_GH, beta) - 0.5 * soft_dtw(D_GG, beta) - 0.5 * soft_dtw(D_HH, beta)
