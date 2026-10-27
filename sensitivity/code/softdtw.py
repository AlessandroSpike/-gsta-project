"""
Soft-DTW (Cuturi & Blondel 2017) with exact gradient E = grad A_beta(Delta) and exact
Hessian-vector products d E[dDelta] by forward-mode differentiation of both recursions.
Exhaustive path enumeration is provided for verification (Delannoy-many paths).
"""
import itertools
import numpy as np

INF = np.inf


def _softmin3(a, b, c, beta):
    m = min(a, b, c)
    if m == INF:
        return INF, (0.0, 0.0, 0.0)
    ea, eb, ec = (np.exp(-(x - m) / beta) if x < INF else 0.0 for x in (a, b, c))
    z = ea + eb + ec
    return m - beta * np.log(z), (ea / z, eb / z, ec / z)


def soft_dtw(D, beta, dD=None):
    """Return value, E and (if dD given) the directional derivatives of value and E."""
    T, S = D.shape
    R = np.full((T + 2, S + 2), INF)
    R[0, 0] = 0.0
    Rd = np.zeros((T + 2, S + 2))
    for i in range(1, T + 1):
        for j in range(1, S + 1):
            v, (wa, wb, wc) = _softmin3(R[i - 1, j - 1], R[i - 1, j], R[i, j - 1], beta)
            R[i, j] = D[i - 1, j - 1] + v
            if dD is not None:
                Rd[i, j] = dD[i - 1, j - 1] + wa * Rd[i - 1, j - 1] + wb * Rd[i - 1, j] + wc * Rd[i, j - 1]
    # backward recursion for E (and its tangent)
    Dp = np.zeros((T + 2, S + 2)); Dp[1:T + 1, 1:S + 1] = D
    dDp = np.zeros((T + 2, S + 2))
    if dD is not None:
        dDp[1:T + 1, 1:S + 1] = dD
    E = np.zeros((T + 2, S + 2)); Ed = np.zeros((T + 2, S + 2))
    E[T, S] = 1.0
    for i in range(T, 0, -1):
        for j in range(S, 0, -1):
            if i == T and j == S:
                continue
            e = ed = 0.0
            for (ii, jj) in ((i + 1, j), (i, j + 1), (i + 1, j + 1)):
                if ii > T or jj > S:
                    continue
                # weight of predecessor (i,j) in the softmin defining R[ii,jj]
                w = np.exp((R[ii, jj] - Dp[ii, jj] - R[i, j]) / beta)
                e += E[ii, jj] * w
                if dD is not None:
                    wd = w * (Rd[ii, jj] - dDp[ii, jj] - Rd[i, j]) / beta
                    ed += Ed[ii, jj] * w + E[ii, jj] * wd
            E[i, j], Ed[i, j] = e, ed
    val = R[T, S]
    if dD is None:
        return val, E[1:T + 1, 1:S + 1]
    return val, E[1:T + 1, 1:S + 1], Rd[T, S], Ed[1:T + 1, 1:S + 1]


def enumerate_paths(T, S):
    """All monotone alignment paths as binary (n_paths, T*S) matrix."""
    paths = []

    def rec(i, j, cells):
        if i == T - 1 and j == S - 1:
            paths.append(cells.copy()); return
        for di, dj in ((1, 0), (0, 1), (1, 1)):
            ii, jj = i + di, j + dj
            if ii < T and jj < S:
                cells.append(ii * S + jj); rec(ii, jj, cells); cells.pop()
    rec(0, 0, [0])
    A = np.zeros((len(paths), T * S))
    for k, c in enumerate(paths):
        A[k, c] = 1.0
    return A


def gibbs_moments(D, beta, A=None):
    """Exact value, E and Hessian (= -Cov/beta) by enumeration."""
    T, S = D.shape
    A = enumerate_paths(T, S) if A is None else A
    cost = A @ D.ravel()
    m = cost.min()
    w = np.exp(-(cost - m) / beta)
    Z = w.sum()
    p = w / Z
    val = m - beta * np.log(Z)
    E = p @ A
    Cov = (A * p[:, None]).T @ A - np.outer(E, E)
    return val, E.reshape(T, S), -Cov / beta, p, A


# ----------------------------------------------------------------------------------------
# lead-lag index (Definition 3) with the level adjustment and data-dependent beta of 4.3
# ----------------------------------------------------------------------------------------
def offsets(T, S):
    return np.arange(S)[None, :] - np.arange(T)[:, None]


def leadlag(Dgh, beta=None, beta_frac=0.1, level_adjust=True):
    Dt = Dgh - Dgh.min() if level_adjust else Dgh.copy()
    b = beta_frac * np.median(Dt) if beta is None else beta
    _, E = soft_dtw(Dt, b)
    O = offsets(*Dgh.shape)
    return float(np.sum(O * E) / np.sum(E)), b, E


def leadlag_grad(Dgh, beta_frac=0.1, level_adjust=True, fixed_beta=None):
    """Exact gradient d tau / d Delta^{GH} (T x S), including the level adjustment (min) and
    the data-dependent temperature beta = beta_frac * median(level-adjusted matrix).
    Computed column-by-column with Hessian-vector products (TS products)."""
    T, S = Dgh.shape
    O = offsets(T, S)
    Dt = Dgh - Dgh.min() if level_adjust else Dgh.copy()
    b = beta_frac * np.median(Dt) if fixed_beta is None else fixed_beta
    _, E = soft_dtw(Dt, b)
    Z = E.sum()
    tau = float(np.sum(O * E) / Z)
    # d tau / d (Dt / b) via HVPs: d tau = <O - tau, dE>/Z
    g_u = np.zeros((T, S))
    for k in range(T * S):
        dU = np.zeros(T * S); dU[k] = 1.0
        _, _, _, Ed = soft_dtw(Dt / b, 1.0, dU.reshape(T, S))
        g_u.ravel()[k] = np.sum((O - tau) * Ed) / Z
    # chain rule through u = Dt / b with Dt = D - min D and b = beta_frac * median(Dt)
    g_Dt = g_u / b                                   # partial wrt Dt at fixed b
    g_b = -np.sum(g_u * Dt) / b ** 2                 # partial wrt b
    G = g_Dt.copy()
    flat = Dgh.ravel()
    if fixed_beta is None:
        # median of TS entries (even count: average of the two middle order statistics)
        order = np.argsort(Dt.ravel(), kind="stable")
        nc = T * S
        mids = [order[nc // 2]] if nc % 2 else [order[nc // 2 - 1], order[nc // 2]]
        for k in mids:
            G.ravel()[k] += g_b * beta_frac / len(mids)
    if level_adjust:
        kmin = int(np.argmin(flat))
        # Dt_ij = D_ij - D_kmin  -> total derivative wrt D_kmin collects -sum of all partials
        tot_Dt = np.sum(g_Dt)
        G.ravel()[kmin] -= tot_Dt
        if fixed_beta is None:
            # median(Dt) = median(D) - D_kmin  -> d b / d D_kmin = -beta_frac
            G.ravel()[kmin] -= g_b * beta_frac
    return tau, G, b, E


def corrected_lead(tau, T):
    return T * tau / (T - 1 - abs(tau))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for T, S in ((6, 6), (5, 7), (4, 4)):
        D = rng.random((T, S))
        beta = 0.3
        v, E = soft_dtw(D, beta)
        v2, E2, H, p, A = gibbs_moments(D, beta)
        dD = rng.standard_normal((T, S))
        _, _, vd, Ed = soft_dtw(D, beta, dD)
        print(T, S, "paths", A.shape[0], "|v-v_enum|", abs(v - v2), "|E-E_enum|", np.abs(E - E2).max(),
              "|dv - <E,dD>|", abs(vd - np.sum(E * dD)),
              "|HVP - Cov|", np.abs(Ed.ravel() - H @ dD.ravel()).max())
    # gradient of the level-adjusted lead-lag index vs central differences
    D = rng.random((6, 6)) + 0.5
    tau, G, b, _ = leadlag_grad(D)
    h = 1e-6
    errs = []
    for k in range(36):
        Dp = D.copy(); Dp.ravel()[k] += h
        Dm = D.copy(); Dm.ravel()[k] -= h
        fd = (leadlag(Dp)[0] - leadlag(Dm)[0]) / (2 * h)
        errs.append(abs(fd - G.ravel()[k]))
    print("tau", tau, "max |grad - FD|", max(errs), "max|grad|", np.abs(G).max())
    # invariances: sum of gradient (level) and <G, D> (scale) vanish
    print("sum G (level invariance):", G.sum(), " <G,D> (scale invariance):", np.sum(G * D))
