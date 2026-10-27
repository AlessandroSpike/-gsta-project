"""Data-generating processes for the simulation studies (Section 4)."""
import numpy as np
from .data import GraphState


def base_network(n, rng, kappa=0.5, offset=2.0, mass_sd=1.0):
    """Directed base geometry C0 = S + A (+offset): S symmetric (latent positions),
    A antisymmetric (node-level direction effects); log-normal masses (total 1)
    and persistent one-dimensional features."""
    x = rng.normal(size=(n, 2))
    S = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)) + offset
    u = rng.normal(size=n)
    A = kappa * (u[:, None] - u[None, :])
    C0 = S + A
    np.fill_diagonal(C0, 0.0)
    mu = rng.lognormal(0.0, mass_sd, n)
    mu = mu / mu.sum()
    f = rng.normal(size=(n, 1))
    return dict(S=S, A=A, C0=C0, mu=mu, f=f)


def noisy_state(net, rng, sigma=0.25, feat_sd=0.05, mass_cv=0.02, C=None, f=None, mu=None):
    n = net["mu"].shape[0]
    C = net["C0"] if C is None else C
    f = net["f"] if f is None else f
    mu = net["mu"] if mu is None else mu
    Ct = np.maximum(C + sigma * rng.normal(size=(n, n)), 0.0)
    np.fill_diagonal(Ct, 0.0)
    ft = f + feat_sd * rng.normal(size=f.shape)
    mt = mu * np.exp(mass_cv * rng.normal(size=n))
    return GraphState(mt, ft, Ct, None)


def shocked_inputs(net, scenario, delta, hubs):
    """(C, f, mu) at the shock date for a scenario with effect size delta."""
    C = net["C0"].copy()
    f = net["f"].copy()
    mu = net["mu"].copy()
    H = np.zeros_like(C, dtype=bool)
    H[np.ix_(hubs, hubs)] = True
    np.fill_diagonal(H, False)
    if scenario == "reversal":          # only the antisymmetric part changes
        C = np.where(H, net["S"] + (1 - 2 * delta) * net["A"], C)
    elif scenario == "corridor":        # cost surge on hub corridors
        C = np.where(H, C * (1 + delta), C)
    elif scenario == "feature":         # connectivity shock at hubs
        f[hubs] = f[hubs] + 2.0 * delta
    elif scenario == "mass":            # throughput contraction at hubs
        mu[hubs] = mu[hubs] * (1 - 0.5 * delta)
    else:
        raise ValueError(scenario)
    np.fill_diagonal(C, 0.0)
    return C, f, mu


def shock_trajectory(net, rng, T, t0, scenario, delta, hubs, sigma=0.25):
    states = []
    for t in range(T):
        if t == t0 and delta > 0:
            C, f, mu = shocked_inputs(net, scenario, delta, hubs)
            states.append(noisy_state(net, rng, sigma, C=C, f=f, mu=mu))
        else:
            states.append(noisy_state(net, rng, sigma))
    return states


def pulse(T, center, width):
    t = np.arange(T)
    return np.exp(-((t - center) ** 2) / (2.0 * width ** 2))


def pulse_trajectory(net, rng, p, delta=1.0, sigma=0.2, mass_drop=0.2, feat_drop=0.5):
    """Common-shock trajectory: multiplicative cost surge, throughput dip and
    connectivity loss, all following the pulse profile p(t)."""
    out = []
    for pt in p:
        C = net["C0"] * (1 + delta * pt)
        mu = net["mu"] * (1 - mass_drop * pt)
        f = net["f"] - feat_drop * pt
        out.append(noisy_state(net, rng, sigma, C=C, f=f, mu=mu))
    return out


def spread_trajectory(net, rng, p, delta=0.8, sigma=0.25):
    """Mean-preserving spread of the cost geometry around its mass-weighted mean:
    C_t = m_w + (1 + delta p(t)) (C0 - m_w); the mass-weighted mean cost is unchanged."""
    w = np.outer(net["mu"], net["mu"]); np.fill_diagonal(w, 0)
    mw = (w * net["C0"]).sum() / w.sum()
    out = []
    for pt in p:
        C = np.maximum(mw + (1 + delta * pt) * (net["C0"] - mw), 0.0)
        np.fill_diagonal(C, 0.0)
        out.append(noisy_state(net, rng, sigma, C=C))
    return out


def relabel(states, rng):
    """Non-persistent identities: an independent random relabelling of the nodes in each period."""
    out = []
    for s in states:
        p = rng.permutation(s.mu.shape[0])
        out.append(GraphState(s.mu[p], s.F[p], s.C[np.ix_(p, p)], None if s.W is None else s.W[np.ix_(p, p)]))
    return out


def summary_vector(s):
    """Permutation-invariant network summaries: mass-weighted mean and sd of costs, asymmetry index,
    mass-weighted mean attribute, total mass, entropy of the normalised masses."""
    w = np.outer(s.mu, s.mu); np.fill_diagonal(w, 0); w = w / w.sum()
    m = (w * s.C).sum(); sd = np.sqrt((w * (s.C - m) ** 2).sum())
    ai = np.linalg.norm(s.C - s.C.T) / np.linalg.norm(s.C)
    pi = s.mu / s.mu.sum()
    return np.array([m, sd, ai, float((pi * s.F[:, 0]).sum()), s.mu.sum(), float(-(pi * np.log(pi)).sum())])


def spectral_signature(s):
    """Sorted singular values of the mass-weighted geometry diag(sqrt mu) C diag(sqrt mu)."""
    r = np.sqrt(s.mu)
    return np.linalg.svd(r[:, None] * s.C * r[None, :], compute_uv=False)
