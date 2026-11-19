"""Tile scoring (Phase 2 'Selection score') and the selection baselines."""
import numpy as np

EPS, LAMBDA_C, TAU_MS = 0.1, 1.0, 100.0
FEEDBACK_EXPIRY_MS = 200.0
# fixed forward critical-region proxy: 0 < x < 40 m and |y| < 10 m (ego frame)
CRIT_X, CRIT_Y = (0.0, 40.0), 10.0


def critical_mask(xy):
    """xy: (N, 2) tile centres in the ego frame -> bool (N,)."""
    xy = np.asarray(xy, dtype=float)
    return (xy[:, 0] > CRIT_X[0]) & (xy[:, 0] < CRIT_X[1]) & (np.abs(xy[:, 1]) < CRIT_Y)


def ego_confidence(e, feedback_age_ms):
    """Missing or expired (>200 ms) feedback sets e = 0."""
    e = np.asarray(e, dtype=float)
    if feedback_age_ms is None or feedback_age_ms > FEEDBACK_EXPIRY_MS:
        return np.zeros_like(e)
    return e


def score(p, r, a, e, c, age_ms, eps=EPS, lam_c=LAMBDA_C, tau_ms=TAU_MS):
    """s = a r p (1 + lam_c c) [eps + (1 - eps)(1 - e)] exp(-age / tau)."""
    p, r, a, e, c = (np.asarray(x, dtype=float) for x in (p, r, a, e, c))
    return a * r * p * (1.0 + lam_c * c) * (eps + (1.0 - eps) * (1.0 - e)) * np.exp(-age_ms / tau_ms)


def top_k(scores, k, valid=None):
    """Indices of the k highest scores (stable; ties broken by index). Invalid/zero-score tiles are never picked."""
    s = np.asarray(scores, dtype=float).copy()
    if valid is not None:
        s[~np.asarray(valid, dtype=bool)] = -np.inf
    order = np.argsort(-s, kind="stable")
    order = [i for i in order[:max(k, 0)] if np.isfinite(s[i]) and s[i] > 0]
    return np.array(order, dtype=int)


# ---- baselines (same packet format, same byte cap, same detector) ----
def select_confidence_only(p, a, k):
    """B3: rank by foreground probability only (quality, need, critical region and age ignored)."""
    return top_k(np.asarray(p, float) * np.asarray(a, float), k)


def select_random(a, k, rng):
    """B2: uniformly random among available tiles."""
    avail = np.flatnonzero(np.asarray(a) > 0)
    if k <= 0 or len(avail) == 0:
        return np.array([], dtype=int)
    return np.sort(rng.choice(avail, size=min(k, len(avail)), replace=False))
