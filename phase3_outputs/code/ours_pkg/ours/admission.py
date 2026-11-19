"""Deadline admission (Phase 2 'Deadline admission and receiver behavior')."""
from . import serializer as S

D_MS = 100.0
RATE_INIT_BPS = 5e6
RATE_EMA_WEIGHT = 0.2


class RateEstimator:
    """Causal link-rate estimate: exponential moving average (weight 0.2) of delivered-packet observations."""

    def __init__(self, init_bps=RATE_INIT_BPS, weight=RATE_EMA_WEIGHT):
        self.rate, self.w = float(init_bps), weight

    def update(self, delivered_bytes, service_time_s):
        if delivered_bytes > 0 and service_time_s > 0:
            obs = 8.0 * delivered_bytes / service_time_s
            self.rate = (1 - self.w) * self.rate + self.w * obs
        return self.rate


def predicted_completion_ms(k, t_ready_ms, queue_ms, rate_bps, t_reserve_ms, bits=8):
    """t_ready + Q_est + 8 Ldata(K) / R_est + T_reserve  (ms)."""
    return t_ready_ms + queue_ms + 1000.0 * 8.0 * S.wire_cost(k, bits) / rate_bps + t_reserve_ms


def max_admissible_k(k_wanted, t_ready_ms, queue_ms, rate_bps, t_reserve_ms, deadline_ms=D_MS,
                     budget_bytes=None, bits=8, feedback=True):
    """Largest K <= k_wanted that satisfies the deadline rule and the per-frame byte budget."""
    k_cap = k_wanted
    if budget_bytes is not None:
        k_cap = min(k_cap, S.max_tokens_in_budget(budget_bytes, bits, feedback))
    k = 0
    for cand in range(1, k_cap + 1):
        if predicted_completion_ms(cand, t_ready_ms, queue_ms, rate_bps, t_reserve_ms, bits) <= deadline_ms:
            k = cand
        else:
            break  # cost is monotone in K
    return k
