"""Replay of the Phase 2 link conditions C0-C3 with common random numbers."""
import hashlib
from dataclasses import dataclass
import numpy as np
from . import serializer as S


@dataclass(frozen=True)
class Link:
    rate_bps: float
    loss: float
    queue_ms: float

GOOD = Link(20e6, 0.0, 0.0)
BAD = Link(5e6, 0.10, 20.0)
CONDITIONS = {  # (sensor_state, link)
    "C0": ("clean", GOOD),
    "C1": ("degraded", GOOD),
    "C2": ("clean", BAD),
    "C3": ("degraded", BAD),
}


def loss_draw(scene, frame, trace, packet_idx):
    """Uniform(0,1) indexed by (scene, frame, trace, packet ordinal): identical for every method."""
    h = hashlib.sha256(f"{scene}|{frame}|{trace}|{packet_idx}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def transmit(k, link, t_ready_ms, scene="s", frame=0, trace=0, bits=8, tokens_per_pkt=None):
    """Send K tokens. Packets are serialised back-to-back after the queue delay; each is lost with prob link.loss
    (a lost packet still consumes its bytes and time). Returns arrival times and which tokens arrived."""
    sizes = S.packet_sizes(k, bits)
    n = S.tokens_per_packet(bits)
    t = t_ready_ms + link.queue_ms
    arrivals, token_ok, bytes_sent = [], [], 0
    for i, sz in enumerate(sizes):
        t += 1000.0 * 8.0 * sz / link.rate_bps
        bytes_sent += sz
        ok = loss_draw(scene, frame, trace, i) >= link.loss
        arrivals.append(t)
        n_tok = min(n, k - i * n)
        token_ok.extend([ok] * n_tok)
    return {"packet_arrival_ms": arrivals, "token_delivered": np.array(token_ok, dtype=bool),
            "bytes_sent": bytes_sent, "last_arrival_ms": arrivals[-1] if arrivals else t_ready_ms}


def delivered_before(result, deadline_ms, t_proc_ms=0.0):
    """Token mask of tokens that arrived intact early enough for decode+fusion+detection to finish by the deadline."""
    n_tok = len(result["token_delivered"])
    if n_tok == 0:
        return np.zeros(0, dtype=bool)
    n = S.tokens_per_packet(8)
    on_time = np.array([result["packet_arrival_ms"][i // n] + t_proc_ms <= deadline_ms for i in range(n_tok)])
    return result["token_delivered"] & on_time
