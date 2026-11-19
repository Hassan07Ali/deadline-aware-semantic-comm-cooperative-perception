"""Digital tile format and wire-byte accounting (Phase 2, 'Digital tile representation' + 'Budget accounting')."""
import math

TILE_VALUES = 256            # 2 x 2 BEV cells x 64 channels
TOKEN_META_BYTES = 6         # 2 B tile index + 1 B modality id + 2 B fp16 scale + 1 B flags
PAYLOAD_CAP = 1200           # application payload cap per packet (includes the 32 B header)
APP_HEADER = 32
LINK_OVERHEAD = 40           # modeled link/transport overhead per packet
PER_PACKET_OVERHEAD = APP_HEADER + LINK_OVERHEAD   # 72 B
FEEDBACK_BYTES = 512 + 32 + 40                     # 32x16 int8 confidence map + header + link = 584 B


def token_bytes(bits=8):
    """v = 256 q / 8 + 6 bytes."""
    return TILE_VALUES * bits // 8 + TOKEN_META_BYTES


def tokens_per_packet(bits=8):
    """n = floor((1200 - 32) / v)."""
    return (PAYLOAD_CAP - APP_HEADER) // token_bytes(bits)


def num_packets(k, bits=8):
    return 0 if k <= 0 else math.ceil(k / tokens_per_packet(bits))


def wire_cost(k, bits=8):
    """Ldata = K v + 72 ceil(K / n); K = 0 sends no data packet."""
    if k <= 0:
        return 0
    return k * token_bytes(bits) + PER_PACKET_OVERHEAD * num_packets(k, bits)


def packet_sizes(k, bits=8):
    """Wire bytes of each data packet, in order (last one may be smaller)."""
    if k <= 0:
        return []
    n, v = tokens_per_packet(bits), token_bytes(bits)
    sizes, left = [], k
    while left > 0:
        t = min(n, left)
        sizes.append(t * v + PER_PACKET_OVERHEAD)
        left -= t
    return sizes


def max_tokens_in_budget(budget_bytes, bits=8, feedback=True):
    """Largest K such that data + feedback wire bytes fit the per-frame budget."""
    fb = FEEDBACK_BYTES if feedback else 0
    k = 0
    while wire_cost(k + 1, bits) + fb <= budget_bytes:
        k += 1
    return k
