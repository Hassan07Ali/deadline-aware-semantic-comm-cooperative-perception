"""Which output is published at the deadline (Phase 2 'Timing definition')."""


def choose_output(coop_ready_ms, ego_ready_ms, deadline_ms):
    """Latest eligible cooperative result; else completed ego-only; else an empty prediction. Late packets are discarded."""
    if coop_ready_ms is not None and coop_ready_ms <= deadline_ms:
        return "cooperative"
    if ego_ready_ms is not None and ego_ready_ms <= deadline_ms:
        return "ego_only"
    return "empty"
