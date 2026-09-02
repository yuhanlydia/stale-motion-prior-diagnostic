"""Action-chunk latency telemetry around detected dynamics changes."""

from __future__ import annotations


def change_chunk_telemetry(*, pending_actions: int, last_chunk_size: int | None) -> dict[str, float | int | None]:
    pending = max(0, int(pending_actions))
    size = int(last_chunk_size) if last_chunk_size else 0
    if pending == 0 or size <= 0:
        return {
            "actions_remaining_at_change": pending,
            "steps_until_next_policy_query": pending,
            "chunk_phase": None,
        }
    completed = max(0, size - pending)
    return {
        "actions_remaining_at_change": pending,
        "steps_until_next_policy_query": pending,
        "chunk_phase": completed / size,
    }
