"""Diagnostics for RESET_CHANGE sensitivity to action-chunk position."""

from __future__ import annotations

import numpy as np


def _design_key(row: dict) -> tuple[str, int, int]:
    return (
        str(row["task"]),
        int(row["level"]),
        int(row.get("requested_start_seed", row["seed"])),
    )


def summarize_reset_by_chunk_phase(
    result_rows: list[dict],
    full_log_rows: list[dict],
) -> dict:
    """Relate paired RESET-FULL MS to the first change's chunk position.

    A larger ``actions_remaining_at_change`` means the reset happened earlier
    inside the currently committed action chunk, leaving more simulator frames
    for the history buffer to rebuild before the next policy query.  If reset
    gains increase strongly with this quantity, the main limitation may be
    post-reset cold start rather than evidence that pre-change history is always
    useful.
    """
    first_change: dict[tuple[str, int, int], int] = {}
    ordered_logs = sorted(
        full_log_rows,
        key=lambda row: (
            str(row.get("task", "")),
            int(row.get("level", -1)),
            int(row.get("requested_start_seed", row.get("seed", -1))),
            int(row.get("query", 0)),
        ),
    )
    for row in ordered_logs:
        if not bool(row.get("change_point")) or not bool(
            row.get("pre_grasp", False)
        ):
            continue
        remaining = row.get("actions_remaining_at_change")
        if remaining is None:
            continue
        first_change.setdefault(_design_key(row), int(remaining))

    by_cell = {
        (_design_key(row), str(row["condition"])): row
        for row in result_rows
    }
    pairs: list[tuple[int, float]] = []
    design_keys = {
        key for key, condition in by_cell if condition == "full"
    }
    for key in sorted(design_keys):
        full = by_cell.get((key, "full"))
        reset = by_cell.get((key, "reset_change"))
        if full is None or reset is None or key not in first_change:
            continue
        pairs.append(
            (
                first_change[key],
                float(reset["ms"]) - float(full["ms"]),
            )
        )

    if not pairs:
        return {"n": 0}

    remaining = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    delta_ms = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    median_remaining = float(np.median(remaining))
    low = delta_ms[remaining <= median_remaining]
    high = delta_ms[remaining > median_remaining]
    if (
        len(pairs) >= 2
        and float(np.std(remaining)) > 0.0
        and float(np.std(delta_ms)) > 0.0
    ):
        correlation = float(np.corrcoef(remaining, delta_ms)[0, 1])
    else:
        correlation = None

    return {
        "n": int(len(pairs)),
        "mean_delta_ms": float(delta_ms.mean()),
        "median_actions_remaining": median_remaining,
        "pearson_actions_remaining_delta_ms": correlation,
        "low_remaining_n": int(len(low)),
        "low_remaining_mean_delta_ms": (
            float(low.mean()) if len(low) else None
        ),
        "high_remaining_n": int(len(high)),
        "high_remaining_mean_delta_ms": (
            float(high.mean()) if len(high) else None
        ),
    }
