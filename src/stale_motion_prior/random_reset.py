"""Phase- and chunk-matched random-reset controls from FULL telemetry."""

from __future__ import annotations

from collections import defaultdict
import random
from typing import Iterable


def _identity(row: dict) -> tuple[str, int, int, int]:
    return (
        str(row["task"]),
        int(row["level"]),
        int(row.get("requested_start_seed", row.get("seed", -1))),
        int(row["seed"]),
    )


def _change_event(row: dict, *, max_query: int) -> dict:
    remaining = row.get("actions_remaining_at_change")
    if remaining is None:
        raise ValueError("change telemetry lacks actions_remaining_at_change")
    return {
        "query": int(row["query"]),
        "phase": int(row["query"]) / max(1, max_query),
        "actions_remaining": max(0, int(remaining)),
    }


def build_phase_matched_schedule(
    records: Iterable[dict],
    *,
    cooldown: int = 2,
    rng_seed: int = 0,
    stationary_reference_level: int | None = None,
) -> list[dict]:
    """Build outcome-blind reset controls matched in phase and chunk position.

    v3 matched only the policy-query phase.  That was insufficient because a
    real change reset happens at the simulator step of the change, while the
    old random control reset immediately before the policy query.  DynamicWAM's
    history buffer uses a policy stride of four environment frames, so these two
    interventions could expose the model to very different amounts of fresh
    post-reset history.

    v4 therefore carries ``actions_remaining_at_change`` from FULL telemetry.
    Each control event is scheduled at a non-change policy-query phase and at
    the same pending-action count inside the preceding action chunk.  Stationary
    Level-1 controls borrow both normalized phase and chunk position from the
    same task/requested-seed Level-3 FULL episode.
    """

    grouped: dict[tuple[str, int, int, int], list[dict]] = defaultdict(list)
    for row in records:
        grouped[_identity(row)].append(dict(row))

    reference_events: dict[tuple[str, int], list[dict]] = defaultdict(list)
    episode_data: list[tuple[tuple[str, int, int, int], list[dict], list[dict], int, str | None]] = []

    for identity, rows in grouped.items():
        rows.sort(key=lambda row: int(row["query"]))
        max_query = max((int(row["query"]) for row in rows), default=0)
        change_rows = [
            row
            for row in rows
            if bool(row.get("change_point")) and bool(row.get("pre_grasp", False))
        ]
        unavailable_reason: str | None = None
        true_events: list[dict] = []
        try:
            true_events = [
                _change_event(row, max_query=max_query)
                for row in change_rows
            ]
        except ValueError:
            unavailable_reason = "missing_chunk_phase_telemetry"

        task, level, requested_seed, _ = identity
        if (
            stationary_reference_level is not None
            and level == stationary_reference_level
            and unavailable_reason is None
        ):
            reference_events[(task, requested_seed)].extend(true_events)
        episode_data.append(
            (identity, rows, true_events, max_query, unavailable_reason)
        )

    rng = random.Random(rng_seed)
    schedule: list[dict] = []
    for identity, rows, true_events, max_query, unavailable_reason in sorted(episode_data):
        task, level, requested_seed, episode_seed = identity
        borrowed_reference = False

        if unavailable_reason is None and stationary_reference_level is not None and level != stationary_reference_level:
            target_events = list(reference_events.get((task, requested_seed), ()))
            borrowed_reference = True
            if not target_events:
                unavailable_reason = "missing_reference_change"
        elif unavailable_reason is None and true_events:
            # Use the episode's own true change descriptors. This exactly
            # matches reset count, normalized phase target and chunk position.
            target_events = list(true_events)
        elif unavailable_reason is None:
            # Backward-compatible zero-reset behavior for a no-change episode
            # when no cross-level stationary reference is requested.
            schedule.append(
                {
                    "task": task,
                    "level": level,
                    "seed": requested_seed,
                    "episode_seed": episode_seed,
                    "random_reset_queries": [],
                    "random_reset_events": [],
                    "unavailable": False,
                    "pre_grasp_matched": True,
                    "chunk_phase_matched": True,
                    "borrowed_reference_level": None,
                }
            )
            continue
        else:
            target_events = []

        if unavailable_reason is not None:
            schedule.append(
                {
                    "task": task,
                    "level": level,
                    "seed": requested_seed,
                    "episode_seed": episode_seed,
                    "random_reset_queries": [],
                    "random_reset_events": [],
                    "unavailable": True,
                    "unavailable_reason": unavailable_reason,
                    "pre_grasp_matched": True,
                    "chunk_phase_matched": False,
                    "borrowed_reference_level": (
                        stationary_reference_level if borrowed_reference else None
                    ),
                }
            )
            continue

        pregrasp_queries = sorted(
            {
                int(row["query"])
                for row in rows
                if bool(row.get("pre_grasp", False))
            }
        )
        true_change_queries = [] if borrowed_reference else [
            int(event["query"]) for event in true_events
        ]
        candidates = [
            query
            for query in pregrasp_queries
            if all(
                abs(query - change) > cooldown
                for change in true_change_queries
            )
        ]

        selected: list[dict] = []
        unavailable = False
        denom = max(1, max_query)
        for target in target_events:
            remaining = int(target["actions_remaining"])
            used_queries = {int(event["query"]) for event in selected}
            available = [
                query
                for query in candidates
                if query not in used_queries
                # query 0 has no preceding action chunk in which an event with
                # pending actions could occur.
                and (remaining == 0 or query > 0)
            ]
            if not available:
                unavailable = True
                selected = []
                break
            distances = [
                abs(query / denom - float(target["phase"]))
                for query in available
            ]
            best = min(distances)
            tied = [
                query
                for query, distance in zip(available, distances)
                if abs(distance - best) < 1e-12
            ]
            selected_query = rng.choice(tied)
            selected.append(
                {
                    "query": selected_query,
                    "actions_remaining": remaining,
                    "source_query": int(target["query"]),
                    "source_phase": float(target["phase"]),
                }
            )

        selected.sort(key=lambda event: (event["query"], event["actions_remaining"]))
        schedule.append(
            {
                "task": task,
                "level": level,
                "seed": requested_seed,
                "episode_seed": episode_seed,
                # Kept for audit/backward compatibility. v4 execution uses
                # random_reset_events, not query-boundary resets.
                "random_reset_queries": [
                    int(event["query"]) for event in selected
                ],
                "random_reset_events": selected,
                "unavailable": unavailable,
                "unavailable_reason": (
                    "no_chunk_phase_matched_query" if unavailable else None
                ),
                "pre_grasp_matched": True,
                "chunk_phase_matched": not unavailable,
                "borrowed_reference_level": (
                    stationary_reference_level if borrowed_reference else None
                ),
            }
        )
    return schedule
