"""Phase-matched random-reset controls built only from FULL query telemetry."""

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


def build_phase_matched_schedule(
    records: Iterable[dict], *, cooldown: int = 2, rng_seed: int = 0
) -> list[dict]:
    """Build pre-grasp random resets matched to the empirical change-time phase.

    The function consumes only FULL query telemetry.  It never reads outcomes.
    For each task/level, true pre-grasp change phases define the target phase
    distribution.  Random controls are selected only from pre-grasp queries,
    outside every true-change cooldown window, and without replacement.
    """

    grouped: dict[tuple[str, int, int, int], list[dict]] = defaultdict(list)
    for row in records:
        grouped[_identity(row)].append(dict(row))

    phase_pool: dict[tuple[str, int], list[float]] = defaultdict(list)
    episode_data: list[tuple[tuple[str, int, int, int], list[dict], list[int], int]] = []

    for identity, rows in grouped.items():
        rows.sort(key=lambda r: int(r["query"]))
        max_query = max((int(r["query"]) for r in rows), default=0)
        true_changes = [
            int(r["query"])
            for r in rows
            if bool(r.get("change_point")) and bool(r.get("pre_grasp", False))
        ]
        task, level, _, _ = identity
        denom = max(1, max_query)
        for query in true_changes:
            phase_pool[(task, level)].append(query / denom)
        episode_data.append((identity, rows, true_changes, max_query))

    rng = random.Random(rng_seed)
    schedule: list[dict] = []
    for identity, rows, true_changes, max_query in sorted(episode_data):
        if not true_changes:
            continue
        task, level, requested_seed, episode_seed = identity
        pregrasp_queries = sorted(
            {int(r["query"]) for r in rows if bool(r.get("pre_grasp", False))}
        )
        candidates = [
            query
            for query in pregrasp_queries
            if all(abs(query - change) > cooldown for change in true_changes)
        ]
        selected: list[int] = []
        unavailable = False
        denom = max(1, max_query)
        pool = phase_pool[(task, level)] or [change / denom for change in true_changes]

        for _ in true_changes:
            available = [query for query in candidates if query not in selected]
            if not available:
                unavailable = True
                selected = []
                break
            target_phase = rng.choice(pool)
            distances = [abs(query / denom - target_phase) for query in available]
            best = min(distances)
            tied = [q for q, d in zip(available, distances) if abs(d - best) < 1e-12]
            selected.append(rng.choice(tied))

        schedule.append(
            {
                "task": task,
                "level": level,
                "seed": requested_seed,
                "episode_seed": episode_seed,
                "random_reset_queries": sorted(selected),
                "unavailable": unavailable,
                "pre_grasp_matched": True,
            }
        )
    return schedule
