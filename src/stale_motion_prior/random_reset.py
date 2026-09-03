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
    records: Iterable[dict],
    *,
    cooldown: int = 2,
    rng_seed: int = 0,
    stationary_reference_level: int | None = None,
) -> list[dict]:
    """Build outcome-blind, pre-grasp phase-matched reset controls.

    For episodes that contain true changes, the control preserves the existing
    design: use the same reset count and the empirical task/level change-time
    phase distribution while staying outside all true-change cooldown windows.

    A stationary episode has no native change time.  When
    ``stationary_reference_level`` is provided, its reset count and normalized
    phases are borrowed from the same task/requested-seed episode at that
    reference level (Level 3 in the v3 protocol).  This creates a real
    stationary-history intervention instead of the old zero-reset no-op.
    """

    grouped: dict[tuple[str, int, int, int], list[dict]] = defaultdict(list)
    for row in records:
        grouped[_identity(row)].append(dict(row))

    phase_pool: dict[tuple[str, int], list[float]] = defaultdict(list)
    reference_phases: dict[tuple[str, int], list[float]] = defaultdict(list)
    episode_data: list[
        tuple[tuple[str, int, int, int], list[dict], list[int], int]
    ] = []

    for identity, rows in grouped.items():
        rows.sort(key=lambda r: int(r["query"]))
        max_query = max((int(r["query"]) for r in rows), default=0)
        true_changes = [
            int(r["query"])
            for r in rows
            if bool(r.get("change_point")) and bool(r.get("pre_grasp", False))
        ]
        task, level, requested_seed, _ = identity
        denom = max(1, max_query)
        for query in true_changes:
            phase = query / denom
            phase_pool[(task, level)].append(phase)
            if stationary_reference_level is not None and level == stationary_reference_level:
                reference_phases[(task, requested_seed)].append(phase)
        episode_data.append((identity, rows, true_changes, max_query))

    rng = random.Random(rng_seed)
    schedule: list[dict] = []
    for identity, rows, true_changes, max_query in sorted(episode_data):
        task, level, requested_seed, episode_seed = identity
        borrowed_reference = False

        if stationary_reference_level is not None and level != stationary_reference_level:
            target_phases = list(reference_phases.get((task, requested_seed), ()))
            borrowed_reference = True
            if not target_phases:
                schedule.append(
                    {
                        "task": task,
                        "level": level,
                        "seed": requested_seed,
                        "episode_seed": episode_seed,
                        "random_reset_queries": [],
                        "unavailable": True,
                        "pre_grasp_matched": True,
                        "borrowed_reference_level": stationary_reference_level,
                    }
                )
                continue
        elif true_changes:
            target_phases = [query / max(1, max_query) for query in true_changes]
        else:
            # Backward-compatible behavior for callers that do not request a
            # stationary reference, and for no-change reference-level episodes.
            schedule.append(
                {
                    "task": task,
                    "level": level,
                    "seed": requested_seed,
                    "episode_seed": episode_seed,
                    "random_reset_queries": [],
                    "unavailable": False,
                    "pre_grasp_matched": True,
                    "borrowed_reference_level": None,
                }
            )
            continue

        pregrasp_queries = sorted(
            {int(r["query"]) for r in rows if bool(r.get("pre_grasp", False))}
        )
        # A stationary control intentionally ignores native detector events;
        # exclude cooldown windows only for genuine same-level changes.
        excluded_changes = [] if borrowed_reference else true_changes
        candidates = [
            query
            for query in pregrasp_queries
            if all(abs(query - change) > cooldown for change in excluded_changes)
        ]
        selected: list[int] = []
        unavailable = False
        denom = max(1, max_query)
        pool = (
            target_phases
            if borrowed_reference
            else (phase_pool[(task, level)] or target_phases)
        )

        for _ in target_phases:
            available = [query for query in candidates if query not in selected]
            if not available:
                unavailable = True
                selected = []
                break
            target_phase = rng.choice(pool)
            distances = [abs(query / denom - target_phase) for query in available]
            best = min(distances)
            tied = [
                query
                for query, distance in zip(available, distances)
                if abs(distance - best) < 1e-12
            ]
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
                "borrowed_reference_level": (
                    stationary_reference_level if borrowed_reference else None
                ),
            }
        )
    return schedule
