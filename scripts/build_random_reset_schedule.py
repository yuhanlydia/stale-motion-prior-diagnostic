#!/usr/bin/env python3
"""Freeze matched random-reset queries from FULL per-query logs.

For each episode, draw normalized change times from other FULL episodes in the
same task/level stratum, map them to that episode length, and reject candidates
within the detector cooldown of a true change. Outcome metrics are never read.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--cooldown", type=int, default=2)
    args = parser.parse_args()
    episodes = []
    distribution = defaultdict(list)
    for path in sorted(args.logs.glob("*_full.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            continue
        identity = (
            rows[0]["task"],
            int(rows[0]["level"]),
            int(rows[0]["requested_start_seed"]),
        )
        episode_seed = int(rows[0]["seed"])
        changes = [int(row["query"]) for row in rows if row.get("change_point") and row.get("pre_grasp", False)]
        length = max(int(row["query"]) for row in rows) + 1
        normalized = [query / max(1, length - 1) for query in changes]
        distribution[identity[:2]].extend(normalized)
        episodes.append((identity, episode_seed, length, changes))
    rng = random.Random(args.seed)
    schedule = []
    for (task, level, requested_seed), episode_seed, length, true_changes in episodes:
        pool = distribution[(task, level)]
        selected = []
        used_fallback = False
        for _ in true_changes:
            candidates = list(pool)
            rng.shuffle(candidates)
            chosen = None
            for normalized in candidates:
                query = min(length - 1, max(0, round(normalized * max(1, length - 1))))
                if all(abs(query - change) > args.cooldown for change in true_changes) and query not in selected:
                    chosen = query
                    break
            if chosen is None:
                # A degenerate change-time distribution can leave no matched
                # quantile outside the cooldown window.  Keep the control
                # usable, but make this deviation explicit in the manifest.
                safe = [
                    query for query in range(length)
                    if all(abs(query - change) > args.cooldown for change in true_changes)
                    and query not in selected
                ]
                if not safe:
                    # This very short episode has no query outside the true
                    # change cooldown; it cannot support the control.
                    used_fallback = True
                    selected = []
                    break
                chosen = safe[rng.randrange(len(safe))]
                used_fallback = True
            selected.append(chosen)
        schedule.append({"task": task, "level": level, "seed": requested_seed, "episode_seed": episode_seed, "random_reset_queries": sorted(selected), "fallback": used_fallback, "unavailable": not selected})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in schedule), encoding="utf-8")
    print(f"wrote {len(schedule)} frozen episode schedules to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
