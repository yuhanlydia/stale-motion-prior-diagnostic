"""Generate paired Gate-0 run manifests without executing training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CORE = ("full", "reset_change", "random_reset", "stale_hold")


def build_matrix(tasks: list[str], levels: list[int], seeds: list[int]) -> list[dict]:
    return [
        {"task": task, "level": level, "seed": seed, "condition": condition}
        for task in tasks
        for level in levels
        for seed in seeds
        for condition in CORE
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build_matrix(args.tasks, args.levels, args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} paired runs to {args.output}")
    return 0

