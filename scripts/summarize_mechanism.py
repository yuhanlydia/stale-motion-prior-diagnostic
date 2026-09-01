#!/usr/bin/env python3
"""Compute per-change local distance AUC and recovery lag from query logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stale_motion_prior.metrics import post_change_distance_auc, recovery_lag


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=3)
    args = parser.parse_args()
    summaries = []
    for path in sorted(args.logs.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        distances = [float(row["ee_target_dist"]) for row in rows if "ee_target_dist" in row]
        if len(distances) != len(rows):
            continue
        for index, row in enumerate(rows):
            if not row.get("change_point") or not row.get("pre_grasp", False):
                continue
            summaries.append({
                "task": row["task"],
                "level": int(row["level"]),
                "seed": int(row["seed"]),
                "condition": row["history_mode"],
                "change_query": int(row["query"]),
                "change_angle_deg": row.get("change_angle_deg"),
                "speed_ratio": row.get("speed_ratio"),
                "distance_auc_h3": post_change_distance_auc(distances, index, args.horizon),
                "recovery_lag": recovery_lag(distances, index),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in summaries), encoding="utf-8")
    print(f"wrote {len(summaries)} change summaries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

