#!/usr/bin/env python3
"""Harvest one-episode DOMINO SR/MS outputs for paired analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("queue", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    harvested = []
    missing = []
    for line in args.queue.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        command = row["command"]
        try:
            config_path = Path(command[command.index("--config") + 1])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"queue command lacks --config: {command}") from exc
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        result_root = Path(config["eval_output_root"])
        metrics_paths = sorted(result_root.glob("**/_metrics.json"))
        episode_paths = sorted(result_root.glob("**/_episodes_detail.json"))
        if len(metrics_paths) != 1 or len(episode_paths) != 1:
            missing.append({"task": row["task"], "level": row["level"], "seed": row["seed"], "condition": row["condition"], "metrics_files": len(metrics_paths), "episode_files": len(episode_paths)})
            continue
        summary = json.loads(metrics_paths[0].read_text(encoding="utf-8"))
        episodes = json.loads(episode_paths[0].read_text(encoding="utf-8"))
        if summary.get("total_episodes") != 1 or len(episodes) != 1:
            raise RuntimeError(f"expected exactly one episode under {result_root}")
        episode = episodes[0]
        if int(episode["seed"]) != int(row["seed"]):
            raise RuntimeError(f"reported seed mismatch for {result_root}: {episode['seed']} != {row['seed']}")
        harvested.append({
            "task": row["task"],
            "level": int(row["level"]),
            "seed": int(row["seed"]),
            "condition": row["condition"],
            "sr": 100.0 if bool(episode["success"]) else 0.0,
            "ms": float(episode["manipulation_score"]),
            "route_completion": float(episode["route_completion"]),
            "fail_reason": episode.get("fail_reason"),
            "metrics_path": str(metrics_paths[0]),
            "episodes_path": str(episode_paths[0]),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in harvested), encoding="utf-8")
    missing_path = args.output.with_suffix(".missing.json")
    missing_path.write_text(json.dumps(missing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"harvested={len(harvested)} missing={len(missing)} output={args.output}")
    return 0 if not missing else 3


if __name__ == "__main__":
    raise SystemExit(main())

