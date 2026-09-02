#!/usr/bin/env python3
"""Inject a frozen random-reset schedule into random-reset queue rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("queue", type=Path)
    parser.add_argument("schedule", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    schedules = {}
    for line in args.schedule.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            schedules[(row["task"], int(row["level"]), int(row["seed"]))] = list(row["random_reset_queries"])
    output = []
    missing = []
    for line in args.queue.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["condition"] == "random_reset":
            identity = (row["task"], int(row["level"]), int(row["seed"]))
            if identity not in schedules:
                missing.append(identity)
                continue
            if not schedules[identity]:
                # No safe query exists for this episode; omit it from the
                # random-reset control rather than silently running FULL.
                continue
            row["random_reset_queries"] = schedules[identity]
        output.append(row)
    if missing:
        raise RuntimeError(f"missing frozen schedules for {sorted(set(missing))}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in output), encoding="utf-8")
    print(f"wrote {len(output)} scheduled rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
