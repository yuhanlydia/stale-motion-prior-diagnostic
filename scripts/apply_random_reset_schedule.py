#!/usr/bin/env python3
"""Inject a frozen random-reset schedule into random-reset queue rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _encode_events(events: list[dict]) -> str:
    return ",".join(
        f"{int(event['query'])}:{int(event['actions_remaining'])}"
        for event in events
    )


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
            key = (row["task"], int(row["level"]), int(row["seed"]))
            schedules[key] = {
                "queries": list(row.get("random_reset_queries", [])),
                "events": list(row.get("random_reset_events", [])),
                "unavailable": bool(row.get("unavailable", False)),
            }
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
            schedule = schedules[identity]
            if schedule["unavailable"]:
                # No valid matched event exists; omit this control rather than
                # silently running FULL.
                continue
            if schedule["events"]:
                # v4 path: execute at matched environment-step/chunk positions.
                row.setdefault("env", {})["SMP_RANDOM_RESET_EVENTS"] = (
                    _encode_events(schedule["events"])
                )
                row["random_reset_events"] = schedule["events"]
                row.pop("random_reset_queries", None)
            else:
                # Legacy/no-event compatibility. A genuine zero-reset schedule
                # remains a no-op; old schedules may still use query resets.
                row["random_reset_queries"] = schedule["queries"]
        output.append(row)
    if missing:
        raise RuntimeError(
            f"missing frozen schedules for {sorted(set(missing))}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output),
        encoding="utf-8",
    )
    print(f"wrote {len(output)} scheduled rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
