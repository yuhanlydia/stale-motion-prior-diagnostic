#!/usr/bin/env python3
"""Create phase queues without changing paired command/config identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("queue", type=Path)
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--levels", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.queue.read_text().splitlines() if line.strip()]
    selectors = {
        "condition": set(args.conditions or []),
        "task": set(args.tasks or []),
        "level": set(args.levels or []),
        "seed": set(args.seeds or []),
    }
    if not any(selectors.values()):
        raise ValueError("provide at least one selector")
    selected = [
        row for row in rows
        if all(not wanted or row[field] in wanted for field, wanted in selectors.items())
    ]
    for field, wanted in selectors.items():
        unknown = wanted - {row[field] for row in rows}
        if unknown:
            raise ValueError(f"{field}s absent from queue: {sorted(unknown)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected), encoding="utf-8")
    print(f"wrote {len(selected)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
