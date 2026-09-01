#!/usr/bin/env python3
"""Create phase queues without changing paired command/config identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("queue", type=Path)
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    wanted = set(args.conditions)
    rows = [json.loads(line) for line in args.queue.read_text().splitlines() if line.strip()]
    selected = [row for row in rows if row["condition"] in wanted]
    unknown = wanted - {row["condition"] for row in rows}
    if unknown:
        raise ValueError(f"conditions absent from queue: {sorted(unknown)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected), encoding="utf-8")
    print(f"wrote {len(selected)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

