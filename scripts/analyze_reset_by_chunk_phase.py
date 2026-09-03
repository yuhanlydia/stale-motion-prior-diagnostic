#!/usr/bin/env python3
"""Analyze whether RESET_CHANGE effects depend on chunk position."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stale_motion_prior.chunk_analysis import summarize_reset_by_chunk_phase


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _collect_logs(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        files = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
        for file_path in files:
            if file_path.is_file():
                rows.extend(_read_jsonl(file_path))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--full-logs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = summarize_reset_by_chunk_phase(
        _read_jsonl(args.results),
        _collect_logs(args.full_logs),
    )
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
