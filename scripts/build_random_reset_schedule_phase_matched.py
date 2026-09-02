#!/usr/bin/env python3
"""Freeze a pre-grasp, phase-matched RANDOM_RESET schedule from FULL logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stale_motion_prior.random_reset import build_phase_matched_schedule


def _jsonl_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob('*.jsonl')))
        elif path.is_file():
            files.append(path)
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cooldown', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    records: list[dict] = []
    for path in _jsonl_files(args.inputs):
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            required = {'task', 'level', 'seed', 'query', 'pre_grasp', 'change_point'}
            if required.issubset(row):
                records.append(row)
    schedule = build_phase_matched_schedule(records, cooldown=args.cooldown, rng_seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        ''.join(json.dumps(row, sort_keys=True) + '\n' for row in schedule),
        encoding='utf-8',
    )
    available = sum(not row['unavailable'] for row in schedule)
    print(f'wrote {len(schedule)} schedules ({available} available) to {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
