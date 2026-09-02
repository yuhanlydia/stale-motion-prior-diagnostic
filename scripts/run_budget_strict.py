#!/usr/bin/env python3
"""Strict, resumable queue runner for paired DOMINO diagnostics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from stale_motion_prior.strict_completion import completion_status


def _append(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(value, sort_keys=True) + '\n')


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _eval_output_root(command: list[str]) -> Path | None:
    try:
        config = Path(command[command.index('--config') + 1])
    except (ValueError, IndexError):
        return None
    if not config.is_file():
        return None
    for line in config.read_text(encoding='utf-8').splitlines():
        if line.startswith('eval_output_root:'):
            value = line.split(':', 1)[1].strip().strip("'\"")
            return Path(value).expanduser().resolve() if value else None
    return None


def _metric_files(root: Path | None) -> set[Path]:
    if root is None or not root.exists():
        return set()
    return set(root.rglob('*_metrics.json'))


def _key(row: dict) -> tuple:
    return (str(row['task']), int(row['level']), int(row['seed']), str(row['condition']))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('queue', type=Path)
    parser.add_argument('--hours', type=float, required=True)
    parser.add_argument('--journal', type=Path, default=Path('runs/journal.jsonl'))
    parser.add_argument('--logs', type=Path, default=Path('runs/logs'))
    parser.add_argument(
        '--official-baseline', action='store_true',
        help='Accept uninstrumented official adapter logs; metrics remain mandatory.',
    )
    args = parser.parse_args()

    rows = _read_jsonl(args.queue)
    prior = _read_jsonl(args.journal)
    completed = {
        _key(row)
        for row in prior
        if row.get('status') == 'complete'
        and len(row.get('official_metrics', [])) == 1
        and Path(row['official_metrics'][0]).is_file()
    }
    deadline = time.monotonic() + max(0.0, args.hours) * 3600.0
    args.logs.mkdir(parents=True, exist_ok=True)

    for row in rows:
        if time.monotonic() >= deadline:
            break
        if _key(row) in completed:
            continue
        command = row.get('command')
        if not isinstance(command, list) or not command:
            raise ValueError(f"queue row has no argv-only command: {_key(row)}")

        stem = f"{row['task']}_{row['level']}_{row['seed']}_{row['condition']}"
        log_path = (args.logs / f'{stem}.log').resolve()
        query_log = (args.logs / f'{stem}.jsonl').resolve()
        cwd = Path(row.get('cwd', '.')).expanduser().resolve()
        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in row.get('env', {}).items()})
        env.update({
            'SMP_MODE': str(row['condition']),
            'SMP_LOG': str(query_log),
            'SMP_TASK': str(row['task']),
            'SMP_LEVEL': str(row['level']),
            'SMP_REQUESTED_SEED': str(row['seed']),
        })
        if row.get('random_reset_queries'):
            env['SMP_RANDOM_RESETS'] = ','.join(map(str, row['random_reset_queries']))
        if 'stale_queries' in row:
            env['SMP_STALE_QUERIES'] = str(row['stale_queries'])

        output_root = _eval_output_root(command)
        metrics_before = _metric_files(output_root)
        started = time.time()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('w', encoding='utf-8') as log_handle:
            process = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        log_text = log_path.read_text(encoding='utf-8', errors='replace')
        new_metrics = sorted(str(path) for path in (_metric_files(output_root) - metrics_before))
        complete, failure_reason = completion_status(
            returncode=process.returncode,
            query_log=query_log,
            log_text=log_text,
            new_metrics=new_metrics,
            require_diagnostic_telemetry=not args.official_baseline,
        )
        _append(args.journal, {
            **{k: row[k] for k in ('task', 'level', 'seed', 'condition')},
            'status': 'complete' if complete else 'failed',
            'failure_reason': failure_reason,
            'returncode': process.returncode,
            'started_unix': started,
            'finished_unix': time.time(),
            'log': str(log_path),
            'query_log': str(query_log),
            'official_metrics': new_metrics,
        })
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
