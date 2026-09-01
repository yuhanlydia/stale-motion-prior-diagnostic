#!/usr/bin/env python3
"""Run a JSONL command queue under a hard wall-time budget.

Each input row must contain an argv list in `command`, plus task/level/seed/
condition metadata. A completed row is appended atomically to the journal.
The active child receives SIGTERM at the deadline, then SIGKILL after grace.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def append(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def key(row: dict) -> tuple:
    return row["task"], int(row["level"]), int(row["seed"]), row["condition"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("queue", type=Path)
    parser.add_argument("--hours", type=float, default=10.0)
    parser.add_argument("--journal", type=Path, default=Path("runs/journal.jsonl"))
    parser.add_argument("--logs", type=Path, default=Path("runs/logs"))
    parser.add_argument("--grace-seconds", type=float, default=30.0)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.queue.read_text().splitlines() if line.strip()]
    completed = set()
    if args.journal.exists():
        completed = {key(json.loads(line)) for line in args.journal.read_text().splitlines() if line.strip() and json.loads(line).get("status") == "complete"}
    deadline = time.monotonic() + args.hours * 3600
    args.logs.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if key(row) in completed or time.monotonic() >= deadline:
            continue
        command = row.get("command")
        if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
            raise ValueError(f"queue row lacks argv command: {row}")
        label = "_".join(map(str, key(row)))
        log_path = args.logs / f"{label}.log"
        env = os.environ.copy()
        env.update({"SMP_MODE": row["condition"], "SMP_LOG": str(args.logs / f"{label}.jsonl")})
        if row.get("random_reset_queries"):
            env["SMP_RANDOM_RESETS"] = ",".join(map(str, row["random_reset_queries"]))
        started = time.time()
        with log_path.open("ab") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(5)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=args.grace_seconds)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        append(args.journal, {**{k: row[k] for k in ("task", "level", "seed", "condition")}, "status": "complete" if process.returncode == 0 else "failed", "returncode": process.returncode, "started_unix": started, "finished_unix": time.time(), "log": str(log_path)})
        if time.monotonic() >= deadline:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

