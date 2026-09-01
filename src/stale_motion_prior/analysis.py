"""Summarize paired episode result JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .statistics import paired_bootstrap


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, help="JSONL with task, level, seed, condition, sr, ms")
    parser.add_argument("--draws", type=int, default=10_000)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
    by_key = {(r["task"], int(r["level"]), int(r["seed"]), r["condition"]): r for r in rows}
    output = {}
    for level in sorted({int(r["level"]) for r in rows}):
        differences = []
        base_keys = {(r["task"], int(r["seed"])) for r in rows if int(r["level"]) == level}
        for task, seed in sorted(base_keys):
            reset = by_key.get((task, level, seed, "reset_change"))
            full = by_key.get((task, level, seed, "full"))
            if reset and full and reset.get("ms") is not None and full.get("ms") is not None:
                differences.append(float(reset["ms"]) - float(full["ms"]))
        if differences:
            output[f"reset_minus_full_ms_L{level}"] = paired_bootstrap(differences, draws=args.draws)
    if "reset_minus_full_ms_L1" in output and "reset_minus_full_ms_L3" in output:
        output["interaction_L3_minus_L1"] = (
            output["reset_minus_full_ms_L3"]["estimate"] - output["reset_minus_full_ms_L1"]["estimate"]
        )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0

