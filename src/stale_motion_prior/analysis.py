"""Summarize paired episode result JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .statistics import paired_bootstrap


def analyze_rows(rows: list[dict], *, draws: int = 10_000) -> dict:
    by_key = {
        (r["task"], int(r["level"]), int(r["seed"]), r["condition"]): r
        for r in rows
    }
    output: dict = {}
    levels = sorted({int(r["level"]) for r in rows})
    comparisons = {
        "reset_minus_full": ("reset_change", "full"),
        "random_minus_full": ("random_reset", "full"),
        "stale_minus_full": ("stale_hold", "full"),
    }
    paired_values: dict[tuple[str, int, str, str, int], float] = {}
    for level in levels:
        base_keys = {
            (r["task"], int(r["seed"]))
            for r in rows
            if int(r["level"]) == level
        }
        for label, (left_condition, right_condition) in comparisons.items():
            for metric in ("ms", "sr"):
                differences = []
                for task, seed in sorted(base_keys):
                    left = by_key.get((task, level, seed, left_condition))
                    right = by_key.get((task, level, seed, right_condition))
                    if left and right and left.get(metric) is not None and right.get(metric) is not None:
                        difference = float(left[metric]) - float(right[metric])
                        differences.append(difference)
                        paired_values[(task, seed, label, metric, level)] = difference
                if differences:
                    output[f"{label}_{metric}_L{level}"] = paired_bootstrap(
                        differences, draws=draws
                    )
    if 1 in levels and 3 in levels:
        for label in comparisons:
            interaction = []
            identities = {
                (task, seed)
                for task, seed, comparison, metric, level in paired_values
                if comparison == label and metric == "ms" and level == 1
            }
            for task, seed in sorted(identities):
                low = paired_values.get((task, seed, label, "ms", 1))
                high = paired_values.get((task, seed, label, "ms", 3))
                if low is not None and high is not None:
                    interaction.append(high - low)
            if interaction:
                output[f"{label}_ms_interaction_L3_minus_L1"] = paired_bootstrap(
                    interaction, draws=draws
                )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, help="JSONL with task, level, seed, condition, sr, ms")
    parser.add_argument("--draws", type=int, default=10_000)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
    output = analyze_rows(rows, draws=args.draws)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0
