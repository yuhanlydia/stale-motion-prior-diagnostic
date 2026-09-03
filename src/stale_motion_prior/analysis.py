"""Summarize paired episode result JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .statistics import paired_bootstrap


def _design_seed(row: dict) -> int:
    """Return the experiment-design seed, not DOMINO's resolved feasible seed."""
    return int(row.get("requested_start_seed", row["seed"]))


def analyze_rows(rows: list[dict], *, draws: int = 10_000) -> dict:
    by_key = {
        (r["task"], int(r["level"]), _design_seed(r), r["condition"]): r
        for r in rows
    }
    output: dict = {}
    levels = sorted({int(r["level"]) for r in rows})
    comparisons = {
        "reset_minus_full": ("reset_change", "full"),
        "random_minus_full": ("random_reset", "full"),
        "stale_minus_full": ("stale_hold", "full"),
        "reset_minus_random": ("reset_change", "random_reset"),
    }
    paired_values: dict[tuple[str, int, str, str, int], float] = {}
    for level in levels:
        base_keys = {
            (r["task"], _design_seed(r))
            for r in rows
            if int(r["level"]) == level
        }
        for label, (left_condition, right_condition) in comparisons.items():
            for metric in ("ms", "sr"):
                differences = []
                for task, seed in sorted(base_keys):
                    left = by_key.get((task, level, seed, left_condition))
                    right = by_key.get((task, level, seed, right_condition))
                    if (
                        left
                        and right
                        and left.get(metric) is not None
                        and right.get(metric) is not None
                    ):
                        difference = float(left[metric]) - float(right[metric])
                        differences.append(difference)
                        paired_values[(task, seed, label, metric, level)] = difference
                if differences:
                    output[f"{label}_{metric}_L{level}"] = paired_bootstrap(
                        differences, draws=draws
                    )

    if 1 in levels and 3 in levels:
        identities = {
            (task, seed)
            for task, seed, comparison, metric, level in paired_values
            if comparison == "reset_minus_full" and metric == "ms" and level == 3
        }
        legacy_interaction = []
        stale_prior_reversal = []
        for task, seed in sorted(identities):
            high = paired_values.get((task, seed, "reset_minus_full", "ms", 3))
            low_reset = paired_values.get((task, seed, "reset_minus_full", "ms", 1))
            low_stationary = paired_values.get((task, seed, "random_minus_full", "ms", 1))
            if high is not None and low_reset is not None:
                legacy_interaction.append(high - low_reset)
            if high is not None and low_stationary is not None:
                stale_prior_reversal.append(high - low_stationary)
        if legacy_interaction:
            output["reset_minus_full_ms_interaction_L3_minus_L1"] = paired_bootstrap(
                legacy_interaction, draws=draws
            )
        if stale_prior_reversal:
            output[
                "stale_prior_reversal_ms_L3_change_minus_L1_stationary"
            ] = paired_bootstrap(stale_prior_reversal, draws=draws)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results", type=Path,
        help="JSONL with task, level, seed, condition, sr, ms and optional requested_start_seed",
    )
    parser.add_argument("--draws", type=int, default=10_000)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.results.read_text().splitlines()
        if line.strip()
    ]
    output = analyze_rows(rows, draws=args.draws)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0
