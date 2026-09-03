#!/usr/bin/env python3
"""Publish a small, tracked audit summary from gitignored DOMINO run artifacts."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from stale_motion_prior.analysis import analyze_rows


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _log_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.jsonl")))
        elif path.is_file():
            files.append(path)
    return files


def _event_coverage(paths: list[Path]) -> dict:
    episodes: dict[tuple, list[dict]] = defaultdict(list)
    for path in _log_files(paths):
        for row in _read_jsonl(path):
            if row.get("history_mode") not in (None, "full"):
                continue
            key = (
                str(row.get("task", "unknown")),
                int(row.get("level", -1)),
                int(row.get("requested_start_seed", row.get("seed", -1))),
                int(row.get("seed", -1)),
            )
            episodes[key].append(row)
    by_level: dict[int, dict[str, int | float]] = {}
    for level in sorted({key[1] for key in episodes}):
        selected = [rows for key, rows in episodes.items() if key[1] == level]
        with_change = sum(
            any(bool(row.get("change_point")) and bool(row.get("pre_grasp", False)) for row in rows)
            for rows in selected
        )
        total = len(selected)
        by_level[level] = {
            "episodes": total,
            "episodes_with_pregrasp_change": with_change,
            "coverage": (with_change / total if total else 0.0),
        }
    return {str(level): value for level, value in by_level.items()}


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _markdown(summary: dict) -> str:
    lines = [
        f"# {summary['label']}",
        "",
        f"- generated_utc: `{summary['generated_utc']}`",
        f"- git_sha: `{summary.get('git_sha')}`",
        f"- result_rows: {summary['result_rows']}",
        "",
        "## Paired DOMINO analysis",
        "",
        "| Statistic | Estimate | 95% CI | n |",
        "|---|---:|---:|---:|",
    ]
    for key, value in sorted(summary["analysis"].items()):
        lines.append(
            f"| `{key}` | {value['estimate']:.6g} | "
            f"[{value['ci_low']:.6g}, {value['ci_high']:.6g}] | {value.get('n', '')} |"
        )
    if summary["event_coverage"]:
        lines.extend(["", "## FULL event coverage", "", "| Level | Episodes | With pre-grasp change | Coverage |", "|---|---:|---:|---:|"])
        for level, value in sorted(summary["event_coverage"].items(), key=lambda x: int(x[0])):
            lines.append(
                f"| L{level} | {value['episodes']} | "
                f"{value['episodes_with_pregrasp_change']} | {value['coverage']:.1%} |"
            )
    lines.extend([
        "",
        "## Interpretation guardrail",
        "",
        "This file is a compact audit artifact. Raw simulator runs remain gitignored. "
        "Do not change preregistered thresholds or task subsets after reading these outcomes and then relabel the same run confirmatory.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--full-logs", nargs="*", type=Path, default=[])
    parser.add_argument("--label", required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path, default=Path("published_results"))
    args = parser.parse_args()

    rows = _read_jsonl(args.results)
    summary = {
        "label": args.label,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "result_rows": len(rows),
        "analysis": analyze_rows(rows, draws=args.draws),
        "event_coverage": _event_coverage(args.full_logs),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.label}.json"
    md_path = args.output_dir / f"{args.label}.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(summary), encoding="utf-8")
    print(f"wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
