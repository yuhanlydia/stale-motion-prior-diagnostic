"""Strict completion predicate for expensive DOMINO runs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def completion_status(
    *, returncode: int, query_log: Path, log_text: str, new_metrics: Sequence[str]
) -> tuple[bool, str | None]:
    if returncode != 0:
        return False, f"returncode_{returncode}"
    if not query_log.is_file() or query_log.stat().st_size <= 0:
        return False, "missing_query_log"
    if "SYNC-EPISODE-SUMMARY" not in log_text:
        return False, "missing_episode_summary"
    if len(new_metrics) == 0:
        return False, "missing_official_metrics"
    if len(new_metrics) != 1:
        return False, "ambiguous_official_metrics"
    return True, None
