"""Post-change mechanism metrics."""

from __future__ import annotations

from collections.abc import Sequence
import math


def post_change_distance_auc(distances: Sequence[float], change_index: int, horizon: int = 3) -> float:
    values = [float(x) for x in distances[change_index + 1 : change_index + 1 + horizon]]
    values = [x for x in values if math.isfinite(x)]
    return float(sum(values) / len(values)) if values else float("nan")


def recovery_lag(distances: Sequence[float], change_index: int, consecutive: int = 2) -> float:
    run = 0
    for index in range(change_index + 1, len(distances)):
        previous, current = float(distances[index - 1]), float(distances[index])
        run = run + 1 if math.isfinite(previous) and math.isfinite(current) and current < previous else 0
        if run >= consecutive:
            return float(index - change_index - consecutive + 1)
    return float("nan")

