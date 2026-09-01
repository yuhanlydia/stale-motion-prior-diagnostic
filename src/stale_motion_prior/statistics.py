"""Paired and task-balanced bootstrap utilities."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
import numpy as np


def paired_bootstrap(differences: Iterable[float], *, draws: int = 10_000, seed: int = 0) -> dict[str, float]:
    values = np.asarray(list(differences), dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("no finite paired differences")
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"estimate": float(values.mean()), "ci_low": float(low), "ci_high": float(high), "n": int(len(values))}


def hierarchical_bootstrap(rows: Iterable[dict], *, value_key: str, draws: int = 10_000, seed: int = 0) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[value_key])
        if np.isfinite(value):
            grouped[str(row["task"])].append(value)
    tasks = sorted(grouped)
    if not tasks or any(not grouped[task] for task in tasks):
        raise ValueError("hierarchical bootstrap requires finite rows grouped by task")
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        selected = rng.choice(tasks, size=len(tasks), replace=True)
        task_means = []
        for task in selected:
            values = np.asarray(grouped[str(task)], dtype=np.float64)
            task_means.append(float(rng.choice(values, size=len(values), replace=True).mean()))
        estimates[draw] = np.mean(task_means)
    observed = float(np.mean([np.mean(grouped[t]) for t in tasks]))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"estimate": observed, "ci_low": float(low), "ci_high": float(high), "tasks": len(tasks)}

