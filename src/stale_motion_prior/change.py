"""Simulator-GT change-point detection, independent of policy outputs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any

import numpy as np


def commanded_segment_task(task_env: Any) -> dict[str, Any] | None:
    """Return DOMINO's active Level-3 task for the configured target."""
    try:
        actor = task_env.get_dynamic_motion_config()["target_actor"]
    except (AttributeError, KeyError, TypeError):
        return None
    supported = {
        "velocity",
        "extended_velocity",
        "trajectory",
        "extended_trajectory",
        "segmented",
        "extended_segmented",
    }
    for task in getattr(task_env, "active_kinematic_tasks", ()):
        if task.get("type") not in supported:
            continue
        component = task.get("component")
        entity = getattr(component, "entity", None)
        same_name = (
            entity is not None
            and callable(getattr(entity, "get_name", None))
            and callable(getattr(actor, "get_name", None))
            and entity.get_name() == actor.get_name()
        )
        if entity is actor or same_name:
            return task
    return None


def commanded_segment_index(task_env: Any) -> int | None:
    """Read DOMINO's ground-truth regime for an active Level-3 target."""
    task = commanded_segment_task(task_env)
    if task is None:
        return None
    return int(task.get("current_segment_idx", 0))


def commanded_segment_spec(task_env: Any) -> list[dict[str, Any]] | None:
    """Return the immutable commanded trajectory parameters in JSON form."""
    task = commanded_segment_task(task_env)
    if task is None or "segments" not in task:
        return None
    fields = ("type", "duration", "start_pos", "end_pos", "velocity", "poly_x", "poly_y")
    result: list[dict[str, Any]] = []
    for segment in task["segments"]:
        clean: dict[str, Any] = {}
        for key in fields:
            if key not in segment:
                continue
            value = segment[key]
            clean[key] = value.tolist() if hasattr(value, "tolist") else value
        result.append(clean)
    return result


@dataclass(frozen=True)
class ChangeDetectorConfig:
    direction_deg: float = 45.0
    speed_ratio: float = 0.5
    speed_epsilon: float = 1e-4
    minimum_speed: float = 1e-3
    median_window: int = 4
    cooldown_queries: int = 2


@dataclass(frozen=True)
class ChangeEvent:
    query: int
    changed: bool
    eligible: bool
    direction_deg: float | None
    speed_ratio: float | None
    speed: float | None
    reason: str | None


class ChangeDetector:
    """Detect direction/speed changes from target position and simulator time."""

    def __init__(self, config: ChangeDetectorConfig | None = None) -> None:
        self.config = config or ChangeDetectorConfig()
        self._last_position: np.ndarray | None = None
        self._last_time: float | None = None
        self._last_velocity: np.ndarray | None = None
        self._recent_speeds: deque[float] = deque(maxlen=self.config.median_window)
        self._last_change_query = -10**9

    def reset(self) -> None:
        self.__init__(self.config)

    def update(
        self,
        *,
        query: int,
        position_xyz: np.ndarray,
        time_seconds: float,
        pre_grasp: bool = True,
        in_workspace: bool = True,
    ) -> ChangeEvent:
        position = np.asarray(position_xyz, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("position_xyz must be finite shape [3]")
        now = float(time_seconds)
        if not math.isfinite(now):
            raise ValueError("time_seconds must be finite")
        if self._last_time is None:
            self._last_position, self._last_time = position.copy(), now
            return ChangeEvent(query, False, False, None, None, None, "warmup")
        dt = now - self._last_time
        if dt <= 0:
            raise ValueError("simulator time must strictly increase")
        velocity = (position - self._last_position) / dt
        speed = float(np.linalg.norm(velocity))
        angle: float | None = None
        ratio: float | None = None
        changed = False
        reason: str | None = None
        eligible = bool(
            pre_grasp
            and in_workspace
            and speed >= self.config.minimum_speed
            and self._last_velocity is not None
            and query - self._last_change_query > self.config.cooldown_queries
        )
        if self._last_velocity is not None:
            previous_speed = float(np.linalg.norm(self._last_velocity))
            denominator = previous_speed * speed + self.config.speed_epsilon
            cosine = float(np.dot(self._last_velocity, velocity) / denominator)
            angle = math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
            baseline = (
                float(np.median(self._recent_speeds))
                if self._recent_speeds
                else previous_speed
            )
            ratio = abs(speed - previous_speed) / (baseline + self.config.speed_epsilon)
            if eligible:
                direction_hit = (
                    previous_speed >= self.config.minimum_speed
                    and angle > self.config.direction_deg
                )
                speed_hit = ratio > self.config.speed_ratio
                changed = direction_hit or speed_hit
                if changed:
                    reason = "direction+speed" if direction_hit and speed_hit else (
                        "direction" if direction_hit else "speed"
                    )
                    self._last_change_query = query
        self._recent_speeds.append(speed)
        self._last_velocity = velocity
        self._last_position, self._last_time = position.copy(), now
        return ChangeEvent(query, changed, eligible, angle, ratio, speed, reason)
