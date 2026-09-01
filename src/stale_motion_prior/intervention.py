"""Inference-time interventions over DynamicWAM's explicit history buffer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np


class Mode(str, Enum):
    FULL = "full"
    RESET_CHANGE = "reset_change"
    RANDOM_RESET = "random_reset"
    STALE_HOLD = "stale_hold"
    SHORT = "short"


@dataclass(frozen=True)
class InterventionConfig:
    mode: Mode = Mode.FULL
    stale_queries: int = 2
    short_effective_history: int = 1
    random_reset_queries: tuple[int, ...] = ()


class HistoryIntervention:
    """Wrap a DynamicWAM HeadFlowBuffer without changing model weights.

    The wrapped buffer must expose push(), observation(), and reset().  RESET
    clears before the current frame is pushed, retaining the current RGB while
    zero/invalid-padding all pre-change intervals. STALE_HOLD freezes only the
    motion packet; the current policy RGB remains supplied by the adapter.
    """

    def __init__(self, buffer_factory: Callable[[], Any], config: InterventionConfig):
        self._factory = buffer_factory
        self.config = config
        self.buffer = buffer_factory()
        self.query = -1
        self._stale_packet: Any | None = None
        self._stale_remaining = 0
        self._latest: tuple[np.ndarray, float] | None = None

    def reset_episode(self) -> None:
        self.buffer.reset()
        self.query = -1
        self._stale_packet = None
        self._stale_remaining = 0
        self._latest = None

    def push(self, frame: np.ndarray, *, simulator_time_seconds: float, change_point: bool) -> None:
        """Push every simulator/action observation and apply true-change resets."""
        if self.config.mode is Mode.RESET_CHANGE and change_point:
            self.buffer.reset()
        if self.config.mode is Mode.STALE_HOLD and change_point:
            try:
                self._stale_packet = self.buffer.observation()
            except RuntimeError:
                self._stale_packet = None
            self._stale_remaining = self.config.stale_queries
        frame_copy = np.ascontiguousarray(frame).copy()
        timestamp = float(simulator_time_seconds)
        self.buffer.push(frame_copy, simulator_time_seconds=timestamp)
        self._latest = (frame_copy, timestamp)

    def observation(self) -> tuple[Any, dict[str, Any]]:
        """Return one policy-query motion packet and intervention telemetry."""
        self.query += 1
        mode = self.config.mode
        random_reset = mode is Mode.RANDOM_RESET and self.query in self.config.random_reset_queries
        short_reset = mode is Mode.SHORT and self.config.short_effective_history == 0
        if random_reset or short_reset:
            if self._latest is None:
                raise RuntimeError("history observation requested before first push")
            frame, timestamp = self._latest
            self.buffer.reset()
            self.buffer.push(frame, simulator_time_seconds=timestamp)
        fresh = self.buffer.observation()
        stale_applied = bool(mode is Mode.STALE_HOLD and self._stale_packet is not None and self._stale_remaining > 0)
        packet = self._stale_packet if stale_applied else fresh
        if stale_applied:
            self._stale_remaining -= 1
        return packet, {
            "history_mode": mode.value,
            "history_reset": bool(random_reset or short_reset),
            "stale_applied": stale_applied,
            "stale_queries_remaining": self._stale_remaining,
        }

    def update(
        self,
        frame: np.ndarray,
        *,
        simulator_time_seconds: float,
        change_point: bool,
    ) -> tuple[Any, dict[str, Any]]:
        self.push(frame, simulator_time_seconds=simulator_time_seconds, change_point=change_point)
        packet, metadata = self.observation()
        if self.config.mode is Mode.RESET_CHANGE and change_point:
            metadata["history_reset"] = True
        return packet, metadata
