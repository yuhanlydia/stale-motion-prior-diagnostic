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
    # Legacy query-boundary reset support. v4 controls use environment-step
    # random reset events so that post-reset history rebuild time is matched.
    random_reset_queries: tuple[int, ...] = ()


def parse_random_reset_events(raw: str) -> tuple[tuple[int, int], ...]:
    """Parse ``query:actions_remaining`` event specifications.

    A v4 random reset occurs inside the action chunk leading to ``query`` when
    the pending action count equals ``actions_remaining``.  This matches the
    exact point at which a real dynamics change reset would have happened.
    """
    events: list[tuple[int, int]] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            query_text, remaining_text = token.split(":", 1)
            query = int(query_text)
            remaining = int(remaining_text)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"invalid random reset event {token!r}; expected query:actions_remaining"
            ) from exc
        if query < 0 or remaining < 0:
            raise ValueError("random reset event indices must be non-negative")
        events.append((query, remaining))
    if len(events) != len(set(events)):
        raise ValueError("duplicate random reset events are not allowed")
    return tuple(sorted(events))


class HistoryIntervention:
    """Wrap a DynamicWAM HeadFlowBuffer without changing model weights.

    RESET_CHANGE clears the history at the simulator step where the regime
    changes, then preserves every subsequent observation before the next policy
    query.  v4 RANDOM_RESET uses the same environment-step semantics through
    ``random_reset_point``.  This avoids the old confound where random controls
    reset immediately before inference and therefore had zero rebuild time.

    STALE_HOLD freezes only the motion packet; the current policy RGB remains
    supplied by the adapter.
    """

    def __init__(self, buffer_factory: Callable[[], Any], config: InterventionConfig):
        self._factory = buffer_factory
        self.config = config
        self.buffer = buffer_factory()
        self.query = -1
        self._stale_packet: Any | None = None
        self._stale_remaining = 0
        self._latest: tuple[np.ndarray, float] | None = None
        self._change_reset_pending = False
        self._random_reset_pending = False

    def reset_episode(self) -> None:
        self.buffer.reset()
        self.query = -1
        self._stale_packet = None
        self._stale_remaining = 0
        self._latest = None
        self._change_reset_pending = False
        self._random_reset_pending = False

    def push(
        self,
        frame: np.ndarray,
        *,
        simulator_time_seconds: float,
        change_point: bool,
        random_reset_point: bool = False,
    ) -> None:
        """Push every simulator/action observation and apply event-step resets."""
        if self.config.mode is Mode.RESET_CHANGE and change_point:
            self.buffer.reset()
            self._change_reset_pending = True
        if self.config.mode is Mode.RANDOM_RESET and random_reset_point:
            self.buffer.reset()
            self._random_reset_pending = True
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
        # Backward compatibility only. v4 schedules leave this empty and reset
        # through push(random_reset_point=True) at a matched environment step.
        legacy_random_reset = (
            mode is Mode.RANDOM_RESET
            and self.query in self.config.random_reset_queries
        )
        short_reset = mode is Mode.SHORT and self.config.short_effective_history == 0
        if legacy_random_reset or short_reset:
            if self._latest is None:
                raise RuntimeError("history observation requested before first push")
            frame, timestamp = self._latest
            self.buffer.reset()
            self.buffer.push(frame, simulator_time_seconds=timestamp)
        fresh = self.buffer.observation()
        stale_applied = bool(
            mode is Mode.STALE_HOLD
            and self._stale_packet is not None
            and self._stale_remaining > 0
        )
        packet = self._stale_packet if stale_applied else fresh
        if stale_applied:
            self._stale_remaining -= 1
        metadata = {
            "history_mode": mode.value,
            "history_reset": bool(
                legacy_random_reset
                or short_reset
                or self._change_reset_pending
                or self._random_reset_pending
            ),
            "random_reset_event_applied": bool(self._random_reset_pending),
            "stale_applied": stale_applied,
            "stale_queries_remaining": self._stale_remaining,
        }
        self._change_reset_pending = False
        self._random_reset_pending = False
        return packet, metadata

    def update(
        self,
        frame: np.ndarray,
        *,
        simulator_time_seconds: float,
        change_point: bool,
        random_reset_point: bool = False,
    ) -> tuple[Any, dict[str, Any]]:
        self.push(
            frame,
            simulator_time_seconds=simulator_time_seconds,
            change_point=change_point,
            random_reset_point=random_reset_point,
        )
        packet, metadata = self.observation()
        if self.config.mode is Mode.RESET_CHANGE and change_point:
            metadata["history_reset"] = True
        if self.config.mode is Mode.RANDOM_RESET and random_reset_point:
            metadata["history_reset"] = True
            metadata["random_reset_event_applied"] = True
        return packet, metadata
