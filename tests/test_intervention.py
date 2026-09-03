from dataclasses import dataclass
import numpy as np
import pytest

from stale_motion_prior.intervention import (
    HistoryIntervention,
    InterventionConfig,
    Mode,
    parse_random_reset_events,
)


@dataclass
class Packet:
    values: tuple[int, ...]


class Buffer:
    def __init__(self):
        self.values = []

    def reset(self):
        self.values.clear()

    def push(self, frame, *, simulator_time_seconds):
        self.values.append(int(frame[0]))

    def observation(self):
        if not self.values:
            raise RuntimeError
        return Packet(tuple(self.values))


def test_reset_forgets_prechange_but_keeps_current():
    intervention = HistoryIntervention(
        Buffer,
        InterventionConfig(mode=Mode.RESET_CHANGE),
    )
    intervention.update(
        np.array([1]), simulator_time_seconds=1, change_point=False
    )
    packet, metadata = intervention.update(
        np.array([2]), simulator_time_seconds=2, change_point=True
    )
    assert packet.values == (2,)
    assert metadata["history_reset"]


def test_stale_hold_freezes_motion_for_two_queries():
    intervention = HistoryIntervention(
        Buffer,
        InterventionConfig(mode=Mode.STALE_HOLD, stale_queries=2),
    )
    intervention.update(
        np.array([1]), simulator_time_seconds=1, change_point=False
    )
    first, _ = intervention.update(
        np.array([2]), simulator_time_seconds=2, change_point=True
    )
    second, _ = intervention.update(
        np.array([3]), simulator_time_seconds=3, change_point=False
    )
    fresh, _ = intervention.update(
        np.array([4]), simulator_time_seconds=4, change_point=False
    )
    assert first.values == second.values == (1,)
    assert fresh.values == (1, 2, 3, 4)


def test_change_reset_is_latched_until_policy_query():
    intervention = HistoryIntervention(
        Buffer,
        InterventionConfig(mode=Mode.RESET_CHANGE),
    )
    intervention.push(
        np.array([1]), simulator_time_seconds=1, change_point=False
    )
    intervention.push(
        np.array([2]), simulator_time_seconds=2, change_point=True
    )
    intervention.push(
        np.array([3]), simulator_time_seconds=3, change_point=False
    )
    packet, metadata = intervention.observation()
    assert packet.values == (2, 3)
    assert metadata["history_reset"]


def test_random_reset_can_happen_mid_chunk_and_rebuild_before_query():
    intervention = HistoryIntervention(
        Buffer,
        InterventionConfig(mode=Mode.RANDOM_RESET),
    )
    intervention.push(
        np.array([1]), simulator_time_seconds=1, change_point=False
    )
    intervention.push(
        np.array([2]),
        simulator_time_seconds=2,
        change_point=False,
        random_reset_point=True,
    )
    intervention.push(
        np.array([3]), simulator_time_seconds=3, change_point=False
    )
    packet, metadata = intervention.observation()
    assert packet.values == (2, 3)
    assert metadata["history_reset"]
    assert metadata["random_reset_event_applied"]


def test_parse_random_reset_events():
    assert parse_random_reset_events("3:7,5:0") == ((3, 7), (5, 0))
    with pytest.raises(ValueError):
        parse_random_reset_events("3")
    with pytest.raises(ValueError):
        parse_random_reset_events("-1:4")
