from dataclasses import dataclass
import numpy as np

from stale_motion_prior.intervention import HistoryIntervention, InterventionConfig, Mode


@dataclass
class Packet:
    values: tuple[int, ...]


class Buffer:
    def __init__(self): self.values = []
    def reset(self): self.values.clear()
    def push(self, frame, *, simulator_time_seconds): self.values.append(int(frame[0]))
    def observation(self):
        if not self.values: raise RuntimeError
        return Packet(tuple(self.values))


def test_reset_forgets_prechange_but_keeps_current():
    intervention = HistoryIntervention(Buffer, InterventionConfig(mode=Mode.RESET_CHANGE))
    intervention.update(np.array([1]), simulator_time_seconds=1, change_point=False)
    packet, metadata = intervention.update(np.array([2]), simulator_time_seconds=2, change_point=True)
    assert packet.values == (2,)
    assert metadata["history_reset"]


def test_stale_hold_freezes_motion_for_two_queries():
    intervention = HistoryIntervention(Buffer, InterventionConfig(mode=Mode.STALE_HOLD, stale_queries=2))
    intervention.update(np.array([1]), simulator_time_seconds=1, change_point=False)
    first, _ = intervention.update(np.array([2]), simulator_time_seconds=2, change_point=True)
    second, _ = intervention.update(np.array([3]), simulator_time_seconds=3, change_point=False)
    fresh, _ = intervention.update(np.array([4]), simulator_time_seconds=4, change_point=False)
    assert first.values == second.values == (1,)
    assert fresh.values == (1, 2, 3, 4)

