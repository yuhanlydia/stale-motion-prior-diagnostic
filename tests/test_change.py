import numpy as np

from stale_motion_prior.change import ChangeDetector


def test_detects_abrupt_direction_change():
    detector = ChangeDetector()
    events = [
        detector.update(query=i, position_xyz=np.array(p), time_seconds=float(i))
        for i, p in enumerate(([0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0]))
    ]
    assert events[-1].changed
    assert events[-1].direction_deg > 45


def test_stationary_linear_motion_does_not_trigger():
    detector = ChangeDetector()
    events = [
        detector.update(query=i, position_xyz=np.array([i, 0, 0]), time_seconds=float(i))
        for i in range(6)
    ]
    assert not any(event.changed for event in events)


def test_motion_start_is_speed_change_not_direction_change():
    detector = ChangeDetector()
    detector.update(query=0, position_xyz=np.array([0.0, 0, 0]), time_seconds=0.0)
    detector.update(query=1, position_xyz=np.array([0.0, 0, 0]), time_seconds=1.0)
    event = detector.update(query=2, position_xyz=np.array([1.0, 0, 0]), time_seconds=2.0)
    assert event.changed
    assert event.reason == "speed"
