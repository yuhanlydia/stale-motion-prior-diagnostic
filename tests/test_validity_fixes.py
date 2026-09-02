from types import SimpleNamespace

import numpy as np

from stale_motion_prior.change import ChangeDetector, ChangeDetectorConfig, commanded_segment_index


class _Actor:
    def __init__(self, name='target'):
        self.name = name
    def get_name(self):
        return self.name


class _Env:
    def __init__(self, task):
        self.actor = _Actor()
        task = dict(task)
        task['component'] = SimpleNamespace(entity=self.actor)
        self.active_kinematic_tasks = [task]
    def get_dynamic_motion_config(self):
        return {'target_actor': self.actor}


def test_commanded_segment_index_requires_explicit_regime_index():
    env = _Env({'type': 'extended_velocity'})
    assert commanded_segment_index(env) is None


def test_commanded_segment_index_uses_explicit_regime_index():
    env = _Env({'type': 'segmented', 'current_segment_idx': 2, 'segments': [{}, {}, {}]})
    assert commanded_segment_index(env) == 2


def test_low_speed_same_direction_is_not_a_fake_turn():
    detector = ChangeDetector(ChangeDetectorConfig(direction_deg=45.0, minimum_speed=1e-3))
    detector.update(query=0, position_xyz=np.array([0.0, 0.0, 0.0]), time_seconds=0.0)
    detector.update(query=1, position_xyz=np.array([0.01, 0.0, 0.0]), time_seconds=1.0)
    event = detector.update(query=2, position_xyz=np.array([0.02, 0.0, 0.0]), time_seconds=2.0)
    assert event.direction_deg is not None
    assert event.direction_deg < 1.0
    assert not event.changed


def test_low_speed_reverse_is_detected_as_direction_change():
    detector = ChangeDetector(ChangeDetectorConfig(direction_deg=45.0, minimum_speed=1e-3))
    detector.update(query=0, position_xyz=np.array([0.0, 0.0, 0.0]), time_seconds=0.0)
    detector.update(query=1, position_xyz=np.array([0.01, 0.0, 0.0]), time_seconds=1.0)
    event = detector.update(query=2, position_xyz=np.array([0.0, 0.0, 0.0]), time_seconds=2.0)
    assert event.direction_deg is not None
    assert event.direction_deg > 179.0
    assert event.changed
