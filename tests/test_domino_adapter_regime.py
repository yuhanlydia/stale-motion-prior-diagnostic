from types import SimpleNamespace

from stale_motion_prior.change import commanded_segment_index, commanded_segment_spec


class Env:
    def __init__(self, actor, tasks):
        self.actor = actor
        self.active_kinematic_tasks = tasks

    def get_dynamic_motion_config(self):
        return {"target_actor": self.actor}


def test_reads_target_commanded_segment():
    target = object()
    other = object()
    tasks = [
        {"type": "extended_segmented", "component": SimpleNamespace(entity=other), "current_segment_idx": 9},
        {"type": "extended_segmented", "component": SimpleNamespace(entity=target), "current_segment_idx": 2},
    ]
    assert commanded_segment_index(Env(target, tasks)) == 2


def test_returns_none_for_non_segmented_motion():
    target = object()
    tasks = [{"type": "extended_velocity", "component": SimpleNamespace(entity=target)}]
    assert commanded_segment_index(Env(target, tasks)) is None


def test_matches_distinct_wrappers_for_same_named_actor():
    class Actor:
        def get_name(self):
            return "roller"

    target = Actor()
    wrapped = Actor()
    tasks = [{"type": "extended_segmented", "component": SimpleNamespace(entity=wrapped), "current_segment_idx": 1}]
    assert commanded_segment_index(Env(target, tasks)) == 1


def test_serializes_commanded_trajectory():
    target = object()
    tasks = [{
        "type": "extended_segmented",
        "component": SimpleNamespace(entity=target),
        "current_segment_idx": 0,
        "segments": [{"type": "velocity", "duration": 1.5, "velocity": [1, 2, 3], "ignored": object()}],
    }]
    assert commanded_segment_spec(Env(target, tasks)) == [
        {"type": "velocity", "duration": 1.5, "velocity": [1, 2, 3]}
    ]
