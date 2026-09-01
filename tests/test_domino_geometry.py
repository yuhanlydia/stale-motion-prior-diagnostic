import numpy as np

from stale_motion_prior.geometry import ee_geometry, is_grasped


class Robot:
    def get_left_ee_pose(self): return [0, 0, 0, 1, 0, 0, 0]
    def get_right_ee_pose(self): return [2, 0, 0, 1, 0, 0, 0]


class Env:
    robot = Robot()
    _first_target_gripper_contact_valid = np.array([False, False])


def test_ee_geometry_uses_nearest_arm():
    geometry = ee_geometry(Env(), np.array([0.5, 0, 0]))
    assert geometry["left_target_dist"] == 0.5
    assert geometry["right_target_dist"] == 1.5
    assert geometry["ee_target_dist"] == 0.5


def test_first_contact_marks_post_grasp():
    env = Env()
    env._first_target_gripper_contact_valid = np.array([True, False])
    assert is_grasped(env)
